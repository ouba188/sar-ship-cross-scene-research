"""Bounded utility644/bank64 algebraic diagnostic; no classifier or selection."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util
import json
import time
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ['MPLCONFIGDIR'] = str(HERE / '.mplcache')
spec = importlib.util.spec_from_file_location('diagnostic_edge', ROOT / 'edge_difference/run.py')
edge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(edge)


def load():
    table = pd.read_csv(ROOT.parent / 'regional_utility_pilot/upstream/table.csv',
                        dtype={'mmsi': str, 'instance_id': str})
    utility = np.flatnonzero(table.role.eq('utility_train'))
    proj = np.load(ROOT / 'projection_layer1.npz')
    bank = proj['bank'].ravel().astype(int)
    ids = np.unique(np.r_[utility, bank])
    with np.load(ROOT / 'tokens.npz') as z:
        # Detect an index mismatch before attributing tokens to ships or classes.
        np.testing.assert_array_equal(z['instance_ids'].astype(str), table.instance_id)
        raw, mass = z['layer1'][ids], z['mass'][ids]
    x = edge.support.unit((raw - proj['mean']) @ proj['components'].T).astype(np.float32)
    x[mass <= 0] = 0
    t = table.iloc[ids].copy().reset_index(names='table_index')
    assert len(utility) == 644 and len(np.unique(bank)) == 64
    return table, t, x, mass, np.searchsorted(ids, utility), np.searchsorted(ids, bank)


def reconstruction(b, n):
    m = len(b)
    errors = dict(ridge_sse=0., mean_sse=0., nearest_body_sse=0.)
    span = []
    for j in range(m):
        keep = np.arange(m) != j
        bt, nt = b[keep], n[keep]
        bm, nm = bt.mean(0), nt.mean(0)
        bt, nt = bt - bm, nt - nm
        k = len(bt)
        lam = np.sum(bt * bt) / k**2
        delta = b[j] - bm
        if lam > 0:
            # Dual form of (Cbb + lambda I)^-1 Cbn, with fold-only centering.
            pred = nm + delta @ bt.T @ np.linalg.solve(bt @ bt.T + k * lam * np.eye(k), nt)
        else:
            pred = nm
        near = np.argmin(np.sum((b[keep] - b[j])**2, axis=1))
        for name, value in [('ridge', pred), ('mean', nm), ('nearest_body', n[keep][near])]:
            errors[name + '_sse'] += float(np.sum((n[j] - value)**2))
        _, s, vt = np.linalg.svd(bt, full_matrices=False)
        rank = int(np.sum(s > s[0] * 1e-8)) if s[0] > 0 else 0
        span.append(float(np.sum((delta @ vt[:rank].T)**2) / max(np.sum(delta**2), 1e-30)))
    errors['ridge_skill'] = 1 - errors['ridge_sse'] / errors['mean_sse']
    errors['nearest_body_skill'] = 1 - errors['nearest_body_sse'] / errors['mean_sse']
    errors['held_body_in_train_span'] = np.mean(span)
    return errors


def kernel_report(t, qids, rids, cov, spectra, norms, valid, group, source):
    q, r = t.iloc[qids], t.iloc[rids]
    allowed = ((q.mmsi.to_numpy()[:, None] != r.mmsi.to_numpy()[None]) &
               (q.pass_group.to_numpy()[:, None] != r.pass_group.to_numpy()[None]) &
               valid[qids, None] & valid[rids][None])
    raw = cov[qids] @ cov[rids].T
    amplitude = norms[qids, None] * norms[rids][None]
    cosine = np.divide(raw, amplitude, out=np.zeros_like(raw), where=amplitude > 0)
    spectral = spectra[qids] @ spectra[rids].T
    spectral_cosine = np.divide(spectral, amplitude, out=np.zeros_like(raw), where=amplitude > 0)
    kernels = dict(raw=raw, cosine=cosine, amplitude_only=amplitude,
                   singular_values=spectral, singular_values_cosine=spectral_cosine)
    counts = np.stack([(allowed & (r.class_id.to_numpy()[None] == c)).sum(1) for c in range(8)], 1)
    eligible = (counts > 0).all(1)
    coverage = []
    for j, qi in enumerate(qids):
        for c in range(8):
            coverage.append(dict(group=group, reference_source=source, query_obs=int(t.iloc[qi].obs_id),
                query_class=int(t.iloc[qi].class_id), reference_class=c,
                valid_references=int(counts[j, c]), query_valid=bool(valid[qi]), eligible=bool(eligible[j])))
    summary, gaps = [], []
    for name, kernel in kernels.items():
        means = np.stack([np.divide((kernel * allowed * (r.class_id.to_numpy()[None] == c)).sum(1),
                                   counts[:, c], out=np.full(len(qids), np.nan), where=counts[:, c] > 0)
                          for c in range(8)], 1)
        y = q.class_id.to_numpy(int)
        same = means[np.arange(len(qids)), y]
        gap = same - (means.sum(1) - same) / 7
        rms = np.sqrt(np.mean(kernel[allowed]**2))
        classgaps = []
        for c in range(8):
            mask = eligible & (y == c)
            v = float(gap[mask].mean()) if mask.any() else np.nan
            classgaps.append(v)
            summary.append(dict(group=group, reference_source=source, kernel=name, class_id=c,
                eligible_queries=int(mask.sum()), same_minus_other=v, kernel_rms=rms,
                gap_over_rms=v/rms, valid_pairs=int(allowed[mask].sum())))
        summary.append(dict(group=group, reference_source=source, kernel=name, class_id='macro',
            eligible_queries=int(eligible.sum()), same_minus_other=float(np.mean(classgaps)),
            kernel_rms=rms, gap_over_rms=float(np.mean(classgaps))/rms, valid_pairs=int(allowed.sum())))
        for j in np.flatnonzero(eligible):
            gaps.append(dict(group=group, reference_source=source, kernel=name,
                query_obs=int(q.iloc[j].obs_id), query_class=int(y[j]), same_minus_other=float(gap[j])))
    return summary, coverage, gaps, eligible, (raw, cosine, allowed)


def main():
    tick = time.perf_counter()
    table, t, x, mass, qids, rids = load()
    b, n, legal, _ = edge.edges(x, mass)
    rows, summaries, coverage, gaps, distributions = [], [], [], [], []
    groupdata = {}
    for g, group in enumerate(edge.GROUPS):
        cov, spectra = np.zeros((len(t), 1024)), np.zeros((len(t), 32))
        valid = np.zeros(len(t), bool)
        for i in range(len(t)):
            keep = legal[i, g]
            m = int(keep.sum())
            if m < 2:
                continue
            bb, nn = b[i, g, keep].astype(float), n[i, g, keep].astype(float)
            bc, nc = bb - bb.mean(0), nn - nn.mean(0)
            cbn, cbb = bc.T @ nc / m, bc.T @ bc / m
            s = np.linalg.svd(cbn, compute_uv=False)
            e = np.maximum(np.linalg.eigvalsh(cbb), 0)
            lam = np.trace(cbb) / m
            operator = np.linalg.solve(cbb + lam*np.eye(32), cbn)
            cov[i], spectra[i], valid[i] = cbn.ravel(), s, np.linalg.norm(cbn) > 1e-15
            rec = dict(group=group, local_index=i, table_index=int(t.iloc[i].table_index),
                obs_id=int(t.iloc[i].obs_id), class_id=int(t.iloc[i].class_id),
                class_name=t.iloc[i].class_name, is_utility=bool(i in qids), is_bank=bool(i in rids),
                legal_edges=m, crosscov_norm=float(np.linalg.norm(cbn)),
                crosscov_rank=int(np.sum(s > s[0]*1e-8)),
                body_rank=int(np.sum(e > e[-1]*1e-8)),
                body_effective_rank=float(e.sum()**2 / (e@e)),
                crosscov_effective_rank=float(np.sum(s*s)**2 / np.sum(s**4)),
                top_singular_energy=float(s[0]**2 / np.sum(s*s)), ridge_lambda=lam,
                ridge_effective_df=float(np.sum(e / (e+lam))),
                operator_norm=float(np.linalg.norm(operator)))
            if i in qids:
                rec.update(reconstruction(bb, nn))
            rows.append(rec)
        norms = np.linalg.norm(cov, axis=1)
        for source, refs in [('utility', qids), ('bank64', rids)]:
            su, co, ga, eligible, kernels = kernel_report(t, qids, refs, cov, spectra, norms, valid, group, source)
            summaries.extend(su); coverage.extend(co); gaps.extend(ga)
            for c in list(range(8)) + ['all']:
                take = eligible & ((t.iloc[qids].class_id.to_numpy() == c) if c != 'all' else True)
                values = norms[qids[take]]
                distributions.append(dict(group=group, population='eligible_query_to_' + source, class_id=c,
                    n=len(values), mean=values.mean(), p10=np.quantile(values, .1),
                    p50=np.median(values), p90=np.quantile(values, .9)))
            if source == 'bank64':
                groupdata[group] = kernels
        for source, ids in [('utility_valid', qids), ('bank_valid', rids)]:
            for c in list(range(8)) + ['all']:
                take = valid[ids] & ((t.iloc[ids].class_id.to_numpy() == c) if c != 'all' else True)
                values = norms[ids[take]]
                distributions.append(dict(group=group, population=source, class_id=c, n=len(values),
                    mean=values.mean(), p10=np.quantile(values, .1), p50=np.median(values), p90=np.quantile(values, .9)))
        print('completed', group, flush=True)
    images = pd.DataFrame(rows)
    ks = pd.DataFrame(summaries)
    for name, data in [('image_diagnostics', images), ('class_kernels', ks),
                       ('reference_coverage', coverage), ('query_gaps', gaps), ('norm_distributions', distributions)]:
        pd.DataFrame(data).to_csv(HERE / (name + '.csv'), index=False)
    rs = []
    u = images[images.is_utility]
    for group, frame in u.groupby('group', sort=False):
        for c in list(range(8)) + ['macro', 'pooled']:
            f = frame if isinstance(c, str) else frame[frame.class_id == c]
            def agg(col):
                return f.groupby('class_id')[col].mean().mean() if c == 'macro' else f[col].mean()
            rs.append(dict(group=group, class_id=c, n=len(f), ridge_skill_mean=agg('ridge_skill'),
                ridge_skill_median=f.ridge_skill.median(), positive_fraction=agg('positive') if 'positive' in f else
                (f.assign(pos=f.ridge_skill > 0).groupby('class_id').pos.mean().mean() if c == 'macro' else (f.ridge_skill > 0).mean()),
                nearest_body_skill_mean=agg('nearest_body_skill'),
                ridge_skill_ratio_of_sse=1-f.ridge_sse.sum()/f.mean_sse.sum(),
                nearest_skill_ratio_of_sse=1-f.nearest_body_sse.sum()/f.mean_sse.sum(),
                body_rank_mean=agg('body_rank'), body_effective_rank_mean=agg('body_effective_rank'),
                ridge_df_mean=agg('ridge_effective_df'), held_body_span_mean=agg('held_body_in_train_span')))
    pd.DataFrame(rs).to_csv(HERE / 'reconstruction_summary.csv', index=False)
    cases = make_panels(table, t, qids, rids, images, groupdata, mass)
    receipt = dict(status='complete', utility_images=len(qids), frozen_bank_images=len(rids),
        union_images=len(t), utility_mmsi=int(t.iloc[qids].mmsi.nunique()), utility_passes=int(t.iloc[qids].pass_group.nunique()),
        bank_mmsi=int(t.iloc[rids].mmsi.nunique()), bank_passes=int(t.iloc[rids].pass_group.nunique()),
        seconds=time.perf_counter()-tick, no_training=True, no_evaluation=True, no_git=True,
        kernel_macro=ks[ks.class_id.eq('macro')].to_dict('records'), cases=cases)
    (HERE / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(ks[ks.class_id.eq('macro')][['group','reference_source','kernel','same_minus_other','gap_over_rms','eligible_queries']].to_string(index=False))
    print(pd.DataFrame(rs).query("class_id == 'macro'").to_string(index=False))
    print('seconds', receipt['seconds'], 'cases', cases, flush=True)


def make_panels(table, t, qids, rids, images, groupdata, mass):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from extract_tokens import geo, slots
    u = images[images.is_utility]
    scores = u.groupby('local_index').agg(n=('group','size'), ridge=('ridge_sse','sum'), mean=('mean_sse','sum'))
    scores = scores[scores.n == 4].copy()
    scores['skill'] = 1-scores.ridge/scores['mean']
    scores['obs_id'] = t.iloc[scores.index].obs_id.to_numpy()
    scores = scores.sort_values(['skill', 'obs_id'], kind='stable')
    selected = [('low', int(scores.index[0])), ('high', int(scores.index[-1]))]
    snaps = {int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    cases, refs, provenance = [], [], []
    for label, qi in selected:
        # Fixed VV-side context, highest raw-C same-class and other-class bank references.
        raw, cosine, allowed = groupdata['VV_side']
        j = int(np.flatnonzero(qids == qi)[0])
        c = int(t.iloc[qi].class_id)
        views = [(qi, 'QUERY')]
        for same, role in [(True, 'SAME CLASS'), (False, 'OTHER CLASS')]:
            mask = allowed[j] & ((t.iloc[rids].class_id.to_numpy() == c) == same)
            choices = np.flatnonzero(mask)
            chosen = int(choices[np.argmax(raw[j, choices])])
            ri = int(rids[chosen])
            views.append((ri, role + ' REF'))
            refs.append(dict(query_obs=int(t.iloc[qi].obs_id), role=role, reference_obs=int(t.iloc[ri].obs_id),
                reference_class=int(t.iloc[ri].class_id), group='VV_side', raw_kernel=raw[j, chosen],
                cosine_kernel=cosine[j, chosen], eligible_references=int(mask.sum()),
                same_mmsi=bool(t.iloc[qi].mmsi == t.iloc[ri].mmsi),
                same_pass=bool(t.iloc[qi].pass_group == t.iloc[ri].pass_group)))
        fig, axes = plt.subplots(2, 3, figsize=(12, 9))
        for p, pol in enumerate(('vv', 'vh')):
            for col, (ix, role) in enumerate(views):
                original = int(t.iloc[ix].table_index)
                pix, fp, geometry, place = geo.native(table, snaps, original, p)
                ax = axes[p, col]
                ax.imshow(pix, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                if mass[ix, p].sum() > 0:
                    masks = slots(geometry, fp)
                    for (lo, hi), color in zip(((0,8),(8,16),(16,20)), ('#24a878','#2c84c6','#d39222')):
                        region = masks[lo:hi].sum(0)
                        if region.min() < .5 < region.max():
                            ax.contour(region, levels=[.5], colors=[color], linewidths=.7)
                row = t.iloc[ix]
                ax.set(title=f'{pol.upper()} {role} obs {int(row.obs_id)}\n{row.class_name}; {row.port}\nvalid={int(mass[ix,p].sum()>0)}',
                       xlim=(-.5,159.5), ylim=(159.5,-.5), xticks=[0,80,159], yticks=[0,80,159])
                provenance.append(dict(case=label, query_obs=int(t.iloc[qi].obs_id), obs_id=int(row.obs_id),
                    role=role, pol=pol, path=row['image_'+pol], placement=place))
        sub = u[u.local_index == qi]
        skills = '; '.join(f'{r.group}={r.ridge_skill:+.3f}' for r in sub.itertuples())
        fig.suptitle(f'{label.upper()} within-image LOO skill: obs {int(t.iloc[qi].obs_id)} | pooled={scores.loc[qi,"skill"]:+.3f}\n{skills}', fontsize=11)
        fig.text(.02,.015,'Native160 crop/pad, no resize; fixed 0..255 display; layer1 RF35.\nReferences: highest VV-side raw-C similarity within eligible same/other class, not classifier predictions.',fontsize=9)
        fig.subplots_adjust(top=.84,bottom=.10,wspace=.22,hspace=.36)
        filename = f'case_{label}_{int(t.iloc[qi].obs_id)}.png'
        fig.savefig(HERE / filename, dpi=125)
        plt.close(fig)
        cases.append(dict(kind=label, obs_id=int(t.iloc[qi].obs_id), class_name=t.iloc[qi].class_name,
            skill=float(scores.loc[qi,'skill']), panel=filename, selection='extrema pooled SSE skill, all four groups available; stable obs_id tie order'))
    pd.DataFrame(refs).to_csv(HERE / 'case_references.csv', index=False)
    (HERE / 'image_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    return cases


if __name__ == '__main__':
    main()
