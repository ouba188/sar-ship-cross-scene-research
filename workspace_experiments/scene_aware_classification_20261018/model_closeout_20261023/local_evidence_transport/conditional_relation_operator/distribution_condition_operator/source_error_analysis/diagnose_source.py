"""Frozen source-only error diagnosis. Run with ClearSAR Python -X utf8 -B.

No fitting, parameter selection, evaluation scoring, or Git operations.
All outputs, including matplotlib's cache, remain beside this script.
"""
import os
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
HERE = OUT.parent
sys.dont_write_bytecode = True
os.environ['MPLCONFIGDIR'] = str(OUT / '.mplcache')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'

import importlib.util
import json
import time
import numpy as np
import pandas as pd
import torch

spec = importlib.util.spec_from_file_location('source_error_frozen_run', HERE / 'run.py')
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
exp = run.exp
GROUPS = exp.edge.GROUPS
SEEDS = (17, 43)


def wrong_class(score, y):
    return np.where(np.arange(8)[None, :] == y[:, None], -np.inf, score).argmax(1)


@torch.inference_mode()
def decompose(reader, ids, matrices, references=False):
    z, raw, valid, detail = reader.scores(ids, *matrices, trace=True)
    token, energy, reference = [], [], []
    for ak, qk, sk, kind in (('A', 'Bqs', 'Bss', 0), ('An', 'Lqs', 'Lss', 1)):
        A, L, S = (detail[k] for k in (ak, qk, sk))
        allocation = .5 * (2 * A * L - A * (A @ S[:, None])) / detail['mq'][:, :, None, None, None]
        allocation *= valid.permute(2, 0, 1)[..., None, None]
        token.append(allocation.sum(-1).permute(1, 2, 0, 3).numpy())
        if references:
            reference.append(allocation.reshape(4, len(ids), 8, 8, 8, 8).sum(-1).permute(1, 2, 0, 3, 4).numpy())
        q = detail['qb' if kind == 0 else 'qn']
        s = detail['sb' if kind == 0 else 'sn'].reshape(4, 8, 64, 32)
        _, _, gram = reader.output_grams(q, s, reader.legal[:, ids], reader.legal[:, reader.bank].reshape(4, 8, 8, 8), kind)
        energy.append((gram.diagonal(dim1=-2, dim2=-1).sum(-1) / detail['mq']).T.numpy())
    token = np.stack(token, -1)  # query, class, group, query-edge, channel
    energy = np.stack(energy, -1)
    error = float(np.max(np.abs(token.sum((-1, -2)) - raw.numpy())))
    assert error < 1e-10, ('token allocation does not replay raw score', error)
    result = dict(z=z.numpy(), raw=raw.numpy(), valid=valid.numpy(), token=token,
                  channel=token.sum(-2), energy=energy, replay_error=error)
    if references:
        result['reference'] = np.stack(reference, -1)
        np.testing.assert_allclose(result['reference'].sum(-2), token, atol=1e-12, rtol=1e-12)
        result['reference_valid'] = detail['rv'].numpy()
    return result


def json_write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    tick = time.perf_counter()
    torch.set_num_threads(2)
    # Parent loader reads shared containers. Discard evaluation rows before Reader
    # construction; never extract their predictions or use them for case selection.
    table_all, x, mass, oldbank, idx, tr, va, _ = exp.source_data()
    locs = np.r_[tr, va]
    source = idx[locs]
    keep = np.unique(np.r_[source, oldbank.ravel()])
    remap = np.full(len(table_all), -1, int)
    remap[keep] = np.arange(len(keep))
    table = table_all.iloc[keep].reset_index(drop=True)
    ids, bank = remap[source], remap[oldbank]
    x, mass = x[keep], mass[keep]
    assert set(table.iloc[ids].role) == {'utility_train', 'selection'}
    assert not set(source).intersection(oldbank.ravel())
    reader = run.Reader(x, mass, bank, 'independent', device='cpu', dtype=torch.float64)
    scales = pd.read_csv(HERE / 'scales.csv')
    div = scales.query("arm == 'independent'").sort_values('feature').divisor.to_numpy()
    reader.divisor = torch.tensor(div, dtype=torch.float64)
    weights = np.load(HERE / 'weights.npz')
    matrices = {'identity': (torch.eye(32, dtype=torch.float64),) * 2}
    for seed in SEEDS:
        matrices[str(seed)] = tuple(torch.tensor(weights[f'independent_{seed}_{key}'], dtype=torch.float64) for key in ('Mb', 'Mn'))
    results = {}
    for state, pair in matrices.items():
        chunks = [decompose(reader, ids[start:start + 32], pair) for start in range(0, len(ids), 32)]
        results[state] = {key: np.concatenate([c[key] for c in chunks]) for key in chunks[0] if key != 'replay_error'}
        results[state]['replay_error'] = max(c['replay_error'] for c in chunks)
        print('completed source reconstruction', state, len(ids), flush=True)

    meta = table.iloc[ids].reset_index(drop=True)
    y, pos = meta.class_id.to_numpy(int), np.arange(len(ids))
    sample_rows, group_rows, token_rows = [], [], []
    # Group/channel comparisons use their own strongest wrong class, not a mean
    # wrong class or a previously selected rival that might cease to be strongest.
    for state, result in results.items():
        for g, group in enumerate(GROUPS):
            for p, channel in enumerate(('body', 'neighbor', 'combined')):
                value = result['channel'][:, :, g, p] * 2 if p < 2 else result['raw'][:, :, g]
                energy = result['energy'][:, g, p] if p < 2 else result['energy'][:, g].mean(-1)
                rival = wrong_class(value, y)
                truth = value[pos, y]
                for j in np.flatnonzero(result['valid'][pos, y, g]):
                    group_rows.append(dict(state=state, role=meta.role[j], obs_id=int(meta.obs_id[j]),
                        mmsi=str(meta.mmsi[j]), port=meta.port[j], class_id=int(y[j]), group=group, channel=channel,
                        rival=int(rival[j]), true_score=float(truth[j]), rival_score=float(value[j, rival[j]]),
                        mean_wrong_score=float((value[j].sum() - truth[j]) / 7),
                        best_gap=float(truth[j] - value[j, rival[j]]),
                        mean_gap=float(truth[j] - (value[j].sum() - truth[j]) / 7),
                        query_energy=float(energy[j]), true_skill=float(truth[j] / max(energy[j], 1e-12)),
                        rival_skill=float(value[j, rival[j]] / max(energy[j], 1e-12)),
                        top1=int(value[j].argmax() == y[j])))
    groups = pd.DataFrame(group_rows)
    groups.to_csv(OUT / 'source_group_scores.csv', index=False)
    measures = ['true_score', 'rival_score', 'true_skill', 'rival_skill', 'mean_gap', 'best_gap', 'top1']
    classes = groups.groupby(['state', 'role', 'group', 'channel', 'class_id'])[measures].mean()
    classes.to_csv(OUT / 'source_class_scores.csv')
    summary = classes.groupby(['state', 'role', 'group', 'channel'])[measures].mean()
    summary.to_csv(OUT / 'source_macro_scores.csv')
    paired_rows = []
    keys = ['role', 'obs_id', 'group', 'channel']
    identity = groups[groups.state == 'identity'].set_index(keys)
    for seed in SEEDS:
        learned = groups[groups.state == str(seed)].set_index(keys)
        difference = (learned[['true_skill', 'best_gap', 'mean_gap']] - identity[['true_skill', 'best_gap', 'mean_gap']]).reset_index()
        for (role, group, channel), f in difference.groupby(['role', 'group', 'channel']):
            paired_rows.append(dict(seed=seed, role=role, group=group, channel=channel, n=len(f),
                true_skill_up_best_gap_down=int(((f.true_skill > 0) & (f.best_gap < 0)).sum()),
                mean_gap_up_best_gap_down=int(((f.mean_gap > 0) & (f.best_gap < 0)).sum())))
    pd.DataFrame(paired_rows).to_csv(OUT / 'source_paired_diagnostics.csv', index=False)

    prediction_errors = {}
    for seed in SEEDS:
        saved = np.load(HERE / f'predictions_independent_{seed}.npz')
        np.testing.assert_array_equal(saved['idx'][locs], source)
        old, new = saved['full'][locs], saved['original'][locs]
        a = weights[f'independent_{seed}_a'].astype(float)
        z0, z1 = results['identity']['z'], results[str(seed)]['z']
        increment = ((z1 - z0) * a).sum(-1)
        increment -= increment.mean(-1, keepdims=True)
        error = float(np.max(np.abs(new - old - increment)))
        assert error < 4e-5, ('cached prediction mismatch', seed, error)
        prediction_errors[str(seed)] = error
        old_rival, new_rival = wrong_class(old, y), wrong_class(new, y)
        score0, score1 = (z0 * a).sum(-1), (z1 * a).sum(-1)
        lr0, lr1 = wrong_class(score0, y), wrong_class(score1, y)
        for j, i in enumerate(ids):
            before, after = old[j].argmax() == y[j], new[j].argmax() == y[j]
            fixed_rival = int(new_rival[j] if before and not after else old_rival[j])
            tok = results[str(seed)]['token'][j]
            local_gap_tokens = (tok[y[j]] - tok[lr1[j]]) * (a / div)[:, None, None]
            legal = reader.numpy_legal[i, :, :, None].repeat(2, axis=2)
            legal &= results[str(seed)]['valid'][j, y[j], :, None, None]
            vals = local_gap_tokens[legal]
            sample_rows.append(dict(seed=seed, role=meta.role[j], obs_id=int(meta.obs_id[j]),
                mmsi=str(meta.mmsi[j]), port=meta.port[j], class_id=int(y[j]), row_index=int(source[j]),
                query_index=j, compact_index=int(i), full_pred=int(old[j].argmax()), selected_pred=int(new[j].argmax()),
                outcome=('CC' if after else 'harmed') if before else ('rescued' if after else 'WW'),
                full_rival=int(old_rival[j]), selected_rival=int(new_rival[j]), fixed_rival=fixed_rival,
                full_best_margin=float(old[j, y[j]] - old[j, old_rival[j]]),
                selected_best_margin=float(new[j, y[j]] - new[j, new_rival[j]]),
                fixed_margin_before=float(old[j, y[j]] - old[j, fixed_rival]),
                fixed_margin_after=float(new[j, y[j]] - new[j, fixed_rival]),
                fixed_margin_delta=float(increment[j, y[j]] - increment[j, fixed_rival]),
                identity_local_rival=int(lr0[j]), selected_local_rival=int(lr1[j]),
                identity_local_gap=float(score0[j, y[j]] - score0[j, lr0[j]]),
                selected_local_gap=float(score1[j, y[j]] - score1[j, lr1[j]]),
                positive_token_count=int((vals > 1e-10).sum()), negative_token_count=int((vals < -1e-10).sum()),
                positive_token_mass=float(vals[vals > 0].sum()), negative_token_mass=float(vals[vals < 0].sum()),
                legal_token_channel_count=len(vals)))
    samples = pd.DataFrame(sample_rows)
    samples.to_csv(OUT / 'source_samples.csv', index=False)
    counts = samples.groupby(['seed', 'role', 'outcome']).size().rename('n').reset_index()
    counts.to_csv(OUT / 'source_outcomes.csv', index=False)
    samples[samples.selected_local_gap < 0].groupby(['seed', 'role', 'class_id', 'selected_local_rival']).agg(
        n=('obs_id', 'size'), vessels=('mmsi', 'nunique'), ports=('port', 'nunique'), local_gap=('selected_local_gap', 'mean'),
        full_margin=('selected_best_margin', 'mean')).to_csv(OUT / 'source_confusions.csv')
    token_summary = []
    for (seed, role), f in samples.groupby(['seed', 'role']):
        wrong = f[f.selected_local_gap < 0]
        token_summary.append(dict(seed=seed, role=role, queries=len(f), local_wrong=len(wrong),
            wrong_with_positive_and_negative=int(((wrong.positive_token_count > 0) & (wrong.negative_token_count > 0)).sum()),
            wrong_without_positive=int((wrong.positive_token_count == 0).sum()),
            identity_local_macro_top1=float(f.groupby('class_id').identity_local_gap.apply(lambda v: (v > 0).mean()).mean()),
            selected_local_macro_top1=float(f.groupby('class_id').selected_local_gap.apply(lambda v: (v > 0).mean()).mean())))
    pd.DataFrame(token_summary).to_csv(OUT / 'source_token_conflicts.csv', index=False)

    cases = []
    for outcome in ('rescued', 'harmed'):
        candidates = samples[(samples.role == 'selection') & (samples.outcome == outcome)].copy()
        cross = candidates.groupby('obs_id').agg(seed_count=('seed', 'nunique'),
            min_abs_change=('fixed_margin_delta', lambda v: abs(v).min()))
        cross = cross.sort_values(['seed_count', 'min_abs_change'], ascending=[False, False])
        if cross.empty:
            candidates = samples[samples.outcome == outcome].copy()
            cross = candidates.groupby('obs_id').agg(seed_count=('seed', 'nunique'),
                min_abs_change=('fixed_margin_delta', lambda v: abs(v).min())).sort_values(
                    ['seed_count', 'min_abs_change'], ascending=[False, False])
        obs = int(cross.index[0])
        chosen = candidates[candidates.obs_id == obs].assign(strength=lambda f: abs(f.fixed_margin_delta)).sort_values(
            ['strength', 'seed'], ascending=[False, True]).iloc[0]
        case = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in chosen.items() if k != 'strength'}
        case['same_outcome_seeds'] = int(cross.loc[obs, 'seed_count'])
        cases.append(case)
    print('source cases', [(c['outcome'], c['obs_id'], c['seed']) for c in cases], flush=True)

    channel_rows, reference_rows, case_scores = [], [], []
    for case in cases:
        j, i, c, r = (int(case[k]) for k in ('query_index', 'compact_index', 'class_id', 'fixed_rival'))
        seed = int(case['seed'])
        a = weights[f'independent_{seed}_a'].astype(float)
        detailed = {state: decompose(reader, np.array([i]), matrices[state], references=True)
                    for state in ('identity', str(seed))}
        allocated = detailed[str(seed)]['reference'][0] - detailed['identity']['reference'][0]
        change = (allocated[c] - allocated[r]).sum(2) * (a / div)[:, None, None]
        total_by_group = change.sum((1, 2))
        g = int(np.abs(total_by_group).argmax())
        case['dominant_group'] = GROUPS[g]
        case['group_index'] = g
        case['reference_indices'] = []
        case['reference_classes'] = list(dict.fromkeys([c, r, int(case['selected_local_rival'])]))
        for cl in case['reference_classes']:
            v = allocated[cl, g].sum((0, 2)) * a[g] / div[g]
            valid = detailed[str(seed)]['reference_valid'][g, cl]
            ref = int(np.argmax(np.where(valid, np.abs(v), -np.inf)))
            case['reference_indices'].append(int(bank[cl, ref]))
        for state, d in detailed.items():
            for cl in case['reference_classes']:
                for gg, name in enumerate(GROUPS):
                    for ref in range(8):
                        ri = int(bank[cl, ref])
                        row = table.iloc[ri]
                        for p, channel in enumerate(('body', 'neighbor')):
                            value = float(d['reference'][0, cl, gg, :, ref, p].sum())
                            reference_rows.append(dict(obs_id=case['obs_id'], seed=seed, state=state,
                                class_id=cl, group=name, channel=channel, ref_index=ref, ref_obs=int(row.obs_id),
                                ref_mmsi=str(row.mmsi), ref_port=row.port, ref_role=row.role,
                                same_vessel=bool(str(row.mmsi) == str(meta.mmsi[j])),
                                same_pass=bool(row.pass_group == meta.pass_group[j]),
                                valid=bool(d['reference_valid'][gg, cl, ref]), score_allocation=value,
                                weighted_allocation=value * a[gg] / div[gg]))
        for other_seed in SEEDS:
            aa = weights[f'independent_{other_seed}_a'].astype(float)
            for state in ('identity', str(other_seed)):
                result = results[state]
                for cl in range(8):
                    case_scores.append(dict(obs_id=case['obs_id'], seed=other_seed, state=state, class_id=cl,
                        class_name=table.loc[table.class_id == cl, 'class_name'].iloc[0],
                        weighted_local_score=float((result['z'][j, cl] * aa).sum()),
                        **{GROUPS[gg] + '_' + name: float(result['channel'][j, cl, gg, p])
                           for gg in range(4) for p, name in enumerate(('body', 'neighbor'))}))
        for rival in dict.fromkeys([r, int(case['selected_local_rival'])]):
            for gg, name in enumerate(GROUPS):
                for p, channel in enumerate(('body', 'neighbor')):
                    before, after = (detailed[s]['channel'][0, :, gg, p] for s in ('identity', str(seed)))
                    channel_rows.append(dict(obs_id=case['obs_id'], seed=seed, rival=rival, group=name, channel=channel,
                        true_identity=float(before[c]), rival_identity=float(before[rival]),
                        true_selected=float(after[c]), rival_selected=float(after[rival]),
                        weighted_margin_delta=float((after[c] - after[rival] - before[c] + before[rival]) * a[gg] / div[gg])))
                    for e in np.flatnonzero(reader.numpy_legal[i, gg]):
                        t0, t1 = (detailed[s]['token'][0, :, gg, e, p] for s in ('identity', str(seed)))
                        slot = int(exp.edge.BODY[gg % 2][e] if p == 0 else exp.edge.NEIGHBOR[gg % 2][e])
                        token_rows.append(dict(obs_id=case['obs_id'], seed=seed, rival=rival, group=name, channel=channel,
                            edge=e, slot=slot, true_identity=float(t0[c]), rival_identity=float(t0[rival]),
                            true_selected=float(t1[c]), rival_selected=float(t1[rival]),
                            weighted_identity_gap=float((t0[c] - t0[rival]) * a[gg] / div[gg]),
                            weighted_selected_gap=float((t1[c] - t1[rival]) * a[gg] / div[gg]),
                            weighted_margin_delta=float((t1[c] - t1[rival] - t0[c] + t0[rival]) * a[gg] / div[gg])))
    for name, rows in (('case_channels', channel_rows), ('case_tokens', token_rows), ('case_references', reference_rows), ('case_class_scores', case_scores)):
        pd.DataFrame(rows).to_csv(OUT / f'{name}.csv', index=False)
    json_write('cases.json', cases)
    make_panels(table, cases, reader, pd.DataFrame(token_rows))

    scope = []
    for role, f in meta.groupby('role'):
        scope.append(dict(role=role, queries=len(f), vessels=int(f.mmsi.nunique()), ports=int(f.port.nunique()),
            passes=int(f.pass_group.nunique()), classes=f.groupby('class_name').size().to_dict(),
            valid_groups={GROUPS[g]: int((reader.counts[g, ids[f.index]].numpy() >= 2).sum()) for g in range(4)}))
    receipt = dict(status='complete', source_only=True, roles=scope, selected_steps={str(s):20 for s in SEEDS},
        query_source='parent source_data; only utility_train and selection retained',
        reference_source='original frozen fit bank; no new reference selection', reference_roles=table.iloc[bank.ravel()].role.value_counts().to_dict(),
        query_reference_vessel_overlap=sorted(set(meta.mmsi) & set(table.iloc[bank.ravel()].mmsi)),
        max_score_allocation_error=max(r['replay_error'] for r in results.values()),
        cached_prediction_max_errors=prediction_errors, seconds=time.perf_counter()-tick,
        case_selection='Prefer selection; outcome shared by more seeds, then largest minimum absolute fixed-rival margin change; display seed with larger absolute change.',
        no_training=True, no_tuning=True, no_evaluation_scoring=True, no_git=True,
        caveat='Source descriptive evidence only; token oracle is label-dependent, not a learned classifier or proof of a new algorithm.')
    json_write('execution.json', receipt)
    print(counts.to_string(index=False), flush=True)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


def make_panels(table, cases, reader, tokens):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from extract_tokens import geo
    wanted = {int(table.iloc[i].obs_id) for case in cases for i in [case['compact_index'], *case['reference_indices']]}
    snapshots = {int(s['obs_id']): s for s in json.loads((geo.OLD / 'local_condition_module/view_snapshot.json').read_text(encoding='utf-8')) if int(s['obs_id']) in wanted}
    provenance = []
    for case in cases:
        columns = 1 + len(case['reference_indices'])
        fig = plt.figure(figsize=(4.5 * columns, 11))
        gs = fig.add_gridspec(3, columns, height_ratios=[1, 1, .65], hspace=.48, wspace=.25)
        for col, (i, label) in enumerate(zip([case['compact_index'], *case['reference_indices']], ('QUERY', 'TRUE REF', 'FULL-RIVAL REF', 'LOCAL-RIVAL REF'))):
            for p, pol in enumerate(('vv', 'vh')):
                pixels, footprint, geometry, place = geo.native(table, snapshots, i, p)
                ax = fig.add_subplot(gs[p, col])
                ax.imshow(pixels, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                row = table.iloc[i]
                ax.set(xticks=[0,80,159], yticks=[0,80,159], xlim=(-.5,159.5), ylim=(159.5,-.5))
                ax.set_title(f'{pol.upper()} {label} {int(row.obs_id)}\n{row.class_name}\n{row.port} | native 160 crop/pad', fontsize=10)
                provenance.append(dict(query_obs=case['obs_id'], obs_id=int(row.obs_id), pol=pol,
                    image_path=row['image_'+pol], placement=place, geometry=None if geometry is None else geometry.tolist(),
                    mmsi=str(row.mmsi), role=row.role, pass_group=row.pass_group,
                    valid_side=bool(reader.counts[p*2,i] >= 2), valid_end=bool(reader.counts[p*2+1,i] >= 2)))
        rivals = list(dict.fromkeys([case['fixed_rival'], case['selected_local_rival']]))
        for k, rival in enumerate(rivals):
            ax = fig.add_subplot(gs[2, k*2:(k+1)*2] if len(rivals) == 2 else gs[2, :])
            t = tokens[(tokens.obs_id == case['obs_id']) & (tokens.group == case['dominant_group']) & (tokens.rival == rival)]
            labels = [f'{r.channel[0].upper()}{r.slot}' for r in t.itertuples()]
            at = np.arange(len(t))
            ax.bar(at-.18, t.weighted_identity_gap, .36, label='identity true - rival', color='#476a9c')
            ax.bar(at+.18, t.weighted_selected_gap, .36, label='selected true - rival', color='#b66255')
            ax.axhline(0, color='black', linewidth=.7)
            ax.set(xticks=at, xticklabels=labels, ylabel='Weighted score allocation', title=f"{case['dominant_group']}: true vs class {rival}; B=body / N=neighbor")
            ax.tick_params(axis='x', labelsize=8)
            ax.legend(fontsize=8)
        fig.suptitle(f"SOURCE {case['role']} | {case['outcome']} {case['obs_id']} | seed {case['seed']} / step20\nFull pred {case['full_pred']} -> {case['selected_pred']}; true {case['class_id']}; best margin {case['full_best_margin']:+.3f} -> {case['selected_best_margin']:+.3f}", fontsize=13)
        fig.text(.04,.025,'Actual original-bank references, chosen by largest absolute score-increment allocation in the dominant group.\nDisplay PNG pixels 0..255, no spatial rescaling/contrast stretch; allocations are not physical pixel causes.', fontsize=9)
        fig.subplots_adjust(top=.89, bottom=.15)
        fig.savefig(OUT / f"source_{case['outcome']}_{case['obs_id']}.png", dpi=130)
        plt.close(fig)
    json_write('image_provenance.json', provenance)


if __name__ == '__main__':
    main()
