"""Frozen seed17 external query/reference panels and exact cached evidence.

Run after external_eval.py. No fitting, feature extraction, or source case review.
All outputs stay in external_confirmation/analysis.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
EXT = HERE / 'external_confirmation'
UP = HERE.parent / 'regional_utility_pilot/upstream'
OUT = EXT / 'analysis'
ARMS = ('anchored', 'independent', 'body_match', 'factorized')
SEED = 17


def select_cases(table, scores):
    y = table.class_id.to_numpy(int)
    correct = scores['anchored'].argmax(1) == y
    rows = []
    for comparison in ('full', 'body_match'):
        other = scores[comparison].argmax(1) == y
        groups = {'rescued': correct & ~other, 'harmed': ~correct & other}
        if comparison == 'full':
            groups.update(CC=correct & other, WW=~correct & ~other)
        for outcome, mask in groups.items():
            ids = np.flatnonzero(mask)
            chosen = None
            if len(ids):
                # Numeric obs_id, then row order: independent of margin size.
                chosen = int(ids[np.argmin(table.iloc[ids].obs_id.to_numpy(np.int64))])
            rows.append(dict(case=f'{comparison}_{outcome}', comparison=comparison,
                outcome=outcome, seed=SEED, evaluation_n=len(y), group_n=len(ids),
                group_fraction=len(ids)/len(y), local_index=chosen,
                obs_id=None if chosen is None else int(table.iloc[chosen].obs_id),
                selected=chosen is not None))
    return rows


def project(features, mass, projection):
    from match_readout import unit
    x = unit((features-projection['mean']) @ projection['components'].T).astype(np.float32)
    x[mass <= 0] = 0
    return x


def replay_head(arm, local, base, saved):
    from train_readout import Head
    ck = torch.load(HERE / 'models' / f'layer1_{arm}_{SEED}.pt',
                    map_location='cpu', weights_only=False)
    local = local.copy()
    if arm == 'body_match':
        local[:, :, :, 1:] = 0
    z = base-base.mean(-1, keepdims=True)
    competitor = np.stack([np.delete(z, c, axis=1).max(1) for c in range(8)], 1)
    raw = np.concatenate((local.reshape(len(base), 8, 12), z[..., None],
                          competitor[..., None], (z-competitor)[..., None]), -1)
    x = np.where(ck['active'], (raw-ck['mu'])/ck['scale'], 0).astype(np.float32)
    model = Head().eval()
    model.load_state_dict(ck['state'])
    with torch.inference_mode():
        replay = base+model(torch.tensor(x)).numpy()
    error = float(np.max(np.abs(replay.astype(float)-saved)))
    np.testing.assert_allclose(replay, saved, atol=5e-5, rtol=1e-5)
    np.testing.assert_array_equal(replay.argmax(1), saved.argmax(1))
    return dict(arm=arm, selected_step=int(ck['selected_step']),
                replay_max_abs=error, replay_top1_exact=True)


def evidence(q, s, qm, sm):
    from match_readout import operations, pair_score, END, IS_END
    args = [torch.tensor(a) for a in (q, s, qm, sm)]
    with torch.inference_mode():
        values, t = operations(*args)
        mass = t.sum((-1, -2))
        outer = t.sum(-1)[..., :, None]*t.sum(-2)[..., None, :]
        outer = outer/mass[..., None, None].clamp_min(1e-20)
        qq, ss, aa, bb = args
        sims = torch.stack([
            torch.einsum('bpid,bckpjd->bckpij', qq[:, :, :8], ss[..., :8, :]),
            torch.einsum('bpid,bckpjd->bckpij', qq[:, :, 8:16], ss[..., 8:16, :]),
            torch.einsum('bpid,bckpjd->bckpij', qq[:, :, END], ss[..., END, :])], -3)
        query_mass = torch.stack([aa[:, :, :8], aa[:, :, 8:16],
                                 aa[:, :, END]*torch.tensor(IS_END)], -2)
        support_mass = torch.stack([bb[..., :8], bb[..., 8:16],
                                   bb[..., END]*torch.tensor(IS_END)], -2)
        valid = ((query_mass[:, None, None, ..., :, None] > 0) &
                 (support_mass[..., None, :] > 0))
        plan = t[..., None, :, :]*valid
        factor = outer[..., None, :, :]*valid
        contribution = plan*sims
        factor_contribution = factor*sims
        values['factorized'] = torch.stack([
            pair_score(outer, sims[..., r, :, :],
                       query_mass[:, None, None, ..., r, :], support_mass[..., r, :])
            for r in range(3)], -2)
        np.testing.assert_allclose(contribution.sum((-1, -2)).numpy(),
                                   values['anchored'][..., 0].numpy(), atol=2e-6)
        np.testing.assert_allclose(outer.sum(-1).numpy(), t.sum(-1).numpy(), atol=2e-6)
        np.testing.assert_allclose(outer.sum(-2).numpy(), t.sum(-2).numpy(), atol=2e-6)
    return {k: v.numpy() for k, v in values.items()}, {
        'body_T': t.numpy(), 'body_outer': outer.numpy(), 'similarity': sims.numpy(),
        'regional_valid': valid.numpy(), 'regional_plan': plan.numpy(),
        'regional_contribution': contribution.numpy(),
        'factorized_plan': factor.numpy(),
        'factorized_contribution': factor_contribution.numpy()}


def render(case, table, source, refs, snapshots, tokens, source_tokens, scores, epochs):
    from extract_tokens import geo, slots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    i = case['local_index']
    row = table.iloc[i]
    true = int(row.class_id)
    rival = int(np.argmax(np.where(np.arange(8) == true, -np.inf, scores['anchored'][i])))
    views = [(table, i, tokens, 'QUERY', None)]
    for cls, label in ((true, 'TRUE reference'), (rival, 'RIVAL reference')):
        for k, idx in enumerate(refs[i, cls]):
            views.append((source, int(idx), source_tokens, f'{label} {k+1}', cls))
    fig, axes = plt.subplots(2, 5, figsize=(23, 12))
    fig.subplots_adjust(left=.025, right=.985, bottom=.17, top=.83, wspace=.14, hspace=.42)
    images = []
    for p, pol in enumerate(('vv', 'vh')):
        for j, (tab, idx, cache, label, refclass) in enumerate(views):
            rr = tab.iloc[idx]
            pix, footprint, g, place = geo.native(tab, snapshots, idx, p)
            assert pix.shape == (160, 160)
            ax = axes[p, j]
            ax.imshow(pix, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            ax.set_xlim(-.5, 159.5)
            ax.set_ylim(159.5, -.5)
            ax.set_aspect('equal')
            ax.set_xticks([0, 40, 80, 120, 159])
            ax.set_yticks([0, 40, 80, 120, 159])
            if g is not None:
                masks = slots(g, footprint)
                for region, color in ((slice(0, 8), '#24a878'),
                                      (slice(8, 16), '#2c84c6'),
                                      (slice(16, 20), '#d39222')):
                    field = masks[region].sum(0)
                    if field.min() < .5 < field.max():
                        ax.contour(field, levels=[.5], colors=[color], linewidths=.8)
                cx, cy, length, width, angle = g[:5]
                uv = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])*[length/2, width/2]
                rot = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
                ax.add_patch(Polygon(uv @ rot.T+[cx, cy], fill=False,
                                     edgecolor='white', linewidth=.8, linestyle=':'))
                for slot, field in enumerate(masks[:8]):
                    area = field.sum()
                    if area > 0:
                        ax.text((field*geo.XX).sum()/area, (field*geo.YY).sum()/area,
                                str(slot), color='#ff514f', ha='center', fontsize=8)
            usable = bool(cache['valid'][idx, p])
            if not usable:
                ax.text(80, 80, 'NO VALID LOCAL TOKENS', ha='center', fontsize=10,
                        color='red', bbox=dict(facecolor='white', alpha=.8))
            classline = textwrap.fill(str(rr.class_name), 27)
            portline = textwrap.fill(str(rr.port), 27)
            ax.set_title(f'{pol.upper()} {label}\nobs {int(rr.obs_id)} | {classline}\n{portline}', fontsize=10)
            images.append(dict(case=case['case'], panel_column=j, pol=pol,
                table_scope='external' if j == 0 else 'source', local_index=idx,
                obs_id=int(rr.obs_id), instance_id=str(rr.instance_id), mmsi=str(rr.mmsi),
                class_id=int(rr.class_id), class_name=str(rr.class_name), port=str(rr.port),
                pass_group=str(rr.pass_group), reference_class=refclass,
                path=str(rr['image_'+pol]), valid_local_tokens=usable,
                native_h=snapshots[int(rr.obs_id)]['images'][pol]['h'],
                native_w=snapshots[int(rr.obs_id)]['images'][pol]['w'],
                geometry=None if g is None else g.tolist(), placement=place))
    labels = []
    for arm, s in scores.items():
        pred = int(s[i].argmax())
        margin = float(s[i, true]-s[i, rival])
        step = '' if arm == 'full' else f', step {epochs[arm]}'
        labels.append(f'{arm}: pred {pred}, margin {margin:+.4f}{step}')
    fig.suptitle(f"EXTERNAL | {case['outcome']} vs {case['comparison']} | query obs {int(row.obs_id)} | seed17 layer1\n"
                 f"true={true}; anchored strongest wrong class={rival}; group {case['group_n']}/{case['evaluation_n']}", fontsize=15)
    fig.text(.03, .105, '\n'.join([' | '.join(labels[:2]), ' | '.join(labels[2:])]), fontsize=11)
    fig.text(.03, .025,
        'All ten panels: the actual 160 x 160 native crop/pad input; no display crop or input resize.\n'
        'Green body, blue side, amber end; dotted white target OBB. Numbers identify body slots.\n'
        'Layer1 theoretical RF=35: within-box token pooling is not box-pure pixel evidence.\n'
        'Matching heads add residuals to frozen full; these are incremental comparisons, not isolated ship-body classifiers.', fontsize=11)
    path = OUT / f"case_{case['case']}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return rival, images, str(path)


def main():
    needed = [EXT/'table.csv', EXT/'tokens.npz', EXT/'strategies_seed17.npz',
              EXT/'local_scores.npz', UP/'table.csv', HERE/'tokens.npz',
              HERE/'projection_layer1.npz']
    needed += [EXT/'results'/f'{arm}_{SEED}.npz' for arm in ARMS]
    needed += [HERE/'models'/f'layer1_{arm}_{SEED}.pt' for arm in ARMS]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        print(json.dumps(dict(status='not_ready', missing=missing), ensure_ascii=False, indent=2))
        return
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(OUT / '.mplcache')
    from extract_tokens import geo
    table = pd.read_csv(EXT/'table.csv', dtype={'mmsi': str, 'instance_id': str})
    source = pd.read_csv(UP/'table.csv', dtype={'mmsi': str, 'instance_id': str})
    tokens, source_tokens = np.load(EXT/'tokens.npz'), np.load(HERE/'tokens.npz')
    for tab, cache in ((table, tokens), (source, source_tokens)):
        np.testing.assert_array_equal(cache['instance_ids'].astype(str), tab.instance_id.to_numpy())
    strategy = np.load(EXT/'strategies_seed17.npz')
    np.testing.assert_array_equal(strategy['instance_ids'].astype(str), table.instance_id.to_numpy())
    np.testing.assert_array_equal(strategy['strategy_names'], ['s0', 'body', 'full'])
    scores = {'full': strategy['logits'][:, 2]}
    scores.update({arm: np.load(EXT/'results'/f'{arm}_{SEED}.npz')['score'] for arm in ARMS})
    for s in scores.values():
        assert s.shape == (len(table), 8) and np.isfinite(s).all()
    local = np.load(EXT/'local_scores.npz')
    refs = local['refs']
    assert refs.shape == (len(table), 8, 2)
    assert refs.min() >= 0 and refs.max() < len(source)
    np.testing.assert_array_equal(source.class_id.to_numpy()[refs],
                                 np.broadcast_to(np.arange(8)[None, :, None], refs.shape))
    selections = select_cases(table, scores)
    pd.DataFrame(selections).to_csv(OUT/'case_selection.csv', index=False)
    cases = [r for r in selections if r['selected']]
    indices = np.array([r['local_index'] for r in cases], dtype=int)
    if not len(indices):
        print('No cases in the six predeclared groups. See case_selection.csv.')
        return
    checks = []
    for arm in ARMS:
        raw = local['anchored' if arm == 'body_match' else arm][indices]
        checks.append(replay_head(arm, raw, scores['full'][indices].astype(np.float32), scores[arm][indices]))
    epochs = {r['arm']: r['selected_step'] for r in checks}
    wanted = set(table.iloc[indices].obs_id) | set(source.iloc[np.unique(refs[indices])].obs_id)
    snapshots = {s['obs_id']: s for s in json.loads(
        (geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))
        if s['obs_id'] in wanted}
    projection = np.load(HERE/'projection_layer1.npz')
    source_features, source_mass = source_tokens['layer1'], source_tokens['mass']
    query_features, query_mass = tokens['layer1'], tokens['mass']
    numbers, paths, links, region_rows, case_rows = [], [], [], [], []
    max_local_error = 0.
    for case in cases:
        i = case['local_index']
        row = table.iloc[i]
        true = int(row.class_id)
        rival, images, panel = render(case, table, source, refs, snapshots, tokens,
                                      source_tokens, scores, epochs)
        paths.extend(images)
        classes = np.array([true, rival])
        ref = refs[i, classes]
        q = project(query_features[i:i+1], query_mass[i:i+1], projection)
        s = project(source_features[ref], source_mass[ref], projection)[None]
        values, parts = evidence(q, s, query_mass[i:i+1], source_mass[ref][None])
        for arm in ('anchored', 'independent', 'factorized'):
            recomputed = values[arm].mean(2)[0]
            saved = local[arm][i, classes]
            max_local_error = max(max_local_error, float(np.max(np.abs(recomputed-saved))))
            np.testing.assert_allclose(recomputed, saved, atol=5e-5, rtol=1e-5)
        np.savez_compressed(OUT/f"evidence_{case['case']}.npz", **parts,
            class_ids=classes, source_indices=ref, source_obs_ids=source.obs_id.to_numpy()[ref],
            query_index=i, query_obs_id=int(row.obs_id),
            method_names=np.array(list(scores)), scores=np.stack([s[i] for s in scores.values()]),
            **{f'recomputed_{arm}': values[arm] for arm in ('anchored', 'independent', 'factorized')})
        for arm, score in scores.items():
            r = score[i].copy()
            r[true] = -np.inf
            numbers.append(dict(case=case['case'], obs_id=int(row.obs_id), arm=arm,
                true_class=true, prediction=int(score[i].argmax()),
                strongest_wrong_class=int(r.argmax()), fixed_rival=rival,
                true_rival_margin=float(score[i, true]-score[i, rival]),
                selected_step=None if arm == 'full' else epochs[arm],
                **{f's{c}': float(score[i, c]) for c in range(8)}))
        for c, cls in enumerate(classes):
            for k, idx in enumerate(ref[c]):
                for p, pol in enumerate(('vv', 'vh')):
                    common = dict(case=case['case'], query_obs=int(row.obs_id),
                        reference_class=int(cls), reference_slot=k, reference_source_index=int(idx),
                        reference_obs=int(source.iloc[idx].obs_id), pol=pol)
                    for r, region in enumerate(('body', 'side', 'end')):
                        contrib = parts['regional_contribution'][0, c, k, p, r]
                        fac = parts['factorized_contribution'][0, c, k, p, r]
                        weight = parts['regional_plan'][0, c, k, p, r]
                        fweight = parts['factorized_plan'][0, c, k, p, r]
                        region_rows.append(dict(**common, region=region,
                            anchored_score=float(contrib.sum()), anchored_mass=float(weight.sum()),
                            factorized_score=float(fac.sum()), factorized_mass=float(fweight.sum()),
                            score_T_minus_outer=float((contrib-fac).sum())))
                        for a in range(8):
                            for b in range(8):
                                links.append(dict(**common, region=region, query_body_slot=a,
                                    reference_body_slot=b, inherited_weight=float(weight[a, b]),
                                    cosine=float(parts['similarity'][0, c, k, p, r, a, b]),
                                    contribution=float(contrib[a, b]),
                                    outer_weight=float(fweight[a, b]), outer_contribution=float(fac[a, b])))
        case_rows.append(dict(**case, instance_id=str(row.instance_id), true_class=true,
                              rival_class=rival, panel=panel))
        print(f"rendered {case['case']} query={int(row.obs_id)}", flush=True)
    pd.DataFrame(case_rows).to_csv(OUT/'case_panels.csv', index=False)
    pd.DataFrame(numbers).to_csv(OUT/'case_scores.csv', index=False)
    pd.DataFrame(region_rows).to_csv(OUT/'case_region_scores.csv', index=False)
    pd.DataFrame(links).to_csv(OUT/'case_correspondences.csv', index=False)
    (OUT/'case_paths.json').write_text(json.dumps(paths, ensure_ascii=False, indent=2), encoding='utf-8')
    receipt = dict(seed=SEED, layer='layer1', evaluation_n=len(table), figures=len(cases),
        selection='minimum numeric obs_id in each predeclared group; zero-count groups retained',
        outcome_direction='comparison -> anchored', reference_table=str(UP/'table.csv'),
        reference_rule='actual shared refs for true class and anchored strongest wrong class; two each',
        canvas=[160, 160], display_limits=[-.5, 159.5], head_checks=checks,
        local_replay_max_abs=max_local_error, atol=5e-5, rtol=1e-5,
        new_fits=0, new_backbone_forwards=0, scope='historical external coverage, not a new blind collection',
        interpretation='T-weighted similarities are local inputs, not additive final-logit or physical contributions; layer1 RF35 mixes across masks',
        image_review='panels generated; this receipt does not claim a human/model has viewed them')
    (OUT/'external_cases_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
