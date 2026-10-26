"""Posthoc analysis only, after source choices and evaluation predictions freeze."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from run import HERE, WEIGHT, SEEDS, load, batch, predict, write, op, RegionResNet50
from analyze import margin

OUT = HERE/'tuning/analysis'
REGIONS = ('body', 'end', 'side')


def restored(folder, mode):
    ck = torch.load(folder/'selected.pt', map_location='cpu', weights_only=True)
    model = RegionResNet50(WEIGHT, mode=mode,
                          temperature=ck.get('config', {}).get('temperature', 1.)).cuda().eval()
    missing = model.load_state_dict(ck['state'], strict=False)
    assert not missing.unexpected_keys and all(k.startswith('backbone.') for k in missing.missing_keys)
    return model


def occupancy(masks, valid, merged=False):
    if merged:
        masks = masks.sum(1, keepdim=True).clamp(0, 1).expand(-1, 3, -1, -1)
    return torch.minimum(masks.clamp(0, 1), valid.clamp(0, 1))


def bootstrap(change, y, passes):
    groups = [np.flatnonzero(passes == g) for g in pd.unique(passes)]
    count = np.array([[np.count_nonzero(y[g] == c) for c in range(8)] for g in groups])
    delta = np.array([[change[g][y[g] == c].sum() for c in range(8)] for g in groups])
    draws = np.random.default_rng(171).multinomial(len(groups), np.full(len(groups), 1/len(groups)), 2000)
    n, d = draws @ count, draws @ delta
    oa = d.sum(1)/n.sum(1)
    complete = (n > 0).all(1)
    ba = (d[complete]/n[complete]).mean(1)
    return dict(delta_OA_pp=100*change.mean(),
                delta_BA_pp=100*np.mean([change[y == c].mean() for c in range(8)]),
                OA_low_pp=100*np.quantile(oa, .025), OA_high_pp=100*np.quantile(oa, .975),
                BA_low_pp=float(100*np.quantile(ba, .025)) if len(ba) else None,
                BA_high_pp=float(100*np.quantile(ba, .975)) if len(ba) else None,
                passes=len(groups), draws=2000, BA_complete_class_draws=int(complete.sum()))


def diagnostics(model, arrays, ids, saved, mode, seed):
    totals = np.zeros((2, 7), float)
    handles = []
    def hook(p):
        def collect(block, args, output):
            x, masks, valid = args
            m = occupancy(masks, valid)
            # Recover the sigmoid gate before occupancy and overlap normalization.
            _, details = block.forward(x, masks, valid, return_details=True)
            active = m > 0
            gates = (details['weights']*m.sum(1, keepdim=True).clamp_min(1)/m.clamp_min(1e-12))[active]
            support = active.any(1)
            totals[p] += [gates.numel(), float(gates.sum()), int((gates <= .01).sum()),
                          int((gates >= .99).sum()), int(support.sum()),
                          float(details['update'].norm(dim=1)[support].sum()),
                          float(x.norm(dim=1)[support].sum())]
        return collect
    for p in range(2):
        handles.append(model.blocks[f'{p}_2'].register_forward_hook(hook(p)))
    try:
        replay = predict(model, arrays, ids)
    finally:
        for handle in handles:
            handle.remove()
    np.testing.assert_allclose(replay, saved, atol=2e-5, rtol=1e-5)
    np.testing.assert_array_equal(replay.argmax(1), saved.argmax(1))
    stages = model.stages
    try:
        model.stages = ()
        bypass = predict(model, arrays, ids)
    finally:
        model.stages = stages
    rows = []
    for p, pol in enumerate(('VV', 'VH')):
        n, gate_sum, low, high, cells, update_sum, input_sum = totals[p]
        rows.append(dict(mode=mode, seed=seed, polarization=pol, evaluated=len(ids),
            available_views=int((arrays['available'][ids, p] > 0).sum()), active_region_cells=int(n),
            gate_mean=gate_sum/n if n else None, saturated_low=low/n if n else None,
            saturated_high=high/n if n else None, active_spatial_cells=int(cells),
            mean_update_norm=update_sum/cells if cells else None,
            relative_update=update_sum/input_sum if input_sum else None,
            gain=float(model.blocks[f'{p}_2'].gain), temperature=model.blocks[f'{p}_2'].temperature,
            replay_max_error=float(abs(replay-saved).max())))
    return bypass, rows


def case_panel(model, arrays, images, idx, row, seed, kind, old, new, truth):
    inputs = batch(arrays, [idx])
    contributions, stats = [], []
    fig, axes = plt.subplots(2, 5, figsize=(15, 7.5))
    extent = (-.5, 159.5, 159.5, -.5)
    for p, pol in enumerate(('VV', 'VH')):
        x, masks, valid, available = inputs
        block = model.blocks[f'{p}_2']
        m = occupancy(masks[:, p], valid[:, p])
        _, d = block(x[:, p], m, valid[:, p], return_details=True)
        z = F.layer_norm(x[:, p].permute(0, 2, 3, 1), (block.channels,)).permute(0, 3, 1, 2)
        tokens = torch.einsum('brhw,bchw->brc', m, z)/d['mass'].clamp_min(1e-6)[..., None]
        values = torch.einsum('brd,rde->bre', block.value(tokens), block.region_value)
        parts = block.gain*block.output(d['weights'][0, :, None]*values[0, :, :, None, None])
        torch.testing.assert_close(parts.sum(0), d['update'][0], atol=2e-5, rtol=2e-5)
        contributions.append(parts)
        update = d['update'][0].norm(dim=0).cpu().numpy()
        overlays = [None, *m[0].cpu().numpy(), update]
        titles = [f'{pol} native; available={int(available[0, p])}',
                  *[f'{pol} {r} occupancy' for r in REGIONS], f'{pol} update norm\nmax={update.max():.4g}']
        for a, overlay, title in zip(axes[p], overlays, titles):
            a.imshow(images[idx, p, 0], cmap='gray', vmin=0, vmax=255, interpolation='nearest', extent=extent)
            if overlay is not None:
                a.imshow(np.ma.masked_less_equal(overlay, 0), extent=extent, interpolation='nearest',
                         cmap='magma' if overlay is update else 'viridis', alpha=.65,
                         vmin=0, vmax=max(float(overlay.max()), 1e-12) if overlay is update else 1)
            a.set_title(title, fontsize=9, pad=8)
            a.axis('off')
        stats.append(dict(polarization=pol, available=bool(available[0, p]),
                          region_mass=d['mass'][0].cpu().tolist(), update_max=float(update.max())))
    fig.subplots_adjust(left=.02, right=.98, bottom=.10, top=.82, hspace=.32, wspace=.12)
    fig.suptitle(f'{kind}: obs {row.obs_id}, seed {seed}, true {truth}; matched none {old.argmax()} -> regional {new.argmax()}\n'
                 f'True margin {margin(old[None], np.array([truth]))[0]:+.4f} -> {margin(new[None], np.array([truth]))[0]:+.4f}', fontsize=12, y=.97)
    fig.text(.02, .035, 'Stored native crop/pad pixels; overlays are feature-cell occupancy and update norm, not physical attribution.', fontsize=9)
    panel = f'{kind}_{row.obs_id}_{seed}.png'
    fig.savefig(OUT/panel, dpi=130)
    plt.close(fig)
    effects = []
    for intervention in ('full', 'no_feedback', 'drop_VV', 'drop_VH', *['drop_'+r for r in REGIONS]):
        handles = []
        for p, pol in enumerate(('VV', 'VH')):
            if intervention in ('no_feedback', 'drop_'+pol):
                handles.append(model.blocks[f'{p}_2'].register_forward_hook(lambda block, args, output: args[0]))
            elif intervention.removeprefix('drop_') in REGIONS:
                part = contributions[p][REGIONS.index(intervention.removeprefix('drop_'))][None]
                handles.append(model.blocks[f'{p}_2'].register_forward_hook(lambda block, args, output, part=part: output-part))
        try:
            score = model.forward_from(*inputs)[0].cpu().numpy()
        finally:
            for handle in handles:
                handle.remove()
        if intervention == 'full':
            replay_error = float(abs(score-new).max())
            assert replay_error < 1e-4, (int(row.obs_id), seed, replay_error)
        effects.append(dict(kind=kind, obs_id=int(row.obs_id), seed=seed, intervention=intervention,
            true=truth, prediction=int(score.argmax()), correct=bool(score.argmax() == truth),
            true_margin=float(margin(score[None], np.array([truth]))[0]),
            delta_margin=float(margin(score[None], np.array([truth]))[0]-margin(new[None], np.array([truth]))[0]),
            scores=json.dumps(score.tolist())))
    return dict(kind=kind, obs_id=int(row.obs_id), table_index=idx, seed=seed, true=truth,
                before=int(old.argmax()), after=int(new.argmax()), old_score=old.tolist(), new_score=new.tolist(),
                image_vv=row.image_vv, image_vh=row.image_vh, pixel_cache=str(HERE/'cache/images.npy'),
                panel=panel, details=stats, saved_full_vs_fresh_max_error=replay_error), effects


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    selected = json.loads((HERE/'tuning/selected_config.json').read_text())
    frozen = json.loads((HERE/'tuning/frozen_choices.json').read_text())
    config = selected['config']
    if config == 'original':
        assert not frozen
        OUT.mkdir(parents=True, exist_ok=True)
        message = 'Original retained by source-only selection; no new target grid evaluated.'
        write(OUT/'analysis_complete.json', dict(status='original_retained', config=config,
            source_means=selected.get('source_means', {}), evaluated=0, cases=0,
            testlabelread=False, training=False, target_selection=False, note=message))
        (OUT/'summary.md').write_text(message+'\n', encoding='utf-8')
        print(message, flush=True)
        return
    assert len(frozen) == 9 and all(r['config'] == config for r in frozen)
    assert {(r['mode'], r['seed']) for r in frozen} == {(m, s) for m in ('none', 'merged', 'regional') for s in SEEDS}
    root = HERE/'tuning'/config
    folders = {(m, s): root/f'{m}_{s}' for m in ('none', 'merged', 'regional') for s in SEEDS}
    folders.update({('original_regional', s): HERE/'training'/f'regional_{s}' for s in SEEDS})
    assert all((p/'evaluation.npz').is_file() and (p/'selected.pt').is_file() for p in folders.values()), 'Run frozen evaluation first.'
    # Evaluation labels are read only below this boundary, for posthoc analysis.
    data = {}
    for key, folder in folders.items():
        with np.load(folder/'evaluation.npz', allow_pickle=False) as z:
            data[key] = {k: z[k].copy() for k in z.files}
    table, arrays = load()
    ids, y = data['none', SEEDS[0]]['ids'], data['none', SEEDS[0]]['y']
    assert table.iloc[ids].role.eq('evaluation').all() and set(y) == set(range(8))
    for d in data.values():
        for key, expected in (('ids', ids), ('y', y), ('obs_id', table.iloc[ids].obs_id.to_numpy())):
            np.testing.assert_array_equal(d[key], expected)
    OUT.mkdir(parents=True, exist_ok=True)
    scores = {k: d['score'] for k, d in data.items()}
    gates = []
    for mode in ('merged', 'regional'):
        for seed in SEEDS:
            model = restored(folders[mode, seed], mode)
            scores[mode+'_no_feedback', seed], rows = diagnostics(model, arrays, ids, scores[mode, seed], mode, seed)
            gates.extend(rows)
            del model
    metrics = pd.DataFrame([dict(mode=m, seed=s, **op.metrics(y, score)) for (m, s), score in scores.items()])
    metrics.to_csv(OUT/'metrics.csv', index=False)
    metrics.groupby('mode')[['OA', 'BA', 'NLL', 'Brier']].agg(['mean', 'std']).to_csv(OUT/'summary.csv')
    paired, perclass, uncertainty = [], [], []
    for baseline in ('none', 'merged', 'original_regional', 'regional_no_feedback'):
        changes = []
        for seed in SEEDS:
            a, b = scores['regional', seed].argmax(1) == y, scores[baseline, seed].argmax(1) == y
            delta = a.astype(float)-b
            changes.append(delta)
            paired.append(dict(baseline=baseline, seed=seed, rescued=int((a & ~b).sum()), harmed=int((~a & b).sum()),
                               delta_OA_pp=100*delta.mean(), delta_BA_pp=100*np.mean([delta[y == c].mean() for c in range(8)])))
            perclass.extend(dict(baseline=baseline, seed=seed, class_id=c, n=int((y == c).sum()),
                recall=a[y == c].mean(), baseline_recall=b[y == c].mean(), delta_recall_pp=100*delta[y == c].mean()) for c in range(8))
        uncertainty.append(dict(baseline=baseline, **bootstrap(np.mean(changes, axis=0), y, table.iloc[ids].pass_group.to_numpy())))
    for name, rows in (('paired', paired), ('by_class', perclass), ('uncertainty', uncertainty), ('gate_saturation', gates)):
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv', index=False)
    np.savez_compressed(OUT/'scores.npz', ids=ids, obs_id=table.iloc[ids].obs_id.to_numpy(), y=y,
                        **{f'{mode}_{seed}': score for (mode, seed), score in scores.items()})
    images = np.load(HERE/'cache/images.npy', mmap_mode='r')
    cases, effects, absent = [], [], []
    for kind in ('rescued', 'harmed', 'CC', 'WW'):
        pool = []
        for seed in SEEDS:
            old, new = scores['none', seed], scores['regional', seed]
            a, b = new.argmax(1) == y, old.argmax(1) == y
            flag = dict(rescued=a & ~b, harmed=~a & b, CC=a & b, WW=~a & ~b)[kind]
            delta = abs(margin(new, y)-margin(old, y))
            pool.extend((float(delta[k]), seed, int(k)) for k in np.flatnonzero(flag))
        if not pool:
            absent.append(kind)
            continue
        _, seed, k = sorted(pool)[len(pool)//2]
        model = restored(folders['regional', seed], 'regional')
        case, rows = case_panel(model, arrays, images, int(ids[k]), table.iloc[ids[k]], seed, kind,
                               scores['none', seed][k], scores['regional', seed][k], int(y[k]))
        cases.append(case)
        effects.extend(rows)
        del model
    write(OUT/'cases.json', cases)
    pd.DataFrame(effects).to_csv(OUT/'case_interventions.csv', index=False)
    write(OUT/'analysis_complete.json', dict(config=config, evaluated=len(ids), seeds=list(SEEDS), testlabelread='posthocanalysis_only',
        training=False, target_selection=False, cases=len(cases), absent_categories=absent, case_baseline='matched_none',
        case_selection='Median absolute true-margin change within each outcome, pooled over fixed seeds; representative, not prevalence.',
        saturation='Unweighted active region-cell sigmoid gates, low<=0.01/high>=0.99; all evaluation rows, per polarization.',
        bootstrap='2000 paired pass-cluster draws of seed-mean correctness changes; BA excludes draws missing any of eight classes; not seed uncertainty.',
        interventions='Same selected checkpoint/head; drop region subtracts its fixed additive contribution, retaining full-input tokens/gates and other regions. Polar/no_feedback bypass retains image features and availability.',
        note='Stored native pixels, display-only overlay scaling. No physical-causal attribution. Outputs do not modify original results.'))
    print(metrics.to_string(index=False), flush=True)
    print('ANALYSIS_COMPLETE', OUT, flush=True)


if __name__ == '__main__':
    main()
