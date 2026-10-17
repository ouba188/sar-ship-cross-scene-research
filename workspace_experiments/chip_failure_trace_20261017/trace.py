"""Trace matched decision changes to native OSS1 chips and actual supports."""
from pathlib import Path
import os
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import pandas as pd
import torch
import tifffile
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STUDY = ROOT/'support_task_alignment_20261017'
PREVIOUS = ROOT/'local_evidence_reliability_20261017'
GEOMETRY = ROOT/'oss1_joint_geometry_20261017'
sys.path.insert(0, str(PREVIOUS))
import train_reliability as prior
from local_reader import regions

KEYS = ['fold', 'seed', 'origin', 'draw', 'global_row']
ARMS = ['fixed_task', 'diverse_aux', 'diverse_task']
CASES = [(786, 'fixed_task', 0, 1), (745, 'fixed_task', 1, 2),
         (953, 'diverse_aux', 0, 3), (995, 'fixed_task', 1, 1)]


def paired_events(predictions):
    base = predictions[predictions.arm.eq('fixed_aux')]
    renamed = {name: 'reference_'+name for name in base.columns if name not in KEYS}
    paired = predictions[predictions.arm.isin(ARMS)].merge(
        base.rename(columns=renamed), on=KEYS, validate='many_to_one')
    paired['change'] = (paired.prediction.eq(paired.truth).astype(int)
                        - paired.reference_prediction.eq(paired.truth).astype(int))
    cohort = base[base.origin.eq('cross')].drop_duplicates(['fold', 'global_row'])
    counts = cohort.groupby(['fold', 'truth']).size().rename('query_class_count').reset_index()
    per_fold = cohort.groupby('fold').agg(query_count=('global_row', 'size'),
                                        query_class_count_total=('truth', 'nunique'))
    paired = paired.merge(counts, on=['fold', 'truth'], validate='many_to_one')
    paired = paired.join(per_fold, on='fold')
    tasks = base[base.origin.eq('cross')][['fold', 'seed', 'draw']].drop_duplicates()
    assert tasks.groupby('fold').size().nunique() == 1
    denominator = len(tasks)
    paired['oa_contribution_pp'] = 100*paired.change/(denominator*paired.query_count)
    paired['ba_contribution_pp'] = 100*paired.change/(
        denominator*paired.query_class_count_total*paired.query_class_count)
    reference_p = paired[[f'reference_p{k}' for k in range(8)]].to_numpy()
    sorted_p = np.sort(reference_p, axis=1)
    paired['reference_top2_gap'] = sorted_p[:, -1]-sorted_p[:, -2]
    paired.to_csv(HERE/'paired_predictions.csv', index=False)
    changed = paired[paired.change.ne(0)].copy()
    changed.to_csv(HERE/'changed_events.csv', index=False)
    summary = paired.groupby(['origin', 'arm']).agg(
        delta_oa=('oa_contribution_pp', 'sum'), delta_ba=('ba_contribution_pp', 'sum'),
        rescue_events=('change', lambda v: (v > 0).sum()),
        harm_events=('change', lambda v: (v < 0).sum()))
    summary.to_csv(HERE/'paired_effects.csv')
    for (origin, arm), row in summary.iterrows():
        m = pd.read_csv(STUDY/'metrics.csv')
        expected = (m[m.origin.eq(origin)&m.arm.eq(arm)][['oa', 'ba']].mean()
                    -m[m.origin.eq(origin)&m.arm.eq('fixed_aux')][['oa', 'ba']].mean())
        np.testing.assert_allclose(row[['delta_oa', 'delta_ba']].astype(float),
                                   expected.to_numpy(), atol=1e-10)
    query_summary = changed.groupby(['origin', 'arm', 'fold', 'global_row', 'truth']).agg(
        delta_oa=('oa_contribution_pp', 'sum'), delta_ba=('ba_contribution_pp', 'sum'),
        rescue_events=('change', lambda v: (v > 0).sum()),
        harm_events=('change', lambda v: (v < 0).sum()),
        maximum_reference_gap=('reference_top2_gap', 'max')).reset_index()
    query_summary.to_csv(HERE/'query_contributions.csv', index=False)
    return paired, changed, cohort


def solve(raw, valid, task):
    probability, h = prior.old.predict(raw[task['ids']], valid[task['ids']], task, 1.)
    ns = task['ns']
    ss, qs = task['ss']+h[:ns]@h[:ns].T, task['qs']+h[ns:]@h[:ns].T
    root = task['root']
    coefficients = root[:, None]*torch.linalg.solve(
        root[:, None]*ss*root[None, :]+8*torch.eye(ns, dtype=torch.float64),
        root[:, None]*(task['desired']-task['prior_s']))
    logits = task['prior_q']+qs@coefficients
    reconstructed = .5*(task['base']+logits.softmax(-1))
    active = valid[task['q']] & (valid[task['s']].sum() >= 2)
    reconstructed = torch.where(active[:, None], reconstructed, task['old'])
    np.testing.assert_allclose(probability.numpy(), reconstructed.numpy(), atol=1e-12)
    return probability.numpy(), qs.numpy(), coefficients.numpy(), logits.numpy(), h.numpy()


def chip_inventory(ids, geometry, detection, metadata):
    rows = []
    for index in sorted(ids):
        g, d, m = geometry.loc[index], detection.loc[index], metadata.loc[index]
        body, weights, pos, cov = regions(d.to_dict(), float(g.radar_axis_rad),
            float(g.incidence_rad), bool(g.radar_valid), bool(g.incidence_valid), 'joint')
        record = dict(global_row=index, sample_id=g.sample_id, class_id=int(g.class_id),
            class_name=g.class_name, mmsi=str(g.mmsi), product=g['product'],
            detector_confidence=d.confidence, axis_valid=d.axis_valid,
            views_same_size=d.views_same_size, native_h=d.native_h, native_w=d.native_w,
            mean_region_coverage=cov.mean(), min_region_coverage=cov.min(),
            empty_regions=int((pos[:, 3] == 0).sum()), body_grid_cells=int(body.sum()),
            ais_sog_knots=m.ais_sog_knots, ais_abs_timegap_s=m.ais_abs_timegap_s,
            ais_length_m=m.ais_length_m, marine_length_m=m.marine_length_overall_m,
            sar_manual_length_m=m.sar_manual_length_m, incidence_deg=g.incidence_deg,
            chip_path_vv=g.chip_path_vv, chip_path_vh=g.chip_path_vh,
            raw_patch_path=g.raw_patch_path, calibrated_patch_path=g.calibrated_patch_path,
            ship_xml_path=g.ship_xml_path, xml_ship_index=int(g.xml_ship_index))
        for pol in ('vv', 'vh'):
            image = tifffile.imread(g['chip_path_'+pol])
            assert image.ndim == 2 and image.shape == (int(d.native_h), int(d.native_w))
            yy, xx = np.indices(image.shape)
            x, y = xx-d.cx, yy-d.cy
            u = x*np.cos(d.ship_axis)+y*np.sin(d.ship_axis)
            v = -x*np.sin(d.ship_axis)+y*np.cos(d.ship_axis)
            mask = ((abs(u) <= max(d.length*1.25, 4)/2)
                    & (abs(v) <= max(d.width*1.25, 3)/2))
            record[pol+'_saturated_body_fraction'] = float((image[mask] == 255).mean())
            record[pol+'_display_body_mean'] = float(image[mask].mean())
            record[pol+'_display_outside_mean'] = float(image[~mask].mean())
        rows.append(record)
    return pd.DataFrame(rows)


def draw_chip(ax, index, pol, calibrated, geometry, detection, overlay=True):
    g, d = geometry.loc[index], detection.loc[index]
    if calibrated:
        array = tifffile.imread(g.calibrated_patch_path)[:, :, 1 if pol == 'vv' else 0]
        image = 10*np.log10(array.clip(1e-8))
        ax.imshow(image, cmap='gray', vmin=-35, vmax=5, interpolation='nearest')
    else:
        image = tifffile.imread(g['chip_path_'+pol])
        ax.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    if overlay:
        body, weights, _, _ = regions(d.to_dict(), float(g.radar_axis_rad),
            float(g.incidence_rad), bool(g.radar_valid), bool(g.incidence_valid), 'joint')
        corners = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*np.array([d.length, d.width])*1.25/2
        co, si = np.cos(d.ship_axis), np.sin(d.ship_axis)
        corners = corners@np.array([[co,si],[-si,co]])+np.array([d.cx,d.cy])
        ax.add_patch(Polygon(corners, fill=False, edgecolor='#2ae0ab', linewidth=.9))
        xx = (np.arange(56)+.25)*d.native_w/56-.5
        yy = (np.arange(56)+.25)*d.native_h/56-.5
        for scale, color in enumerate(('#ffcf4a', '#f26b8a', '#70b8ff')):
            weight = weights[scale*4:(scale+1)*4].sum(0)
            if weight.max() > 0:
                ax.contour(xx, yy, weight, levels=[weight.max()*.3], colors=[color], linewidths=.6)
    ax.set_xlim(-.5, image.shape[1]-.5)
    ax.set_ylim(image.shape[0]-.5, -.5)
    ax.tick_params(labelsize=6)


def plot_case(event, edges, geometry, detection):
    index = int(event['query_global_row'])
    positive = int(edges.loc[edges.total_margin_change.idxmax(), 'support_global_row'])
    negative = int(edges.loc[edges.total_margin_change.idxmin(), 'support_global_row'])
    same = edges[edges.support_class.eq(event['truth'])].sort_values('kernel_reference', ascending=False)
    same_index = int(same.iloc[0].support_global_row)
    columns = [('QUERY', index), ('Most negative delta', negative),
               ('Most positive delta', positive), ('Same-class support', same_index)]
    fig, axes = plt.subplots(3, 4, figsize=(13, 9))
    for column, (role, sample) in enumerate(columns):
        for row, (pol, calibrated) in enumerate([('vv', False), ('vh', False), ('vv', True)]):
            draw_chip(axes[row, column], sample, pol, calibrated, geometry, detection)
            if column == 0:
                axes[row, column].set_ylabel(['Native uint8 VV', 'Native uint8 VH', 'Calibrated VV (dB)'][row])
        g = geometry.loc[sample]
        axes[0, column].set_title(f'{role}\nrow {sample} | {g.class_name}', fontsize=9)
        contribution = edges[edges.support_global_row.eq(sample)]
        if len(contribution):
            axes[2, column].set_xlabel(f"Expert margin delta {contribution.iloc[0].total_margin_change:+.5f}", fontsize=8)
    fig.suptitle(f"Outcome-selected query {index}: fixed_aux -> {event['arm']} | seed {event['seed']}, draw {event['draw']}\n"
                 f"Truth {geometry.loc[index].class_name}; predicted {event['reference_prediction']} -> {event['prediction']} | "
                 'green: readout body proxy; yellow/pink/blue: three local scales', fontsize=10)
    fig.tight_layout(rect=(0, .02, 1, .93))
    fig.savefig(HERE/'figures'/f'case_{index}.png', dpi=170)
    plt.close(fig)
    return [dict(query_global_row=index, role=role, global_row=sample) for role, sample in columns]


@torch.no_grad()
def trace(predictions, paired, changed, geometry, detection, metadata):
    quality = dict(np.load(PREVIOUS/'quality.npz'))
    manifest = pd.read_csv(prior.old.MANIFEST, dtype={'mmsi': str})
    all_edges, replay, cases, task_supports, galleries = [], [], [], [], []
    selected = {case for case in CASES}
    for args in prior.old.folds():
        data, features, _, _, tasks, _, _ = prior.prepare_fold(args, manifest, quality)
        rows = data['rows']
        for task in tasks:
            if task['origin'] != 'cross':
                continue
            for support in task['s']:
                task_supports.append(dict(fold=data['fold'], draw=task['draw'],
                    global_row=int(rows[support]), class_id=int(data['labels'][support]),
                    local_valid=bool(features['valid'][support])))
        for seed in range(3):
            raw = {}
            for arm in ['fixed_aux']+ARMS:
                cache = np.load(STUDY/data['fold']/f'{arm}_{seed}'/'readouts.npz')
                np.testing.assert_array_equal(cache['global_rows'], rows)
                raw[arm] = torch.tensor(cache['raw'])
            for task in tasks:
                if task['origin'] != 'cross':
                    continue
                solved = {arm: solve(raw[arm], features['valid'], task) for arm in raw}
                for arm, (prob, *_rest) in solved.items():
                    saved = predictions[predictions.fold.eq(data['fold'])&predictions.seed.eq(seed)
                        &predictions.origin.eq('cross')&predictions.draw.eq(task['draw'])
                        &predictions.arm.eq(arm)].set_index('global_row').loc[rows[task['q']]]
                    error = float(np.max(abs(prob-saved[[f'p{k}' for k in range(8)]].to_numpy())))
                    assert error < 1e-8, (data['fold'], seed, task['draw'], arm, error)
                    replay.append(dict(fold=data['fold'], seed=seed, draw=task['draw'], arm=arm, max_error=error))
                events = changed[changed.fold.eq(data['fold'])&changed.seed.eq(seed)
                    &changed.origin.eq('cross')&changed.draw.eq(task['draw'])]
                p0, k0, c0, l0, h0 = solved['fixed_aux']
                for event in events.itertuples():
                    j = int(np.flatnonzero(rows[task['q']]==event.global_row)[0])
                    opponent = event.prediction if event.change < 0 else event.reference_prediction
                    p1, k1, c1, l1, h1 = solved[event.arm]
                    a0, a1 = c0[:, event.truth]-c0[:, opponent], c1[:, event.truth]-c1[:, opponent]
                    # Symmetric exact decomposition avoids privileging either refit order.
                    kernel = .5*(k1[j]-k0[j])*(a0+a1)
                    refit = .5*(k1[j]+k0[j])*(a1-a0)
                    total = kernel+refit
                    delta = (l1[j,event.truth]-l1[j,opponent])-(l0[j,event.truth]-l0[j,opponent])
                    np.testing.assert_allclose(total.sum(), delta, atol=1e-10)
                    edge_rows = []
                    for pos, support in enumerate(task['s']):
                        edge_rows.append(dict(fold=data['fold'], seed=seed, draw=task['draw'], arm=event.arm,
                            query_global_row=event.global_row, support_global_row=int(rows[support]),
                            support_class=int(data['labels'][support]), truth=event.truth, opponent=opponent,
                            change=event.change, kernel_reference=k0[j,pos], kernel_new=k1[j,pos],
                            local_kernel_reference=float(h0[task['ns']+j]@h0[pos]),
                            local_kernel_new=float(h1[task['ns']+j]@h1[pos]),
                            coefficient_reference=a0[pos], coefficient_new=a1[pos],
                            kernel_margin_change=kernel[pos], support_refit_margin_change=refit[pos],
                            total_margin_change=total[pos]))
                    all_edges.extend(edge_rows)
                    if (event.global_row, event.arm, seed, task['draw']) in selected:
                        record = dict(query_global_row=event.global_row, fold=data['fold'], arm=event.arm,
                            seed=seed, draw=task['draw'], truth=event.truth, opponent=opponent,
                            reference_prediction=event.reference_prediction, prediction=event.prediction,
                            change=event.change, same_class_support=event.same_class_support,
                            ap_reference=event.reference_ap, ap_new=event.ap,
                            final_margin_reference=p0[j,event.truth]-p0[j,opponent],
                            final_margin_new=p1[j,event.truth]-p1[j,opponent],
                            base_margin=float(task['base'][j,event.truth]-task['base'][j,opponent]),
                            expert_logit_margin_reference=l0[j,event.truth]-l0[j,opponent],
                            expert_logit_margin_new=l1[j,event.truth]-l1[j,opponent],
                            expert_margin_delta=delta, kernel_part=kernel.sum(), refit_part=refit.sum(),
                            ba_contribution_pp=event.ba_contribution_pp)
                        cases.append(record)
                        galleries.extend(plot_case(record, pd.DataFrame(edge_rows), geometry, detection))
        print('TRACED', data['fold'], flush=True)
    pd.DataFrame(all_edges).to_csv(HERE/'support_margin_edges.csv', index=False)
    pd.DataFrame(replay).to_csv(HERE/'replay.csv', index=False)
    pd.DataFrame(cases).to_csv(HERE/'case_summary.csv', index=False)
    pd.DataFrame(galleries).to_csv(HERE/'gallery_selection.csv', index=False)
    supports = pd.DataFrame(task_supports).drop_duplicates()
    supports.to_csv(HERE/'actual_supports.csv', index=False)
    return supports, pd.DataFrame(cases)


def main():
    torch.set_num_threads(1)
    (HERE/'figures').mkdir(exist_ok=True)
    predictions = pd.read_csv(STUDY/'predictions.csv')
    paired, changed, cohort = paired_events(predictions)
    geometry = pd.read_csv(GEOMETRY/'geometry.csv', dtype={'mmsi': str}).set_index('global_row')
    detection = pd.read_csv(GEOMETRY/'detections.csv').set_index('global_row')
    metadata = pd.read_csv(ROOT/'all_factor_probe_20261017/metadata/oss1_factor_metadata.csv',
                           dtype={'mmsi': str}).set_index('global_row')
    for index in geometry.index:
        assert geometry.loc[index, 'sample_id'] == detection.loc[index, 'sample_id']
    with threadpool_limits(limits=1):
        supports, cases = trace(predictions, paired, changed, geometry, detection, metadata)
    ids = set(cohort.global_row)|set(supports.global_row)
    inventory = chip_inventory(ids, geometry, detection, metadata)
    inventory['is_query'] = inventory.global_row.isin(cohort.global_row)
    inventory['cross_changed_query'] = inventory.global_row.isin(changed[changed.origin.eq('cross')].global_row)
    inventory.to_csv(HERE/'chip_inventory.csv', index=False)
    query = cohort[['global_row','fold','truth']].merge(inventory, on='global_row', validate='one_to_one')
    query.to_csv(HERE/'query_inventory.csv', index=False)
    query.groupby('cross_changed_query')[['detector_confidence','native_h','mean_region_coverage',
        'min_region_coverage','vv_saturated_body_fraction','vh_saturated_body_fraction']].agg(['count','median','mean']).to_csv(HERE/'quality_groups.csv')
    selected = predictions[predictions.global_row.isin([case[0] for case in CASES])].copy()
    selected.to_csv(HERE/'selected_query_all_contexts.csv', index=False)
    print(cases.to_string(index=False), flush=True)
    print('Distinct queries', len(cohort), 'query/support inventory', len(inventory), flush=True)


if __name__ == '__main__':
    main()
