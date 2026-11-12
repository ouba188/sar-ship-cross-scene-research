"""Recover fixed historical spatial-M3 region updates, CPU-only, no training."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROOT.mkdir(parents=True, exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLCONFIGDIR'] = str(ROOT / 'mplconfig')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
sys.dont_write_bytecode = True

import json
import shutil
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
WORK = ROOT.parents[2]
HISTORY = WORK / 'm3_frequency_increment_20261028/m3_true_rescue_12457_20261031'
EXPORTED = ROOT.parent / 'expanded'
sys.path.insert(0, str(HISTORY.parent / 'm3_explained_20261029'))
import trace_relations as tr


def error(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def score_record(name, score):
    z = tr.arr(score)[0]
    return dict(condition=name, margin=float(z[0] - z[1:].max()),
                prediction=int(z.argmax()), logits=z.tolist())


@torch.no_grad()
def main():
    exp, joint = tr.exp, tr.j
    exp.DEVICE = 'cpu'
    exp.current.setup('cpu')
    rows, _, _, rid, arrays = exp.load()
    idx = next(i for i, row in enumerate(rows) if str(row['id']) == '12457')
    assets = json.loads((EXPORTED / 'l3_assets.json').read_text(encoding='utf-8'))
    query = next(x for x in assets['images'] if x['id'] == '12457')
    receipt = json.loads((HISTORY / 'REPLAY_RECEIPT.json').read_text(encoding='utf-8'))
    saved = np.load(HISTORY / 'intermediate.npz')
    removals = pd.read_csv(HISTORY / 'local_removal.csv')
    print('Restoring historical model on CPU; cached query and 16 references only.', flush=True)
    model = tr.review.restore('N_level_pattern', 17)
    model.eval().requires_grad_(False)
    for op in model.m3[0]:
        op.frequency = False
        assert op.output.bias is None
    scores, details = joint.forward(model, 'N_level_pattern', joint.pair_cache(), arrays,
                                   np.array([idx]), rid, rows, details=True)
    original = exp.tensor(arrays['x'], [idx])
    updates = torch.stack([d['update'] for d in details], 1)
    combined = original + updates
    checks = dict(query_feature_max_abs=error(tr.arr(original), saved['query_feature']),
                  total_update_max_abs=error(tr.arr(updates), saved['update']),
                  spatial_logits_max_abs=error(tr.arr(scores)[0], receipt['conditions'][1]['logits']))
    records = [score_record('spatial_full', scores)]
    recovered, summary = {}, {}
    image_dir = ROOT / 'images'
    image_dir.mkdir(exist_ok=True)
    for p, pol in enumerate(('VV', 'VH')):
        d = details[p]
        checks[pol + '_T_max_abs'] = max(error(tr.arr(d['transport'][r]), saved[f'{pol}_transport_{r}']) for r in range(3))
        checks[pol + '_beta_max_abs'] = error(tr.arr(d['reference_weights']), saved[f'{pol}_reference_weights'])
        component, checks[pol + '_decomposition_max_abs'] = tr.decompose(model, p, d, arrays, idx, rid)
        parts = [component(slots=s) for s in exp.tokens.REGIONS]
        checks[pol + '_region_sum_max_abs'] = float((sum(parts) - d['update']).abs().max())
        for name, part in zip(('body', 'side', 'ends'), parts):
            recovered[f'{pol}_{name}_update'] = tr.arr(part)[0]
        axis = parts[2]
        modified = combined.clone()
        modified[:, p] -= axis
        record = score_record('remove_' + pol + '_ends', tr.downstream(model, arrays, idx, modified))
        historical = next(r for r in receipt['interventions'] if r['condition'] == f'移除{pol}轴端回写')
        checks[pol + '_removal_logits_max_abs'] = error(record['logits'], historical['logits'])
        csv_margin = float(removals.loc[removals.condition.eq(f'移除{pol}轴端回写'), 'margin'].iloc[0])
        checks[pol + '_removal_margin_abs'] = abs(record['margin'] - csv_margin)
        record['historical_margin'] = csv_margin
        record['margin_change'] = record['margin'] - records[0]['margin']
        records.append(record)
        view = query['views'][pol]
        geometry = np.load(view['geometry_npz'])
        masks = exp.tensor(arrays['slots'], [idx])[:, p]
        valid = exp.tensor(arrays['valid'], [idx])[:, p]
        clipped = torch.minimum(masks.clamp(0, 1), joint.grid_occupancy(valid, (10, 10)).clamp(0, 1))
        write = clipped / clipped.sum(1, keepdim=True).clamp_min(1)
        integrated = joint.grid_occupancy(torch.from_numpy(geometry['slots'])[None], (10, 10))
        checks[pol + '_native_to_cached_slots_max_abs'] = error(tr.arr(integrated), tr.arr(masks))
        axis_l2 = tr.arr(axis[0].norm(dim=0))
        total_l2 = tr.arr(d['update'][0].norm(dim=0))
        coverage = tr.arr(clipped[0, 16:20].sum(0))
        write_weight = tr.arr(write[0, 16:20].sum(0))
        peak = np.unravel_index(axis_l2.argmax(), axis_l2.shape)
        live = axis_l2 > 1e-8
        summary[pol] = dict(axis_l2_grid=axis_l2.tolist(), total_l2_grid=total_l2.tolist(),
                            axis_coverage_grid=coverage.tolist(), axis_write_weight_grid=write_weight.tolist(),
                            axis_global_l2=float(axis.norm()), total_global_l2=float(d['update'].norm()),
                            axis_peak=float(axis_l2.max()), total_peak=float(total_l2.max()),
                            peak_yx=list(map(int, peak)), nonzero_cells=int(live.sum()),
                            cells=[dict(y=int(y), x=int(x), axis_l2=float(axis_l2[y, x]),
                                        total_l2=float(total_l2[y, x]), write_weight=float(write_weight[y, x]))
                                   for y, x in zip(*np.where(live))])
        recovered.update({f'{pol}_total_update': tr.arr(d['update'])[0],
                          f'{pol}_axis_l2': axis_l2, f'{pol}_total_l2': total_l2,
                          f'{pol}_axis_coverage': coverage, f'{pol}_axis_write_weight': write_weight})
        for kind in ('raw', 'native160'):
            dest = image_dir / f'12457_{pol}_{kind}.png'
            shutil.copyfile(view[kind], dest)
            view[kind] = str(dest)
        shutil.copyfile(view['geometry_npz'], image_dir / f'12457_{pol}_geometry.npz')
        for name, grid in [('axis_l2', axis_l2), ('total_l2', total_l2), ('axis_write_weight', write_weight)]:
            np.savetxt(ROOT / f'{pol}_{name}.csv', grid, delimiter=',', fmt='%.9g')
    absent = tr.downstream(model, arrays, idx, original)
    checks['no_m3_logits_max_abs'] = error(tr.arr(absent)[0], receipt['conditions'][0]['logits'])
    records.append(score_record('all_writeback_off', absent))
    # These checks distinguish exact slot decomposition from a masked total-update picture.
    assert max(checks.values()) < 2e-5, checks
    assert all(r['prediction'] == 0 for r in records[:3]) and records[3]['prediction'] == 3
    assert not torch.cuda.is_initialized()
    np.savez_compressed(ROOT / 'recovered_updates.npz', **recovered)
    result = dict(case='12457', seed=17, candidate='N_level_pattern', m3_frequency=False,
                  status='exact_region_writeback_recovered', device='cpu', threads=2,
                  cuda_initialized=torch.cuda.is_initialized(),
                  historical_source=str(HISTORY), checkpoint=str(joint.HERE / 'training/N_level_pattern_17/m3.pt'),
                  base_checkpoint=str(exp.PACKAGE / 'weights/selected_s17.pt'),
                  cache=str(exp.HERE / 'cache'), pair_cache=str(joint.HERE / 'pair_cache'),
                  reference_ids=[str(rows[i]['id']) for i in rid], checks=checks,
                  scores=records, summary=summary, query=query,
                  definition='Fixed spatial T and beta; keep slots16:20 transported values; keep original all-slot write denominator; apply original bias-free output projection and gain.',
                  scope='Historical cached feature replay only; no current-model or population claim. Norm is not signed class evidence. Ends are detected-axis geometry, not labeled bow/stern.')
    (ROOT / 'recovery.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(checks=checks, scores=records,
                         summary={p: {k: v for k, v in s.items() if not k.endswith('grid') and k != 'cells'}
                                  for p, s in summary.items()}), ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
