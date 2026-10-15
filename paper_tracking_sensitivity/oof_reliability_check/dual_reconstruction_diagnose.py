"""Bounded F0921 negative-result diagnosis: prior versus support-kernel path."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from dual_reconstruction_core import OUT as PARENT, ROOT, Inputs, reconstruct, write, BATCH, KINDS
from dual_reconstruction_predict import view, kernel, base_block
from backbone_ablation_core import r50_coordinates, r50_views
from unified_head_core import coefficients, late
from backbone_ablation_report import ba

OUT = PARENT/'f0921_diagnosis'
ARMS = ('RR', 'SR', 'RS', 'SS')
FOLDER = ROOT/'learned_frequency_residual_20261009/tasks/f0921'


def main():
    write(OUT/'PROTOCOL.json', dict(dataset='f0921', reason='Both fixed source-only reconstruction replacements hurt matched R50-only BA',
        factors='R=R50-only; S=student reconstruction; first letter prior, second kernel; both mapper kinds',
        fixed='Same source fits, original5x64 labels, ridge8 and full base; no parameter changes',
        goal='Distinguish source-prior versus support-relationship computational contribution to failure',
        feature_shift='Average source-standardized squared input coordinate and fraction beyond3 source standard deviations',
        boundary='Posthoc failure diagnosis, not a target-selected replacement or a physical causal explanation'))
    inputs = Inputs(FOLDER)
    dst = PARENT/'targets/f0921/f0921'
    with np.load(FOLDER/'target.npz') as z:
        data = {k: z[k] for k in ('phi', 'mean', 'conditions', 'prior', 'support', 'support_labels', 'eval_masks', 'rows', 'ids')}
    with np.load(FOLDER/'selective_embedding.npz') as z:
        embed = z['embedding'].astype(float)
    with np.load(dst/'mapper.npz') as z:
        mapper = {k: z[k] for k in z.files}
    with np.load(dst/'coefficients.npz') as z:
        student_heads = {k: dict(coef=z[k+'_source_coef'], intercept=z[k+'_source_intercept']) for k in KINDS}
    with np.load(ROOT/'backbone_ablation_20261010/targets/f0921/f0921/coefficients.npz') as z:
        rh = dict(coef=z['r50_coef'], intercept=z['r50_intercept'])
    with np.load(dst/'predictions.npz') as z:
        student_predictions, student_modes = z['predictions'], list(z['modes'])
    with np.load(ROOT/'backbone_ablation_20261010/targets/f0921/f0921/predictions.npz') as z:
        old_predictions, old_modes = z['predictions'], list(z['modes'])
    def rview(index):
        raw = inputs.target_raw_block(index)
        ret, comp = r50_views(raw, inputs.model, inputs.ctx.stats)
        prior = (r50_coordinates(raw, inputs.model)@rh['coef']+rh['intercept']).astype(np.float32).astype(float)
        return dict(ret=ret, comp=comp, prior=prior)
    ref = np.unique(data['support'])
    locs = [np.searchsorted(ref, s) for s in data['support']]
    rv = rview(ref)
    kr = kernel(rv, rv)
    kb = base_block(data, embed, ref, ref, inputs.info['eta'])
    n = len(inputs.rows)
    predictions = np.empty((2, 5, 4, n), np.int8)
    shifts = []
    for domain, count, reader in [('source', len(inputs.source_rows), inputs.source_x), ('target', n, inputs.target_x)]:
        sq = outside = 0.
        for lo in range(0, count, BATCH):
            x = (reader(slice(lo, min(lo+BATCH, count)))-mapper['x_mean'])/mapper['x_std']
            sq += float(np.sum(x.astype(float)**2))
            outside += int(np.sum(np.abs(x) > 3))
        shifts.append(dict(domain=domain, mean_squared_standardized_input=sq/(count*4096),
                           fraction_coordinates_outside3std=outside/(count*4096)))
    for ki, kind in enumerate(KINDS):
        sv = view(inputs, reconstruct(inputs.target_x(ref), mapper, kind), ref,
                  student_heads[kind], None, None)
        ks = kernel(sv, sv)
        states = []
        for draw, loc in enumerate(locs):
            sy = data['support_labels'][draw]
            prior = {'R': rv['prior'][loc], 'S': sv['prior'][loc]}
            kernels = {'R': kr[np.ix_(loc, loc)], 'S': ks[np.ix_(loc, loc)]}
            ab = coefficients(kb[np.ix_(loc, loc)], data['prior'][data['support'][draw]], sy, 8.)
            states.append((ab, {arm: coefficients(kernels[arm[1]], prior[arm[0]], sy, 8.) for arm in ARMS}))
        for lo in range(0, n, BATCH):
            s = slice(lo, min(lo+BATCH, n))
            r = rview(s)
            v = view(inputs, reconstruct(inputs.target_x(s), mapper, kind), s,
                     student_heads[kind], None, None)
            priors = {'R': r['prior'], 'S': v['prior']}
            kernels = {'R': kernel(r, rv), 'S': kernel(v, sv)}
            b = base_block(data, embed, s, ref, inputs.info['eta'])
            for draw, loc in enumerate(locs):
                ab, alphas = states[draw]
                base = data['prior'][s]+b[:, loc]@ab
                for ai, arm in enumerate(ARMS):
                    e = priors[arm[0]]+kernels[arm[1]][:, loc]@alphas[arm]
                    predictions[ki, draw, ai, s] = late(base, e).argmax(1)
        np.testing.assert_array_equal(predictions[ki, :, 0], old_predictions[:, old_modes.index('r50_late')])
        np.testing.assert_array_equal(predictions[ki, :, 3], student_predictions[:, student_modes.index(kind+'_late')])
        print('F NEGATIVE DIAGNOSIS', kind, 'diagonals exact', flush=True)
    np.savez_compressed(OUT/'predictions.npz', predictions=predictions, arms=np.array(ARMS),
        kinds=np.array(KINDS), rows=data['rows'], ids=data['ids'], support=data['support'], eval_masks=data['eval_masks'])
    write(OUT/'FROZEN.json', dict(diagonals_exact=True, source_mapper_unchanged=True))
    manifest = pd.read_csv(ROOT/'f0921_transfer_20261007/target_manifest.csv', usecols=['sample_id', 'class_id'], keep_default_na=False)
    np.testing.assert_array_equal(manifest.sample_id.to_numpy(str)[data['rows']], data['ids'])
    truth = manifest.class_id.to_numpy(int)[data['rows']]
    records = []
    for ki, kind in enumerate(KINDS):
        for draw, ev in enumerate(data['eval_masks']):
            for ai, arm in enumerate(ARMS):
                p, y = predictions[ki, draw, ai, ev], truth[ev]
                records.append(dict(kind=kind, draw=draw, arm=arm, oa=100*np.mean(p == y), ba=ba(y, p)))
    table = pd.DataFrame(records)
    table.to_csv(OUT/'metrics.csv', index=False)
    summary = table.groupby(['kind', 'arm'])[['oa', 'ba']].mean().reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    write(OUT/'COMPLETE.json', dict(diagonals_exact=True, input_shift=shifts,
        posthoc_diagnostic=True, no_tuning=True, target_best_replacement_selected=False))
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(shifts), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
