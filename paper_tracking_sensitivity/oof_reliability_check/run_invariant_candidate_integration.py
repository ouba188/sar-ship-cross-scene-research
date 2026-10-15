"""Matched development integration of the conditional-mean penalty candidate."""

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'

import gc
import json
import pickle

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from projection_residual_expert import discarded_coordinates, freeze_kernel_draw
from run_ports_projection_completion import matched_views
from run_stable_port_pilot import selective_archive, standardized_rows
from run_conditional_invariance_probe import ROOT, OLD, OUT
from run_generic_transfer import metrics
from sar_joint_completion import SOURCE, write_json
from stable_port_directions import _project


DEST = OUT/'integration'
PARENT = ROOT/'module_completion_20261008/ports_projection'


def freeze(port, base, frame):
    parent = PARENT/port
    old = selective_archive(parent/'matched_joint/joint_policies.npz',
        ('query_rows', 'source_rows', 'support_rows', 'base_scores', 'eval_mask', 'cap'))
    q, source = old['query_rows'], old['source_rows']
    with np.load(OUT/'ports'/port/'frozen.npz') as bank:
        np.testing.assert_array_equal(source, bank['source_rows'])
        np.testing.assert_array_equal(q, bank['query_rows'])
        basis = bank['conditional_penalty_basis']
        center, scale = bank['feature_center'], bank['feature_scale']
    xs = standardized_rows(base, source, center, scale)
    zs = _project(xs, basis)
    zcenter, zscale = zs.mean(0), np.maximum(zs.std(0), 1e-6)
    del xs, zs
    gc.collect()
    z = (_project(standardized_rows(base, q, center, scale), basis)-zcenter)/zscale/np.sqrt(basis.shape[1])
    with (parent/'source_model.pkl').open('rb') as handle:
        model = pickle.load(handle)
    with np.load(parent/'deployment_complement.npz') as saved:
        stats = {k: saved[k] for k in saved.files if k != 'source_rows'}
    with np.load(ROOT/'pred_cache2'/f'{port}.npz') as saved:
        xs = saved['Xs'][q]
    extra = np.load(parent/'matched_joint/joint_extra.npy', mmap_mode='r')
    bd = ((discarded_coordinates(base[q], model['pca'])-stats['discarded_center'][:base.shape[1]])/
          stats['discarded_scale'][:base.shape[1]])
    prior, retained, complement = matched_views(xs, bd, extra, model, stats, True)
    support_local = np.searchsorted(q, old['support_rows'])
    union = np.unique(support_local)
    kernel = .75*(.5*(retained@retained[union].T)+.5*(complement@complement[union].T))
    kernel += .25*(z@z[union].T)+1.
    results = []
    for draw, support in enumerate(support_local):
        np.testing.assert_array_equal(q[support], old['support_rows'][draw])
        result = freeze_kernel_draw(old['base_scores'][draw], prior, kernel[:, np.searchsorted(union, support)],
            support, frame.class_id.to_numpy()[q[support]], model['gate'], q)
        np.testing.assert_array_equal(result['eval_mask'], old['eval_mask'][draw])
        assert result['cap'] == old['cap'][draw]
        results.append(result)
    np.savez_compressed(DEST/f'{port}.npz', **{k: np.stack([r[k] for r in results]) for k in results[0]},
        query_rows=q, source_rows=source, support_rows=old['support_rows'], base_scores=old['base_scores'],
        projected_center=zcenter, projected_scale=zscale)
    print('FROZEN integration '+port, flush=True)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    if (DEST/'COMPLETE.json').exists():
        raise ValueError('Preserve completed experiment')
    write_json(DEST/'PROTOCOL.json', dict(candidate='conditional_penalty',
        kernel='.75 Kjoint + .25 Kcandidate + 1; same weight as prior stable-direction pilot',
        baseline='Unchanged full joint, Base64, 64 supports, ridge8, fixed gate, 5percent cap',
        scope='23-port development integration after descriptive diagnostic; not independent confirmation',
        query_selected_parameters=False, unchanged_main_model=True))
    inventory = json.loads((ROOT/'oof_base_adapt_results_20261007/POLICIES_SAVED.json').read_text(encoding='utf-8'))
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id', 'port'], keep_default_na=False)
    base = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    with threadpool_limits(limits=2):
        for item in inventory:
            if not (DEST/f'{item["port"]}.npz').exists():
                freeze(item['port'], base, frame)
    write_json(DEST/'ALL_FROZEN.json', dict(ports=23))
    rows = []
    for item in inventory:
        port = item['port']
        with np.load(DEST/f'{port}.npz') as p:
            y = frame.class_id.to_numpy()[p['query_rows']]
            for draw, ev in enumerate(p['eval_mask']):
                full = metrics(y[ev], p['predictions'][draw, ev], 8)
                expert = metrics(y[ev], p['expert_predictions'][draw, ev], 8)
                rows.append(dict(port=port, draw=draw, arm='conditional_penalty',
                    acc=full['acc'], ba=full['ba'], expert_acc=expert['acc'], expert_ba=expert['ba'],
                    selected=int(p['selected'][draw].sum()), evaluated=int(ev.sum())))
    df = pd.concat([pd.read_csv(OLD/'metrics.csv'), pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(DEST/'metrics.csv', index=False, encoding='utf-8-sig')
    pm = df.groupby(['port', 'arm'])[['acc', 'ba', 'expert_acc', 'expert_ba']].mean().reset_index()
    pm.to_csv(DEST/'port_means.csv', index=False, encoding='utf-8-sig')
    mean = pm.groupby('arm')[['acc', 'ba', 'expert_acc', 'expert_ba']].mean().reset_index()
    mean.to_csv(DEST/'equal_port_means.csv', index=False, encoding='utf-8-sig')
    write_json(DEST/'COMPLETE.json', dict(ports=23, new_rows=len(rows), comparisons=mean.to_dict('records'),
        independent_confirmation=False, main_model_changed=False))
    print(mean.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
