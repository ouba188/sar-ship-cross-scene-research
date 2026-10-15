"""Mine source-port-consistent R50 contrasts and test the frozen joint expert."""

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'

import argparse
import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from projection_residual_expert import discarded_coordinates, freeze_kernel_draw
from run_ports_projection_completion import matched_views
from run_generic_transfer import metrics
from sar_joint_completion import SOURCE, write_json
from stable_port_directions import fit_direction_bank, transform_direction


ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
PARENT = ROOT/'module_completion_20261008/ports_projection'
OUT = ROOT/'stable_port_directions_20261008'
ARMS = ('joint_reference', 'kernel_shrink_control', 'pooled_direction', 'stable_direction')


def selective_archive(path, keys):
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k] for k in keys}


def standardized_rows(base, rows, center, scale):
    result = np.array(base[rows], dtype=np.float32, copy=True)
    np.subtract(result, center, out=result)
    np.divide(result, scale, out=result)
    return result


def freeze_port(port, out, base, frame):
    started = time.monotonic()
    destination = out/'ports'/port
    destination.mkdir(parents=True, exist_ok=True)
    parent = PARENT/port
    old = selective_archive(parent/'matched_joint/joint_policies.npz',
        ('query_rows', 'source_rows', 'support_rows', 'base_scores', 'predictions',
         'expert_predictions', 'expert_scores', 'eval_mask', 'selected', 'cap'))
    q, source = old['query_rows'], old['source_rows']
    assert np.intersect1d(q, source).size == 0
    assert (frame.port.to_numpy()[q] == port).all()
    assert (frame.port.to_numpy()[source] != port).all()
    with (parent/'source_model.pkl').open('rb') as handle:
        model = pickle.load(handle)
    with np.load(parent/'deployment_complement.npz', allow_pickle=False) as saved:
        stats = {k: saved[k] for k in saved.files if k != 'source_rows'}
    # R50 is unchanged by the per-port M3 training; no mismatched R18 source
    # features are used to discover a direction for the current joint expert.
    with np.load(parent/'source_geometry.npz', allow_pickle=False) as geometry:
        np.testing.assert_array_equal(geometry['source_rows'], source)
        center, scale = geometry['raw_center'][:base.shape[1]], geometry['raw_scale'][:base.shape[1]]
    xsource = standardized_rows(base, source, center, scale)
    bank = fit_direction_bank(xsource, frame.class_id.to_numpy()[source], frame.port.to_numpy()[source])
    del xsource
    gc.collect()
    np.savez_compressed(destination/'source_direction_bank.npz', **bank, source_rows=source,
                        feature_center=center, feature_scale=scale)
    xquery = standardized_rows(base, q, center, scale)
    projected = {kind: transform_direction(xquery, bank, kind) for kind in ('pooled', 'stable')}
    del xquery
    with np.load(ROOT/'pred_cache2'/f'{port}.npz', allow_pickle=False) as saved:
        xs = saved['Xs'][q]
    extra = np.load(parent/'matched_joint/joint_extra.npy', mmap_mode='r')
    bd = ((discarded_coordinates(base[q], model['pca'])-stats['discarded_center'][:base.shape[1]])/
          stats['discarded_scale'][:base.shape[1]])
    prior, retained, complement = matched_views(xs, bd, extra, model, stats, True)
    union = np.unique(np.searchsorted(q, old['support_rows'].ravel()))
    kernel = .5*(retained@retained[union].T)+.5*(complement@complement[union].T)
    kernels = {'joint_reference': kernel+1., 'kernel_shrink_control': .75*kernel+1.}
    for kind in ('pooled', 'stable'):
        kernels[kind+'_direction'] = .75*kernel+.25*(projected[kind]@projected[kind][union].T)+1.
    arrays = {name: [] for name in ARMS}
    local_supports = np.searchsorted(q, old['support_rows'])
    max_error = 0.
    for draw, support in enumerate(local_supports):
        np.testing.assert_array_equal(q[support], old['support_rows'][draw])
        labels = frame.class_id.to_numpy()[q[support]]
        columns = np.searchsorted(union, support)
        for name in ARMS:
            result = freeze_kernel_draw(old['base_scores'][draw], prior, kernels[name][:, columns],
                                        support, labels, model['gate'], q)
            np.testing.assert_array_equal(result['eval_mask'], old['eval_mask'][draw])
            assert result['cap'] == old['cap'][draw]
            assert result['selected'].sum() <= result['cap']
            assert not result['selected'][~result['eval_mask']].any()
            if name == 'joint_reference':
                np.testing.assert_array_equal(result['predictions'], old['predictions'][draw])
                np.testing.assert_array_equal(result['expert_predictions'], old['expert_predictions'][draw])
                np.testing.assert_array_equal(result['selected'], old['selected'][draw])
                error = float(np.max(np.abs(result['expert_scores']-old['expert_scores'][draw])))
                max_error = max(error, max_error)
                np.testing.assert_allclose(result['expert_scores'], old['expert_scores'][draw], atol=1e-8, rtol=0)
            arrays[name].append(result)
    for name, records in arrays.items():
        np.savez_compressed(destination/f'{name}_policies.npz',
            **{k: np.stack([r[k] for r in records]) for k in records[0]}, query_rows=q,
            source_rows=source, support_rows=old['support_rows'], base_scores=old['base_scores'])
    np.savez_compressed(destination/'heldout_direction_scores.npz', query_rows=q,
        support_local_rows=local_supports, eval_mask=old['eval_mask'], pairs=bank['pairs'], **projected)
    descriptor = dict(port=port, query_rows=len(q)-64, target_rows=len(q), source_rows=len(source),
        directions=len(bank['pairs']), source_only_direction_fit=True,
        source_preprocessing='Existing outer-source R50 mean/std; same outer source identity purge',
        joint_replay_exact=True, max_joint_score_error=max_error,
        source_mean_leave_port_cosine=float(np.mean(bank['source_cosine_mean'])),
        source_mean_positive_cosine_fraction=float(np.mean(bank['source_cosine_positive_fraction'])),
        source_mean_coherence=float(np.mean(bank['coherence_mean'])),
        elapsed_seconds=time.monotonic()-started)
    write_json(destination/'FROZEN.json', descriptor)
    print(f'FROZEN {port}: {descriptor["directions"]} directions, exact joint replay, '
          f'{descriptor["elapsed_seconds"]:.1f}s', flush=True)
    return descriptor


def evaluate(out, records, frame):
    rows, recall_rows, pair_rows = [], [], []
    labels = frame.class_id.to_numpy(int)
    for record in records:
        folder = out/'ports'/record['port']
        for arm in ARMS:
            policy = selective_archive(folder/f'{arm}_policies.npz',
                ('query_rows', 'base_scores', 'predictions', 'expert_predictions', 'eval_mask', 'selected'))
            y = labels[policy['query_rows']]
            for draw in range(5):
                ev = policy['eval_mask'][draw]
                pred, expert = policy['predictions'][draw], policy['expert_predictions'][draw]
                base = policy['base_scores'][draw].argmax(1)
                m, em = metrics(y[ev], pred[ev], 8), metrics(y[ev], expert[ev], 8)
                rows.append(dict(port=record['port'], draw=draw, arm=arm, acc=m['acc'], ba=m['ba'],
                    expert_acc=em['acc'], expert_ba=em['ba'], selected=int(policy['selected'][draw].sum()),
                    evaluated=int(ev.sum()), rescued=int(np.sum(ev & (pred == y) & (base != y))),
                    harmed=int(np.sum(ev & (pred != y) & (base == y)))))
                for c in range(8):
                    recall_rows.append(dict(port=record['port'], draw=draw, arm=arm, class_id=c,
                        query_count=int(np.sum(ev & (y == c))), recall=m['class_recall'][c],
                        expert_recall=em['class_recall'][c]))
        with np.load(folder/'heldout_direction_scores.npz', allow_pickle=False) as saved:
            y = labels[saved['query_rows']]
            for draw, ev in enumerate(saved['eval_mask']):
                for i, pair in enumerate(saved['pairs']):
                    c, d = pair
                    selected = ev & ((y == c) | (y == d))
                    if np.sum(ev & (y == c)) < 5 or np.sum(ev & (y == d)) < 5:
                        continue
                    truth = y[selected] == c
                    pair_rows.append(dict(port=record['port'], draw=draw, class_c=int(c), class_d=int(d),
                        class_c_query=int(truth.sum()), class_d_query=int((~truth).sum()),
                        pooled_auc=float(roc_auc_score(truth, saved['pooled'][selected, i])),
                        stable_auc=float(roc_auc_score(truth, saved['stable'][selected, i]))))
    pd.DataFrame(rows).to_csv(out/'metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(recall_rows).to_csv(out/'class_recall.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(pair_rows).to_csv(out/'heldout_pair_auc.csv', index=False, encoding='utf-8-sig')
    assert len(rows) == 23*5*len(ARMS)
    write_json(out/'COMPLETE.json', dict(status='complete', ports=23, draws=5, arms=list(ARMS),
        evaluation_rows=len(rows), target_pair_auc_rows=len(pair_rows),
        joint_replay_exact_all_ports=True, query_labels_used_in_fitting=False,
        no_new_target_labels=True, no_encoder_training=True, causal_identification=False))
    print('COMPLETE: 23 ports, five supports, four matched arms', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out/'COMPLETE.json').exists():
        raise ValueError('Preserve completed experiment; specify a new output for a new protocol')
    inventory = json.loads((ROOT/'oof_base_adapt_results_20261007/POLICIES_SAVED.json').read_text(encoding='utf-8'))
    assert len(inventory) == 23
    protocol = dict(date='2026-10-08', task='Source-port stable discriminative direction discovery',
        scope='All23 eligible ports from original24 corpus; external sets not run',
        backbone='Frozen R50 raw2048; shared coordinates across all ports. Existing M3 R18 unchanged.',
        discovery='Outer source-only, same MMSI/product purge; equal-port within-class-pair mean contrasts',
        contrast='For class c<d, port delta=mu_c-mu_d; each class>=5 rows; at least3 eligible ports',
        pooled='L2-normalized equal-port mean delta',
        stable='L2-normalized mean_delta * mean_delta^2 / mean(delta^2), coordinatewise',
        normalization='Parent outer-source R50 mean/std; each projected direction source-centered/std-scaled; divide sqrt(pair count)',
        arms=list(ARMS), baseline='Exact current full joint M2+M3, unchanged Base64, source prior, 64labels and gate',
        kernels={'joint_reference': 'Kjoint+1', 'kernel_shrink_control': '.75Kjoint+1',
                 'pooled_direction': '.75Kjoint+.25Kpooled+1', 'stable_direction': '.75Kjoint+.25Kstable+1'},
        kernel_joint='.5Kretained+.5Kcomplement, bias excluded here; exactly one shared bias',
        residual_ridge=8., labels_per_task_draw=64, draws=list(range(5)), budget=.05,
        target_pair_readout='ROC AUC of fixed oriented class-pair direction on held-port query, pair counts>=5 each',
        source_cosine_scope='Descriptive leave-port contrast direction, conditional on outer-source preprocessing',
        no_target_query_model_selection=True, no_parameter_grid=True, encoder_training=False,
        labels_available_only_to_fit='Purged outer-source labels and the original64 support labels',
        independent_confirmation=False, causal_identification=False,
        main_model_unchanged=True, all_policies_frozen_before_query_readout=True)
    if (out/'PROTOCOL.json').exists():
        assert json.loads((out/'PROTOCOL.json').read_text(encoding='utf-8')) == protocol
    write_json(out/'PROTOCOL.json', protocol)
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id', 'class_name', 'port'], keep_default_na=False)
    base = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    frozen = []
    with threadpool_limits(limits=2):
        for item in inventory:
            folder = out/'ports'/item['port']
            if (folder/'FROZEN.json').exists():
                frozen.append(json.loads((folder/'FROZEN.json').read_text(encoding='utf-8')))
            else:
                frozen.append(freeze_port(item['port'], out, base, frame))
        write_json(out/'POLICIES_SAVED.json', frozen)
        evaluate(out, frozen, frame)


if __name__ == '__main__':
    main()
