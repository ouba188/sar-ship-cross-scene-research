"""Bounded independent replay of the fixed visual-tail expert experiment."""

import argparse
import csv
import hashlib
import json
import pickle
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.linalg import qr, solve, solve_triangular
from threadpoolctl import threadpool_limits

from verify_residual_expert import archive, close, read_json, require
from verify_receiver_consistent_handoff import confidence_parts, replay_mask, replay_score
from verify_local_metric_expert import compare_metric_rows, metric_replay, MEASURES


ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = Path('E:/临时会话/visual_reliable_baseline')
VV_PATH = WORKSPACE / 'artifacts/eight_class_adaptive_20260916/pooled_features.npy'
VH_PATH = WORKSPACE / 'b2_ssl_feats_38091.npz'
MANIFEST = VV_PATH.with_name('manifest.csv')
PARENT = ROOT / 'oof_residual_expert_results_20261007_v2'
FROZEN = ROOT / 'oof_receiver_consistent_results_20261007'
GATE = ROOT / 'oof_context_results_20261007'
ADVANTAGE = ROOT / 'oof_action_advantage_results_20261007'
NEW_ARMS = ('visual_tail', 'energy_control')
EXPERT_ARMS = NEW_ARMS + ('linear', 'bias')
BUDGET_ARMS = EXPERT_ARMS + ('advantage_reference',)
REFIT_PORTS = ('Antwerp-Bruges', 'Mombasa')


def input_receipts(ports, config):
    inputs = {'manifest': MANIFEST, 'vv_features': VV_PATH, 'vh_features': VH_PATH}
    digests = {}
    for name, path in inputs.items():
        require(Path(config[name]).resolve() == path.resolve(), 'configured input changed: ' + name)
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
                digest.update(block)
        digests[name + '_sha256'] = digest.hexdigest()
    require(Path(config['source_receipts']).resolve() == (ROOT / 'oof_predictions_v1').resolve(),
            'existing OOF receipt directory changed')
    for port in ports:
        with np.load(ROOT / 'oof_predictions_v1' / (port + '.npz'), allow_pickle=False) as data:
            receipt = json.loads(str(data['metadata_json'].item()))
        require(receipt['complete'] and receipt['outer_port'] == port, port + ': complete original OOF receipt')
        for key, value in digests.items():
            require(receipt[key] == value, port + ': original input receipt mismatch: ' + key)


def retained_geometry(raw, source, parent, refit, context):
    center, scale = parent['source_center'], parent['source_scale']
    require(raw.shape == (38091, 192) and np.all(scale > 0), context + ': PCA geometry')
    if refit:
        close(center, raw[source].mean(0), context + ': inherited source center refit')
        close(scale, raw[source].std(0) + 1e-6, context + ': inherited source scale refit')
    # NumPy 2.x promotes the explicit np.float64 sqrt scalar after standardization.
    standardized = (raw - center) / scale
    return standardized.astype(np.float64) / np.sqrt(192.)


def replay_tail(raw, low, source, query, geometry, prefix, refit, context):
    def saved(name):
        return geometry[prefix + '_' + name]

    center, low_center, decoder = saved('source_center'), saved('retained_center'), saved('decoder')
    require(center.shape == (raw.shape[1],) and low_center.shape == (low.shape[1],)
            and decoder.shape == (low.shape[1], raw.shape[1]), context + ': decoder dimensions')
    multiplier = float(saved('multiplier'))
    require(np.isfinite(decoder).all() and np.isfinite(multiplier) and multiplier > 0,
            context + ': finite decoder and positive normalization')
    energy, residual_energy = float(saved('source_retained_energy')), float(saved('source_tail_energy'))
    require(energy > 0 and residual_energy > 0, context + ': positive energies')
    close(multiplier ** 2 * residual_energy, energy, context + ': saved energy normalization')
    errors = {}
    if refit:
        high_source = np.asarray(raw[source], dtype=np.float64)
        source_center = high_source.mean(axis=0)
        high_source -= source_center
        low_source = low[source].copy()
        source_low_center = low_source.mean(axis=0)
        source_energy = np.einsum('ij,ij->', low_source, low_source) / len(source)
        low_source -= source_low_center
        # QR least squares avoids the producer's normal-equation decoder solve.
        basis, triangular = qr(low_source, mode='economic')
        fitted = solve_triangular(triangular, basis.T @ high_source)
        close(center, source_center, context + ': raw source center', atol=1e-11)
        close(low_center, source_low_center, context + ': low source center', atol=1e-12)
        close(decoder, fitted, context + ': independent QR decoder', atol=1e-9)
        errors['decoder'] = float(np.max(np.abs(decoder - fitted)))
        high_source -= low_source @ fitted
        tail_energy = np.einsum('ij,ij->', high_source, high_source) / len(source)
        normalized = np.sqrt(source_energy / tail_energy)
        cross = np.max(np.abs(low_source.T @ high_source / len(source)))
        close(energy, source_energy, context + ': source retained energy')
        close(residual_energy, tail_energy, context + ': independently refit tail energy', atol=1e-10)
        close(multiplier, normalized, context + ': independently refit multiplier')
        close(saved('source_orthogonality_max'), cross, context + ': orthogonality diagnostic', atol=1e-10)
        require(cross < 1e-8, context + ': source tail is not orthogonal to retained coordinates')
        errors['source_orthogonality'] = float(cross)
    high_query = np.asarray(raw[query], dtype=np.float64) - center
    low_query = low[query] - low_center
    return (high_query - low_query @ decoder) * multiplier, errors


def support_replay(design, support, labels, prior):
    counts = np.bincount(labels, minlength=8)
    weights = 64. / (np.count_nonzero(counts) * counts[labels])
    response = -np.ones((64, 8)) - prior[support]
    response[np.arange(64), labels] += 2.
    x = design[support]
    # Unscaled dual system, distinct from the producer's sqrt-weight formulation.
    dual = solve(x @ x.T + np.diag(8. / weights), response, assume_a='pos')
    coefficient = x.T @ dual
    return prior + design @ coefficient, coefficient


def verify_summary(summary, expected, ports):
    require(summary['n_ports'] == 23 and summary['contexts'] == 115
            and summary['policy_metric_rows'] == 1035 and summary['target_label_budget'] == 64,
            'summary coverage')
    require(summary['independent_confirmation'] is False and summary['novelty_established'] is False,
            'summary interpretation')
    values = {}
    for mode, arms in (('full', EXPERT_ARMS), ('budget', BUDGET_ARMS)):
        for arm in arms:
            fields = MEASURES + (('admission_fraction',) if mode == 'budget' else ())
            values[mode, arm] = {field: np.array([[expected[port, seed, mode, arm][field]
                                                  for seed in range(5)] for port in ports])
                                 for field in fields}
    require(len(summary['means']) == 9, 'summary mean count')
    seen = set()
    for row in summary['means']:
        key = row['mode'], row['arm']
        require(key in values and key not in seen, 'summary mean identity')
        seen.add(key)
        for field, array in values[key].items():
            close(row[field], array.mean(1).mean(), str(key) + ': mean ' + field)
    pairs = [('visual_tail', 'linear'), ('visual_tail', 'energy_control'),
             ('visual_tail', 'bias'), ('energy_control', 'linear')]
    keys = {(mode, a, b) for mode in ('full', 'budget') for a, b in pairs}
    keys.add(('budget', 'visual_tail', 'advantage_reference'))
    require(len(summary['comparisons']) == len(keys), 'summary contrast count')
    samples = np.random.default_rng(0).integers(0, 23, (10000, 23))
    seen = set()
    for row in summary['comparisons']:
        key = row['mode'], row['candidate'], row['control']
        require(key in keys and key not in seen, 'summary contrast identity')
        seen.add(key)
        mode, a, b = key
        delta = values[mode, a]['ba'] - values[mode, b]['ba']
        by_port = delta.mean(1)
        measures = dict(mean_delta_ba_pp=100 * by_port.mean(),
                        mean_delta_acc_pp=100 * (values[mode, a]['acc'] - values[mode, b]['acc']).mean(1).mean(),
                        positive_ports=int((by_port > 1e-12).sum()),
                        negative_ports=int((by_port < -1e-12).sum()),
                        draw_deltas_pp=100 * delta.mean(0),
                        port_bootstrap_95ci_pp=100 * np.percentile(by_port[samples].mean(1), [2.5, 97.5]))
        for field, value in measures.items():
            close(row[field], value, str(key) + ': ' + field)
        require(set(row['port_deltas_pp']) == set(ports), 'contrast port inventory')
        close([row['port_deltas_pp'][port] for port in ports], 100 * by_port, str(key) + ': port deltas')
    primary = all((values['budget', 'visual_tail']['ba'] - values['budget', arm]['ba']).mean() > 0
                  for arm in ('linear', 'energy_control', 'advantage_reference'))
    require(summary['passes_development_screen'] == bool(primary), 'three-control primary screen')


def verify(run):
    require((run / 'POLICIES_SAVED.json').is_file() and (run / 'COMPLETE.json').is_file(),
            'wait for all policies and final readout before verification')
    inventory = read_json(run / 'POLICIES_SAVED.json')
    require(inventory == read_json(PARENT / 'POLICIES_SAVED.json'), 'parent inventory changed')
    ports = [item['port'] for item in inventory]
    require(len(ports) == len(set(ports)) == 23 and set(REFIT_PORTS).issubset(ports), 'fixed port inventory')
    config = read_json(run / 'RUN_CONFIG.json')
    require(tuple(config['expert_arms']) == EXPERT_ARMS
            and tuple(config['budget_arms']) == BUDGET_ARMS, 'configured arms')
    for key, value in dict(labels=64, strength=8., budget=.05).items():
        require(config[key] == value, 'fixed configuration: ' + key)
    complete = read_json(run / 'COMPLETE.json')
    require(complete['status'] == 'complete' and complete['new_target_fits'] == 230
            and complete['source_decoder_fits'] == 46 and complete['new_backbone_pca_head_fits'] == 0
            and complete['policy_metric_rows'] == 1035, 'completion coverage')
    input_receipts(ports, config)
    vv = np.asarray(np.load(VV_PATH, mmap_mode='r', allow_pickle=False), dtype=np.float32)
    with np.load(VH_PATH, allow_pickle=False) as data:
        vh = np.asarray(data['F'], dtype=np.float32)
    require(vv.shape == (38091, 2048) and vh.shape == (38091, 512), 'raw feature shapes')
    expected, fits, refits = {}, 0, []
    errors = dict(score=0., coefficient=0., gate=0., decoder=0., source_orthogonality=0.)
    for port in ports:
        original = archive(ROOT / 'pred_cache' / (port + '.npz'))
        visual = archive(ROOT / 'pred_cache2' / (port + '.npz'))
        parent = archive(PARENT / port / 'policies.npz')
        frozen = archive(FROZEN / port / 'policies.npz')
        advantage = archive(ADVANTAGE / port / 'policies.npz')
        saved = archive(run / port / 'policies.npz')
        geometry = archive(run / port / 'geometry.npz')
        source, q = original['spos'], original['qpos']
        n = len(q)
        require(np.array_equal(q, np.unique(q)) and len(source) == len(np.unique(source))
                and not np.isin(source, q).any(), port + ': source/query rows')
        require(np.all(original['ports_all'][source] != port) and np.all(original['ports_all'][q] == port),
                port + ': source/query ports')
        require(np.array_equal(geometry['source_rows'], source), port + ': geometry source rows')
        for data in (saved, parent, frozen, advantage):
            require(np.array_equal(data['query_rows'], q) and np.array_equal(data['seeds'], np.arange(5)),
                    port + ': query/seeds alignment')
            require(data['support_rows'].shape == (5, 64) and data['eval_masks'].shape == (5, n)
                    and data['eval_masks'].dtype == np.bool_, port + ': support/eval dimensions')
        require(tuple(saved['expert_names']) == EXPERT_ARMS and tuple(saved['budget_names']) == BUDGET_ARMS,
                port + ': policy arm axes')
        for key, shape in (('scores', (5, 4, n, 8)), ('visual_tail_correction', (5, 2753, 8)),
                           ('energy_control_correction', (5, 193, 8)), ('gate_scores', (5, 4, n)),
                           ('expert_predictions', (5, 4, n)), ('selected', (5, 5, n)),
                           ('predictions', (5, 5, n)), ('caps', (5,))):
            require(saved[key].shape == shape and np.isfinite(saved[key]).all(), port + ': shape/finite ' + key)
        require(saved['selected'].dtype == np.bool_, port + ': boolean selected masks')
        for key in ('expert_predictions', 'predictions', 'base_prediction'):
            require(np.issubdtype(saved[key].dtype, np.integer)
                    and np.all((saved[key] >= 0) & (saved[key] < 8)), port + ': prediction columns')
        require(visual['Xs'].shape == (38091, 128) and visual['Z'].shape == (38091, 64),
                port + ': retained feature shapes')
        raw = np.column_stack((visual['Xs'], visual['Z']))
        low = retained_geometry(raw, source, parent, port in REFIT_PORTS, port)
        blocks = []
        for prefix, high, retained in (('vv', vv, low[:, :128]), ('vh', vh, low[:, 128:])):
            tail, geometry_errors = replay_tail(high, retained, source, q, geometry, prefix,
                                                port in REFIT_PORTS, port + '/' + prefix)
            blocks.append(tail)
            for key, value in geometry_errors.items():
                errors[key] = max(errors[key], value)
        if port in REFIT_PORTS:
            refits.append(port)
        designs = (np.column_stack((low[q], *blocks, np.ones(n))),
                   np.column_stack((np.sqrt(2.) * low[q], np.ones(n))))
        prior = original['expert_logits'][q].astype(np.float64)
        base_logits = original['base_logits'][q]
        base = base_logits.argmax(1)
        require(np.array_equal(saved['base_prediction'], base), port + ': unchanged base')
        with (GATE / port / 'source_model.pkl').open('rb') as handle:
            gate = pickle.load(handle)['prior']
        for field in ('center', 'scale', 'lr'):
            close(frozen['source_' + field], gate[field], port + ': inherited gate ' + field, atol=0, rtol=0)
        base_parts = confidence_parts(base_logits)
        budget_index = list(frozen['budgets']).index(.05)
        control_names = ('anchor_visual64', 'anchor_bias64')
        score_indices = [list(parent['score_names']).index(name) for name in control_names]
        expert_indices = [list(frozen['expert_names']).index(name) for name in control_names]
        advantage_index = list(advantage['arm_names']).index('advantage_visual')
        for seed in range(5):
            context = port + '/' + str(seed)
            support = np.random.default_rng(seed).choice(q, 64, replace=False)
            ev = ~np.isin(q, support)
            for data in (saved, parent, frozen, advantage):
                require(np.array_equal(data['support_rows'][seed], support)
                        and np.array_equal(data['eval_masks'][seed], ev), context + ': support/evaluation')
            cap = int(np.floor(.05 * ev.sum()))
            require(saved['caps'][seed] == frozen['caps'][seed, budget_index] == advantage['caps'][seed] == cap,
                    context + ': fixed 5% cap')
            local = np.searchsorted(q, support)
            require(np.array_equal(q[local], support), context + ': support position mapping')
            labels = original['y_ids'][support]
            observed = np.bincount(labels, minlength=8) > 0
            for ai, (design, coefficient_name) in enumerate(zip(designs, ('visual_tail_correction', 'energy_control_correction'))):
                scores, coefficients = support_replay(design, local, labels, prior)
                close(saved[coefficient_name][seed], coefficients, context + ': independent coefficients', atol=1e-9)
                close(saved['scores'][seed, ai], scores, context + ': independent scores', atol=1e-9)
                require(np.array_equal(saved['expert_predictions'][seed, ai], scores.argmax(1)),
                        context + ': independent receiver argmax')
                errors['coefficient'] = max(errors['coefficient'], float(np.abs(saved[coefficient_name][seed] - coefficients).max()))
                errors['score'] = max(errors['score'], float(np.abs(saved['scores'][seed, ai] - scores).max()))
                fits += 1
            for offset, (si, ei) in enumerate(zip(score_indices, expert_indices)):
                ai = 2 + offset
                require(np.array_equal(saved['scores'][seed, ai], parent['scores'][seed, si]), context + ': unchanged control scores')
                for key in ('selected', 'predictions'):
                    require(np.array_equal(saved[key][seed, ai], frozen[key][seed, budget_index, ei]),
                            context + ': unchanged control ' + key)
                close(saved['gate_scores'][seed, ai], frozen['gate_scores'][seed, ei],
                      context + ': unchanged control gate', atol=0, rtol=0)
            for ai, arm in enumerate(EXPERT_ARMS):
                scores = saved['scores'][seed, ai]
                receiver = scores.argmax(1)
                require(np.array_equal(saved['expert_predictions'][seed, ai], receiver), context + ': receiver argmax')
                score = replay_score(base_parts, scores, gate)
                close(saved['gate_scores'][seed, ai], score, context + '/' + arm + ': gate', atol=1e-11)
                errors['gate'] = max(errors['gate'], float(np.abs(saved['gate_scores'][seed, ai] - score).max()))
                mask = replay_mask(q, ev, base, receiver, observed, score, cap)
                require(np.array_equal(saved['selected'][seed, ai], mask), context + '/' + arm + ': ranked mask')
                require(np.array_equal(saved['predictions'][seed, ai], np.where(mask, receiver, base)),
                        context + '/' + arm + ': handoff prediction')
            for key in ('selected', 'predictions'):
                require(np.array_equal(saved[key][seed, 4], advantage[key][seed, advantage_index]),
                        context + ': unchanged advantage ' + key)
        # Evaluation labels enter only after all support fits and policy checks.
        truth = original['y_ids'][q]
        port_expected = {}
        parent_metrics = read_json(PARENT / port / 'metrics.json')
        frozen_metrics = read_json(FROZEN / port / 'metrics.json')
        advantage_metrics = read_json(ADVANTAGE / port / 'metrics.json')
        for seed in range(5):
            ev = saved['eval_masks'][seed]
            linear = saved['expert_predictions'][seed, 2]
            for mode, arms, predictions in (('full', EXPERT_ARMS, saved['expert_predictions'][seed]),
                                             ('budget', BUDGET_ARMS, saved['predictions'][seed])):
                for ai, arm in enumerate(arms):
                    pred = predictions[ai]
                    result = metric_replay(truth[ev], base[ev], linear[ev], pred[ev])
                    result.update(evaluated_rows=int(ev.sum()), admitted_rows=None, changed_rows=None,
                                  cap=None, admission_fraction=None)
                    if mode == 'budget':
                        mask = saved['selected'][seed, ai]
                        count = int(mask.sum())
                        changed = int(np.count_nonzero(pred[ev] != base[ev]))
                        require(not mask[~ev].any() and changed == count <= saved['caps'][seed],
                                port + ': budget accounting')
                        result.update(admitted_rows=count, changed_rows=changed, cap=int(saved['caps'][seed]),
                                      admission_fraction=count / int(ev.sum()))
                    if arm in ('linear', 'bias'):
                        name = control_names[('linear', 'bias').index(arm)]
                        control = (parent_metrics['seeds'][seed]['metrics'][name] if mode == 'full' else
                                   next(row for row in frozen_metrics if row['seed'] == seed
                                        and row['budget'] == .05 and row['expert'] == name))
                    elif arm == 'advantage_reference':
                        control = next(row for row in advantage_metrics if row['seed'] == seed
                                       and row['arm'] == 'advantage_visual')
                    else:
                        control = None
                    if control is not None:
                        for field in ('ba', 'acc'):
                            close(result[field], control[field], port + ': inherited metric ' + field)
                    port_expected[port, seed, mode, arm] = result
        compare_metric_rows(read_json(run / port / 'metrics.json'), port_expected)
        expected.update(port_expected)
        print('VERIFIED ' + port + ': raw tails, 10 independent fits, 20 gates/masks, 45 metric rows', flush=True)
    require(fits == 230 and tuple(refits) == REFIT_PORTS and len(expected) == 1035, 'verification coverage')
    with (run / 'metrics.csv').open(encoding='utf-8', newline='') as handle:
        compare_metric_rows(list(csv.DictReader(handle)), expected, csv_mode=True)
    verify_summary(read_json(run / 'summary.json'), expected, ports)
    return dict(status='passed', ports=23, contexts=115, source_geometry_refit_ports=refits,
                independent_source_decoder_refits=4, raw_query_tail_replays=46,
                existing_oof_input_receipts_checked=23, existing_input_file_digests_replayed=3,
                independent_weighted_ridge_fits=fits, gate_and_mask_replays=460,
                unchanged_budget_controls=345, full_metric_rows=460, budget_metric_rows=575,
                policy_metric_rows=len(expected), max_absolute_errors=errors,
                query_labels_used_only_for_readout=True, producer_functions_used=False,
                additional_experiments=0, verifier_numpy=np.__version__,
                limitations=['Source decoders and source normalization independently refit for two fixed ports only.',
                             'Other source decoders are trusted; raw query tails and target fits replayed for every port.',
                             'Raw encoded-row provenance, archived PCA/source heads, gate and advantage policy are inherited.',
                             'Artifact replay does not independently prove policy-freeze timing or encoder training isolation.',
                             'Energy control matches source energy, not effective rank or capacity.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', '--out', dest='run', type=Path,
                        default=ROOT / 'oof_visual_tail_results_20261007')
    args = parser.parse_args()
    require(args.run.is_dir(), 'run directory is not available yet')
    start = time.perf_counter()
    receipt = {}
    previous = read_json(args.run / 'VERIFICATION.json') if (args.run / 'VERIFICATION.json').is_file() else None
    try:
        with threadpool_limits(limits=1):
            receipt = verify(args.run)
    except Exception as exc:
        receipt = dict(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        if previous is not None and previous.get('status') == 'failed':
            receipt['previous_failed_attempt'] = previous
        receipt.update(verifier=Path(__file__).name, run=str(args.run.resolve()), seconds=time.perf_counter() - start)
        (args.run / 'VERIFICATION.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(json.dumps(receipt, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
