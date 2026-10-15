"""Replay a fixed joint expert at different admission caps; train nothing."""

import argparse
import json
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'

import numpy as np
import pandas as pd

from budgeted_residual_handoff import ranked_mask
from run_generic_transfer import metrics


ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
COMPLETION = ROOT/'module_completion_20261008'
DEFAULT_OUT = ROOT/'handoff_budget_sweep_20261008'
SOURCE = Path('E:/\u4e34\u65f6\u4f1a\u8bdd/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
POLICIES = ('gate_000', 'gate_005', 'gate_010', 'gate_020', 'gate_050', 'gate_100', 'all_expert')
BUDGETS = (0., .05, .10, .20, .50, 1., 1.)
DOMAINS = {'pacs': ('art_painting', 'cartoon', 'photo', 'sketch'),
           'officehome': ('Art', 'Clipart', 'Product', 'Real World')}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def load_policy(path):
    keys = ('expert_predictions', 'predictions', 'selected', 'eval_mask', 'gate_scores', 'cap', 'support_rows')
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in keys}


def task_inventory():
    items = []
    ports = json.loads((ROOT/'oof_base_adapt_results_20261007/POLICIES_SAVED.json').read_text(encoding='utf-8'))
    if len(ports) != 23:
        raise ValueError('Expected all23 eligible target ports')
    for item in ports:
        directory = COMPLETION/'ports_projection'/item['port']/'matched_joint'
        items.append(dict(dataset='self24ports', task=item['port'], n_classes=8,
            policy_path=str(directory/'joint_policies.npz'),
            base_path=str(directory/'joint_policies.npz'),
            manifest_path=str(SOURCE/'manifest.csv'), original_metrics=str(directory/'metrics.csv'),
            reference_protocol='matched_ports_joint_reference'))
    f0921 = ROOT/'joint_residual_pilot_20261008'
    config = json.loads((f0921/'protocol.json').read_text(encoding='utf-8'))
    items.append(dict(dataset='F0921', task='F0921', n_classes=8,
        policy_path=str(f0921/'joint/policies.npz'),
        base_path=str(Path(config['previous'])/'frozen_replay/policies.npz'),
        manifest_path=str(Path(config['target'])/'target_manifest.csv'),
        original_metrics=str(f0921/'metrics.csv'), reference_protocol='matched_F0921_reference'))
    for short, previous in [('oss1', 'opensarship1_transfer_20261007'), ('oss2', 'opensarship2_transfer_20261007')]:
        items.append(dict(dataset=short.upper(), task=short.upper(), n_classes=8,
            policy_path=str(COMPLETION/short/'joint/policies.npz'),
            base_path=str(ROOT/previous/'policies.npz'),
            manifest_path=str(ROOT/previous/'target_manifest.csv'),
            original_metrics=str(COMPLETION/short/'metrics.csv'), reference_protocol='original_reference'))
    for dataset, domains in DOMAINS.items():
        for domain in domains:
            slug = domain.replace(' ', '_')
            directory = COMPLETION/'generic/rgb_retrained'/dataset/slug
            previous = ROOT/f'{dataset}_handoff_transfer_20261007'/slug
            items.append(dict(dataset=dataset, task=domain, n_classes=7 if dataset == 'pacs' else 65,
                policy_path=str(directory/'joint_policies.npz'),
                base_path=str(previous/'policies.npz'), manifest_path=str(previous/'target_manifest.csv'),
                original_metrics=str(directory/'metrics.csv'), reference_protocol='retrained_rgb_reference'))
    return items


def load_task(item):
    policy = load_policy(item['policy_path'])
    frame = pd.read_csv(item['manifest_path'], keep_default_na=False)
    with np.load(item['policy_path'], allow_pickle=False) as new, np.load(item['base_path'], allow_pickle=False) as old:
        if item['dataset'] == 'self24ports':
            row_ids = new['query_rows']
            labels = frame.class_id.to_numpy(int)[row_ids]
            base = new['base_scores'].argmax(2)
            support = np.searchsorted(row_ids, policy['support_rows'])
            np.testing.assert_array_equal(row_ids[support], policy['support_rows'])
        else:
            base = old['predictions'][:, 1]
            labels = frame.class_id.to_numpy(int)
            support = policy['support_rows']
            np.testing.assert_array_equal(support, old['support_rows'])
            if item['dataset'] in DOMAINS:
                row_ids = new['target_rows']
                np.testing.assert_array_equal(row_ids, old['target_rows'])
                np.testing.assert_array_equal(row_ids, frame.row_id.to_numpy(int))
            else:
                row_ids = frame.target_manifest_row.to_numpy(int)
                np.testing.assert_array_equal(new['sample_ids'], frame.sample_id.to_numpy(str))
                np.testing.assert_array_equal(new['sample_ids'], old['sample_ids'])
    if base.shape != (5, len(labels)) or support.shape != (5, 64):
        raise ValueError('Expected the original five64-label tasks')
    return policy, base, support, labels, row_ids


def freeze_budget_draw(base, expert, score, evaluation, support_labels, row_ids, n_classes):
    """Routing reads support labels only, never query labels."""
    observed = np.bincount(support_labels, minlength=n_classes) > 0
    eligible = evaluation & (base != expert) & observed[expert] & (score > 0)
    n = int(evaluation.sum())
    masks = [ranked_mask(score, eligible, row_ids, int(np.floor(budget*n))) for budget in BUDGETS[:-1]]
    masks.append(evaluation.copy())
    masks = np.stack(masks)
    predictions = np.where(masks, expert[None, :], base[None, :])
    return predictions, masks, eligible


def freeze_task(item, out):
    old, base, support, labels, row_ids = load_task(item)
    predictions, selections, candidates = [], [], []
    for draw in range(5):
        ev = ~np.isin(np.arange(len(labels)), support[draw])
        np.testing.assert_array_equal(ev, old['eval_mask'][draw])
        pred, selected, eligible = freeze_budget_draw(base[draw], old['expert_predictions'][draw],
            old['gate_scores'][draw], ev, labels[support[draw]], row_ids, item['n_classes'])
        np.testing.assert_array_equal(pred[1], old['predictions'][draw])
        np.testing.assert_array_equal(selected[1], old['selected'][draw])
        if selected[:, ~ev].any() or not np.all(np.isfinite(old['gate_scores'][draw])):
            raise ValueError('Support rows entered routing or scores were nonfinite')
        for index, budget in enumerate(BUDGETS[:-1]):
            if selected[index].sum() > int(np.floor(budget*ev.sum())):
                raise ValueError('Budget exceeded')
            if index and np.any(selected[index-1] & ~selected[index]):
                raise ValueError('Fixed-rank caps must be nested')
        np.testing.assert_array_equal(selected[-2], eligible)
        predictions.append(pred)
        selections.append(selected)
        candidates.append(eligible)
    directory = out/'policies'/item['dataset']/item['task'].replace(' ', '_')
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'policies.npz'
    np.savez_compressed(path, predictions=np.stack(predictions), selected=np.stack(selections),
        eligible=np.stack(candidates), eval_mask=old['eval_mask'], base_predictions=base,
        expert_predictions=old['expert_predictions'], gate_scores=old['gate_scores'],
        row_ids=row_ids, support_local_rows=support, policy_names=np.asarray(POLICIES), budgets=BUDGETS)
    record = dict(**item, frozen_path=str(path), exact_5pct_prediction_and_selection_replay=True,
        query_labels_used_in_routing=False, draws=5, target_rows=len(labels), queries_per_draw=len(labels)-64)
    write_json(directory/'PROVENANCE.json', record)
    print(f'FROZEN {item["dataset"]}/{item["task"]}: five draws, seven policies, exact5 replay', flush=True)
    return record


def readout(record):
    frame = pd.read_csv(record['manifest_path'], keep_default_na=False)
    labels = frame.class_id.to_numpy(int)
    rows, classes = [], []
    original = pd.read_csv(record['original_metrics'])
    original = original[original.arm == 'joint'].set_index('draw')
    with np.load(record['frozen_path'], allow_pickle=False) as policy:
        if record['dataset'] == 'self24ports':
            labels = labels[policy['row_ids']]
        for draw in range(5):
            ev, base = policy['eval_mask'][draw], policy['base_predictions'][draw]
            n = int(ev.sum())
            base_m = metrics(labels[ev], base[ev], record['n_classes'])
            for index, name in enumerate(POLICIES):
                pred, selected = policy['predictions'][draw, index], policy['selected'][draw, index]
                value = metrics(labels[ev], pred[ev], record['n_classes'])
                rescued = int(np.sum(ev & (pred == labels) & (base != labels)))
                harmed = int(np.sum(ev & (pred != labels) & (base == labels)))
                selected_n = int(selected.sum())
                delta_oa = 100*(value['acc']-base_m['acc'])
                np.testing.assert_allclose(delta_oa, 100*(rescued-harmed)/n, atol=1e-12, rtol=0)
                if name in ('gate_005', 'all_expert'):
                    prefix = '' if name == 'gate_005' else 'expert_'
                    np.testing.assert_allclose([value['acc'], value['ba']],
                        original.loc[draw, [prefix+'acc', prefix+'ba']].to_numpy(float), atol=1e-12, rtol=0)
                rows.append(dict(dataset=record['dataset'], task=record['task'], draw=draw,
                    policy=name, budget=BUDGETS[index], acc=value['acc'], ba=value['ba'],
                    selected=selected_n, evaluated=n, eligible=int(policy['eligible'][draw].sum()),
                    rescued=rescued, harmed=harmed, changed=int(np.sum(ev & (pred != base))),
                    realized_handoff_rate=selected_n/n, selection_net_correct=rescued-harmed,
                    delta_acc_vs_base_pp=delta_oa, delta_ba_vs_base_pp=100*(value['ba']-base_m['ba']),
                    reference_protocol=record['reference_protocol']))
                for cls, recall in enumerate(value['class_recall']):
                    classes.append(dict(dataset=record['dataset'], task=record['task'], draw=draw,
                        policy=name, class_id=cls, query_count=int(np.sum(ev & (labels == cls))), recall=recall))
    return rows, classes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out/'COMPLETE.json').exists():
        raise ValueError('Preserve completed sweep; select a new output for another protocol')
    tasks = task_inventory()
    write_json(out/'PROTOCOL.json', dict(date='2026-10-08', expert='fixed joint M2+M3',
        policies=list(POLICIES), budgets=list(BUDGETS), targets=tasks,
        target_support='Same64 per task; original draws0..4; no new labels',
        frozen='Encoders, source geometry, residual heads, Base64, expert outputs and gate scores',
        gate='Disagreement AND expert class observed in support AND score>0; same stable row-ID tie break',
        gate100='No effective cap; eligibility and positive-score filters retained',
        all_expert='Every query uses expert, bypassing eligibility; admission differs from changed prediction',
        metrics='OA and equal-present-class BA; task means over5draws then equal task means per dataset',
        training=False, gpu=False, compute_saving_measured=False, independent_confirmation=False,
        target_model_selection=False, all_policies_frozen_before_query_readout=True,
        interpretation='Descriptive budget sensitivity, not query-selected deployment budget'))
    frozen = [freeze_task(item, out) for item in tasks]
    write_json(out/'POLICIES_SAVED.json', frozen)
    rows, classes = [], []
    for record in frozen:
        result, recalls = readout(record)
        rows.extend(result)
        classes.extend(recalls)
    pd.DataFrame(rows).to_csv(out/'metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(classes).to_csv(out/'class_recall.csv', index=False, encoding='utf-8-sig')
    write_json(out/'COMPLETE.json', dict(status='complete', tasks=len(tasks), expected_tasks=34,
        draws_per_task=5, policies=7, evaluation_rows=len(rows), expected_rows=1190,
        exact_5pct_prediction_and_selection_replay=True, original_expert_and5pct_metrics_reproduced=True,
        no_retraining=True, no_new_labels=True, no_query_labels_used_for_routing=True))
    print(f'COMPLETE: {len(tasks)} targets, {len(rows)} matched rows', flush=True)


if __name__ == '__main__':
    main()
