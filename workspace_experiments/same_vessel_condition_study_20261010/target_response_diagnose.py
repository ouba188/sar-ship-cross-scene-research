"""Replay representative rescues/harms and account for exact score changes."""

import numpy as np
import pandas as pd
import torch
from channel_operator import BASE, OUT as PARENT, PORTS, build_blocks, save_json
from target_response_adapter import ResponseAdapter, residual_head, phi_prior, encode

OUT = BASE/'target_response_adapter_balanced_20261010'


def coverage_accounting():
    classes = pd.read_csv(OUT/'by_class.csv')
    records = []
    for (port, draw), rows in classes.groupby(['port', 'draw']):
        tag = port.replace(' ', '_')+f'_draw{draw}'
        with np.load(OUT/f'{tag}_split.npz') as saved:
            counts = np.bincount(saved['support_labels'], minlength=8)
        for row in rows.to_dict('records'):
            count = int(counts[row['class_id']])
            row['support_examples_of_class'] = count
            row['coverage_group'] = 'absent' if count == 0 else 'singleton' if count == 1 else 'multiple'
            records.append(row)
    detail = pd.DataFrame(records)
    detail.to_csv(OUT/'label_coverage_class_detail.csv', index=False)
    cells = detail.groupby(['port', 'draw', 'arm', 'coverage_group']).agg(
        recall=('recall', 'mean'), classes=('class_id', 'size'), queries=('n', 'sum')).reset_index()
    reference = cells[cells.arm.eq('head_only')][['port', 'draw', 'coverage_group', 'recall']].rename(columns={'recall': 'head_recall'})
    cells = cells.merge(reference, on=['port', 'draw', 'coverage_group'])
    cells['delta_recall_vs_head'] = cells.recall-cells.head_recall
    cells.to_csv(OUT/'label_coverage_group_detail.csv', index=False)
    cells.groupby(['arm', 'coverage_group']).agg(mean_recall=('recall', 'mean'),
        delta_vs_head=('delta_recall_vs_head', 'mean'), task_groups=('draw', 'size')).reset_index().to_csv(
            OUT/'label_coverage_group_summary.csv', index=False)


def run():
    torch.set_num_threads(1)
    coverage_accounting()
    obs = pd.read_csv(BASE/'observations.csv')
    maps = np.load(PARENT/'boundary_maps.npy', mmap_mode='r')
    block = build_blocks()[1]
    records, replays = [], []
    for port in PORTS:
        tag = port.replace(' ', '_')+'_draw0'
        with np.load(OUT/f'{tag}_split.npz') as split:
            support, query = split['support_map_rows'], split['query_map_rows']
            labels, query_ids = split['support_labels'], split['query_obs_ids']
        qm = obs.iloc[query_ids].reset_index(drop=True)
        truth = qm.class_id.to_numpy()
        with np.load(OUT/f'{tag}_models.npz') as saved:
            model = {k: saved[k] for k in saved.files}
        base_pred, guided_pred = model['head_only_pred'], model['scene_guided_adapter_pred']
        rescue = np.flatnonzero((base_pred != truth) & (guided_pred == truth))[:4]
        harm = np.flatnonzero((base_pred == truth) & (guided_pred != truth))[:4]
        chosen = np.concatenate([rescue, harm])
        if not len(chosen):
            continue
        all_rows = np.concatenate([support, query[chosen]])
        center, scale, weight, bias = [torch.from_numpy(model[k]) for k in
            ('source_center', 'source_scale', 'source_coef', 'source_bias')]
        identity = ResponseAdapter(model['channel_scale']).eval()
        adapted = ResponseAdapter(model['channel_scale']).eval()
        with torch.no_grad():
            adapted.log_gain.copy_(torch.from_numpy(model['scene_guided_adapter_log_gain']))
            adapted.shift.copy_(torch.from_numpy(model['scene_guided_adapter_shift']))
        original_features = encode(block, identity, maps, all_rows)
        changed_features = encode(block, adapted, maps, all_rows)
        with torch.no_grad():
            p0, prior0 = phi_prior(torch.from_numpy(original_features), center, scale, weight, bias)
            p1, prior1 = phi_prior(torch.from_numpy(changed_features), center, scale, weight, bias)
            beta0, _, _ = residual_head(p0[:64], prior0[:64], torch.from_numpy(labels), balanced=True)
            beta1 = torch.from_numpy(model['scene_guided_adapter_residual_coef'])
            s0 = (prior0[64:]+p0[64:]@beta0).numpy()
            s1 = (prior1[64:]+p1[64:]@beta1).numpy()
            prior_change = (prior1[64:]-prior0[64:]).numpy()
            feature_change = ((p1[64:]-p0[64:])@beta0).numpy()
            head_change = (p1[64:]@(beta1-beta0)).numpy()
        error = float(np.max(abs((s1-s0)-(prior_change+feature_change+head_change))))
        if not np.array_equal(s0.argmax(1), base_pred[chosen]) or not np.array_equal(s1.argmax(1), guided_pred[chosen]):
            raise ValueError('Selected case predictions did not replay')
        if error > 1e-5:
            raise ValueError('Score decomposition did not close')
        replays.append(dict(port=port, cases=len(chosen), prediction_disagreements=0, accounting_error=error))
        for j, index in enumerate(chosen):
            y = int(truth[index])
            if base_pred[index] != y:
                rival = int(base_pred[index])
                kind = 'rescue'
            else:
                rival = int(guided_pred[index])
                kind = 'harm'
            margin = lambda a: float(a[j, y]-a[j, rival])
            records.append(dict(port=port, draw=0, kind=kind, instance_id=qm.iloc[index].instance_id,
                true_class=y, fixed_rival=rival, old_margin=margin(s0), new_margin=margin(s1),
                source_prior_term=margin(prior_change), feature_with_old_head_term=margin(feature_change),
                support_head_refit_term=margin(head_change)))
    pd.DataFrame(records).to_csv(OUT/'case_score_accounting.csv', index=False)
    save_json(OUT/'CASE_REPLAY.json', dict(tasks=replays,
        boundary='Illustrative first4 rescues/first4 harms in each draw0; not a population causal estimate. Fixed path order for score allocation.'))
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == '__main__':
    run()
