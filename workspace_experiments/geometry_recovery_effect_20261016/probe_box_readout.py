"""Frozen query-region sensitivity, not a repaired full-system benchmark.

Keep supports, source weights, B, geometry angles and labels unchanged. The two
queries were selected retrospectively, so this cannot estimate general benefit.
"""
import json
import numpy as np
import pandas as pd
import torch

import run_effect as run

OUT = run.HERE / 'visual_diagnosis/geometry_trace'
FLAVORS = ('historical', 'fresh_old_adapter', 'fresh_fixed_adapter')
KEYS = ('img_tokens', 'ship_tokens', 'img_areas', 'ship_areas')


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    data = run.load('Singapore', True)
    d = data['tensors']
    z = np.load(OUT / 'query_box_tokens.npz')
    global_rows = data['meta'].global_row.to_numpy()
    indices = np.array([np.flatnonzero(global_rows == g)[0] for g in z['rows']])
    ids = torch.as_tensor(indices, device=run.base.DEVICE)
    old_tokens = {key: d[key][ids].clone() for key in KEYS}
    results, checks = [], []
    for seed in range(3):
        run.prepare(data, seed)
        for family in ('strict', 'optional'):
            run.set_input(data, family, 'new')
            model = run.model_from(run.HERE, 'Singapore', family, seed)
            raw = run.base.raw_descriptors(model, data)
            changed_raw = {}
            for flavor in FLAVORS:
                for key in KEYS:
                    d[key][ids] = torch.as_tensor(z[flavor+'_'+key], device=run.base.DEVICE)
                changed_raw[flavor] = run.base.descriptor(model, d, ids)
            for key in KEYS:
                d[key][ids] = old_tokens[key]
            torch.testing.assert_close(changed_raw['historical'], raw[ids], atol=1e-6, rtol=0)
            archive = pd.read_csv(run.HERE/'evaluation/Singapore'/f'{family}_new_seed{seed}_predictions.csv')
            archive = archive.set_index(['task', 'global_row'])
            replay_error = 0.
            for ti, item in enumerate(data['cache']['target']):
                qids = item['q'].cpu().numpy()
                positions_np = np.flatnonzero(np.isin(qids, indices))
                if not len(positions_np):
                    continue
                positions = torch.as_tensor(positions_np, device=run.base.DEVICE)
                query_ids = item['q'][positions]
                assert not set(query_ids.cpu().tolist()) & set(item['s'].cpu().tolist())
                p0 = run.fit.predict(model, data, item, positions=positions, raw=raw)[0]
                for j, qi in enumerate(query_ids.cpu().tolist()):
                    expected = archive.loc[(ti, int(global_rows[qi])), [f'p{c}' for c in range(8)]].to_numpy(float)
                    actual = p0[j].cpu().numpy()
                    replay_error = max(replay_error, float(np.abs(actual-expected).max()))
                    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
                for flavor in FLAVORS:
                    raw_query_changed = raw.clone()
                    source_positions = [int(np.flatnonzero(indices == qi)[0]) for qi in query_ids.cpu().tolist()]
                    raw_query_changed[query_ids] = changed_raw[flavor][source_positions]
                    # Important: do not modify this task's support descriptors.
                    torch.testing.assert_close(raw_query_changed[item['s']], raw[item['s']], atol=0, rtol=0)
                    prob = run.fit.predict(model, data, item, positions=positions, raw=raw_query_changed)[0]
                    if flavor == 'historical':
                        torch.testing.assert_close(prob, p0, atol=1e-8, rtol=0)
                    for j, qi in enumerate(query_ids.cpu().tolist()):
                        p = prob[j].cpu().numpy(); original = p0[j].cpu().numpy()
                        y = int(d['labels'][qi]); wrong_scores = original.copy(); wrong_scores[y] = -1
                        rival = int(wrong_scores.argmax())
                        results.append(dict(seed=seed, family=family, flavor=flavor, task=ti,
                            origin=item['t']['origin'], draw=item['t']['draw'],
                            global_row=int(global_rows[qi]), truth=y, pred=int(p.argmax()),
                            original_pred=int(original.argmax()), margin=float(p[y]-p[rival]),
                            rival=rival, nll=float(-np.log(max(p[y], 1e-12))),
                            brier=float(((p-np.eye(8)[y])**2).sum()),
                            enabled=bool(d['valid'][qi] and d['valid'][item['s']].sum()>=2),
                            **{f'p{k}':float(p[k]) for k in range(8)}))
            checks.append(dict(seed=seed,family=family,max_archived_prediction_error=replay_error))
    frame = pd.DataFrame(results)
    frame.to_csv(OUT/'query_box_predictions.csv', index=False)
    key = ['seed','family','task','global_row']
    left = frame.loc[frame.flavor=='fresh_old_adapter']
    right = frame.loc[frame.flavor=='fresh_fixed_adapter']
    pair = left.merge(right, on=key, suffixes=('_old','_fixed'), validate='one_to_one')
    pair['rescue'] = (pair.pred_old != pair.truth_old) & (pair.pred_fixed == pair.truth_fixed)
    pair['harm'] = (pair.pred_old == pair.truth_old) & (pair.pred_fixed != pair.truth_fixed)
    pair['margin_delta'] = pair.margin_fixed-pair.margin_old
    pair.to_csv(OUT/'adapter_only_paired.csv', index=False)
    (OUT/'readout_checks.json').write_text(json.dumps(dict(replays=checks,
        labels_used_for_box_or_parameter_selection=False, query_case_selection='posthoc two changed queries',
        trained_models=0, supports_unchanged=True, baseline_B_unchanged=True,
        experiment_type='query_only_frozen_readout_sensitivity'), indent=2), encoding='utf-8')
    print(pair.groupby(['family','origin_old'])[['rescue','harm','margin_delta']].agg(['sum','mean']).to_string())
    print('prediction records', len(frame), 'unique queries', frame.global_row.nunique())


if __name__ == '__main__':
    main()
