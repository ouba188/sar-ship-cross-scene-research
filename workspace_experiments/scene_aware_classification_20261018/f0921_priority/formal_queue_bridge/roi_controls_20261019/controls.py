"""Matched ordinary ROI and target-by-grid controls on the frozen parent tasks."""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
spec = importlib.util.spec_from_file_location('bridge_parent', PARENT / 'run.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)
KEYS = ['fold', 'budget', 'draw', 'row']
PROBS = [f'p{i}' for i in range(7)]


def merged_regions(tokens, mass):
    """Merge cached cell/role averages with their feature-map pixel masses."""
    role = np.arange(tokens.shape[2]) % 3
    output, available = [], []
    for selected in (role == 0, role != 0):
        weights = mass[:, selected].astype(np.float64)
        total = weights.sum(1)
        value = np.einsum('nptd,nt->npd', tokens[:, :, selected].astype(np.float64), weights)
        value /= np.maximum(total[:, None, None], 1e-12)
        value[total == 0] = 0
        output.append(value)
        available.append(total > 0)
    return np.stack(output, 2), np.stack(available, 1)


def local_kernels(features, support):
    valid = features['valid12']
    post, _ = bridge.token_descriptor(features['postpool'], valid, support)
    pre, _ = bridge.token_descriptor(features['prepool'], valid, support)
    target = post.copy()
    target[:, :, ::3] = pre[:, :, ::3]
    merged, ok = merged_regions(features['prepool'], features['mass12'])
    roi, _ = bridge.token_descriptor(merged[:, :, :1], ok[:, :1], support)
    fg_bg, _ = bridge.token_descriptor(merged, ok, support)
    repeated = features['postpool'].copy()
    repeated[:, :, ::3] = merged[:, :, :1]
    repeat_z, _ = bridge.token_descriptor(repeated, valid, support)
    return {name: bridge.kernel(z) for name, z in
            [('postpool', post), ('target_only', target), ('roi_global', roi),
             ('fg_bg_global', fg_bg), ('roi_repeat_target', repeat_z)]}


def paired_tables(pred, met):
    draws, events = [], []
    comparisons = [('target_only', c) for c in
                   ['global_grid', 'global_only', 'roi_global', 'fg_bg_global', 'roi_repeat_target']]
    comparisons += [(c, 'global_grid') for c in ['roi_global', 'fg_bg_global', 'roi_repeat_target']]
    comparisons += [('fg_bg_global', c) for c in ['global_only', 'roi_global']]
    for arm, comparator in comparisons:
        a = pred[pred.arm.eq(arm)].set_index(KEYS).sort_index()
        b = pred[pred.arm.eq(comparator)].set_index(KEYS).sort_index()
        assert a.index.equals(b.index)
        part = a[['truth', 'correct', 'margin', 'ap', 'brier', 'same_class_support', 'target_polarizations']].copy()
        part['baseline_correct'] = b.correct
        part['rescue'] = a.correct & ~b.correct
        part['harm'] = ~a.correct & b.correct
        part['delta_margin'] = a.margin - b.margin
        part['delta_brier'] = a.brier - b.brier
        part['arm'] = arm
        part['comparator'] = comparator
        events.append(part.reset_index())
        mi = ['fold', 'budget', 'draw']
        ma = met[met.arm.eq(arm)].set_index(mi)
        mb = met[met.arm.eq(comparator)].set_index(mi)
        delta = (ma[['oa', 'ba', 'brier', 'nll', 'map_r']] - mb[['oa', 'ba', 'brier', 'nll', 'map_r']])
        delta = delta.groupby(['budget', 'draw']).mean().reset_index()
        delta['arm'], delta['comparator'] = arm, comparator
        draws.append(delta)
    ev, dr = pd.concat(events), pd.concat(draws)
    ev.to_csv(HERE / 'paired_events.csv', index=False)
    dr.to_csv(HERE / 'paired_draws.csv', index=False)
    summaries = []
    for key, group in dr.groupby(['arm', 'comparator', 'budget']):
        selected = ev[ev.arm.eq(key[0]) & ev.comparator.eq(key[1]) & ev.budget.eq(key[2])]
        summaries.append(dict(arm=key[0], comparator=key[1], budget=int(key[2]),
            delta_oa_pp=100*group.oa.mean(), delta_ba_pp=100*group.ba.mean(),
            std_oa_pp=100*group.oa.std(), std_ba_pp=100*group.ba.std(),
            positive_draws=int((group.oa > 1e-12).sum()), negative_draws=int((group.oa < -1e-12).sum()),
            rescues=int(selected.rescue.sum()), harms=int(selected.harm.sum())))
    pd.DataFrame(summaries).to_csv(HERE / 'paired_summary.csv', index=False)
    ev.groupby(['arm', 'comparator', 'budget', 'truth'])[['rescue', 'harm']].sum().to_csv(HERE/'class_flips.csv')
    ev.groupby(['arm', 'comparator', 'budget', 'target_polarizations'])[['rescue', 'harm', 'delta_brier']].agg(['sum','mean','size']).to_csv(HERE/'mask_strata.csv')
    return pd.DataFrame(summaries)


def main():
    started = time.perf_counter()
    meta = pd.read_csv(PARENT/'cohort.csv', dtype={'date': str, 'mmsi': str})
    features = dict(np.load(PARENT/'features.npz'))
    np.testing.assert_array_equal(features['clean_rows'], meta.clean_row)
    episodes = json.loads((PARENT/'episodes.json').read_text())
    availability = pd.read_csv(PARENT/'availability.csv').set_index('row')
    y, nc = meta.label.to_numpy(), int(meta.label.max()+1)
    records, summary, timing, cache = [], [], [], {}
    with threadpool_limits(limits=1):
        for task in episodes:
            source, query = np.array(task['source']), np.array(task['query'])
            fold, budget = task['fold'], task['budget']
            if fold not in cache:
                g = bridge.global_descriptor(features['global'], source)
                b = bridge.global_descriptor(features['r18'], source)
                kg, kb = g@g.T, b@b.T
                cache[fold] = kg, kb, bridge.prior(kg,source,y,nc), bridge.prior(kb,source,y,nc)
            kg, kb, pa, pb = cache[fold]
            for draw, rows in enumerate(task['supports']):
                support = np.array(rows)
                grid = bridge.cosine(features['grid2'].reshape(len(meta),2,-1),support).astype(np.float64)
                kernels = dict(global_only=kg, global_grid=.5*kg+.5*grid)
                tic = time.perf_counter()
                kernels.update({name: .5*kg+.25*grid+.25*k for name,k in local_kernels(features,support).items()})
                timing.append(dict(fold=fold,budget=budget,draw=draw,all_readouts_seconds=time.perf_counter()-tic))
                expert, _ = bridge.adapted(kb,pb,support,y)
                for arm,k in kernels.items():
                    score, _ = bridge.adapted(k,pa,support,y)
                    prob = bridge.late(score[query],expert[query])
                    sim = k[np.ix_(query,support)]
                    ap = np.array([bridge.ap_at_r(v,y[support]==label) for v,label in zip(sim,y[query])])
                    summary.append(dict(fold=fold,budget=budget,draw=draw,arm=arm,queries=len(query),
                        **bridge.metrics(prob,y[query],ap)))
                    for j,i in enumerate(query):
                        order = np.argsort(-sim[j],kind='stable')
                        def top(c, same=True):
                            return next((int(support[t]) for t in order if (y[support[t]]==c)==same),-1)
                        p, truth = prob[j], int(y[i])
                        pred = int(p.argmax())
                        row = dict(fold=fold,budget=budget,draw=draw,arm=arm,row=int(i),clean_row=int(meta.iloc[i].clean_row),
                            truth=truth,prediction=pred,correct=pred==truth,margin=float(p[truth]-np.delete(p,truth).max()),
                            ap=float(ap[j]),brier=float(np.square(p-np.eye(nc)[truth]).sum()),
                            nll=float(-np.log(max(p[truth],1e-12))),top_same_row=top(truth),
                            top_wrong_row=top(truth,False),top_predicted_row=top(pred),
                            same_class_support=int((y[support]==truth).sum()),
                            target_polarizations=int(availability.loc[i,'target_polarizations']))
                        row.update({f'p{c}':float(p[c]) for c in range(nc)})
                        records.append(row)
            print('DONE',fold,budget,flush=True)
    pred, met = pd.DataFrame(records), pd.DataFrame(summary)
    archived = pd.read_csv(PARENT/'predictions.csv',dtype={'fold':str})
    replay = {}
    for arm in ['global_only','global_grid','postpool','target_only']:
        a = pred[pred.arm.eq(arm)].set_index(KEYS).sort_index()[PROBS]
        b = archived[archived.arm.eq(arm)].set_index(KEYS).sort_index()[PROBS]
        assert a.index.equals(b.index)
        np.testing.assert_allclose(a,b,rtol=0,atol=2e-12)
        replay[arm] = float(np.abs(a.to_numpy()-b.to_numpy()).max())
    pred.to_csv(HERE/'predictions.csv',index=False)
    met.to_csv(HERE/'metrics.csv',index=False)
    cols = ['oa','ba','brier','nll','map_r']
    met.groupby(['budget','arm'])[cols].mean().to_csv(HERE/'summary_equal_date.csv')
    met.groupby(['fold','budget','arm'])[cols].mean().to_csv(HERE/'by_date.csv')
    pred.groupby(['budget','arm','truth'])[['correct','brier','ap']].mean().to_csv(HERE/'by_class.csv')
    weighted = pred.groupby(['budget','arm'])[['correct','brier','ap']].mean()
    weighted['ba'] = pred.groupby(['budget','arm','truth']).correct.mean().groupby(['budget','arm']).mean()
    weighted.to_csv(HERE/'summary_query_weighted.csv')
    paired = paired_tables(pred,met)
    pd.DataFrame(timing).to_csv(HERE/'timing.csv',index=False)
    (HERE/'receipt.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,
        replay_max_error=replay,observations=len(meta),tasks=len(episodes),draws=5,
        new_backbone_inference=False,new_neural_training=False,finished=True),indent=2))
    print(met.groupby(['budget','arm'])[['oa','ba','map_r']].mean().to_string())
    print(paired.to_string(index=False))


if __name__=='__main__':
    main()
