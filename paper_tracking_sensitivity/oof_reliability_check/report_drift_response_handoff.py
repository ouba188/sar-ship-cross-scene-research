"""Read frozen policies to diagnose exchanges, without another training run."""

import json

import numpy as np
import pandas as pd

from run_drift_response_handoff import OUT, PILOT, SOURCE, archive


def main():
    frame = pd.read_csv(SOURCE/'manifest.csv',usecols=['class_id'])
    rows, contrasts, signal = [],[],[]
    for port in PILOT:
        p = archive(OUT/'ports'/port/'policies.npz')
        y = frame.class_id.to_numpy()[p['query_rows']]
        names = list(p['arm_names'])
        old_index,new_index = names.index('confidence_net'),names.index('protected_response')
        for draw,ev in enumerate(p['eval_mask']):
            bp,ep = p['base_scores'][draw].argmax(1),p['expert_scores'][draw].argmax(1)
            utility = (ep==y).astype(int)-(bp==y).astype(int)
            old,new = p['selected'][draw,old_index],p['selected'][draw,new_index]
            _,counts = np.unique(y[ev],return_counts=True)
            class_count = np.bincount(y[ev],minlength=8)
            weight = np.zeros(len(y))
            weight[ev] = 1./(len(counts)*class_count[y[ev]])
            for name,mask in [('swapped_in',new&~old),('swapped_out',old&~new)]:
                rows.append(dict(port=port,draw=draw,group=name,n=int(mask.sum()),
                    rescues=int(np.sum(mask&(utility==1))),harms=int(np.sum(mask&(utility==-1))),
                    neutral=int(np.sum(mask&(utility==0))),oa_gain=float(utility[mask].sum()/ev.sum()),
                    ba_gain=float(np.sum(weight[mask]*utility[mask]))))
            pred_old,pred_new = p['predictions'][draw,old_index],p['predictions'][draw,new_index]
            delta = (pred_new==y).astype(int)-(pred_old==y).astype(int)
            direct_ba = float(np.sum(weight*delta))
            exchange_ba = float(np.sum(weight*utility*(new.astype(int)-old.astype(int))))
            np.testing.assert_allclose(direct_ba,exchange_ba,atol=1e-12)
            contrasts.append(dict(port=port,draw=draw,delta_ba=direct_ba,
                selected_retention=float(np.sum(new&old)/max(old.sum(),1))))
        for path in sorted((OUT/'ports'/port).glob('*_source_response.npz')):
            data = archive(path)
            utility = ((data['expert'].argmax(1)==data['labels']).astype(int)-
                       (data['base'].argmax(1)==data['labels']).astype(int))
            x = data['protected']
            difference = x[:,4]-x[:,5]
            for name,mask in [('expert_more_stable',difference>0),('base_more_stable',difference<0),
                              ('equal_flip_rate',difference==0)]:
                mask &= data['eligible']
                if not mask.any():
                    continue
                w = data['weights'][mask]
                signal.append(dict(port=port,held_port=str(data['held_port']),group=name,n=int(mask.sum()),
                    weighted_net_utility=float(np.average(utility[mask],weights=w)),
                    weighted_rescue=float(np.average(utility[mask]==1,weights=w)),
                    weighted_harm=float(np.average(utility[mask]==-1,weights=w))))
    exchanges = pd.DataFrame(rows)
    exchanges.to_csv(OUT/'admission_exchanges.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(signal).to_csv(OUT/'source_flip_utility_groups.csv',index=False,encoding='utf-8-sig')
    result = dict(exchanges=exchanges.groupby('group')[['n','rescues','harms','neutral','oa_gain','ba_gain']].mean().reset_index().to_dict('records'),
        selected_retention_mean=float(pd.DataFrame(contrasts).selected_retention.mean()),
        delta_ba_pp=float(100*pd.DataFrame(contrasts).delta_ba.mean()),
        interpretation='Existing-policy decomposition only; no target refitting/selection; overlapping draws not independent')
    (OUT/'EXCHANGE_DIAGNOSIS.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
