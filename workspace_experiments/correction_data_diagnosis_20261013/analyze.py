"""Data composition and event coverage for the frozen correction selector."""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent
PREV = OUT.parent / 'correction_choice_probe_20261013'
sys.path.insert(0, str(PREV))
from choice_core import class_weights, gain_labels


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def auc(gain, score, w):
    keep = gain != 0
    if len(np.unique(gain[keep])) < 2:
        return np.nan
    return float(roc_auc_score(gain[keep] > 0, score[keep], sample_weight=w[keep]))


def identities(frame):
    name = next(k for k in ('mmsi', 'mmsi_or_ship_id') if k in frame)
    values = frame[name].fillna('').astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    unknown = ~values.str.fullmatch(r'[0-9]{9}') | values.eq('000000000')
    values.loc[unknown] = ['unknown_row_' + str(i) for i in frame.index[unknown]]
    return values.to_numpy(), int(unknown.sum())


def balanced_vessel_weights(y, ships):
    frame = pd.DataFrame({'y': y, 'ship': ships})
    rows = frame.groupby(['y', 'ship']).y.transform('size').to_numpy()
    vessels = frame.groupby('y').ship.transform('nunique').to_numpy()
    return 1. / (len(np.unique(y)) * vessels * rows)


def summarize(frame, filename, group):
    numeric = frame.select_dtypes(include=np.number).columns.difference(group)
    frame.groupby(group)[numeric].mean().reset_index().to_csv(OUT / filename, index=False)


def main():
    units = json.loads((PREV/'PREDICTIONS_FROZEN.json').read_text(encoding='utf-8'))['units']
    tables = {}
    for u in units:
        if u['label_path'] not in tables:
            columns = pd.read_csv(u['label_path'], nrows=0).columns
            keep = [k for k in ('class_id','class_name','port','mmsi','mmsi_or_ship_id',
                                'product','scene','scene_id') if k in columns]
            tables[u['label_path']] = pd.read_csv(u['label_path'], usecols=keep,
                dtype={k: str for k in keep if k != 'class_id'})
    event_rows = []
    for u in units:
        if u['family'] != 'source_fit':
            continue
        z = load(u['path'])
        qm = tables[u['label_path']].iloc[z['query_ids']]
        g = gain_labels(z['raw'], z['initial'], qm.class_id.to_numpy())
        ships, _ = identities(qm)
        w = class_weights(qm.class_id.to_numpy())
        for k in np.flatnonzero(g):
            event_rows.append(dict(context=u['task'], uid=u['uid'], obs_id=int(z['query_ids'][k]),
                ship=ships[k], port=qm.iloc[k]['port'], class_id=int(qm.iloc[k].class_id),
                class_name=qm.iloc[k].class_name, gain=int(g[k]), weight=float(w[k]),
                raw_class=int(z['raw'][k].argmax()), initial_class=int(z['initial'][k].argmax())))
    events = pd.DataFrame(event_rows)
    events.to_csv(OUT/'source_events.csv', index=False)
    for group, name in [(['class_id','class_name'],'source_event_classes'), (['port'],'source_event_ports'),
                        (['ship'],'source_event_ships')]:
        events.groupby(group).agg(exposures=('gain','size'), observations=('obs_id','nunique'),
            ships=('ship','nunique'), rescues=('gain',lambda x:int((x>0).sum())),
            harms=('gain',lambda x:int((x<0).sum())), weight=('weight','sum')).reset_index().to_csv(
                OUT/(name+'.csv'), index=False)
    keys = ['family','task','cohort','uid']
    composition, classes, strata, cover, vessel = [], [], [], [], []
    references = pd.read_csv(PREV/'metrics.csv').set_index(['uid','arm'])
    for u in units:
        z = load(u['path'])
        d = load(PREV/'decisions'/(u['uid']+'.npz'))
        np.testing.assert_array_equal(d['query_ids'], z['query_ids'])
        table = tables[u['label_path']]
        sm, qm = table.iloc[z['support_ids']], table.iloc[z['query_ids']]
        sy, y = sm.class_id.to_numpy(int), qm.class_id.to_numpy(int)
        ships, unknown = identities(qm)
        ss, _ = identities(sm)
        w = class_weights(y)
        vw = balanced_vessel_weights(y, ships)
        r, i = z['raw'].argmax(1), z['initial'].argmax(1)
        g = (i==y).astype(int)-(r==y)
        counts = np.bincount(sy, minlength=8)
        qc = np.bincount(y, minlength=8)
        np.testing.assert_array_equal(counts, z['profile_counts'])
        key = {k:u[k] for k in keys}
        product = next((k for k in ('product','scene','scene_id') if k in table), None)
        row = dict(**key, query_n=len(y), query_ships=len(np.unique(ships)), unknown_identity_rows=unknown,
            support_ships=len(np.unique(ss)), support_classes=int((counts>0).sum()),
            query_classes=int((qc>0).sum()), support_query_class_tv=float(abs(counts/16-qc/len(y)).sum()/2),
            absent_class_query_percent=float(100*np.mean(counts[y]==0)),
            absent_class_ba_mass=float(100*w@(counts[y]==0)),
            query_seen_support_ship_percent=float(100*np.isin(ships,ss).mean()),
            support_largest_class_percent=float(100*counts.max()/16),
            query_largest_class_percent=float(100*qc.max()/len(y)),
            support_oof_events=int((z['profile_oof_gain']!=0).sum()),
            query_events=int((g!=0).sum()))
        if 'port' in table:
            row.update(support_ports=sm.port.nunique(), query_ports=qm.port.nunique(),
                query_support_port_seen_percent=float(100*qm.port.isin(sm.port).mean()))
        if product:
            row.update(support_products=sm[product].nunique(),query_products=qm[product].nunique(),
                query_support_product_seen_percent=float(100*qm[product].isin(sm[product]).mean()))
        composition.append(row)
        delta = {}
        for arm in ('score_only','contextual','confidence'):
            chosen = np.where(d[arm] > (0 if arm=='confidence' else .5), i, r)
            delta[arm] = (chosen==y).astype(int)-(i==y)
            np.testing.assert_allclose(100*w@delta[arm],references.loc[(u['uid'],arm),'versus_initial_ba'],atol=1e-10)
            vessel.append(dict(**key,arm=arm,row_ba_delta=float(100*w@delta[arm]),
                vessel_class_balanced_delta=float(100*vw@delta[arm]),
                raw_vessel_class_balanced=float(100*vw@(r==y)),
                initial_vessel_class_balanced=float(100*vw@(i==y))))
        np.testing.assert_allclose(100*w@(i==y),references.loc[(u['uid'],'initial'),'ba'],atol=1e-10)
        for cl in np.unique(y):
            keep = y==cl
            classes.append(dict(**key,class_id=cl,class_name=qm.loc[qm.class_id==cl,'class_name'].iloc[0],
                query_n=int(keep.sum()),query_ships=len(np.unique(ships[keep])),support_n=int(counts[cl]),
                rescue_n=int(((g==1)&keep).sum()),harm_n=int(((g==-1)&keep).sum()),
                raw_recall=float(100*np.mean(r[keep]==cl)),initial_recall=float(100*np.mean(i[keep]==cl)),
                initial_ba_contribution=float(100*w[keep]@g[keep]),
                score_only_ba_contribution=float(100*w[keep]@delta['score_only'][keep]),
                contextual_ba_contribution=float(100*w[keep]@delta['contextual'][keep]),
                score_only_within_class_auc=auc(g[keep],d['score_only'][keep],w[keep]),
                contextual_within_class_auc=auc(g[keep],d['contextual'][keep],w[keep]),
                support_oof_rescue=int(z['profile_rescue'][cl]),support_oof_harm=int(z['profile_harm'][cl])))
        split_groups = [('support_count',np.minimum(counts[y],2),range(3)),
                        ('support_ship_seen',np.isin(ships,ss).astype(int),range(2)),
                        ('support_class_has_oof_event',
                         ((z['profile_rescue']+z['profile_harm'])[y]>0).astype(int),range(2))]
        for kind, ids, levels in split_groups:
            totals = []
            # Empty strata contribute zero; dropping them would bias across-draw means.
            for st in levels:
                keep = ids==st
                rr = dict(**key,kind=kind,stratum=int(st),n=int(keep.sum()),
                    ba_mass=float(100*w[keep].sum()),initial_ba_contribution=float(100*w[keep]@g[keep]))
                for arm in delta:
                    rr[arm+'_ba_contribution'] = float(100*w[keep]@delta[arm][keep])
                totals.append(rr['contextual_ba_contribution'])
                strata.append(rr)
            np.testing.assert_allclose(sum(totals),100*w@delta['contextual'],atol=1e-10)
        # Coverage counts distinct source observations, not repeated exposures.
        train = events if u['family']=='external' else events[events.context==u['task']]
        pair = train.groupby(['raw_class','initial_class']).obs_id.nunique().to_dict()
        triple = train.groupby(['class_id','raw_class','initial_class']).obs_id.nunique().to_dict()
        for name, count in [('predicted_pair',np.array([pair.get((a,b),0) for a,b in zip(r,i)])),
                            ('true_class_pair',np.array([triple.get((c,a,b),0) for c,a,b in zip(y,r,i)]))]:
            event = g!=0
            ew = w[event]
            weak = count[event]<5
            cover.append(dict(**key,coverage_type=name,event_n=int(event.sum()),
                no_source_event_percent=float(100*np.average(count[event]==0,weights=ew)) if len(ew) else np.nan,
                fewer5_source_obs_percent=float(100*np.average(weak,weights=ew)) if len(ew) else np.nan,
                score_auc=auc(g,d['score_only'],w),context_auc=auc(g,d['contextual'],w)))
            ids = np.where(count==0,0,np.where(count<5,1,2))
            for st in range(3):
                keep = ids==st
                rr = dict(**key,kind=name+'_source_coverage',stratum=st,n=int(keep.sum()),
                    ba_mass=float(100*w[keep].sum()),initial_ba_contribution=float(100*w[keep]@g[keep]))
                for arm in delta:
                    rr[arm+'_ba_contribution'] = float(100*w[keep]@delta[arm][keep])
                strata.append(rr)
        print('ANALYZED',u['uid'],flush=True)
    frames = {'composition':pd.DataFrame(composition),'classes':pd.DataFrame(classes),
              'strata':pd.DataFrame(strata),'event_coverage':pd.DataFrame(cover),'vessel_sensitivity':pd.DataFrame(vessel)}
    group=['family','task','cohort']
    for name, frame in frames.items():
        frame.to_csv(OUT/(name+'.csv'),index=False)
        extra={'classes':['class_id','class_name'],'strata':['kind','stratum'],
               'event_coverage':['coverage_type'],'vessel_sensitivity':['arm']}.get(name,[])
        summarize(frame,name+'_summary.csv',group+extra)
    # Within-class AUC is averaged over estimable class/unit cells, with coverage explicit.
    cf=frames['classes']
    cf.groupby(group).agg(score_within_class_auc=('score_only_within_class_auc','mean'),
        context_within_class_auc=('contextual_within_class_auc','mean'),
        estimable_cells=('contextual_within_class_auc','count'),class_unit_cells=('class_id','size')).reset_index().to_csv(
            OUT/'within_class_ranking.csv',index=False)
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(status='complete',units=len(units),new_fits=0,
        new_predictions=0,diagnostic_only=True,reference_metric_replays=4*len(units),
        partition_checks=3*len(units)),indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
