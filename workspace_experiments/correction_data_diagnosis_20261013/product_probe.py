"""Frozen OSS2 product strata and same-ship/time paired sensitivity."""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PREV = OUT.parent/'correction_choice_probe_20261013'
sys.path.insert(0, str(PREV))
from choice_core import class_weights
from analyze import load

units = json.loads((PREV/'PREDICTIONS_FROZEN.json').read_text(encoding='utf-8'))['units']
meta = pd.read_csv(OUT.parent/'ports_root_cause_20261013/dataset_audit/public_metadata.csv',
                   usecols=['dataset','sample_id','product_type','class_id','incidence_xml'])
meta = meta[meta.dataset=='oss2'].set_index('sample_id',verify_integrity=True)
pairs = pd.read_csv(OUT.parent/'ports_root_cause_20261013/dataset_audit/oss2_grdh_slc_pairs.csv')
pairs = pairs[pairs.same_sar_datetime.astype(str).str.lower().eq('true') &
              pairs.same_label.astype(str).str.lower().eq('true')]
strata, paired = [], []
pair_catalog = None
for u in units:
    if u['family']!='external' or u['task']!='oss2':
        continue
    manifest = pd.read_csv(u['label_path'],usecols=['sample_id','class_id'])
    z = load(u['path'])
    d = load(PREV/'decisions'/(u['uid']+'.npz'))
    qm = manifest.iloc[z['query_ids']]
    md = meta.loc[qm.sample_id]
    y = qm.class_id.to_numpy(int)
    np.testing.assert_array_equal(y,md.class_id.to_numpy(int))
    np.testing.assert_array_equal(d['query_ids'],z['query_ids'])
    products = md.product_type.to_numpy()
    weight = class_weights(y)
    r, i = z['raw'].argmax(1), z['initial'].argmax(1)
    predictions = {'raw':r,'initial':i}
    for arm in ('score_only','contextual','confidence'):
        predictions[arm] = np.where(d[arm]>(0 if arm=='confidence' else .5),i,r)
    for arm, pred in predictions.items():
        partition=[]
        for product in sorted(np.unique(products)):
            keep = products==product
            delta = (pred[keep]==y[keep]).astype(int)-(i[keep]==y[keep])
            contribution = float(100*weight[keep]@delta)
            partition.append(contribution)
            strata.append(dict(cohort=u['cohort'],unit=u['unit'],product=product,arm=arm,
                n=int(keep.sum()),present_classes=len(np.unique(y[keep])),
                oa=float(100*np.mean(pred[keep]==y[keep])),
                ba=float(100*class_weights(y[keep])@(pred[keep]==y[keep])),
                delta_vs_initial_conditional_ba=float(100*class_weights(y[keep])@delta),
                delta_vs_initial_full_ba_contribution=contribution))
        np.testing.assert_allclose(sum(partition),100*weight@((pred==y).astype(int)-(i==y)),atol=1e-10)
    lookup=pd.Index(qm.sample_id)
    valid=pairs[pairs.sample_id_grdh.isin(lookup) & pairs.sample_id_slc.isin(lookup)]
    a,b=lookup.get_indexer(valid.sample_id_grdh),lookup.get_indexer(valid.sample_id_slc)
    np.testing.assert_array_equal(y[a],y[b])
    assert (products[a]=='GRDH').all() and (products[b]=='SLC').all()
    if pair_catalog is None:
        pair_catalog = valid
        valid.to_csv(OUT/'paired_query_catalog.csv',index=False)
    else:
        np.testing.assert_array_equal(valid.sample_id_grdh,pair_catalog.sample_id_grdh)
        np.testing.assert_array_equal(valid.sample_id_slc,pair_catalog.sample_id_slc)
    for arm,pred in predictions.items():
        for product,ids in [('GRDH',a),('SLC',b)]:
            yy=y[ids]
            paired.append(dict(cohort=u['cohort'],unit=u['unit'],arm=arm,product=product,
                pair_n=len(yy),ships=valid.identity.nunique(),classes=len(np.unique(yy)),
                oa=float(100*np.mean(pred[ids]==yy)),
                ba=float(100*class_weights(yy)@(pred[ids]==yy)),
                delta_vs_initial_ba=float(100*class_weights(yy)@((pred[ids]==yy).astype(int)-(i[ids]==yy)))))
for name,rows in [('product_strata',strata),('product_pairs',paired)]:
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/(name+'.csv'),index=False)
    frame.groupby(['cohort','product','arm']).mean(numeric_only=True).reset_index().to_csv(
        OUT/(name+'_summary.csv'),index=False)
print(pd.read_csv(OUT/'product_strata_summary.csv').to_string(index=False))
print('PAIRED')
print(pd.read_csv(OUT/'product_pairs_summary.csv').to_string(index=False))
