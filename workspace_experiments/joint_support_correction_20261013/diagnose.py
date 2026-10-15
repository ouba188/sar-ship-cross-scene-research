"""Paired risk and support-coverage diagnosis from frozen probabilities only."""

import run_joint as r
from run_joint import np, pd, OUT, utility


def main():
    receipt = r.read(OUT/'PREDICTIONS_FROZEN.json')
    labels = {d['dataset']:pd.read_csv(d['manifest'],usecols=['class_id']).class_id.to_numpy(int)
              for d in receipt['datasets']}
    records=[]
    for unit in receipt['units']:
        z=r.load(unit['prediction_path'])
        sy=r.load(unit['weights_path'],['support_labels'])['support_labels']
        y=labels[unit['dataset']][z['query_ids']]
        w=utility.class_weights(y)
        truth=np.eye(8)[y]
        key={k:unit[k] for k in r.KEY}
        for candidate, reference in [('combined','initial'),('initial_guard','initial'),('light','raw')]:
            p,b=z[candidate],z[reference]
            pred,baseline=p.argmax(1),b.argmax(1)
            gain=np.square(b-truth).sum(1)-np.square(p-truth).sum(1)
            change=pred!=baseline
            delta=(pred==y).astype(float)-(baseline==y)
            absent=~np.isin(y,sy)
            row=dict(**key,candidate=candidate,reference=reference,
                risk_gain=float(w@gain),risk_gain_unchanged=float(w@(gain*~change)),
                risk_gain_changed=float(w@(gain*change)),
                delta_ba=100*float(w@delta),delta_oa=100*float(delta.mean()),
                absent_class_delta_ba=100*float(w@(delta*absent)),
                present_class_delta_ba=100*float(w@(delta*~absent)),
                query_changed=int(change.sum()),query_n=len(y),
                support_fishing=int((sy==1).sum()),absent_classes=8-len(np.unique(sy)))
            np.testing.assert_allclose(row['risk_gain_unchanged']+row['risk_gain_changed'],row['risk_gain'],atol=1e-12)
            records.append(row)
    f=pd.DataFrame(records)
    f.to_csv(OUT/'risk_coverage.csv',index=False)
    f.groupby(['dataset','cohort','candidate','reference']).mean(numeric_only=True).to_csv(OUT/'risk_coverage_summary.csv')
    classes=pd.read_csv(OUT/'classes.csv')
    joined=classes[classes.arm=='combined'].merge(classes[classes.arm=='initial'],
        on=r.KEY+['class_id'],suffixes=('_combined','_initial'),validate='one_to_one')
    present=classes[classes.arm=='raw'].groupby(r.KEY).class_id.nunique()
    joined=joined.join(present.rename('class_n'),on=r.KEY)
    joined['ba_contribution']=(joined.recall_combined-joined.recall_initial)/joined.class_n
    joined['risk_change']=joined.brier_combined-joined.brier_initial
    joined.to_csv(OUT/'combined_class_changes.csv',index=False)
    joined.groupby(['dataset','cohort','class_id'])[['ba_contribution','risk_change','support_n_combined']].mean().to_csv(OUT/'combined_class_summary.csv')
    print(f.groupby(['dataset','cohort','candidate'])[['risk_gain','risk_gain_unchanged','delta_ba',
        'absent_class_delta_ba','present_class_delta_ba','support_fishing']].mean().to_string(),flush=True)


if __name__=='__main__':
    main()
