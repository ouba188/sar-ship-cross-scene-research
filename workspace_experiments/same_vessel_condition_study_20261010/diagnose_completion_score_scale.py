"""Fixed score-direction/amplitude attribution, not target-selected calibration."""

from full_teacher_data import BASE, SOURCE, PORTS, write
from unified_head_core import late
import numpy as np
import pandas as pd

OUT=BASE/'support_completion_target_20261012'


def main():
    dest=OUT/'score_scale_diagnostic';dest.mkdir(exist_ok=True)
    write(dest/'PROTOCOL.json',dict(
        question='Does changed expert score amplitude account for fusion degradation?',
        intervention='Per-query centered logit L2 direction and norm exchanged between old and corrected expert',
        shape_only='New direction with old norm; preserves corrected expert argmax',
        amplitude_only='Old direction with new norm; preserves original expert argmax',
        use='Post-result fixed diagnostic, not a source-selected deployable method or hyperparameter sweep',
        no_training=True,no_label_input_to_intervention=True))
    manifest=pd.read_csv(SOURCE/'manifest.csv',usecols=['class_id'])
    records,stats,classes=[],[],[]
    for port in PORTS:
        with np.load(OUT/port/'predictions.npz') as z:
            data={k:z[k] for k in z.files}
        y=manifest.iloc[data['rows']].class_id.to_numpy()
        for draw in range(5):
            old,new=data['expert_raw'][draw],data['expert_corrected'][draw]
            oc,nc=old-old.mean(1,keepdims=True),new-new.mean(1,keepdims=True)
            on,nn=np.linalg.norm(oc,axis=1,keepdims=True),np.linalg.norm(nc,axis=1,keepdims=True)
            assert (on>0).all() and (nn>0).all()
            shape=nc*on/nn;amplitude=oc*nn/on
            np.testing.assert_array_equal(shape.argmax(1),new.argmax(1))
            np.testing.assert_array_equal(amplitude.argmax(1),old.argmax(1))
            variants=dict(raw=old,corrected=new,shape_only=shape,amplitude_only=amplitude)
            mask=data['eval_masks'][draw]
            ratio=(nn/on).ravel()[mask]
            stats.append(dict(port=port,draw=draw,norm_ratio_median=float(np.median(ratio)),
                norm_ratio_q10=float(np.quantile(ratio,.1)),norm_ratio_q90=float(np.quantile(ratio,.9))))
            for name,expert in variants.items():
                fused=late(data['base'][draw],expert)
                if name in ('raw','corrected'):
                    np.testing.assert_allclose(fused,data['fusion_'+name][draw],atol=1e-12,rtol=0)
                for endpoint,scores in [('expert',expert),('fusion',fused)]:
                    truth=y[mask];pred=scores[mask].argmax(1)
                    records.append(dict(port=port,draw=draw,arm=name,endpoint=endpoint,
                        oa=float(100*np.mean(truth==pred)),
                        ba=float(100*np.mean([np.mean(pred[truth==c]==c) for c in np.unique(truth)]))))
                    for c in np.unique(truth):
                        m=truth==c
                        classes.append(dict(port=port,draw=draw,arm=name,endpoint=endpoint,class_id=int(c),
                            recall=float(100*np.mean(pred[m]==c))))
    df=pd.DataFrame(records);df.to_csv(dest/'metrics.csv',index=False)
    pd.DataFrame(classes).to_csv(dest/'classes.csv',index=False)
    pd.DataFrame(stats).to_csv(dest/'norm_ratios.csv',index=False)
    summary=df.groupby(['endpoint','arm'])[['oa','ba']].mean()
    summary.to_csv(dest/'summary.csv')
    df.groupby(['port','endpoint','arm'])[['oa','ba']].mean().to_csv(dest/'ports.csv')
    print(summary.to_string(),flush=True)
    print(pd.DataFrame(stats).groupby('port').norm_ratio_median.mean().to_string(),flush=True)


if __name__=='__main__':
    main()
