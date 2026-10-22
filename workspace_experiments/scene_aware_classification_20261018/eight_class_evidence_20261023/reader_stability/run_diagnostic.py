"""Source-only audit of changing evidence directions and centering references."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '2'
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
BASE = PARENT/'baseline'


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def main():
    cohort = pd.read_csv(PARENT/'cohort.csv', dtype={'mmsi':str})
    raw = np.load(PARENT/'features.npz',allow_pickle=True)
    np.testing.assert_array_equal(raw['instance_ids'].astype(str),cohort.instance_id)
    h = raw['layer3'].astype(np.float64)
    groups = np.arange(h.shape[1]).reshape(16,-1)
    aggregate, directions, samples = [], [], []
    for fold in range(3):
        d = np.load(BASE/f'fold{fold}_inputs.npz')
        outer = np.load(BASE/'models'/f'fold{fold}_outer.npz')
        wo, ho = outer['W'],outer['h0']
        scale = h[d['train']].std(0)
        for inner in range(4):
            model = np.load(BASE/'models'/f'fold{fold}_inner{inner}.npz')
            ix, wi, hi = model['held'], model['W'],model['h0']
            assert set(ix) <= set(d['train']) and not set(ix)&set(d['test'])
            qouter, qinner = d['q'][ix], d['oq'][ix]
            direction = np.stack([(h[ix][:,g]-ho[g])@(wi[:,g]-wo[:,g]).T for g in groups],axis=1)
            reference = np.stack([(ho[g]-hi[g])@wi[:,g].T for g in groups])
            error = float(np.abs(qinner-qouter-direction-reference).max())
            assert error < 1e-8,error
            aggregate.append(dict(fold=fold,inner=inner,n=len(ix),
                q_difference_rms=rms(qinner-qouter),direction_rms=rms(direction),
                reference_rms=rms(reference),outer_q_rms=rms(qouter),error=error))
            for c in range(8):
                for r in range(c):
                    for gi,g in enumerate(groups):
                        u=(wo[c,g]-wo[r,g])*scale[g]
                        v=(wi[c,g]-wi[r,g])*scale[g]
                        cosine=float(u@v/max(np.linalg.norm(u)*np.linalg.norm(v),1e-12))
                        directions.append(dict(fold=fold,inner=inner,class_a=c,class_b=r,group=gi,cosine=cosine))
            y=cohort.iloc[ix].class_id.to_numpy()
            rival=np.where(np.eye(8,dtype=bool)[y],-np.inf,d['base'][ix]).argmax(1)
            row=np.arange(len(ix))[:,None]
            group=np.arange(16)[None,:]
            mo=qouter[row,group,y[:,None]]-qouter[row,group,rival[:,None]]
            mi=qinner[row,group,y[:,None]]-qinner[row,group,rival[:,None]]
            mass=np.sqrt(np.abs(mo*mi))
            flips=(mo*mi)<0
            for j,i in enumerate(ix):
                samples.append(dict(fold=fold,inner=inner,instance_id=cohort.iloc[i].instance_id,
                    class_id=int(y[j]),rival=int(rival[j]),
                    mass_weighted_sign_disagreement=float((mass[j]*flips[j]).sum()/max(mass[j].sum(),1e-12)),
                    inner_correct=bool(d['oof'][i].argmax()==y[j]),
                    outer_in_sample_correct=bool(d['base'][i].argmax()==y[j])))
    pd.DataFrame(aggregate).to_csv(HERE/'decomposition.csv',index=False)
    dd=pd.DataFrame(directions)
    dd.to_csv(HERE/'direction_cosines.csv',index=False)
    ss=pd.DataFrame(samples)
    ss.to_csv(HERE/'source_events.csv',index=False)
    ss.groupby(['fold','class_id']).agg(n=('instance_id','size'),
        sign_disagreement=('mass_weighted_sign_disagreement','mean'),
        held_accuracy=('inner_correct','mean'),in_sample_accuracy=('outer_in_sample_correct','mean')).to_csv(HERE/'class_effects.csv')
    result=dict(source_events=len(ss),independent_observations=int(ss.instance_id.nunique()),
        mean_direction_cosine=float(dd.cosine.mean()),median_direction_cosine=float(dd.cosine.median()),
        direction_cosine_p05=float(dd.cosine.quantile(.05)),
        mean_mass_weighted_sign_disagreement=float(ss.mass_weighted_sign_disagreement.mean()),
        caveat='Outer predicts its own training samples; this is coordinate/direction stability, not unbiased accuracy or a target-set experiment.')
    (HERE/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(pd.DataFrame(aggregate).to_string(index=False))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
