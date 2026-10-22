"""Why class/contrast coincide, and which class margins the residuals move."""
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from run_pool import HERE,NEW,ARMS


def main():
    rows=[]
    with np.load(HERE/'delta_features.npz') as z:
        delta=z['delta']
    for f in range(6):
        with np.load(NEW/'origins'/f'fold{f}_outer.npz') as z:
            w=z['W']
        rows.append(dict(fold=f,W_class_mean_max_abs=float(abs(w.mean(0)).max()),
            W_class_mean_rms=float(np.sqrt(np.mean(w.mean(0)**2))),W_rms=float(np.sqrt(np.mean(w**2))),
            relative_common_rms=float(np.sqrt(np.mean(w.mean(0)**2)/np.mean(w**2))),
            class_contrast_delta_max_diff=float(abs(delta[f,:,0]-delta[f,:,1]).max()),
            class_contrast_delta_rms_diff=float(np.sqrt(np.mean((delta[f,:,0]-delta[f,:,1])**2)))))
    pd.DataFrame(rows).to_csv(HERE/'class_common_diagnosis.csv',index=False)
    maps=[]
    with np.load(HERE/'validation_maps.npz') as z:
        for k in z.files:
            if k.endswith('_v'):
                v=z[k];common=v.mean(0)
                maps.append(dict(map=k,common_max_abs=float(abs(common).max()),common_rms=float(np.sqrt(np.mean(common**2))),
                    v_rms=float(np.sqrt(np.mean(v*v))),relative_common_rms=float(np.sqrt(np.mean(common**2)/np.mean(v*v)))))
    pd.DataFrame(maps).to_csv(HERE/'validation_map_diagnosis.csv',index=False)
    p=pd.read_csv(HERE/'predictions.csv')
    y=p.class_id.to_numpy()
    b=p[['b0','b1','b2']].to_numpy()
    d=p[['delta0','delta1','delta2']].to_numpy()
    wrong=b.copy();wrong[np.arange(len(y)),y]=-np.inf
    wrong=wrong.argmax(1)
    p['fixed_pair_increment']=p.selected_lambda.to_numpy()*(d[np.arange(len(y)),y]-d[np.arange(len(y)),wrong])
    for c in range(3):
        p[f'weighted_delta{c}']=p.selected_lambda*p[f'delta{c}']
    rows=[]
    for arm in ARMS:
        group=p[p.arm==arm]
        for scope,col in [('class','class_id'),('outcome','outcome')]:
            for name,g in group.groupby(col):
                rows.append(dict(arm=arm,scope=scope,name=name,n=len(g),net=int(g.net.sum()),
                    **{f'mean_weighted_delta{c}':g[f'weighted_delta{c}'].mean() for c in range(3)},
                    mean_true_fixed_wrong_increment=g.fixed_pair_increment.mean(),
                    median_true_fixed_wrong_increment=g.fixed_pair_increment.median()))
    pd.DataFrame(rows).to_csv(HERE/'residual_diagnosis.csv',index=False)
    print(pd.read_csv(HERE/'class_common_diagnosis.csv').to_string(index=False))
    print(pd.DataFrame(rows).query('scope == "class"').to_string(index=False))


if __name__=='__main__':
    main()
