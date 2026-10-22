"""Audit actual pooling outputs and source-only scale/selection; never refit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax,logsumexp

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RUN=ROOT/'contrastive_spatial_pool_20261023'
NEW=ROOT/'target_anchored_context_20261023/partition_replication'
EXP=ROOT/'context_expansion_20261022'


def balanced(x,y):
    return float(np.mean([x[y==c].mean() for c in range(3)]))


def loss(s,y):
    return balanced(logsumexp(s,axis=1)-s[np.arange(len(y)),y],y)


def main():
    t=pd.read_csv(EXP/'readout/targets.csv',dtype={'mmsi':str})
    data=np.load(RUN/'delta_features.npz')
    delta=data['delta'];arms=data['arms'].tolist()
    np.testing.assert_array_equal(data['instance_ids'],t.instance_id)
    assert delta.shape==(6,5695,3,3) and np.isfinite(delta).all()
    pred=pd.read_csv(RUN/'predictions.csv')
    choices=pd.read_csv(RUN/'selections.csv').set_index(['fold','arm'])
    scale,derivatives,norms,equiv,checks,byclass=[],[],[],[],[],[]
    for fold in range(6):
        d=np.load(NEW/f'fold{fold}_inputs.npz')
        w=np.load(NEW/'origins'/f'fold{fold}_outer.npz')
        np.testing.assert_array_equal(d['train'],w['fit'])
        np.testing.assert_array_equal(d['instance_ids'],t.instance_id)
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            assert not set(t.iloc[d[a]].mmsi)&set(t.iloc[d[b]].mmsi)
            assert not set(t.iloc[d[a]].pass_group)&set(t.iloc[d[b]].pass_group)
        for pol in range(2):
            wp=w['W'][:,pol*1024:(pol+1)*1024]
            norms.extend(dict(fold=fold,pol=['vv','vh'][pol],class_id=c,L2_norm=float(np.linalg.norm(wp[c]))) for c in range(3))
        equiv.append(dict(fold=fold,max_class_contrast_delta_difference=float(abs(delta[fold,:,0]-delta[fold,:,1]).max()),
            mean_class_contrast_delta_difference=float(abs(delta[fold,:,0]-delta[fold,:,1]).mean())))
        for role in ('train','validation'):
            ix=d[role];y=t.iloc[ix].class_id.to_numpy();base=d['base'][ix].astype(float)
            order=np.argsort(base,axis=1);top,second=order[:,-1],order[:,-2]
            gap=base[np.arange(len(ix)),top]-base[np.arange(len(ix)),second]
            p=softmax(base,axis=1)
            for j,arm in enumerate(arms):
                dd=delta[fold,ix,j]
                pair=dd[np.arange(len(ix)),top]-dd[np.arange(len(ix)),second]
                drange=np.ptp(dd,axis=1)
                rec=dict(fold=fold,role=role,arm=arm,n=len(ix),
                    A_gap_p50=float(np.median(gap)),A_gap_p95=float(np.quantile(gap,.95)),
                    delta_pair_abs_p50=float(np.median(abs(pair))),delta_pair_abs_p95=float(np.quantile(abs(pair),.95)),
                    delta_range_p50=float(np.median(drange)),delta_range_p95=float(np.quantile(drange,.95)),
                    lambda01_pair_abs_p50=float(np.median(.1*abs(pair))),
                    lambda01_exceeds_A_gap_fraction=float((.1*abs(pair)>gap).mean()),
                    lambda01_changes_A_top1_fraction=float(((base+.1*dd).argmax(1)!=base.argmax(1)).mean()))
                scale.append(rec)
                mean=(p*dd).sum(1)
                grad=balanced(mean-dd[np.arange(len(ix)),y],y)
                curv=balanced((p*(dd-mean[:,None])**2).sum(1),y)
                loss0,loss01=loss(base,y),loss(base+.1*dd,y)
                derivatives.append(dict(fold=fold,role=role,arm=arm,NLL0=loss0,NLL01=loss01,
                    balanced_NLL_derivative_at_0=grad,balanced_NLL_curvature_at_0=curv,
                    selected_lambda=float(choices.loc[(fold,arm),'selected_lambda']),
                    possible_subgrid_improvement=bool(grad<0 and loss01>=loss0),
                    nonnegative_lambda_optimum_is_zero=bool(grad>=0)))
                for c in range(3):
                    hit=y==c
                    byclass.append(dict(fold=fold,role=role,arm=arm,class_id=c,n=int(hit.sum()),
                        A_gap_p50=float(np.median(gap[hit])),delta_pair_abs_p50=float(np.median(abs(pair[hit]))),
                        delta_pair_abs_p95=float(np.quantile(abs(pair[hit]),.95)),
                        lambda01_exceeds_gap_fraction=float((.1*abs(pair[hit])>gap[hit]).mean())))
        val,test=d['validation'],d['test'];yy=t.iloc[val].class_id.to_numpy()
        for j,arm in enumerate(arms):
            candidates=[(lam,loss(d['base'][val].astype(float)+lam*delta[fold,val,j],yy)) for lam in (0.,.1,.25,.5,1.)]
            bestloss=float('inf');best=None
            for lam,value in candidates:
                if value<bestloss-1e-12:
                    bestloss,best=value,lam
            assert best==float(choices.loc[(fold,arm),'selected_lambda'])
            rows=pred[pred.fold.eq(fold)&pred.arm.eq(arm)].set_index('instance_id').loc[t.iloc[test].instance_id]
            score=d['base'][test]+best*delta[fold,test,j]
            err=float(abs(rows[['s0','s1','s2']].to_numpy()-score).max())
            assert err<1e-12
            np.testing.assert_array_equal(rows.prediction,score.argmax(1))
            checks.append(dict(fold=fold,arm=arm,source_selection_reproduced=True,prediction_error=err))
    for name,rows in [('source_scale',scale),('source_derivatives',derivatives),('source_scale_by_class',byclass),
                      ('raw_W_norms',norms),('full_delta_equivalence',equiv),('selection_replay',checks)]:
        pd.DataFrame(rows).to_csv(HERE/(name+'.csv'),index=False)
    maps=np.load(RUN/'validation_maps.npz');errors=[]
    for ident in ('pair_0032624','pair_0053959','pair_0030906','pair_0030918'):
        total=np.zeros((3,3));i=int(t.index[t.instance_id.eq(ident)][0]);fold=None
        for pol in ('vv','vh'):
            for rot in range(4):
                prefix=f'{ident}_{pol}_rot{rot}'
                v=maps[prefix+'_v'].reshape(3,-1);r=v-v.mean(0,keepdims=True)
                np.testing.assert_allclose(r,maps[prefix+'_r'].reshape(3,-1),atol=1e-12)
                independent=np.stack([(softmax(v,axis=1)*v).sum(1),(softmax(r,axis=1)*r).sum(1),r.std(1,ddof=0)])
                errors.append(float(abs(independent-maps[prefix+'_delta']).max()))
                total+=independent/4;fold=int(maps[prefix+'_fold'])
        errors.append(float(abs(total-delta[fold,i]).max()))
    assert max(errors)<1e-10
    result=dict(folds=6,all_predictions_replayed=True,all_source_selections_replayed=True,
        pol_c4_replay_cases=4,max_independent_map_error=max(errors),
        max_full_class_contrast_delta_difference=max(x['max_class_contrast_delta_difference'] for x in equiv),
        source_scale_roles=['train','validation'],test_used_for_scale_selection=False,
        lambda_or_temperature_sweeps_added=False,training_or_feature_extraction=False)
    (HERE/'implementation_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(pd.DataFrame(scale).loc[lambda x:x.role.eq('train')].to_string(index=False))
    print(pd.DataFrame(derivatives).loc[lambda x:x.role.eq('validation')].to_string(index=False))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
