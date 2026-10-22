"""Posthoc objective diagnostics. Oracle labels never enter a deployed gate."""
import json
import numpy as np
import pandas as pd
import run as exp

ROOT=exp.ROOT


def main():
    f,splits=exp.old.relation.load_data();rows=[];details=[];mismatch=[]
    prior_oracle=pd.read_csv(exp.PREVIOUS/'oracle_family_rows.csv')
    for fold,(tr,val,test) in enumerate(splits):
        oof=np.load(ROOT/f'oof_f{fold}.npz')
        for seed in range(3):
            full=np.load(exp.PREVIOUS/f'inputs_f{fold}_s{seed}.npz')
            active_oof=np.abs(oof['residual'][seed]).max(1)>1e-8
            active_full=np.abs(full['residual'][tr]).max(1)>1e-8
            mismatch.append(dict(fold=fold,seed=seed,n=len(tr),oof_active=int(active_oof.sum()),
                full_active=int(active_full.sum()),both_active=int((active_oof&active_full).sum()),
                oof_zero_full_active=int((~active_oof&active_full).sum()),
                oof_active_full_zero=int((active_oof&~active_full).sum())))
            a=full['base'][test].astype(np.float64);d=full['residual'][test].astype(np.float64)
            y=f.iloc[test].class_id.to_numpy();target=2*np.eye(8)[y]-1
            denom=np.sum(d*d,1);w=np.ones(len(test))
            active=denom>1e-12
            w[active]=np.clip(np.sum(d[active]*(target[active]-a[active]),1)/denom[active],0,2)
            scores=a+w[:,None]*d
            reference=a+d
            rc=reference.argmax(1)==y;mc=scores.argmax(1)==y
            oracle=prior_oracle[(prior_oracle.fold==fold)&(prior_oracle.seed==seed)].set_index('obs_id').loc[f.iloc[test].obs_id]
            reachable=oracle.reachable.to_numpy(bool)
            # The closed form must not increase each sample's own quadratic loss.
            ref_loss=np.mean((reference-target)**2,1);new_loss=np.mean((scores-target)**2,1)
            assert (new_loss<=ref_loss+1e-10).all()
            rows.append(dict(fold=fold,seed=seed,n=len(test),reference_correct=int(rc.sum()),
                label_informed_mse_correct=int(mc.sum()),reachable_correct=int(reachable.sum()),
                mse_rescue=int((mc&~rc).sum()),mse_harm=int((~mc&rc).sum()),
                reachable_rescue_but_mse_misses=int((reachable&~rc&~mc).sum()),
                no_scalar_solution=int((~reachable&~rc).sum()),
                **{f'oracle_{k}':v for k,v in exp.old.control.metrics(y,scores).items()}))
            for j,i in enumerate(test):
                details.append(dict(fold=fold,seed=seed,obs_id=int(f.iloc[i].obs_id),
                    class_id=int(y[j]),reference_correct=bool(rc[j]),mse_correct=bool(mc[j]),
                    reachable=bool(reachable[j]),posthoc_mse_weight=float(w[j]),
                    reference_mse=float(ref_loss[j]),oracle_mse=float(new_loss[j])))
    pd.DataFrame(rows).to_csv(ROOT/'objective_oracle_summary.csv',index=False)
    pd.DataFrame(details).to_csv(ROOT/'objective_oracle_rows.csv',index=False)
    pd.DataFrame(mismatch).to_csv(ROOT/'residual_signal_mismatch.csv',index=False)
    report=dict(interpretation='Posthoc true-label quadratic oracle, NOT a candidate or deployable performance. No retraining.',
        reason='Separate inability to learn weights from mismatch between quadratic loss and top-1 class boundaries.',
        rowwise_quadratic_nonincrease_verified=True)
    (ROOT/'DIAGNOSTIC_SCOPE.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).groupby('seed')[['reference_correct','label_informed_mse_correct','reachable_correct',
        'mse_rescue','mse_harm','reachable_rescue_but_mse_misses','no_scalar_solution']].sum().to_string())
    print(pd.DataFrame(mismatch).to_string(index=False))


if __name__=='__main__':main()
