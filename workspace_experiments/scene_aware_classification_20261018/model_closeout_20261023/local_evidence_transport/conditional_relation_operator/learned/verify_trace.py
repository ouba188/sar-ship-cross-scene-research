"""Check selected nonidentity trace against saved GPU features and scalar algebra."""
import sys
sys.dont_write_bytecode = True
import json
import numpy as np
import pandas as pd
import run as experiment


def main():
    table,x,mass,bank,idx,tr,va,te = experiment.source_data()
    weights = np.load(experiment.HERE/'weights.npz')
    features = np.load(experiment.HERE/'features_43.npz')
    _,_,legal,_ = experiment.edge.edges(x,mass)
    utility = idx[tr]
    partial = utility[(legal[utility].sum(-1)==3).any(-1)]
    ids = np.unique(np.r_[utility[:1],partial[:1]])
    error,scalar_error = 0.,0.
    for arm in ('independent','excess'):
        mb,mn = [weights[f'{arm}_43_{key}'] for key in ('Mb','Mn')]
        assert min(np.linalg.norm(mb-np.eye(32)),np.linalg.norm(mn-np.eye(32)))>0
        for i in ids:
            trace = experiment.trace_query(int(i),arm,mb,mn,x,mass,bank)
            error = max(error,float(np.max(np.abs(trace['raw_scores']-features[f'{arm}_raw'][i]))))
            for g in range(4):
                if trace['mq'][g,0]<2:
                    continue
                r = int(np.flatnonzero(trace['ms'][g]>=2)[0])
                qm,sm = trace['legal_edges'][0,g],trace['legal_edges'][r+1,g]
                qb,qn = trace['qb'][g,0,qm],trace['qn'][g,0,qm]
                sb,sn = trace['sb'][g,r,sm],trace['sn'][g,r,sm]
                if arm == 'independent':
                    score = .5*(experiment.fixed.transfer(qb,qb,sb,sb)['transfer'].mean()+
                                experiment.fixed.transfer(qn,qn,sn,sn)['transfer'].mean())
                else:
                    score = experiment.fixed.transfer(qb,qn,sb,sn)['excess'].mean()
                scalar_error = max(scalar_error,abs(float(score-trace['reference_scores'][g,0,r//8,r%8])))
    assert error<1e-5 and scalar_error<1e-10,(error,scalar_error)
    receipt = pd.read_csv(experiment.HERE/'training_receipts.csv')
    initial_max = 0.
    for seed in experiment.SEEDS:
        with np.load(experiment.HERE/f'features_{seed}.npz') as z:
            for arm in receipt[(receipt.seed==seed)&(receipt.selected_step==0)].arm:
                initial_max = max(initial_max,float(np.nanmax(np.abs(z[arm]))))
    assert initial_max==0.,initial_max
    result = dict(selected_state='config43 independent/excess step20',utility_indices=ids.tolist(),
        actual_partial_mask=partial.tolist(),saved_GPU_vs_CPU_trace_error=error,
        scalar_vs_vector_nonidentity_error=scalar_error,all_selected_step0_feature_max=initial_max,
        fits=len(receipt),expected_fits=15,no_new_evaluation=True)
    assert len(receipt)==15
    (experiment.HERE/'trace_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
