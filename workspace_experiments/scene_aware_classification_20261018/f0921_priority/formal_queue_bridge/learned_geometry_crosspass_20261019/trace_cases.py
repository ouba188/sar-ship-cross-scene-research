"""Replay selected queries and decompose direct kernel/refit evidence changes."""
import json
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from runner import Study,Reader,HERE


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    study = Study()
    cases = pd.read_csv(HERE/'selected_cases.csv')
    contributions,paths,condition_sensitivity = [],[],[]
    reader_cache = {}
    for case in cases.itertuples():
        task = next(t for t in study.tasks['tasks'] if t['name']==case.task)
        ct = study.context(task['source'],task['supports'][case.draw],task['query'])
        j = list(ct['q']).index(case.row); ns=ct['ns']; qi=ns+j
        results = {}
        for arm in ['C',case.comparator]:
            if arm=='base':
                results[arm] = (ct['base_prob'],ct['base'],ct['base_alpha'],ct['base_score'],None)
            else:
                key = (arm,case.seed)
                if key not in reader_cache:
                    r=Reader(arm).cuda().eval()
                    r.load_state_dict(torch.load(HERE/f'{arm}_seed{case.seed}.pt',weights_only=True))
                    reader_cache[key]=r
                with torch.no_grad(): results[arm] = study.predict(reader_cache[key],ct)
        p,k,alpha,score,w = [v.cpu().numpy() if v is not None else None for v in results['C']]
        p0,k0,a0,s0,w0 = [v.cpu().numpy() if v is not None else None for v in results[case.comparator]]
        np.testing.assert_allclose(p[j].max(),pd.read_csv(HERE/'predictions.csv').query(
            'arm=="C" and task==@case.task and seed==@case.seed and draw==@case.draw and row==@case.row')[[f'p{i}' for i in range(7)]].to_numpy().max(),atol=2e-8)
        truth=case.truth
        rivals=p0[j].copy();rivals[truth]=-np.inf;rival=int(rivals.argmax())
        aa=alpha[:,truth]-alpha[:,rival];ab=a0[:,truth]-a0[:,rival]
        ka,kb=k[qi,:ns],k0[qi,:ns]
        direct=.5*(ka-kb)*(aa+ab)
        refit=.5*(ka+kb)*(aa-ab)
        total=direct+refit
        observed=(score[qi,truth]-score[qi,rival])-(s0[qi,truth]-s0[qi,rival])
        np.testing.assert_allclose(total.sum(),observed,atol=1e-10)
        tags=dict(task=case.task,seed=case.seed,draw=case.draw,row=case.row,comparator=case.comparator,category=case.category)
        paths.append(dict(**tags,truth=truth,fixed_final_rival=rival,direct=float(direct.sum()),refit=float(refit.sum()),
            branch_margin_change=float(observed),same_class_sum=float(total[study.y[ct['s']]==truth].sum()),
            old_final_margin=float(p0[j,truth]-np.delete(p0[j],truth).max()),new_final_margin=float(p[j,truth]-np.delete(p[j],truth).max())))
        for t,s in enumerate(ct['s']):
            contributions.append(dict(**tags,support_row=int(s),support_label=int(study.y[s]),
                old_similarity=float(kb[t]),new_similarity=float(ka[t]),direct=float(direct[t]),refit=float(refit[t]),total=float(total[t])))
        key=70000+case.seed*1000+int(case.task[0])*100+case.draw
        wrong,*_=study.wrong_geometry(key)
        with torch.no_grad(): pm,km,am,sm,wm=study.predict(reader_cache[('C',case.seed)],ct,wrong)
        condition_sensitivity.append(dict(**tags,prob_mean_abs=float(np.abs(p[j]-pm[j].cpu().numpy()).mean()),
            qs_mean_abs=float(np.abs(k[qi,:ns]-km[qi,:ns].cpu().numpy()).mean()),
            attention_mean_abs=float(np.abs(w[qi]-wm[qi].cpu().numpy()).mean())))
    pd.DataFrame(paths).to_csv(HERE/'case_paths.csv',index=False)
    pd.DataFrame(contributions).to_csv(HERE/'case_support_contributions.csv',index=False)
    pd.DataFrame(condition_sensitivity).to_csv(HERE/'case_condition_sensitivity.csv',index=False)
    print(pd.DataFrame(paths).to_string(index=False))


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
