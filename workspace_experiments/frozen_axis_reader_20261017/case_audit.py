"""Review native query/support chips for flips and persistent decisions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from frozen_reader import predict

HERE = Path(__file__).resolve().parent
KEYS = ['fold','seed','origin','draw','global_row']


def load_task(fold,origin,draw):
    with np.load(HERE/fold/'tasks'/f'{origin}_{draw}.npz') as cache:
        task = {k:torch.tensor(cache[k]) for k in ['ss','qs','root','desired','prior_s','prior_q','base','old']}
        for k in ['ids','s','q']:
            task[k] = cache[k].copy()
        task['ns'] = int(cache['ns'])
    return task


def solve(task,h):
    ns = task['ns']
    ss = task['ss']+h[:ns]@h[:ns].T
    qs = task['qs']+h[ns:]@h[:ns].T
    root = task['root']
    coef = root[:,None]*torch.linalg.solve(root[:,None]*ss*root[None,:]+8*torch.eye(ns,dtype=ss.dtype),
                                         root[:,None]*(task['desired']-task['prior_s']))
    return qs,coef


def select_cases(frame):
    f = frame[frame.arm.eq('frozen_common')&frame.origin.eq('cross')].copy()
    cases = []
    for group,change in [('rescued',1),('harmed',-1)]:
        subset = f[f.change.eq(change)]
        if len(subset):
            ordered = subset.sort_values(['ba_contribution_pp','global_row','seed','draw'],
                ascending=[change<0,True,True,True])
            row = ordered.iloc[0].copy()
            row['case_group'],row['selection_reason'] = group,'Largest absolute event BA contribution, then fixed ID/seed/draw order'
            cases.append(row)
    correct = f[f.prediction.eq(f.truth)&f.reference_prediction.eq(f.truth)]
    old_case = correct[correct.global_row.eq(786)&correct.seed.eq(0)&correct.draw.eq(1)]
    if len(correct):
        if len(old_case):
            row,reason = old_case.iloc[0].copy(),'Previously harmed query 786, seed0/draw1: test preservation of effective evidence'
        else:
            ordered = correct.sort_values('max_probability_change')
            row,reason = ordered.iloc[len(ordered)//2].copy(),'Median probability change among persistent correct contexts'
        row['case_group'],row['selection_reason'] = 'persistent_correct',reason
        cases.append(row)
    wrong = f[f.prediction.ne(f.truth)&f.reference_prediction.ne(f.truth)]
    missing = wrong[~wrong.axis_valid&wrong.same_class_support.eq(0)]
    if len(wrong):
        candidates = missing if len(missing) else wrong
        ordered = candidates.sort_values(['global_row','seed','draw'])
        row = ordered.iloc[len(ordered)//2].copy()
        row['case_group'] = 'persistent_wrong'
        row['selection_reason'] = ('Median ordered context among axis-invalid errors without same-class support'
                                   if len(missing) else 'Median ordered persistent-error context')
        cases.append(row)
    return pd.DataFrame(cases)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    predictions = pd.read_csv(HERE/'paired_predictions.csv')
    cases = select_cases(predictions)
    cases.to_csv(HERE/'native_case_selection.csv',index=False)
    geometry = pd.read_csv(HERE.parent/'oss1_joint_geometry_20261017'/'geometry.csv',dtype={'mmsi':str}).set_index('global_row')
    labels = geometry.class_id.to_dict()
    class_names = geometry.drop_duplicates('class_id').set_index('class_id').class_name.to_dict()
    saved = predictions.set_index(['arm']+KEYS)
    mechanisms,edges,inventory = [],[],[]
    errors = []
    for case in cases.to_dict('records'):
        fold,seed,draw,gr = case['fold'],int(case['seed']),int(case['draw']),int(case['global_row'])
        task = load_task(fold,'cross',draw)
        with np.load(HERE/fold/f'frozen_common_{seed}'/'readouts.npz') as cache:
            global_rows = cache['global_rows']
            axis,common = torch.tensor(cache['axis_raw']),torch.tensor(cache['common_raw'])
            av,cv = torch.tensor(cache['axis_valid']),torch.tensor(cache['common_valid'])
        ids,ns = task['ids'],task['ns']
        local = int(np.flatnonzero(global_rows==gr)[0])
        qi = int(np.flatnonzero(task['q']==local)[0])
        before,h0,_ = predict(axis[ids],av[ids],task)
        after,h1,_ = predict(axis[ids],av[ids],task,common[ids],cv[ids],axis_weight=1.)
        for arm,p in [('margin',before),('frozen_common',after)]:
            key = (arm,fold,seed,'cross',draw,gr)
            error = float(abs(p[qi].numpy()-saved.loc[key,[f'p{k}' for k in range(8)]].to_numpy(float)).max())
            assert error<1e-10
            errors.append(error)
        y = int(case['truth'])
        rival = int(before[qi].masked_fill(torch.arange(8)==y,-torch.inf).argmax())
        k0,c0 = solve(task,h0)
        old_axis_active = bool(av[local] and av[task['s']].sum()>=2)
        if not old_axis_active:
            k0,c0 = solve(task,torch.zeros_like(h0))
        replay_before = .5*(task['base']+(task['prior_q']+k0@c0).softmax(-1))
        torch.testing.assert_close(replay_before[qi],before[qi],atol=1e-12,rtol=0)
        k1,c1 = solve(task,h1)
        v0,v1 = c0[:,y]-c0[:,rival],c1[:,y]-c1[:,rival]
        direct = (k1[qi]-k0[qi])*v1
        refit = k0[qi]*(v1-v0)
        expert_change = k1[qi]@v1-k0[qi]@v0
        torch.testing.assert_close(direct.sum()+refit.sum(),expert_change,atol=1e-12,rtol=0)
        support_rows = global_rows[task['s']]
        sy = np.array([labels[int(r)] for r in support_rows])
        contribution = (k1[qi]*v1).numpy()
        picks = []
        for role,target in [('same_class_support',y),('rival_class_support',rival)]:
            choices = np.flatnonzero(sy==target)
            best = int(choices[np.argmax(abs(contribution[choices]))]) if len(choices) else None
            picks.append((role,None if best is None else int(support_rows[best])))
        record = dict(case_group=case['case_group'],selection_reason=case['selection_reason'],fold=fold,seed=seed,draw=draw,
            global_row=gr,class_name=class_names[y],truth=y,rival=rival,rival_name=class_names[rival],
            before_prediction=int(before[qi].argmax()),after_prediction=int(after[qi].argmax()),
            old_true_probability=float(before[qi,y]),new_true_probability=float(after[qi,y]),
            old_margin=float(before[qi,y]-before[qi,rival]),new_margin=float(after[qi,y]-after[qi,rival]),
            axis_valid=bool(av[local]),same_class_support=int((sy==y).sum()),rival_class_support=int((sy==rival).sum()),
            common_direct_expert_margin=float(direct.sum()),coefficient_refit_margin=float(refit.sum()),
            expert_margin_change=float(expert_change),same_support=picks[0][1],rival_support=picks[1][1])
        record['reference_path'] = 'axis_reader' if old_axis_active else 'legacy_fallback'
        record['old_solver_matches_deployed_query'] = True
        mechanisms.append(record)
        for i,s in enumerate(support_rows):
            edges.append(dict(case_group=case['case_group'],fold=fold,seed=seed,draw=draw,query_row=gr,support_row=int(s),
                query_class=y,support_class=int(sy[i]),old_kernel=float(k0[qi,i]),new_kernel=float(k1[qi,i]),
                common_kernel=float(k1[qi,i]-k0[qi,i]),old_coefficient=float(v0[i]),new_coefficient=float(v1[i]),
                direct_margin=float(direct[i]),refit_margin=float(refit[i]),new_edge_margin=float(contribution[i])))
        fig,axes = plt.subplots(3,2,figsize=(9,10))
        for i,(role,r) in enumerate([('query',gr)]+picks):
            for j,pol in enumerate(['vv','vh']):
                ax = axes[i,j]
                if r is None:
                    ax.text(.5,.5,'No support of this class in the actual draw',ha='center',va='center',wrap=True,fontsize=10)
                    ax.axis('off')
                    continue
                meta = geometry.loc[r]
                path = meta['chip_path_'+pol]
                with Image.open(path) as im:
                    native = np.array(im)
                ax.imshow(native,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                ax.set_title(f"{role}: {r} | {meta.class_name}\n{pol.upper()} | native {native.shape[1]}x{native.shape[0]}",fontsize=10)
                ax.axis('off')
                inventory.append(dict(case_group=case['case_group'],role=role,global_row=r,polarization=pol,path=path,
                    width=native.shape[1],height=native.shape[0],dtype=str(native.dtype),minimum=int(native.min()),maximum=int(native.max())))
        fig.suptitle(f"{case['case_group']} | {fold} | seed {seed}, draw {draw}\n"
                     f"truth {class_names[y]} | {class_names[int(before[qi].argmax())]} -> {class_names[int(after[qi].argmax())]}",fontsize=11)
        fig.tight_layout()
        fig.savefig(HERE/f"native_{case['case_group']}.png",dpi=140)
        plt.close(fig)
    pd.DataFrame(mechanisms).to_csv(HERE/'native_case_mechanisms.csv',index=False)
    pd.DataFrame(edges).to_csv(HERE/'native_case_support_edges.csv',index=False)
    pd.DataFrame(inventory).to_csv(HERE/'native_case_inventory.csv',index=False)
    (HERE/'native_case_replay.json').write_text(json.dumps(dict(cases=len(cases),probability_replay_max_error=max(errors)),indent=2),encoding='utf-8')
    print(pd.DataFrame(mechanisms).to_string(index=False))


if __name__=='__main__':
    main()
