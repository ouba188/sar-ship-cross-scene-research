"""Post-result fixed representation swaps and changed-case ridge attribution."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'axis_access_margin_20261017'))
from model import prior, predict


def coefficients(task, h):
    ns = task['ns']
    ss = task['ss']+h[:ns]@h[:ns].T
    qs = task['qs']+h[ns:]@h[:ns].T
    root = task['root']
    coef = root[:,None]*torch.linalg.solve(root[:,None]*ss*root[None,:]+8*torch.eye(ns,dtype=ss.dtype),
                                         root[:,None]*(task['desired']-task['prior_s']))
    return qs, coef


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    manifest = pd.read_csv(prior.old.MANIFEST,dtype={'mmsi':str})
    geometry = pd.read_csv(prior.OLD/'geometry.csv',dtype={'mmsi':str}).set_index('global_row')
    quality = dict(np.load(prior.HERE/'quality.npz'))
    keys = ['fold','seed','origin','draw','global_row']
    saved = pd.read_csv(HERE/'paired_predictions.csv', low_memory=False)
    references = saved.set_index(['arm']+keys)
    changed = saved[saved.arm.eq('additive')&saved.origin.eq('cross')&saved.change.ne(0)]
    swaps, invalids, relations, edges, errors, swap_predictions = [], [], [], [], [], []
    with threadpool_limits(limits=1):
        for args in prior.old.folds():
            data, _, _, _, tasks, _, _ = prior.prepare_fold(args,manifest,quality)
            fold = data['fold']
            for seed in range(3):
                old = np.load(HERE/fold/f'margin_{seed}'/'readouts.npz')
                new = np.load(HERE/fold/f'additive_{seed}'/'readouts.npz')
                np.testing.assert_array_equal(old['global_rows'],data['rows'])
                np.testing.assert_array_equal(new['global_rows'],data['rows'])
                a0, a1, common = [torch.tensor(x) for x in [old['axis_raw'],new['axis_raw'],new['common_raw']]]
                av, cv = torch.tensor(new['axis_valid']), torch.tensor(new['common_valid'])
                for task in tasks:
                    ids, ns, q = task['ids'],task['ns'],task['q']
                    old_p, old_h, _ = predict(a0[ids],av[ids],task)
                    axis_p, axis_h, _ = predict(a1[ids],av[ids],task)
                    new_p, new_h, _ = predict(a1[ids],av[ids],task,common[ids],cv[ids],axis_weight=1.)
                    swap_p, _, _ = predict(a0[ids],av[ids],task,common[ids],cv[ids],axis_weight=1.)
                    truth = data['labels'][q]
                    scores = prior.old.scores(truth,swap_p.numpy())
                    scores['macro_brier'] = float(prior.old.risk(swap_p,torch.tensor(truth)))
                    swaps.append(dict(fold=fold,seed=seed,origin=task['origin'],draw=task['draw'],**scores))
                    task_keys = pd.MultiIndex.from_tuples([(fold,seed,task['origin'],task['draw'],int(data['rows'][r])) for r in q],names=keys)
                    for arm, pp in [('margin',old_p),('additive_axis_only',axis_p),('additive',new_p)]:
                        ref = references.loc[arm].loc[task_keys,[f'p{k}' for k in range(8)]].to_numpy(float)
                        error = float(abs(pp.numpy()-ref).max())
                        assert error < 1e-10
                        errors.append(error)
                    for j,r in enumerate(q):
                        swap_predictions.append(dict(fold=fold,seed=seed,origin=task['origin'],draw=task['draw'],global_row=int(data['rows'][r]),
                            truth=int(truth[j]),prediction=int(swap_p[j].argmax()),reference_prediction=int(old_p[j].argmax()),
                            **{f'p{k}':float(swap_p[j,k]) for k in range(8)}))
                    k0, c0 = coefficients(task,old_h)
                    ka, ca = coefficients(task,axis_h)
                    k1, c1 = coefficients(task,new_h)
                    split = a1.shape[1]
                    common_qs = new_h[ns:,split:]@new_h[:ns,split:].T
                    direct_common = common_qs@c1
                    selected = changed[(changed.fold==fold)&(changed.seed==seed)&(changed.draw==task['draw'])]
                    for j,r in enumerate(q):
                        gr, y = int(data['rows'][r]), int(truth[j])
                        rival = int(old_p[j].masked_fill(torch.arange(8)==y,-torch.inf).argmax())
                        if not bool(av[r]):
                            gap = float(old_p[j,rival]-old_p[j,y])
                            new_margin = float(new_p[j,y]-new_p[j,rival])
                            invalids.append(dict(fold=fold,seed=seed,origin=task['origin'],draw=task['draw'],global_row=gr,
                                truth=y,same_class_support=int((data['labels'][task['s']]==y).sum()),
                                reference_correct=int(old_p[j].argmax()==y),prediction_correct=int(new_p[j].argmax()==y),
                                deficit=gap,margin_change=new_margin+gap,max_probability_change=float(abs(new_p[j]-old_p[j]).max()),
                                common_direct_margin=float(direct_common[j,y]-direct_common[j,rival]),
                                brier_gain=float((old_p[j]-torch.eye(8)[y]).square().sum()-(new_p[j]-torch.eye(8)[y]).square().sum())))
                        if task['origin']!='cross' or gr not in set(selected.global_row):
                            continue
                        v0, va, v1 = c0[:,y]-c0[:,rival], ca[:,y]-ca[:,rival], c1[:,y]-c1[:,rival]
                        kernel_term = .5*(k1[j]-k0[j])*(v1+v0)
                        refit_term = .5*(k1[j]+k0[j])*(v1-v0)
                        expected = k1[j]@v1-k0[j]@v0
                        torch.testing.assert_close(kernel_term.sum()+refit_term.sum(),expected,atol=1e-12,rtol=0)
                        relation = dict(fold=fold,seed=seed,draw=task['draw'],global_row=gr,truth=y,rival=rival,
                            same_class_support=int((data['labels'][task['s']]==y).sum()),
                            old_margin=float(old_p[j,y]-old_p[j,rival]),
                            axis_only_margin=float(axis_p[j,y]-axis_p[j,rival]),
                            additive_margin=float(new_p[j,y]-new_p[j,rival]),
                            frozen_old_axis_margin=float(swap_p[j,y]-swap_p[j,rival]),
                            expert_margin_change=float(expected), kernel_term=float(kernel_term.sum()),refit_term=float(refit_term.sum()),
                            common_direct_margin=float(direct_common[j,y]-direct_common[j,rival]))
                        relations.append(relation)
                        for si,s in enumerate(task['s']):
                            edges.append(dict(fold=fold,seed=seed,draw=task['draw'],query_row=gr,support_row=int(data['rows'][s]),
                                query_class=y,support_class=int(data['labels'][s]),rival=rival,
                                old_axis_kernel=float(k0[j,si]-task['qs'][j,si]),
                                new_axis_kernel=float(ka[j,si]-task['qs'][j,si]),
                                common_kernel=float(common_qs[j,si]),
                                old_coefficient=float(v0[si]),new_coefficient=float(v1[si]),
                                kernel_term=float(kernel_term[si]),refit_term=float(refit_term[si])))
    pd.DataFrame(swaps).to_csv(HERE/'frozen_old_axis_swap_metrics.csv',index=False)
    pd.DataFrame(swap_predictions).to_csv(HERE/'frozen_old_axis_swap_predictions.csv',index=False)
    invalid = pd.DataFrame(invalids)
    invalid.to_csv(HERE/'invalid_query_diagnostics.csv',index=False)
    pd.DataFrame(relations).to_csv(HERE/'changed_case_mechanism.csv',index=False)
    pd.DataFrame(edges).to_csv(HERE/'support_edge_changes.csv',index=False)
    (HERE/'diagnostic_replay.json').write_text(json.dumps(dict(probability_tasks=len(errors),max_error=max(errors)),indent=2),encoding='utf-8')
    # Native, unrotated VV/VH previews; selected cases explain outcomes, not prevalence.
    fig, axes = plt.subplots(5,2,figsize=(8,13))
    for i,gr in enumerate([786,719,745,1038,1088]):
        row = geometry.loc[gr]
        for j,pol in enumerate(['vv','vh']):
            with Image.open(row['chip_path_'+pol]) as im:
                native = np.array(im)
            axes[i,j].imshow(native,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            axes[i,j].set_title(f"{gr} | {row.class_name} | {pol.upper()} | {native.shape[1]}x{native.shape[0]}",fontsize=10)
            axes[i,j].axis('off')
    fig.suptitle('Native OSS1 chips: changed queries and same-class support 719',fontsize=12)
    fig.tight_layout()
    fig.savefig(HERE/'native_case_gallery.png',dpi=130)
    plt.close(fig)
    print(pd.DataFrame(swaps).groupby('origin')[['oa','ba','macro_brier','nll']].mean().to_string())
    print(pd.DataFrame(relations).to_string(index=False))
    wrong = invalid[invalid.origin.eq('cross')&invalid.reference_correct.eq(0)]
    print('INVALID WRONG',len(wrong),'GAP',wrong.deficit.median(),'MARGIN CHANGE',wrong.margin_change.median(),
          'HELP FRACTION',wrong.common_direct_margin.gt(0).mean())


if __name__=='__main__':
    main()
