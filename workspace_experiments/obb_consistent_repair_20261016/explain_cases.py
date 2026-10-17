"""Exact B score-path accounting and native-pixel rescue/harm illustrations."""
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from threadpoolctl import threadpool_limits
import run_consistent as run
from body_context_gate import _coefficients

HERE=run.HERE


def contexts(data,arm,seed):
    saved=torch.load(run.b_path(arm,seed),map_location='cuda',weights_only=False)
    model=run.geo.GeometryReader('B').cuda().eval()
    model.load_state_dict(saved['state'])
    return run.geo.contexts(model,data).double()


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    events=pd.read_csv(HERE/'analysis/changed_cases.csv',dtype={'mmsi':str})
    events=events.loc[events.model.eq('B') & events.comparison.eq('fixed-wrong')]
    old=run.load('wrong',True); new=run.load('fixed',True)
    records=[]; support_records=[]; errors=[]
    for seed in range(3):
        run.prepare(old,'wrong',seed); run.prepare(new,'fixed',seed)
        co,cn=contexts(old,'wrong',seed),contexts(new,'fixed',seed)
        for ti,group in events.loc[events.seed==seed].groupby('task'):
            a,b=old['cache']['target'][ti],new['cache']['target'][ti]
            torch.testing.assert_close(a['base_prob'],b['base_prob'],atol=0,rtol=0)
            torch.testing.assert_close(a['prior_q'],b['prior_q'],atol=0,rtol=0)
            torch.testing.assert_close(a['s'],b['s'],atol=0,rtol=0)
            s,q=a['s'],a['q']
            ao=_coefficients(a['ss'],a['prior_s'],a['sy'])
            an=_coefficients(b['ss'],b['prior_s'],b['sy'])
            ship_o,ship_n=old['tensors']['ship'],new['tensors']['ship']
            dk_ship=1.5*(ship_n[q]@ship_n[s].T-ship_o[q]@ship_o[s].T)
            dk_context=1.5*(cn[q]@cn[s].T-co[q]@co[s].T)
            torch.testing.assert_close(dk_ship+dk_context,b['qs']-a['qs'],atol=1e-11,rtol=1e-11)
            logits_o=a['prior_q']+a['qs']@ao
            logits_n=b['prior_q']+b['qs']@an
            mapping={int(g):i for i,g in enumerate(old['meta'].iloc[q.cpu().numpy()].global_row)}
            for r in group.itertuples():
                i=mapping[r.global_row]; truth=int(r.truth)
                competitor=int(r.pred_ref if r.rescue else r.pred)
                alpha=ao[:,truth]-ao[:,competitor]
                ship=dk_ship[i]*alpha
                context=dk_context[i]*alpha
                refit=b['qs'][i]*((an[:,truth]-an[:,competitor])-alpha)
                delta=float((logits_n[i,truth]-logits_n[i,competitor])-(logits_o[i,truth]-logits_o[i,competitor]))
                actual=float((ship+context+refit).sum())
                assert abs(actual-delta)<1e-10
                errors.append(abs(actual-delta))
                records.append(dict(seed=seed,task=ti,origin=r.origin,draw=r.draw,
                    acquisition=r.acquisition,global_row=r.global_row,mmsi=r.mmsi,
                    truth=truth,competitor=competitor,event='rescue' if r.rescue else 'harm',
                    ship_relation=float(ship.sum()),spatial_relation=float(context.sum()),
                    support_refit=float(refit.sum()),expert_margin_delta=delta,
                    old_full_margin=float(a['old_prob'][i,truth]-a['old_prob'][i,competitor]),
                    new_full_margin=float(b['old_prob'][i,truth]-b['old_prob'][i,competitor])))
                for j,row in enumerate(old['meta'].iloc[s.cpu().numpy()].itertuples()):
                    support_records.append(dict(seed=seed,task=ti,query_global_row=r.global_row,
                        support_global_row=row.global_row,support_class=row.class_id,
                        ship_relation=float(ship[j]),spatial_relation=float(context[j]),
                        support_refit=float(refit[j]),total=float(ship[j]+context[j]+refit[j])))
    table=pd.DataFrame(records); supports=pd.DataFrame(support_records)
    table.to_csv(HERE/'analysis/B_score_paths.csv',index=False)
    supports.to_csv(HERE/'analysis/B_support_paths.csv',index=False)
    run.dump(HERE/'analysis/SCORE_PATH_CHECK.json',dict(events=len(table),max_error=max(errors,default=0),
        scope='Expert true-minus-fixed-competitor logit, not fused probability or physical causal effect',
        order='Change query-support kernel with old coefficients, then recompute support coefficients'))
    plot_cases(table,supports)
    if not table.empty:
        print(table.groupby(['origin','event'])[['ship_relation','spatial_relation','support_refit','expert_margin_delta']].mean().to_string())


def plot_cases(table,supports):
    if table.empty:
        return
    # First changed query of each outcome, seed 0/cross, not largest gains.
    candidates=table.loc[table.seed.eq(0)&table.origin.eq('cross')].sort_values(['global_row','draw'])
    selected=candidates.groupby('event',sort=True).head(1)
    if selected.empty:
        return
    selected.to_csv(HERE/'analysis/visual_case_selection.csv',index=False)
    meta=pd.read_csv(HERE/'inventory.csv').set_index('global_row')
    det={(r['global_row'],r['pol']):r for r in map(json.loads,(HERE/'detections.jsonl').read_text().splitlines())}
    fig,axes=plt.subplots(len(selected),4,figsize=(13,4.3*len(selected)),squeeze=False,layout='constrained')
    for k,r in enumerate(selected.itertuples()):
        parts=supports.loc[(supports.seed==r.seed)&(supports.task==r.task)&(supports.query_global_row==r.global_row)]
        chosen=parts.assign(strength=parts.total.abs()).sort_values('strength',ascending=False).iloc[0]
        for j,(grow,pol,role) in enumerate([(r.global_row,'vv','Query VV'),(r.global_row,'vh','Query VH'),
                      (int(chosen.support_global_row),'vv','Support VV'),(int(chosen.support_global_row),'vh','Support VH')]):
            ax=axes[k,j]; m=meta.loc[grow]; d=det[(grow,pol)]
            with Image.open(m['image_relpath_'+pol]) as im:
                ax.imshow(np.array(im.convert('L')),cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            for arm,color,style in [('wrong','orange','--'),('fixed','lime','-')]:
                x0,y0,x1,y1=d[arm+'_box']
                ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=color,linestyle=style,linewidth=1,label=arm))
            ax.set_title(f'{r.event} | {role} | row {grow}\nclass {m.class_id}; draw {r.draw}',fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        axes[k,0].set_ylabel(f'Fused margin\n{r.old_full_margin:.4f} -> {r.new_full_margin:.4f}',fontsize=9)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Same raw OBB: wrong / fixed adapter; support with largest absolute score-path change',fontsize=10)
    fig.savefig(HERE/'analysis/changed_native_examples.png',dpi=150)
    plt.close(fig)
    print(selected.to_string(index=False))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
