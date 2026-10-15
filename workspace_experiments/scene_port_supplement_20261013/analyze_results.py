"""Paired outcomes, support benefit, class drivers and bounded solver verification."""
import json
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import OUT, OLD, frame, CELLS, MODES, write


def main():
    plans=json.loads((OUT/'ENROLLMENT.json').read_text(encoding='utf-8'))
    d=frame(); yall=d.class_id.to_numpy()
    records=[]; class_rows=[]; transitions=[]; coverage=[]
    checked=0
    for e in plans:
        z=np.load(OUT/'predictions'/f"episode_{e['episode']:03d}.npz")
        q=z['query_rows']; y=yall[q]; classes=np.unique(y)
        p=z['probabilities']; pri=z['source_only']
        assert np.isfinite(p).all() and np.isfinite(pri).all()
        np.testing.assert_allclose(p.sum(-1),1,atol=1e-12)
        np.testing.assert_allclose(pri.sum(-1),1,atol=1e-12)
        base={k:e[k] for k in ['episode','budget','policy','query_port','support_port','acquisition']}
        coverage.append(dict(**base,n=len(q),known_ships=d.iloc[q].mmsi.nunique(),classes=','.join(map(str,classes)),
                             angle_known=int(d.iloc[q].angle.notna().sum())))
        cells=[(-1,'source_only',pri)]+[(draw,cell,p[draw,ci]) for draw in range(5) for ci,cell in enumerate(CELLS)]
        for draw,cell,prob in cells:
            for mi,mode in enumerate(MODES):
                pred=prob[mi].argmax(-1)
                oa=100*float(np.mean(pred==y))
                cr=np.array([np.mean(pred[y==c]==c) for c in classes])
                ba=100*float(cr.mean())
                np.testing.assert_allclose([oa,ba],[100*accuracy_score(y,pred),
                    100*recall_score(y,pred,labels=classes,average='macro',zero_division=0)],atol=1e-10)
                checked+=1
                records.append(dict(**base,draw=draw,cell=cell,mode=mode,n=len(q),classes=len(classes),oa=oa,ba=ba))
                for c,r in zip(classes,cr):
                    class_rows.append(dict(**base,draw=draw,cell=cell,mode=mode,class_id=int(c),
                        n=int((y==c).sum()),recall=100*r,ba_contribution=100*r/len(classes)))
        for draw in range(5):
            for mi,mode in enumerate(MODES):
                for before,after,a,b in [('SS','OS',p[draw,0,mi],p[draw,1,mi]),
                        ('SS','SO',p[draw,0,mi],p[draw,2,mi]),
                        ('source_only','OO',pri[mi],p[draw,3,mi])]:
                    ca=a.argmax(-1)==y; cb=b.argmax(-1)==y
                    for c in range(8):
                        m=y==c
                        transitions.append(dict(**base,draw=draw,mode=mode,contrast=after+'-'+before,class_id=int(c),
                            n=int(m.sum()),rescued=int((~ca&cb&m).sum()),lost=int((ca&~cb&m).sum()),
                            ba_delta_contribution=100*float(np.mean(cb[m])-np.mean(ca[m]))/len(classes) if m.any() else 0.))
    df=pd.DataFrame(records); df.to_csv(OUT/'metrics.csv',index=False)
    cd=pd.DataFrame(class_rows);cd.to_csv(OUT/'classes.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    tr=pd.DataFrame(transitions);tr.to_csv(OUT/'transitions.csv',index=False)
    ix=['episode','budget','policy','query_port','support_port','acquisition','draw','mode']
    tabs={cell:df[df.cell==cell].set_index(ix)[['oa','ba']] for cell in CELLS}
    effects=[]
    for name,val in [('port_same_acq',tabs['OS']-tabs['SS']),('acq_same_port',tabs['SO']-tabs['SS']),
                     ('both',tabs['OO']-tabs['SS']),('port_other_acq',tabs['OO']-tabs['SO']),
                     ('interaction',tabs['OO']-tabs['OS']-tabs['SO']+tabs['SS'])]:
        effects.append(val.rename(columns={'oa':'delta_oa','ba':'delta_ba'}).reset_index().assign(contrast=name))
    mergekeys=[k for k in ix if k!='draw']
    pri=df[df.cell=='source_only'][mergekeys+['oa','ba']]
    for cell in CELLS:
        sub=df[df.cell==cell].merge(pri,on=mergekeys,suffixes=('','_prior'),validate='many_to_one')
        ef=sub[ix].copy();ef['delta_oa']=sub.oa-sub.oa_prior;ef['delta_ba']=sub.ba-sub.ba_prior
        effects.append(ef.assign(contrast=cell+'_minus_source'))
    ef=pd.concat(effects,ignore_index=True);ef.to_csv(OUT/'effects.csv',index=False)
    group=['policy','budget','query_port','support_port','mode','contrast']
    dirs=ef.groupby(group)[['delta_oa','delta_ba']].mean().reset_index()
    dirs.to_csv(OUT/'direction_effects.csv',index=False)
    summary=dirs.groupby(['policy','budget','mode','contrast']).agg(directions=('query_port','size'),
        delta_oa=('delta_oa','mean'),delta_ba=('delta_ba','mean'),negative_ba_directions=('delta_ba',lambda a:int((a<0).sum()))).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    drawdir=ef.groupby(group+['draw'])[['delta_oa','delta_ba']].mean().reset_index()
    drawmean=drawdir.groupby(['policy','budget','mode','contrast','draw'])[['delta_oa','delta_ba']].mean().reset_index()
    drawmean.to_csv(OUT/'draw_summary.csv',index=False)
    drawmean.groupby(['policy','budget','mode','contrast']).agg(
        mean_delta_oa=('delta_oa','mean'),sd_support_draw_oa=('delta_oa','std'),
        mean_delta_ba=('delta_ba','mean'),sd_support_draw_ba=('delta_ba','std'),
        positive_support_draws_ba=('delta_ba',lambda v:int((v>0).sum()))).to_csv(OUT/'support_draw_variability.csv')
    absolute=df.groupby(['policy','budget','query_port','support_port','mode','cell'])[['oa','ba']].mean().groupby(
        ['policy','budget','mode','cell']).mean().reset_index()
    absolute.to_csv(OUT/'absolute_summary.csv',index=False)
    branchix=['episode','budget','policy','query_port','support_port','acquisition','draw','cell']
    branch=(df[df['mode']=='dual_late'].set_index(branchix)[['oa','ba']]-df[df['mode']=='r50_late'].set_index(branchix)[['oa','ba']]).reset_index()
    branch=branch.groupby(['policy','budget','query_port','support_port','cell'])[['oa','ba']].mean().reset_index()
    branch.to_csv(OUT/'branch_direction_gains.csv',index=False)
    branch.groupby(['policy','budget','cell'])[['oa','ba']].mean().to_csv(OUT/'branch_summary.csv')
    classdirs=tr.groupby(['policy','budget','query_port','support_port','mode','contrast','class_id']).agg(
        rescued=('rescued','mean'),lost=('lost','mean'),ba_delta_contribution=('ba_delta_contribution','mean')).reset_index()
    classdirs.to_csv(OUT/'class_direction_effects.csv',index=False)
    classdirs.groupby(['policy','budget','mode','contrast','class_id'])[['rescued','lost','ba_delta_contribution']].mean().to_csv(OUT/'class_summary.csv')
    # Budget comparisons only pair the same query context, never different cohorts.
    budgetix=['policy','query_port','support_port','acquisition','draw','mode','cell']
    paired=df[df.budget==64].merge(df[df.budget==16],on=budgetix,suffixes=('_64','_16'),validate='one_to_one')
    querymap={ (e['policy'],e['query_port'],e['support_port'],e['acquisition'],e['budget']):e['query'] for e in plans }
    for key in {(e['policy'],e['query_port'],e['support_port'],e['acquisition']) for e in plans if e['budget']==64}:
        assert querymap[key+(64,)]==querymap[key+(16,)]
    paired['delta_oa']=paired.oa_64-paired.oa_16;paired['delta_ba']=paired.ba_64-paired.ba_16
    paired.to_csv(OUT/'paired_budget_effects.csv',index=False)
    bg=paired.groupby(['policy','query_port','support_port','mode','cell'])[['delta_oa','delta_ba']].mean().groupby(
        ['policy','mode','cell']).mean().reset_index()
    bg.to_csv(OUT/'paired_budget_summary.csv',index=False)
    # Check the numerical reader independently for one episode per policy and budget.
    f=np.load(OUT/'encoded.npz'); rows=f['rows']; solver_checks=0;maxerror=0.
    for policy in ['class_matched','class_angle_matched']:
        for budget in [16,64]:
            e=next(e for e in plans if e['policy']==policy and e['budget']==budget)
            z=np.load(OUT/'predictions'/f"episode_{e['episode']:03d}.npz")
            q=np.searchsorted(rows,z['query_rows'])
            for ci in range(4):
                sr=z['support_rows'][0,ci];s=np.searchsorted(rows,sr);sy=yall[sr]
                w=compute_sample_weight('balanced',sy);target=2*np.eye(8)[sy]-1
                outs=[]
                for phi,pk in [(f['b'],'pb'),(np.c_[f['r'],f['t']],'pd')]:
                    fitted=KernelRidge(alpha=8.,kernel='precomputed').fit(phi[s]@phi[s].T+1,target-f[pk][s],sample_weight=w)
                    outs.append(softmax(f[pk][q]+fitted.predict(phi[q]@phi[s].T+1),axis=1))
                actual=.5*(outs[0]+outs[1]); expected=z['probabilities'][0,ci,2]
                maxerror=max(maxerror,float(np.max(np.abs(actual-expected))))
                np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-9);solver_checks+=1
    # Reused geometry must remain consistent with previously saved encoded values.
    old=np.load(OLD/'encoded.npz');common=np.intersect1d(rows,old['rows']);newloc=np.searchsorted(rows,common);oldloc=np.searchsorted(old['rows'],common)
    encode_error={}
    for k in ['r','t','b','pr','pd','pb']:
        encode_error[k]=float(np.max(np.abs(f[k][newloc]-old[k][oldloc])))
        np.testing.assert_allclose(f[k][newloc],old[k][oldloc],atol=2e-6,rtol=1e-5)
    write('VERIFIED.json',dict(independent_metric_rows=checked,independent_solver_cases=solver_checks,
        solver_max_probability_error=maxerror,old_encoded_replay_rows=len(common),old_encoded_max_errors=encode_error))
    render(summary,dirs)
    print(absolute[absolute['mode']=='dual_late'].to_string(index=False))
    print(summary[(summary['mode']=='dual_late')&summary.contrast.isin(['port_same_acq','acq_same_port','OO_minus_source'])].to_string(index=False))
    print('PAIRED BUDGET\n',bg[bg['mode']=='dual_late'].to_string(index=False))
    print('VERIFIED',checked,solver_checks,maxerror)


def render(summary,dirs):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    names={'port_same_acq':'Port change\nsame acquisition','acq_same_port':'Acquisition change\nsame port','both':'Both changes'}
    for ax,policy in zip(axes,['class_matched','class_angle_matched']):
        for i,budget in enumerate([16,64]):
            s=summary[(summary.policy==policy)&(summary.budget==budget)&(summary['mode']=='dual_late')].set_index('contrast')
            ax.bar(np.arange(3)+(i-.5)*.34,[s.loc[c,'delta_ba'] for c in names],width=.34,label=f'B={budget}',color=['#257f83','#bb5266'][i])
        ax.set_xticks(range(3),list(names.values()));ax.axhline(0,color='#555',lw=.8)
        ax.set_ylabel('BA change relative to same-port/same-acquisition (pp)')
        ax.set_title('Matched class counts' if policy=='class_matched' else 'Matched class + incidence bins\nNingbo-Shanghai only')
        ax.legend();ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(OUT/'transfer_effects.png',dpi=170);fig.savefig(OUT/'transfer_effects.svg');plt.close(fig)


if __name__=='__main__':main()
