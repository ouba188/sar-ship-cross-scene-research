"""Descriptive class/acquisition analysis and native review after selection."""
import json
import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_joint as run
from run_joint import HERE,OLD,parent,strong
import torch
sys.path.insert(0,str(parent.HERE))
import analyze_results as common


def render(p, branches):
    table=parent.load_table().set_index('instance_id')
    snapshots={s['obs_id']:s for s in json.loads((OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    (HERE/'panels').mkdir(exist_ok=True)
    cases,missing=[],[]
    for arm,g in p[p.seed.eq(17)&p.role.eq('test')].groupby('arm'):
        choices=[]
        for outcome in ['rescued','harmed','CC','WW']:
            h=g[g.outcome.eq(outcome)].sort_values('instance_id')
            if len(h):choices.append(h.iloc[0])
            else:missing.append(dict(arm=arm,outcome=outcome))
        if len(choices)<4:
            for outcome in ['CC','WW']:
                choices.extend(q for _,q in g[g.outcome.eq(outcome)].sort_values('instance_id').iloc[1:2].iterrows())
        for q in choices:
            row=table.loc[q.instance_id]
            y,oldr,newr=int(q.class_id),int(q.old_rival),int(q.final_rival)
            b=q[[f'b{c}' for c in range(8)]].to_numpy(float)
            s=q[[f's{c}' for c in range(8)]].to_numpy(float)
            fig,axes=plt.subplots(2,3,figsize=(15,10),layout='constrained')
            dimensions=[]
            for j,pol in enumerate(['vv','vh']):
                view=snapshots[int(row.obs_id)]['images'][pol]
                with Image.open(row[f'image_{pol}']) as im:raw=np.array(im.convert('L'))
                pix,masks,place=common.crop_pad(raw,common.masks_for(view),160)
                dimensions.append(f'{pol}:{raw.shape[1]}x{raw.shape[0]}')
                axes[j,0].imshow(raw,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                axes[j,0].set_title(f'{pol.upper()} original {raw.shape[1]}x{raw.shape[0]} px')
                axes[j,1].imshow(pix,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                axes[j,1].set_title(f'160 crop/pad; valid={q[f"valid_{pol}"]}; footprint={masks.sum(0).mean():.3f}')
                prior=view['priors'];target=prior.get('target_index');det=prior.get('detections',[])
                if target is not None and 0<=target<len(det):
                    points=np.array(det[target]['corners']);points=np.r_[points,points[:1]]
                    axes[j,0].plot(points[:,0],points[:,1],color='#13c74e',lw=1)
                    points=points-[place['crop_x'],place['crop_y']]+[place['pad_left'],place['pad_top']]
                    axes[j,1].plot(points[:,0],points[:,1],color='#13c74e',lw=1)
                for ax in axes[j,:2]:ax.set_xlabel('pixel x');ax.set_ylabel('pixel y')
            xx=np.arange(8)
            axes[0,2].bar(xx-.18,b,width=.36,label='Frozen B = T(oldAD)',color='#64748b')
            axes[0,2].bar(xx+.18,s,width=.36,label='B + increment',color='#168b65')
            axes[0,2].set_xticks(xx);axes[0,2].set_title('Actual eight-class scores');axes[0,2].legend(fontsize=8)
            axes[1,2].bar(xx,s-b,color=['#168b65' if c==y else '#be5264' for c in xx])
            axes[1,2].axhline(0,color='black',lw=.6);axes[1,2].set_xticks(xx)
            axes[1,2].set_title('New direct residual only, not kappa contribution')
            mechanism=('Joint mean + distribution MLP, 273 parameters' if arm=='joint' else 'Separately learned mean + local branches, 266 parameters')
            contributions = {}
            closure_note = ''
            if arm == 'additive':
                br = branches.loc[(int(q.fold),17,'test',q.instance_id)]
                dm = br[[f'mean_delta{c}' for c in range(8)]].to_numpy(float)
                dl = br[[f'local_delta{c}' for c in range(8)]].to_numpy(float)
                roundoff = br[[f'roundoff{c}' for c in range(8)]].to_numpy(float)
                axes[1,2].clear()
                axes[1,2].bar(xx-.2,dm,width=.2,label='Mean branch',color='#1463a3')
                axes[1,2].bar(xx,dl,width=.2,label='Local branch',color='#c26919')
                axes[1,2].bar(xx+.2,s-b,width=.2,label='Full delta',color='#168b65')
                axes[1,2].set_xticks(xx);axes[1,2].axhline(0,color='black',lw=.6)
                axes[1,2].set_title('Centered branch outputs, no physical attribution')
                axes[1,2].legend(fontsize=8)
                mb,ml=dm[y]-dm[newr],dl[y]-dl[newr]
                err=roundoff[y]-roundoff[newr]
                closure_note=f'\nFinal-rival margin: {b[y]-b[newr]:+.5f} + mean {mb:+.5f} + local {ml:+.5f} = {s[y]-s[newr]:+.5f}; roundoff {err:+.1e}'
                contributions=dict(mean_true=float(dm[y]),mean_finalrival=float(dm[newr]),
                    local_true=float(dl[y]),local_finalrival=float(dl[newr]),
                    mean_margin_contribution=float(mb),local_margin_contribution=float(ml),
                    margin_roundoff=float(err),base_finalrival_margin=float(b[y]-b[newr]),
                    mean_oldrival_margin_contribution=float(dm[y]-dm[oldr]),
                    local_oldrival_margin_contribution=float(dl[y]-dl[oldr]))
            fig.suptitle(f'{arm} / oldseed0 -> seed17 / fold{int(q.fold)} / epoch{int(q.epoch)} / {q.outcome}\n'
                f'{q.instance_id} / {row.class_name} / {row.port}\n{mechanism}; frozen T epoch={int(q.T_epoch)}\n'
                f'true={y}; winner {int(q.base_prediction)} -> {int(q.final_winner)}; B rival={oldr}; final rival={newr}\n'
                f'true margin {b[y]-b[oldr]:+.5f} -> {s[y]-s[newr]:+.5f}; '
                f'fixed B-rival margin change {(s-b)[y]-(s-b)[oldr]:+.5f}'+closure_note,fontsize=10)
            filename = f'{arm}_{q.outcome}.png'
            path=HERE/'panels'/filename
            fig.savefig(path,dpi=100);plt.close(fig)
            cases.append(dict(arm=arm,seed=17,oldseed=0,fold=int(q.fold),epoch=int(q.epoch),T_epoch=int(q.T_epoch),
                outcome=q.outcome,instance_id=q.instance_id,obs_id=int(row.obs_id),class_id=y,class_name=row.class_name,port=row.port,
                old_winner=int(q.base_prediction),final_winner=int(q.final_winner),old_rival=oldr,final_rival=newr,
                old_margin=b[y]-b[oldr],final_margin=s[y]-s[newr],old_rival_delta=(s-b)[y]-(s-b)[oldr],
                image_vv=row.image_vv,image_vh=row.image_vh,dimensions=';'.join(dimensions),valid_vh=bool(q.valid_vh),
                panel=str(path),readout=mechanism,selection='test seed17 first instance_id per arm/outcome',**contributions))
    pd.DataFrame(cases).to_csv(HERE/'review_cases.csv',index=False)
    return cases,missing




def fixed_branch_replay(p):
    table=parent.load_table()
    cache=dict(np.load(parent.HERE/'signed_features.npz'))
    valid=cache['valid'].astype(bool)
    saved=p[p.arm.eq('additive')].set_index(['fold','seed','role','instance_id'])
    frames,checks=[],[]
    for fold in range(3):
        d,_,_=parent.load_fold(fold,table,cache)
        means,_,moments,_=run.load_features(fold,table,cache,d)
        for _,seed in strong.SEED_PAIRS:
            ck,model=run.restore(HERE/'models'/f'fold{fold}_additive_seed{seed}.pt')
            for role in ['validation','test']:
                ix=d[role]
                keys=pd.MultiIndex.from_arrays([[fold]*len(ix),[seed]*len(ix),[role]*len(ix),table.iloc[ix].instance_id])
                g=saved.loc[keys].reset_index()
                base=g[[f'b{c}' for c in range(8)]].to_numpy(float)
                full=g[[f's{c}' for c in range(8)]].to_numpy(float)
                x,_,_=run.inputs(base,means[ix],moments[ix],valid[ix],ck['mu'],ck['scale'])
                with torch.inference_mode():
                    raw_mean=model.mean(x[...,run.MEAN_COLUMNS]).squeeze(-1).numpy().astype(float)
                    raw_local=model.local(x[...,run.LOCAL_COLUMNS]).squeeze(-1).numpy().astype(float)
                    delta=model(x).numpy().astype(float)
                dm=raw_mean-raw_mean.mean(1,keepdims=True)
                dl=raw_local-raw_local.mean(1,keepdims=True)
                roundoff=delta-dm-dl
                error=float(np.abs(base+delta-full).max())
                closure=float(np.abs(roundoff).max())
                assert error<1e-6 and closure<2e-6
                np.testing.assert_array_equal((base+dm+dl).argmax(1),full.argmax(1))
                assert np.isfinite(dm).all() and np.isfinite(dl).all()
                checks.append(dict(fold=fold,seed=seed,role=role,n=len(ix),
                    saved_score_replay_error=error,independent_center_sum_error=closure,
                    decomposed_top1_changes=0,mean_center_error=float(np.abs(dm.sum(1)).max()),
                    local_center_error=float(np.abs(dl.sum(1)).max())))
                h=g[['fold','seed','role','instance_id','class_id','class_name','port','mmsi','pass_group','valid_vh']].copy()
                for c in range(8):
                    h[f'b{c}'],h[f's{c}']=base[:,c],full[:,c]
                    h[f'mean_delta{c}'],h[f'local_delta{c}'],h[f'roundoff{c}']=dm[:,c],dl[:,c],roundoff[:,c]
                h['full_prediction']=full.argmax(1)
                h['delete_local_prediction']=(base+dm).argmax(1)
                h['delete_mean_prediction']=(base+dl).argmax(1)
                h['base_prediction']=base.argmax(1)
                frames.append(h)
    branch=pd.concat(frames,ignore_index=True)
    branch.to_csv(HERE/'branch_predictions.csv',index=False)
    pd.DataFrame(checks).to_csv(HERE/'branch_checks.csv',index=False)
    metrics,classes,roles=[],[],[]
    for (role,seed),g in branch.groupby(['role','seed']):
        y=g.class_id.to_numpy()
        base=g[[f'b{c}' for c in range(8)]].to_numpy()
        full=g[[f's{c}' for c in range(8)]].to_numpy()
        fm=parent.assessment(y,full)
        base_correct=base.argmax(1)==y
        full_correct=full.argmax(1)==y
        for deleted,kept in [('local','mean'),('mean','local')]:
            score=base+g[[f'{kept}_delta{c}' for c in range(8)]].to_numpy()
            correct=score.argmax(1)==y
            dm=parent.assessment(y,score)
            meta=dict(role=role,seed=seed,deleted=deleted,kept=kept)
            def changes(mask):
                return dict(net_vs_full=int((correct[mask].astype(int)-full_correct[mask]).sum()),
                    deletion_rescued=int((mask & correct & ~full_correct).sum()),
                    deletion_harmed=int((mask & ~correct & full_correct).sum()),
                    net_vs_B=int((correct[mask].astype(int)-base_correct[mask]).sum()))
            metrics.append(dict(**meta,**dm,**changes(np.ones(len(g),bool)),
                **{f'delta_{k}_vs_full':dm[k]-fm[k] for k in ['OA','BA','NLL','Brier']}))
            for c in range(8):
                mask=y==c
                cm=parent.assessment(y[mask],score[mask])
                cf=parent.assessment(y[mask],full[mask])
                classes.append(dict(**meta,class_id=c,**cm,**changes(mask),
                    delta_OA_vs_full=cm['OA']-cf['OA']))
                # Role of the deleted branch when added back to the fixed kept branch.
                roles.append(dict(**meta,added_back=deleted,class_id=c,n=int(mask.sum()),
                    protected_B_correct=int((mask & base_correct & ~correct & full_correct).sum()),
                    rescued_B_wrong=int((mask & ~base_correct & ~correct & full_correct).sum()),
                    lost_B_correct=int((mask & base_correct & correct & ~full_correct).sum()),
                    lost_kept_branch_rescue=int((mask & ~base_correct & correct & ~full_correct).sum())))
    for name,rows in [('branch_deletion_metrics',metrics),('branch_deletion_classes',classes),
                      ('branch_roles',roles)]:
        pd.DataFrame(rows).to_csv(HERE/f'{name}.csv',index=False)
    print('BRANCH CHECKS',pd.DataFrame(checks).max(numeric_only=True).to_dict(),flush=True)
    print(pd.DataFrame(metrics)[['role','seed','deleted','OA','BA','net_vs_full',
        'deletion_rescued','deletion_harmed','net_vs_B']].to_string(index=False),flush=True)
    return branch


def main():
    torch.set_num_threads(2)
    p=pd.read_csv(HERE/'predictions.csv',dtype={'mmsi':str})
    refs=pd.concat([pd.read_csv(strong.HERE/'predictions.csv',dtype={'mmsi':str}),
        pd.read_csv(HERE.parent/'mean_only_control/predictions.csv',dtype={'mmsi':str}),p],ignore_index=True)
    strata,paired,ci,classes,unique_val=[],[],[],[],[]
    for (role,arm,seed),g in p.groupby(['role','arm','seed']):
        meta=dict(role=role,arm=arm,seed=seed)
        for unit in ['fold','class_id','port','valid_vh','pass_group','product']:
            for name,h in g.groupby(unit):
                strata.append(dict(**meta,unit=unit,name=name,**common.metrics(h)))
        am=common.metrics(g)
        paired.append(dict(**meta,reference='B',n=len(g),net=int(g.net.sum()),
            **{key:am[key] for key in ['delta_OA','delta_BA','delta_NLL','delta_Brier']}))
        if role=='validation':
            h=g.sort_values(['fold','instance_id']).drop_duplicates('instance_id')
            unique_val.append(dict(**meta,**common.metrics(h)))
        else:
            ci.extend(dict(**meta,**r) for r in common.intervals(g,g.net.to_numpy(),arm+'-B'))
        a=g.set_index(['fold','instance_id']).sort_index()
        for reference in ['mean_only','local','extra_score']+(['joint'] if arm=='additive' else []):
            h=refs[refs.role.eq(role)&refs.seed.eq(seed)&refs.arm.eq(reference)].set_index(['fold','instance_id']).loc[a.index]
            np.testing.assert_array_equal(a.class_id,h.class_id)
            np.testing.assert_array_equal(a[[f'b{c}' for c in range(8)]],h[[f'b{c}' for c in range(8)]])
            diff=a.net.to_numpy()-h.net.to_numpy()
            am,hm=common.metrics(a),common.metrics(h)
            paired.append(dict(**meta,reference=reference,n=len(a),net=int(diff.sum()),
                **{f'delta_{k}':am[k]-hm[k] for k in ['OA','BA','NLL','Brier']}))
            for c,sub in a.groupby('class_id'):
                hh=h.loc[sub.index]
                classes.append(dict(**meta,reference=reference,class_id=c,n=len(sub),
                    net=int(sub.net.sum()-hh.net.sum()),
                    delta_recall=100*float(sub.net.mean()-hh.net.mean())))
            if role=='test':
                ci.extend(dict(**meta,**r) for r in common.intervals(a,diff,arm+'-'+reference))
    for name,rows in [('stratified_metrics',strata),('paired_metrics',paired),
                      ('paired_intervals',ci),('paired_classes',classes),('validation_unique_metrics',unique_val)]:
        pd.DataFrame(rows).to_csv(HERE/f'{name}.csv',index=False)
    s=pd.DataFrame(strata)
    for unit,name in [('fold','fold_metrics'),('class_id','class_metrics'),('port','port_metrics'),
                      ('valid_vh','vh_metrics'),('product','acquisition_metrics')]:
        s[s.unit.eq(unit)].to_csv(HERE/f'{name}.csv',index=False)
    branches=fixed_branch_replay(p)
    cases,missing=render(p,branches.set_index(['fold','seed','role','instance_id']))
    (HERE/'analysis_summary.json').write_text(json.dumps(dict(panels=[r['panel'] for r in cases],
        missing_outcomes=missing,images_viewed=False,new_training_runs=0,
        fixed_deletions=['delete_mean','delete_local'],
        bootstrap_scope='Fixed predictions; pass/MMSI clusters, excludes fitting and selection uncertainty'),indent=2),encoding='utf-8')
    print(pd.DataFrame(paired).to_string(index=False),flush=True)
    print(pd.DataFrame(cases)[['arm','outcome','instance_id','class_id','old_winner','final_winner',
        'old_rival','final_rival','old_margin','final_margin']].to_string(index=False),flush=True)


if __name__=='__main__':main()
