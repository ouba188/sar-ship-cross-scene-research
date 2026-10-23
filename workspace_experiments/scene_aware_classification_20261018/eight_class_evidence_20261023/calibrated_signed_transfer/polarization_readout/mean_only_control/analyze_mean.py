"""Descriptive class/acquisition analysis and native review after selection."""
import json
import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_mean import HERE,OLD,parent,strong
sys.path.insert(0,str(parent.HERE))
import analyze_results as common


def render(p):
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
            mechanism='Auxiliary W GAP means repeated in two slots + B -> shared direct residual'
            fig.suptitle(f'{arm} / oldseed0 -> seed17 / fold{int(q.fold)} / epoch{int(q.epoch)} / {q.outcome}\n'
                f'{q.instance_id} / {row.class_name} / {row.port}\n{mechanism}; frozen T epoch={int(q.T_epoch)}\n'
                f'true={y}; winner {int(q.base_prediction)} -> {int(q.final_winner)}; B rival={oldr}; final rival={newr}\n'
                f'true margin {b[y]-b[oldr]:+.5f} -> {s[y]-s[newr]:+.5f}; '
                f'fixed B-rival margin change {(s-b)[y]-(s-b)[oldr]:+.5f}',fontsize=10)
            filename = f'mean_{q.outcome}.png'
            path=HERE/'panels'/filename
            fig.savefig(path,dpi=100);plt.close(fig)
            cases.append(dict(arm=arm,seed=17,oldseed=0,fold=int(q.fold),epoch=int(q.epoch),T_epoch=int(q.T_epoch),
                outcome=q.outcome,instance_id=q.instance_id,obs_id=int(row.obs_id),class_id=y,class_name=row.class_name,port=row.port,
                old_winner=int(q.base_prediction),final_winner=int(q.final_winner),old_rival=oldr,final_rival=newr,
                old_margin=b[y]-b[oldr],final_margin=s[y]-s[newr],old_rival_delta=(s-b)[y]-(s-b)[oldr],
                image_vv=row.image_vv,image_vh=row.image_vh,dimensions=';'.join(dimensions),valid_vh=bool(q.valid_vh),
                panel=str(path),readout=mechanism,selection='test seed17 first instance_id per arm/outcome'))
    pd.DataFrame(cases).to_csv(HERE/'review_cases.csv',index=False)
    return cases,missing



def main():
    p=pd.read_csv(HERE/'predictions.csv',dtype={'mmsi':str})
    existing=pd.read_csv(strong.HERE/'predictions.csv',dtype={'mmsi':str})
    strata,paired,ci,classes,unique_val=[],[],[],[],[]
    for (role,arm,seed),g in p.groupby(['role','arm','seed']):
        meta=dict(role=role,arm=arm,seed=seed,oldseed=int(g.oldseed.iloc[0]))
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
            ci.extend(dict(**meta,**r) for r in common.intervals(g,g.net.to_numpy(),'mean_only-B'))
        key=['fold','instance_id']
        a=g.set_index(key).sort_index()
        for reference in ['local','extra_score']:
            h=existing[existing.role.eq(role)&existing.seed.eq(seed)&existing.arm.eq(reference)].set_index(key).loc[a.index]
            np.testing.assert_array_equal(a.class_id,h.class_id)
            np.testing.assert_array_equal(a[[f'b{c}' for c in range(8)]],h[[f'b{c}' for c in range(8)]])
            diff=a.net.to_numpy()-h.net.to_numpy()
            am,hm=common.metrics(a),common.metrics(h)
            paired.append(dict(**meta,reference=reference,n=len(a),net=int(diff.sum()),
                delta_OA=am['OA']-hm['OA'],delta_BA=am['BA']-hm['BA'],
                delta_NLL=am['NLL']-hm['NLL'],delta_Brier=am['Brier']-hm['Brier']))
            for c,sub in a.groupby('class_id'):
                hh=h.loc[sub.index]
                classes.append(dict(**meta,reference=reference,class_id=c,n=len(sub),
                    net=int(sub.net.sum()-hh.net.sum()),
                    delta_recall=100*float(sub.net.mean()-hh.net.mean())))
            if role=='test':
                ci.extend(dict(**meta,**r) for r in common.intervals(a,diff,'mean_only-'+reference))
    for name,rows in [('stratified_metrics',strata),('paired_metrics',paired),
                      ('paired_intervals',ci),('paired_classes',classes),
                      ('validation_unique_metrics',unique_val)]:
        pd.DataFrame(rows).to_csv(HERE/f'{name}.csv',index=False)
    s=pd.DataFrame(strata)
    for unit,name in [('fold','fold_metrics'),('class_id','class_metrics'),
                      ('port','port_metrics'),('valid_vh','vh_metrics'),
                      ('product','acquisition_metrics')]:
        s[s.unit.eq(unit)].to_csv(HERE/f'{name}.csv',index=False)
    cases,missing=render(p)
    table=parent.load_table().set_index('instance_id')
    with np.load(parent.HERE/'signed_features.npz') as z:
        names=list(z['reader_names'])
        lookup={v:i for i,v in enumerate(z['instance_ids'])}
        means=z['mean_pol']
    values=[]
    for case in cases:
        mu=means[names.index(f"fold{case['fold']}_outer"),lookup[case['instance_id']]]
        for pol in range(2):
            for c in range(8):
                values.append(dict(instance_id=case['instance_id'],outcome=case['outcome'],
                    pol=['vv','vh'][pol],class_id=c,slot0_mean=mu[pol,c],slot1_mean=mu[pol,c]))
    pd.DataFrame(values).to_csv(HERE/'review_inputs.csv',index=False)
    (HERE/'analysis_summary.json').write_text(json.dumps(dict(
        panels=[r['panel'] for r in cases],missing_outcomes=missing,images_viewed=False,
        seeds_pooled_as_independent=False,
        bootstrap_scope='Fixed predictions; pass/MMSI clusters; excludes fitting and selection uncertainty'),
        indent=2),encoding='utf-8')
    print(pd.DataFrame(paired).to_string(index=False),flush=True)
    print(pd.DataFrame(cases)[['outcome','instance_id','class_id','old_winner','final_winner',
        'old_rival','final_rival','old_margin','final_margin']].to_string(index=False),flush=True)


if __name__=='__main__':main()
