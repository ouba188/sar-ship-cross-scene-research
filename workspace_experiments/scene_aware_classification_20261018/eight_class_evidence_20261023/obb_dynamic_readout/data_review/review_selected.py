"""Fixed selected-head replay, four predeclared cases and region-summary removal."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'2')
import sys
sys.dont_write_bytecode=True
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import run_heads as runner
import common as c
SCORE,BASE=runner.SCORE,runner.BASE


def pick_cases(pred):
    d=pred[pred.arm.eq('dynamic')&pred.seed.eq(17)].copy()
    f=pred[pred.arm.eq('fixed')&pred.seed.eq(17)].copy()
    keys=['fold','instance_id']
    assert not d.duplicated(keys).any() and not f.duplicated(keys).any()
    ref=f.set_index(keys).loc[pd.MultiIndex.from_frame(d[keys])]
    np.testing.assert_array_equal(d.class_id.to_numpy(),ref.class_id.to_numpy())
    d['fixed_prediction']=ref.prediction.to_numpy();d['fixed_epoch']=ref.epoch.to_numpy()
    dc=d.prediction.eq(d.class_id);fc=d.fixed_prediction.eq(d.class_id)
    rows=[]
    for label,mask in [('rescued',dc&~fc),('harmed',~dc&fc),('CC',dc&fc),('WW',~dc&~fc)]:
        group=d[mask].sort_values(['obs_id','instance_id','fold'])
        if len(group)==0:
            rows.append(dict(outcome=label,available=False,group_n=0,active_n=0));continue
        active=group[group.epoch.gt(0)&(group.fixed_epoch.gt(0) if label in ['rescued','harmed'] else True)]
        q=(active if len(active) else group).iloc[0]
        rows.append(dict(outcome=label,comparison='fixed',available=True,group_n=len(group),active_n=len(active),
            allgroup_first_instance=group.iloc[0].instance_id,allgroup_first_obs=int(group.iloc[0].obs_id),
            instance_id=q.instance_id,obs_id=int(q.obs_id),fold=int(q.fold),dynamic_epoch=int(q.epoch),
            fixed_epoch=int(q.fixed_epoch),fallback=not bool(len(active)),
            selection='active eligible first obs_id' if len(active) else 'no active eligible: allgroup first obs_id'))
    bc=d.base_prediction.eq(d.class_id)
    for label,mask,expected_obs in [('B_rescued',dc&~bc,966),('B_harmed',~dc&bc,2217)]:
        group=d[mask].sort_values(['obs_id','instance_id','fold'])
        q=group.iloc[0]
        assert int(q.obs_id)==expected_obs
        rows.append(dict(outcome=label,comparison='baseline',available=True,group_n=len(group),
            active_n=int(group.epoch.gt(0).sum()),allgroup_first_instance=q.instance_id,
            allgroup_first_obs=int(q.obs_id),instance_id=q.instance_id,obs_id=int(q.obs_id),fold=int(q.fold),
            dynamic_epoch=int(q.epoch),fixed_epoch=int(q.fixed_epoch),fallback=False,
            selection='authorized baseline addendum: allgroup first obs_id'))
    return rows


def one_info(info,loc):
    return {key:value[loc:loc+1].copy() for key,value in info.items()}


def replay(base,info,arm,ck,head):
    raw=runner.raw_inputs(base,info,arm)
    x,_,_=runner.standardize(raw,ck['mu'],ck['scale'])
    with torch.no_grad():return base+head(x).numpy().astype(float)


def render(row,case,base,scores,ablation,template,table,snapshots,valid):
    fig,axes=plt.subplots(2,4,figsize=(19,10),layout='constrained')
    colors=['#38db68','#efae24','#28c4f0']
    for p,pol in enumerate(c.POL):
        canvas,fp,g,place=c.native(table,snapshots,int(row.name),p)
        with Image.open(row['image_'+pol]) as im:raw=np.asarray(im.convert('L'))
        axes[p,0].imshow(raw,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
        if g is not None:
            u=np.array([np.cos(g[4]),np.sin(g[4])])*g[2]/2
            v=np.array([-np.sin(g[4]),np.cos(g[4])])*g[3]/2
            center=g[:2]-[place['pad_left']-place['crop_x'],place['pad_top']-place['crop_y']]
            corners=np.array([center-u-v,center+u-v,center+u+v,center-u+v])
            axes[p,0].add_patch(Polygon(corners,fill=False,edgecolor=colors[0],lw=1))
        axes[p,0].set_title(f'{pol.upper()} original {raw.shape[1]}x{raw.shape[0]}; valid={bool(valid[int(row.name),p])}')
        for col,arm in [(1,'dynamic'),(2,'fixed')]:
            lw=None if arm=='dynamic' else template.loc[(f'fold{case["fold"]}_outer',pol),['length','width']].to_numpy(float)
            mask=c.soft_regions(g,lw)*fp
            q=c.region_quality(c.soft_regions(g,lw),fp,g,lw)
            axes[p,col].imshow(canvas,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            for r in range(3):
                rgba=np.zeros((160,160,4));rgba[...,:3]=matplotlib.colors.to_rgb(colors[r]);rgba[...,3]=mask[r]*.5
                axes[p,col].imshow(rgba,interpolation='nearest')
            ls=g[2:4] if lw is None else lw
            axes[p,col].set_title(f'{pol.upper()} {arm}: L/W={ls[0]:.1f}/{ls[1]:.1f}\n'
                f'min n_eff='+','.join(f'{x:.1f}' for x in q['min_neff'])+'; '+
                'coverage='+','.join(f'{x:.2f}' for x in q['coverage']),fontsize=9)
        for ax in axes[p,:3]:ax.set_xlabel('native pixels' if ax is axes[p,0] else 'native160 pixels')
    for name,s in [('B',base),('fixed',scores['fixed']),('dynamic',scores['dynamic'])]:
        axes[0,3].plot(range(8),s,marker='o',lw=1,label=name)
    axes[0,3].set_xticks(range(8));axes[0,3].legend(fontsize=8)
    axes[0,3].set_title(f'Actual scores; y={case["true_class"]}, fixed comparison={case["fixed_rival"]}\n'
        f'dynamic own top rival={case["dynamic_top_rival"]}')
    labels=[];values=[];barcolors=[]
    for arm,color in [('dynamic','#238369'),('fixed','#b98236')]:
        for drop in ['none','body','end','side']:
            a=next(x for x in ablation if x['arm']==arm and x['removed']==drop)
            labels.append(('D' if arm=='dynamic' else 'F')+(' full' if drop=='none' else ' -'+drop)+f'\npred{a["prediction"]}')
            values.append(a['fixed_margin']);barcolors.append(color)
    axes[1,3].bar(labels,values,color=barcolors)
    axes[1,3].axhline(0,color='black',lw=.7);axes[1,3].tick_params(axis='x',labelrotation=60)
    axes[1,3].set_title('Fixed rival margin, not necessarily top1 margin\nRaw summary zero; B / quality / valid / head unchanged')
    fig.suptitle(f'TEST seed17 / {case["outcome"]} dynamic vs {case.get("comparison","fixed")} / obs{case["obs_id"]} / {row.instance_id}\n'
        f'{row.class_name} / {row.port} / fold{case["fold"]}; selected epochs D/F={case["dynamic_epoch"]}/{case["fixed_epoch"]}; '
        f'pred B/D/F={case["base_prediction"]}/{case["dynamic_prediction"]}/{case["fixed_prediction"]}\n'
        f'{case["selection"]}; green body / amber end / cyan side; RF35, not isolated pixel evidence',fontsize=11)
    path=HERE/f'test_{case["outcome"]}.png';fig.savefig(path,dpi=110);plt.close(fig)
    return str(path)


def main():
    tick=time.perf_counter();torch.set_num_threads(2)
    assert (HERE/'SELECTED_CASE_PROTOCOL_ZH.md').exists()
    table,valid,snapshots=c.load_data()
    pred=pd.read_csv(c.HERE/'test_predictions.csv',dtype={'mmsi':str},float_precision='round_trip')
    selections=pick_cases(pred)
    pd.DataFrame(selections).to_csv(HERE/'active_case_selection.csv',index=False)
    indexed=pred.set_index(['arm','seed','fold','instance_id'])
    template=pd.read_csv(c.HERE/'source_templates.csv').set_index(['reader','pol'])
    infos={};heads={};cases=[];ablations=[];slots=[];checks=[]
    todo=selections
    if '--baseline-addendum' in sys.argv:
        todo=[case for case in selections if case.get('comparison')=='baseline']
        cases=pd.read_csv(HERE/'active_cases.csv').to_dict('records')
        for case in cases:case['comparison']='fixed'
        ablations=pd.read_csv(HERE/'region_ablation.csv').to_dict('records')
        slots=pd.read_csv(HERE/'case_region_slots.csv').to_dict('records')
        checks=pd.read_csv(HERE/'selected_replay_checks.csv').to_dict('records')
        assert not set(x['instance_id'] for x in cases)&set(x['instance_id'] for x in todo)
    for case in todo:
        if not case['available']:continue
        fold,identifier=case['fold'],case['instance_id']
        if fold not in infos:infos[fold]=runner.load_info(fold,'test',table)
        q=indexed.loc[('dynamic',17,fold,identifier)]
        idx=int(q.row_index);row=table.iloc[idx];y=int(row.class_id)
        loc=int(np.flatnonzero(infos[fold]['indices']==idx)[0]);info=one_info(infos[fold],loc)
        base=q[BASE].to_numpy(float)[None]
        expected={arm:indexed.loc[(arm,17,fold,identifier),SCORE].to_numpy(float) for arm in ['dynamic','fixed']}
        predictions={arm:int(s.argmax()) for arm,s in expected.items()}
        top={arm:int(np.argmax(np.where(np.arange(8)==y,-np.inf,s))) for arm,s in expected.items()}
        if case['outcome']=='B_rescued':rival=int(base.argmax())
        elif case['outcome']=='rescued':rival=predictions['fixed']
        elif case['outcome'] in ['harmed','B_harmed','WW']:rival=predictions['dynamic']
        else:rival=top['dynamic']
        assert rival!=y
        case.update(true_class=y,fixed_rival=rival,dynamic_top_rival=top['dynamic'],fixed_top_rival=top['fixed'],
            dynamic_prediction=predictions['dynamic'],fixed_prediction=predictions['fixed'],base_prediction=int(base.argmax()),
            dynamic_top1_margin=float(expected['dynamic'][y]-expected['dynamic'][top['dynamic']]),
            fixed_top1_margin=float(expected['fixed'][y]-expected['fixed'][top['fixed']]),
            dynamic_fixed_margin=float(expected['dynamic'][y]-expected['dynamic'][rival]),
            fixed_fixed_margin=float(expected['fixed'][y]-expected['fixed'][rival]),
            base_fixed_margin=float(base[0,y]-base[0,rival]))
        local=[];scores={}
        for arm in ['dynamic','fixed']:
            key=(fold,arm)
            if key not in heads:
                ck=torch.load(c.HERE/'models'/f'fold{fold}_{arm}_seed17.pt',map_location='cpu',weights_only=False)
                head=runner.Head(ck['input_dim']).eval().requires_grad_(False);head.load_state_dict(ck['state'])
                heads[key]=(ck,head)
            ck,head=heads[key]
            assert ck['epoch']==case[arm+'_epoch']
            s=replay(base,info,arm,ck,head)[0];error=float(abs(s-expected[arm]).max())
            assert error<1e-6
            scores[arm]=s
            checks.append(dict(instance_id=identifier,fold=fold,arm=arm,epoch=ck['epoch'],score_replay_error=error))
            offset=1 if arm=='dynamic' else 4
            for region in range(3):
                for p,pol in enumerate(c.POL):
                    for cls in range(8):
                        mu,h,b=info['summary'][0,p,offset+region,cls]
                        slots.append(dict(instance_id=identifier,arm=arm,pol=pol,region=c.REGIONS[region],response_class=cls,
                            mu=mu,h=h,b=b,region_valid=bool(info['region_valid'][0,p,offset+region]),
                            is_true=cls==y,is_fixed_rival=cls==rival))
            for region in [-1,0,1,2]:
                changed={key:val.copy() for key,val in info.items()}
                if region>=0:changed['summary'][:,:,offset+region]=0
                z=replay(base,changed,arm,ck,head)[0]
                if ck['epoch']==0:np.testing.assert_array_equal(z,base[0])
                m=float(z[y]-z[rival]);original=float(s[y]-s[rival])
                local.append(dict(instance_id=identifier,obs_id=case['obs_id'],outcome=case['outcome'],fold=fold,seed=17,
                    arm=arm,epoch=ck['epoch'],removed='none' if region<0 else c.REGIONS[region],
                    fixed_rival=rival,true_class=y,prediction=int(z.argmax()),fixed_margin=m,
                    original_fixed_margin=original,margin_change=m-original,
                    own_top1_margin=float(z[y]-np.max(np.where(np.arange(8)==y,-np.inf,z))),
                    **{f's{cls}':z[cls] for cls in range(8)}))
        case['panel']=render(row,case,base[0],scores,local,template,table,snapshots,valid)
        case.update(image_vv=row.image_vv,image_vh=row.image_vh)
        cases.append(case.copy());ablations.extend(local)
        print(case['outcome'],case['obs_id'],'D/F epochs',case['dynamic_epoch'],case['fixed_epoch'],
            'rival',rival,'D/F margins',case['dynamic_fixed_margin'],case['fixed_fixed_margin'],flush=True)
    for name,rows in [('active_cases',cases),('region_ablation',ablations),('case_region_slots',slots),('selected_replay_checks',checks)]:
        pd.DataFrame(rows).to_csv(HERE/f'{name}.csv',index=False)
    receipt=dict(cases=len(cases),new_fits=0,current_invocation_seconds=time.perf_counter()-tick,
        current_invocation_cases=len(todo),baseline_addendum='--baseline-addendum' in sys.argv,
        max_score_replay_error=max(x['score_replay_error'] for x in checks),seed=17,
        selection='predeclared active preference, first obs_id; no epoch changes',
        region_intervention='raw mu/h/b zero for both pol; fixed geometry/quality/valid/scaler/head',
        physical_intervention=False)
    (HERE/'selected_case_summary.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
