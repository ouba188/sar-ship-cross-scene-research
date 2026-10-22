"""Fixed eight-class test predictions, conditional intervals and actual inputs."""
import sys
sys.dont_write_bytecode=True
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_transfer import HERE,PARENT,OLD,metrics,crop_pad,masks_for


def paired_intervals(p):
    rows=[]
    y=p.class_id.to_numpy()
    diff=p.net.to_numpy()
    for block in ['pass_group','mmsi']:
        codes,names=pd.factorize(p[block],sort=True)
        denom=np.zeros((len(names),8));numer=np.zeros_like(denom)
        np.add.at(denom,(codes,y),1);np.add.at(numer,(codes,y),diff)
        rng=np.random.default_rng(20261023)
        oa=[];ba=[];missing=0
        for _ in range(2000):
            ix=rng.integers(0,len(names),len(names))
            count=denom[ix].sum(0);net=numer[ix].sum(0)
            oa.append(100*net.sum()/count.sum())
            if (count==0).any():
                missing+=1
            else:
                ba.append(100*np.mean(net/count))
        rows.append(dict(block=block,blocks=len(names),draws=2000,seed=20261023,BA_missing_class_draws=missing,
            OA_delta=100*diff.mean(),BA_delta=100*np.mean([diff[y==c].mean() for c in range(8)]),
            OA_low=float(np.quantile(oa,.025)),OA_high=float(np.quantile(oa,.975)),
            BA_low=float(np.quantile(ba,.025)),BA_high=float(np.quantile(ba,.975))))
    pd.DataFrame(rows).to_csv(HERE/'paired_intervals.csv',index=False)


def render(p):
    frame=pd.read_csv(PARENT/'cohort.csv').set_index('instance_id')
    snapshots={s['obs_id']:s for s in json.loads((OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    coverage=pd.read_csv(HERE/'window_coverage.csv').set_index(['instance_id','pol'])
    selections=[];missing=[]
    for outcome in ['rescued','harmed','CC','WW']:
        hit=p[p.outcome==outcome].sort_values('instance_id')
        if len(hit):
            chosen=hit.iloc[0].copy();chosen['selection_rule']='first ID per outcome'
            selections.append(chosen)
        else:
            missing.append(outcome)
    if len(selections)<4:
        for outcome in ['CC','WW']:
            hit=p[p.outcome==outcome].sort_values('instance_id')
            for _,q in hit.iloc[1:2].iterrows():
                q=q.copy();q['selection_rule']='second CC/WW because flip outcome absent'
                selections.append(q)
    # The first-ID cases share Busan; also inspect the largest class tradeoff.
    for c,outcome,port in [(3,'harmed','Jebel Ali'),(5,'rescued',None)]:
        hit=p[(p.class_id==c)&(p.outcome==outcome)]
        if port is not None:
            hit=hit[hit.port==port]
        if len(hit):
            chosen=hit.sort_values('instance_id').iloc[0].copy()
            chosen['selection_rule']=f'class{c} {outcome} first ID'+(' in '+port if port else '')
            if chosen.instance_id not in [q.instance_id for q in selections]:
                selections.append(chosen)
    rows=[];panels=[]
    for pred in selections:
        ident=pred.instance_id;row=frame.loc[ident]
        truth=int(row.class_id)
        base=pred[[f'b{c}' for c in range(8)]].to_numpy(float)
        score=pred[[f's{c}' for c in range(8)]].to_numpy(float)
        delta=pred[[f'delta{c}' for c in range(8)]].to_numpy(float)
        wrong=max((c for c in range(8) if c!=truth),key=lambda c:base[c])
        fig,axes=plt.subplots(2,3,figsize=(15,10),layout='constrained')
        for j,pol in enumerate(['vv','vh']):
            view=snapshots[int(row.obs_id)]['images'][pol]
            with Image.open(row[f'image_{pol}']) as im:
                raw=np.array(im.convert('L'))
            pixels,masks,place=crop_pad(raw,masks_for(view),160)
            support=masks.sum(0)
            weight=F.interpolate(torch.tensor(support)[None,None],size=(10,10),mode='area')[0,0].numpy()
            used=bool(coverage.loc[(ident,pol),'valid'])
            prior=view['priors'];target=prior.get('target_index')
            detections=prior.get('detections',[])
            corners=np.array(detections[target]['corners']) if target is not None and 0<=target<len(detections) else None
            axes[j,0].imshow(raw,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            axes[j,0].set_title(f'{pol.upper()} original {raw.shape[1]}x{raw.shape[0]} px')
            axes[j,1].imshow(pixels,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            axes[j,1].set_title('Actual 160 crop/pad input'+('' if used else ' [NOT USED]'))
            axes[j,2].pcolormesh(np.arange(11)*16,np.arange(11)*16,weight,cmap='Greys_r',vmin=0,vmax=1,
                                 edgecolors='#666666',linewidth=.4,shading='flat')
            axes[j,2].set(xlim=(0,160),ylim=(160,0),aspect='equal')
            axes[j,2].set_title(f'10x10 footprint weights; coverage {support.mean():.3f}; valid={used}')
            if corners is not None:
                rawpoints=np.r_[corners,corners[:1]]
                axes[j,0].plot(rawpoints[:,0],rawpoints[:,1],color='#13c74e',lw=1)
                points=rawpoints-[place['crop_x'],place['crop_y']]+[place['pad_left'],place['pad_top']]
                for ax in axes[j,1:]:
                    ax.plot(points[:,0],points[:,1],color='#13c74e',lw=1)
            for ax in axes[j]:
                ax.set_xlabel('pixel x');ax.set_ylabel('pixel y')
        fig.suptitle(f'{ident} | {row.class_name} | {row.port} | {pred.outcome} | fold {int(pred.fold)}\n'
            f'A pred {int(pred.base_prediction)} -> std pred {int(pred.prediction)}; lambda={pred.selected_lambda:g}; true {truth} vs fixed wrong {wrong}\n'
            f'pair margin {base[truth]-base[wrong]:+.5f} -> {score[truth]-score[wrong]:+.5f}; '
            f'weighted delta {pred.selected_lambda*(delta[truth]-delta[wrong]):+.5f}\n'
            'Green: recorded OBB. Zero-footprint cells excluded from pooling; convolution RF can still mix padding.',fontsize=11)
        path=HERE/f'case_{ident}.png';fig.savefig(path,dpi=120);plt.close(fig)
        panels.append(str(path))
        rows.append(dict(instance_id=ident,selection=pred.selection_rule,
            outcome=pred.outcome,fold=int(pred.fold),class_id=truth,class_name=row.class_name,port=row.port,
            fixed_wrong_class=wrong,base_prediction=int(pred.base_prediction),prediction=int(pred.prediction),
            selected_lambda=pred.selected_lambda,base_margin=float(base[truth]-base[wrong]),margin=float(score[truth]-score[wrong]),
            delta_margin=float(delta[truth]-delta[wrong]),image_vv=row.image_vv,image_vh=row.image_vh,
            vv_valid=bool(coverage.loc[(ident,'vv'),'valid']),vh_valid=bool(coverage.loc[(ident,'vh'),'valid']),panel=str(path)))
    pd.DataFrame(rows).to_csv(HERE/'review_cases.csv',index=False)
    return panels,missing


def main():
    start=time.perf_counter()
    p=pd.read_csv(HERE/'predictions.csv',dtype={'mmsi':str})
    base=p[[f'b{c}' for c in range(8)]].to_numpy()
    replay=base+p.selected_lambda.to_numpy()[:,None]*p[[f'delta{c}' for c in range(8)]].to_numpy()
    assert np.max(abs(replay-p[[f's{c}' for c in range(8)]].to_numpy()))<1e-12
    assert np.array_equal(replay.argmax(1),p.prediction)
    classes=[];acquisition=[];risk_rows=[]
    for c,g in p.groupby('class_id'):
        classes.append(dict(class_id=c,class_name=g.class_name.iloc[0],n=len(g),baseline_correct=int(g.base_prediction.eq(c).sum()),
            correct=int(g.prediction.eq(c).sum()),rescued=int(g.net.eq(1).sum()),harmed=int(g.net.eq(-1).sum()),net=int(g.net.sum()),
            baseline_recall=float(g.base_prediction.eq(c).mean()*100),recall=float(g.prediction.eq(c).mean()*100)))
        for arm,prefix in [('A','b'),('contrast_std','s')]:
            mm=metrics(g.class_id.to_numpy(),g[[f'{prefix}{j}' for j in range(8)]].to_numpy())
            risk_rows.append(dict(class_id=c,arm=arm,NLL=mm['NLL'],Brier=mm['Brier'],
                **{f'mean_delta{j}':g[f'delta{j}'].mean() for j in range(8)}))
    for unit in ['port','pass_group','product','fold']:
        for name,g in p.groupby(unit):
            acquisition.append(dict(unit=unit,name=name,n=len(g),mmsi=g.mmsi.nunique(),
                rescued=int(g.net.eq(1).sum()),harmed=int(g.net.eq(-1).sum()),net=int(g.net.sum()),
                **{f'class_{c}':int(g.class_id.eq(c).sum()) for c in range(8)},
                baseline_OA=float(g.base_prediction.eq(g.class_id).mean()*100),OA=float(g.prediction.eq(g.class_id).mean()*100)))
    pd.DataFrame(classes).to_csv(HERE/'class_effects.csv',index=False)
    pd.DataFrame(risk_rows).to_csv(HERE/'risk_by_class.csv',index=False)
    p.groupby(['outcome','class_id','base_prediction','prediction']).size().rename('n').reset_index().to_csv(HERE/'prediction_transitions.csv',index=False)
    pd.DataFrame(acquisition).to_csv(HERE/'acquisition_effects.csv',index=False)
    paired_intervals(p)
    panels,missing=render(p)
    summary=dict(prediction_replay=True,panels=panels,missing_outcomes=missing,test_rows=len(p),
        test_mmsi=p.mmsi.nunique(),test_passes=p.pass_group.nunique(),test_ports=p.port.nunique(),
        seconds=time.perf_counter()-start,scope='fixed std transfer, not eight-class full A+D')
    (HERE/'analysis_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
