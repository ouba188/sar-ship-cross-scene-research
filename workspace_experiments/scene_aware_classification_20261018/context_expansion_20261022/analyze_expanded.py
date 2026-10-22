"""Paired acquisition/identity summaries and native outcome panels."""
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageDraw,ImageFont
from osgeo import gdal
from train_expanded import HERE,ARMS,SEEDS,MODEL_ARM,Relation,inputs_for,metrics


def clustered_ci(t,difference,key):
    groups=list(t.groupby(key,sort=True).indices.values());sums=[];counts=[]
    for ix in groups:
        sums.append([difference[ix][t.class_id.to_numpy()[ix]==c].sum() for c in range(3)])
        counts.append([(t.class_id.to_numpy()[ix]==c).sum() for c in range(3)])
    sums=np.asarray(sums);counts=np.asarray(counts);rng=np.random.default_rng(20261022)
    draws=rng.integers(0,len(groups),size=(2000,len(groups)));num=sums[draws].sum(1);den=counts[draws].sum(1)
    oa=num.sum(1)/den.sum(1)*100;ba=(num/den).mean(1)*100
    return dict(blocks=len(groups),oa_low=float(np.quantile(oa,.025)),oa_high=float(np.quantile(oa,.975)),
        ba_low=float(np.quantile(ba,.025)),ba_high=float(np.quantile(ba,.975)))


def main():
    torch.set_num_threads(2);gdal.UseExceptions();gdal.SetConfigOption('GDAL_PAM_ENABLED','NO');gdal.SetCacheMax(64*1024**2)
    t=pd.read_csv(HERE/'readout/targets.csv',dtype={'mmsi':str});p=pd.read_csv(HERE/'predictions.csv')
    edges=pd.read_csv(HERE/'readout/neighbors.csv');f=np.load(HERE/'readout/features.npz')
    p=p.merge(t[['instance_id','mmsi','port','pass_group']],on='instance_id',validate='many_to_one')
    p['base_correct']=p.base_prediction.eq(p.class_id);p['correct']=p.prediction.eq(p.class_id)
    p['net']=p.correct.astype(int)-p.base_correct.astype(int)
    p['kind']=np.select([p.net.eq(1),p.net.eq(-1),p.correct],['rescued','harmed','correct'],default='wrong')
    base=p[p.arm.eq(ARMS[0])&p.seed.eq(SEEDS[0])].set_index('instance_id').loc[t.instance_id]
    basemetrics=metrics(t.class_id.to_numpy(),base[['b0','b1','b2']].to_numpy())
    pd.DataFrame([basemetrics]).to_csv(HERE/'baseline_metrics.csv',index=False)
    p.groupby(['arm','seed','port','pass_group']).agg(n=('instance_id','size'),net=('net','sum')).reset_index().to_csv(HERE/'acquisition_effects.csv',index=False)
    p.groupby(['arm','seed','class_id']).agg(n=('instance_id','size'),net=('net','sum'),correct=('correct','sum'),base_correct=('base_correct','sum')).reset_index().to_csv(HERE/'class_effects.csv',index=False)
    p[p.prediction.ne(p.base_prediction)].to_csv(HERE/'prediction_flips.csv',index=False)
    intervals=[]
    for arm,q in p.groupby('arm'):
        diff=q.groupby('instance_id').net.mean().reindex(t.instance_id).to_numpy()
        for key in ['pass_group','mmsi']:
            intervals.append(dict(arm=arm,block=key,**clustered_ci(t,diff,key)))
    pd.DataFrame(intervals).to_csv(HERE/'paired_intervals.csv',index=False)
    chosen=[]
    q=p[p.arm.eq('conditional_role')&p.seed.eq(29)&p.neighbors.gt(0)]
    for kind in ['rescued','harmed','correct','wrong']:
        chosen.extend(q[q.kind.eq(kind)].sort_values('instance_id').head(2).instance_id.tolist())
    cases=p[p.instance_id.isin(chosen)].copy();cases.to_csv(HERE/'review_cases.csv',index=False)
    maxerr=0.
    for fold in range(6):
        d=dict(np.load(HERE/f'fold{fold}_inputs.npz'))
        for arm in ARMS:
            inputs=inputs_for(arm,d,'cuda')
            for seed in SEEDS:
                model=Relation(MODEL_ARM.get(arm,arm)).cuda().eval()
                model.load_state_dict(torch.load(HERE/'models'/f'fold{fold}_{arm}_{seed}.pt',weights_only=True,map_location='cpu'))
                with torch.no_grad():s=d['base'][d['test']]+model(*[x[d['test']] for x in inputs]).cpu().numpy()
                ref=p[p.fold.eq(fold)&p.arm.eq(arm)&p.seed.eq(seed)].set_index('instance_id').loc[t.iloc[d['test']].instance_id]
                err=float(np.abs(s-ref[['s0','s1','s2']].to_numpy()).max());maxerr=max(err,maxerr)
                assert err<1e-4,(fold,arm,seed,err)
                assert np.array_equal(s.argmax(1),ref.prediction.to_numpy())
    render(t,edges,p,chosen)
    checks=dict(replay_max_error=maxerr,heads=108,reviewed_ids=chosen,baseline=basemetrics,
        ci_note='Paired seed-averaged correctness; acquisition and MMSI blocks are separate sensitivity analyses, not a joint two-way bootstrap. Exploratory reused development inventory.')
    (HERE/'analysis_summary.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print('BASE',basemetrics)
    print(p.groupby(['arm','seed']).net.sum().unstack().to_string())
    print(p.groupby(['arm','class_id']).net.sum().unstack().div(3).to_string())
    print(pd.DataFrame(intervals).to_string(index=False))


def render(t,edges,p,ids):
    t=t.set_index('instance_id');font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15);datasets={}
    for page,start in enumerate(range(0,len(ids),2),1):
        chosen=ids[start:start+2];canvas=Image.new('RGB',(1400,len(chosen)*420),'white');draw=ImageDraw.Draw(canvas)
        for rowno,iid in enumerate(chosen):
            y=rowno*420;row=t.loc[iid]
            draw.text((8,y+4),f'{iid} | {row.port} | truth={row.class_name} | N={row.neighbors} | native inputs; seed29',font=font,fill='black')
            scores=p[p.instance_id.eq(iid)&p.seed.eq(29)].set_index('arm')
            for j,arm in enumerate(ARMS):
                a=scores.loc[arm];s=a[['s0','s1','s2']].to_numpy(float);c=int(a.class_id)
                margin=s[c]-np.delete(s,c).max()
                draw.text((8,y+27+j*20),f'{arm}: {int(a.base_prediction)} -> {int(a.prediction)} {a.kind}, true margin {margin:+.3f}',font=font,fill='black')
            for j,pol in enumerate(['vv','vh']):
                im=Image.open(HERE/'readout/native_targets'/f'{iid}_{pol}.png').convert('RGB');canvas.paste(im,(8+j*180,y+185))
                draw.text((8+j*180,y+165),'target '+pol.upper(),font=font,fill='black')
            for j,edge in enumerate(edges[edges.instance_id.eq(iid)].sort_values('slot').head(3).itertuples()):
                x=430+j*310;draw.text((x,y+27),edge.detection_id+f' {edge.distance_m:.0f}m',font=font,fill='black')
                draw.text((x,y+47),edge.land_action,font=font,fill='black')
                for k,pol in enumerate(['vv','vh']):
                    path=getattr(edge,'raster_'+pol)
                    if path not in datasets:datasets[path]=gdal.Open(path)
                    window=json.loads(getattr(edge,'read_window_'+pol));im=Image.fromarray(datasets[path].ReadAsArray(*window)).convert('RGB')
                    pts=np.asarray(json.loads(edge.source_points))-np.asarray(window[:2]);dd=ImageDraw.Draw(im)
                    dd.line([tuple(a) for a in np.vstack([pts,pts[0]])],fill='#ffb020',width=1)
                    draw.text((x,y+72+k*155),pol.upper(),font=font,fill='black');canvas.paste(im,(x,y+94+k*155))
            draw.text((8,y+393),'0=bulk;1=fishing;2=general. No neighbor class labels. Orange=detected OBB. Neighbors shown by distance, not selected for effect.',font=font,fill='black')
        canvas.save(HERE/f'outcome_review_{page}.png')

if __name__=='__main__':main()
