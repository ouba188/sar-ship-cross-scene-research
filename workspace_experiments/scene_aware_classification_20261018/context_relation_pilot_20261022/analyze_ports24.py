"""Replay C and measure edge deletion on actual rescued/harmed/persistent cases."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from osgeo import gdal
from run_ports24 import Baseline, Residual, HERE
from review_ports24 import render
from prepare_ports24 import crop_world


def feature(n,g,m,base):
    count=m.sum(1).clip(1)
    mean=(g*m[:,:,None]).sum(1)/count[:,None]
    var=(((g-mean[:,None])**2)*m[:,:,None]).sum(1)/count[:,None]
    z=base.project(n.reshape(-1,n.shape[-1])).reshape(len(n),16,-1)
    bag=(z*m[:,:,None]).sum(1)/count[:,None]
    return np.c_[mean,var,m.sum(1)/16,bag]


def main():
    torch.set_num_threads(2)
    t=pd.read_csv(HERE/'targets.csv',dtype={'mmsi':str})
    p=pd.read_csv(HERE/'predictions.csv')
    splits=pd.read_csv(HERE/'splits.csv')
    f=np.load(HERE/'features.npz'); h,n,g,m=[f[k] for k in ('target','neighbor','geometry','mask')]
    edges=pd.read_csv(HERE/'neighbors.csv')
    selected=[]
    q=p[p.arm.eq('C')&p.seed.eq(29)].copy()
    q['kind']=np.select([(q.base_prediction!=q.class_id)&(q.prediction==q.class_id),
                        (q.base_prediction==q.class_id)&(q.prediction!=q.class_id),
                        (q.base_prediction==q.class_id)&(q.prediction==q.class_id)],
                       ['rescued','harmed','correct'],default='wrong')
    for kind in ['rescued','harmed','correct','wrong']:
        a=q[q.kind.eq(kind)&q.neighbors.gt(0)].sort_values(['neighbors','instance_id'],ascending=[False,True])
        selected.extend(a.head(2).instance_id.tolist())
    results=[]
    norms=[]
    max_error=0.
    for fold in sorted(q.fold.unique()):
        train_ids=splits[(splits.fold==fold)&splits.role.eq('train')].instance_id
        train=np.where(t.instance_id.isin(train_ids))[0]
        checkpoint=torch.load(HERE/f'fold{fold}_C_seed29.pt',weights_only=False,map_location='cpu')
        base=Baseline(h[train],t.iloc[train].class_id.to_numpy(),t.iloc[train].mmsi.to_numpy(),checkpoint['base_alpha'])
        x=feature(n,g,m,base)
        model=Residual(x.shape[1]); model.load_state_dict(checkpoint['state']); model.eval()
        standardized=(x-checkpoint['mean'])/checkpoint['scale']
        roles=splits[splits.fold.eq(fold)].set_index('instance_id').role.to_dict()
        for i in np.where(m.any(1))[0]:
            if t.iloc[i].instance_id in roles:
                norms.append(dict(fold=fold,instance_id=t.iloc[i].instance_id,role=roles[t.iloc[i].instance_id],
                    norm=float(np.linalg.norm(standardized[i])),maxabs=float(np.abs(standardized[i]).max())))
        def predict(x,mask):
            xx=torch.tensor((x-checkpoint['mean'])/checkpoint['scale'],dtype=torch.float32)
            with torch.no_grad():
                return model(xx,torch.tensor(mask.any(1),dtype=torch.float32)).numpy()
        score=base.predict(h)+predict(x,m)
        for row in q[q.fold==fold].itertuples():
            i=int(np.where(t.instance_id.eq(row.instance_id))[0][0])
            max_error=max(max_error,float(np.max(np.abs(score[i]-np.array([row.s0,row.s1,row.s2])))))
            if row.instance_id not in selected:
                continue
            own=edges[edges.instance_id.eq(row.instance_id)].reset_index(drop=True)
            c=row.class_id
            rival=row.prediction if row.prediction!=c else int(np.argmax(np.where(np.arange(3)==c,-np.inf,np.array([row.b0,row.b1,row.b2]))))
            for k in np.where(m[i])[0]:
                removed=m[i:i+1].copy(); removed[0,k]=False
                rx=feature(n[i:i+1],g[i:i+1],removed,base)
                reduced=base.predict(h[i:i+1])+predict(rx,removed)
                results.append(dict(instance_id=row.instance_id,seed=29,fold=fold,kind=row.kind,
                    class_id=c,rival=rival,detection_id=own.iloc[k].detection_id,
                    base_margin=float(getattr(row,f'b{c}')-getattr(row,f'b{rival}')),
                    full_margin=float(score[i,c]-score[i,rival]),removed_margin=float(reduced[0,c]-reduced[0,rival]),
                    removal_margin_drop=float(score[i,c]-score[i,rival]-reduced[0,c]+reduced[0,rival]),
                    prediction=row.prediction,removed_prediction=int(reduced.argmax(1)[0])))
    if max_error>1e-5:
        raise ValueError(f'Checkpoint replay differs: {max_error}')
    pd.DataFrame(results).to_csv(HERE/'edge_deletion_cases.csv',index=False)
    pd.DataFrame(norms).to_csv(HERE/'context_input_norms.csv',index=False)
    q[q.instance_id.isin(selected)].merge(t[['instance_id','port','original_chip','raster_vv','raster_vh']],on='instance_id').to_csv(HERE/'review_cases.csv',index=False)
    for j in range(0,len(selected),4):
        render(selected[j:j+4],f'outcome_review_{j//4+1}.png')
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    for page,start in enumerate(range(0,len(selected),4)):
        canvas=Image.new('RGB',(1200,280*len(selected[start:start+4])),'white'); d=ImageDraw.Draw(canvas)
        for j,iid in enumerate(selected[start:start+4]):
            row=t[t.instance_id.eq(iid)].iloc[0]
            rank=pd.DataFrame(results); rank=rank[rank.instance_id.eq(iid)].sort_values('removal_margin_drop',key=abs,ascending=False).head(3)
            d.text((8,j*280+3),iid+' | largest absolute edge-deletion effect',font=font,fill='black')
            for col,edge in enumerate(rank.itertuples()):
                er=edges[edges.instance_id.eq(iid)&edges.detection_id.eq(edge.detection_id)].iloc[0]
                d.text((col*400+8,j*280+28),f'{edge.detection_id} | margin drop {edge.removal_margin_drop:+.3f}',font=font,fill='black')
                for pol_idx,pol in enumerate(('vv','vh')):
                    ds=gdal.Open(row['raster_'+pol]); world=gdal.ApplyGeoTransform(ds.GetGeoTransform(),er.cx,er.cy)
                    im,_=crop_world(ds,world,er.side_m)
                    canvas.paste(Image.fromarray(im).convert('RGB'),(col*400+8+pol_idx*180,j*280+70))
                    d.text((col*400+8+pol_idx*180,j*280+48),pol.upper(),font=font,fill='black')
                    ds=None
        canvas.save(HERE/f'neighbor_detail_{page+1}.png')
    (HERE/'replay_check.json').write_text(json.dumps(dict(max_abs_score_error=max_error,selected=selected,
        note='Deletion recomputes all means/counts. It measures model dependence, not causal ship identity or physical mechanism.'),indent=2),encoding='utf-8')
    print(pd.DataFrame(results).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
