"""Replay all fitted heads, compare cases and inspect actual native inputs."""
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.preprocessing import StandardScaler
from osgeo import gdal
from run_relations import HERE, OLD, ARMS, SEEDS, Relation, load_data, tensors_for, summary_geometry, metrics

CASES=['pair_0069192','pair_0069233','pair_0044888','pair_0046003','pair_0047041',
       'pair_0082497','pair_0069110','pair_0082337','pair_0054051','pair_0052425']


def forward(model,inputs):
    with torch.no_grad():
        return model(*inputs).numpy()


def main():
    torch.set_num_threads(2)
    gdal.UseExceptions();gdal.SetConfigOption('GDAL_PAM_ENABLED','NO');gdal.SetCacheMax(64*1024**2)
    t,f,r=load_data();p=pd.read_csv(HERE/'predictions.csv');e=pd.read_csv(OLD/'ports24/neighbors.csv')
    y=t.class_id.to_numpy();records=[];edge_rows=[];branch_rows=[];maxerr=0.;query_rows=[];coverage=[]
    for fold in sorted(p.fold.unique()):
        d=dict(np.load(HERE/f'fold{fold}_inputs.npz'))
        for role in ['train','validation','test']:
            for c in range(3):
                ix=d[role][y[d[role]]==c]
                coverage.append(dict(fold=fold,role=role,class_id=c,n=len(ix),context=int(f['mask'][ix].any(1).sum()),edges=int(f['mask'][ix].sum())))
        scaler=StandardScaler().fit(summary_geometry(f['geometry'],f['mask'])[d['train']])
        for seed in SEEDS:
            for arm in ARMS:
                model=Relation(arm).eval()
                model.load_state_dict(torch.load(HERE/'models'/f'fold{fold}_{arm}_{seed}.pt',weights_only=True,map_location='cpu'))
                inputs=tensors_for(arm,d);s=d['base']+forward(model,inputs)
                ref=p[p.fold.eq(fold)&p.seed.eq(seed)&p.arm.eq(arm)].set_index('instance_id').loc[t.iloc[d['test']].instance_id]
                err=float(np.abs(s[d['test']]-ref[['s0','s1','s2']].to_numpy()).max());maxerr=max(maxerr,err)
                assert err<1e-5,(arm,fold,seed,err)
                if arm=='conditional':
                    changed=[x.clone() for x in inputs];changed[0]=torch.tensor(d['qconstant'],dtype=torch.float32)
                    alt=d['base']+forward(model,changed)
                    ix=d['test']
                    query_rows.append(dict(fold=fold,seed=seed,n=len(ix),top1_changed=int((s[ix].argmax(1)!=alt[ix].argmax(1)).sum()),
                        actual_correct=int((s[ix].argmax(1)==y[ix]).sum()),constant_correct=int((alt[ix].argmax(1)==y[ix]).sum())))
                if seed!=29:
                    continue
                for i in d['test']:
                    iid=t.iloc[i].instance_id
                    if iid not in CASES:
                        continue
                    c=int(y[i]);rival=int(np.argmax(np.where(np.arange(3)==c,-np.inf,d['base'][i])))
                    bp=int(d['base'][i].argmax());pred=int(s[i].argmax())
                    kind='rescued' if bp!=c and pred==c else 'harmed' if bp==c and pred!=c else 'correct' if pred==c else 'wrong'
                    records.append(dict(instance_id=iid,fold=fold,seed=seed,arm=arm,class_id=c,rival=rival,base_prediction=bp,prediction=pred,
                        kind=kind,base_margin=float(d['base'][i,c]-d['base'][i,rival]),margin=float(s[i,c]-s[i,rival]),
                        delta_norm=float(np.linalg.norm(s[i]-d['base'][i]))))
                    if arm not in ['whole_raw','whole_unit','split_unit','conditional']:
                        continue
                    own=e[e.instance_id.eq(iid)].reset_index(drop=True)
                    for j in np.flatnonzero(d['m'][i]):
                        reduced=[x[i:i+1].clone() for x in inputs];reduced[-1][0,j]=False
                        st=scaler.transform(summary_geometry(f['geometry'][i:i+1],reduced[-1].numpy()))
                        reduced[3]=torch.tensor(st,dtype=torch.float32)
                        ss=d['base'][i]+forward(model,reduced)[0]
                        edge_rows.append(dict(instance_id=iid,fold=fold,seed=seed,arm=arm,detection_id=own.iloc[j].detection_id,
                            class_id=c,rival=rival,prediction=pred,removed_prediction=int(ss.argmax()),
                            full_margin=float(s[i,c]-s[i,rival]),removed_margin=float(ss[c]-ss[rival]),
                            margin_drop=float(s[i,c]-s[i,rival]-ss[c]+ss[rival])))
                    if arm in ['split_unit','conditional']:
                        for name,sl in [('fg',slice(0,32)),('bg',slice(32,64))]:
                            changed=[x[i:i+1].clone() for x in inputs];changed[1][:,:,sl]=0
                            ss=d['base'][i]+forward(model,changed)[0]
                            branch_rows.append(dict(instance_id=iid,seed=seed,arm=arm,zeroed=name,prediction=pred,
                                changed_prediction=int(ss.argmax()),margin=float(s[i,c]-s[i,rival]),zeroed_margin=float(ss[c]-ss[rival])))
    cases=pd.DataFrame(records);cases.to_csv(HERE/'case_scores.csv',index=False)
    ed=pd.DataFrame(edge_rows);ed.to_csv(HERE/'edge_deletion_cases.csv',index=False)
    pd.DataFrame(branch_rows).to_csv(HERE/'branch_zero_cases.csv',index=False)
    pd.DataFrame(query_rows).to_csv(HERE/'fixed_head_query_mean.csv',index=False)
    pd.DataFrame(coverage).to_csv(HERE/'context_coverage.csv',index=False)
    details=p.merge(t[['instance_id','port','pass_group']],on='instance_id')
    details[details.prediction.ne(details.base_prediction)].to_csv(HERE/'prediction_flips.csv',index=False)
    byclass=[];bygroup=[]
    for (arm,seed),a in details.groupby(['arm','seed']):
        for c,b in a.groupby('class_id'):
            byclass.append(dict(arm=arm,seed=seed,class_id=c,n=len(b),base_correct=int(b.base_prediction.eq(c).sum()),
                correct=int(b.prediction.eq(c).sum()),net=int(b.prediction.eq(c).sum()-b.base_prediction.eq(c).sum())))
        for group,b in a.groupby('pass_group'):
            bygroup.append(dict(arm=arm,seed=seed,pass_group=group,port=b.port.iloc[0],n=len(b),context=int(b.neighbors.gt(0).sum()),
                net=int(b.prediction.eq(b.class_id).sum()-b.base_prediction.eq(b.class_id).sum())))
    pd.DataFrame(byclass).to_csv(HERE/'class_effects.csv',index=False)
    pd.DataFrame(bygroup).to_csv(HERE/'acquisition_effects.csv',index=False)
    pooled=pd.read_csv(HERE/'pooled.csv');summary=pooled[pooled.scope.eq('all')].groupby('arm')[['OA','BA','macro_F1','rescued','harmed']].agg(['mean','std'])
    summary.to_csv(HERE/'summary.csv')
    render(t,e,cases,ed)
    (HERE/'analysis_checks.json').write_text(json.dumps(dict(heads_replayed=126,max_abs_score_error=maxerr,
        reviewed_ids=CASES,seed=29,note='Edge deletion recomputes count/means/variance. Region zero and query replacement are model-dependence diagnostics, not physical interventions.'),indent=2),encoding='utf-8')
    print('BASE',metrics(y,details[details.arm.eq('whole_unit')&details.seed.eq(29)].set_index('instance_id').loc[t.instance_id][['b0','b1','b2']].to_numpy()))
    print(pd.DataFrame(byclass).groupby(['arm','class_id']).net.mean().unstack().to_string())
    print(details[details.arm.eq('whole_unit')&details.prediction.ne(details.base_prediction)][['seed','instance_id','class_id','base_prediction','prediction']].to_string(index=False))
    print(cases[cases.instance_id.isin(['pair_0069192','pair_0069233'])].to_string(index=False))
    print('Replay error',maxerr)


def render(t,edges,cases,effects):
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    t=t.set_index('instance_id');datasets={}
    for page,start in enumerate(range(0,len(CASES),2),1):
        chosen=CASES[start:start+2]
        canvas=Image.new('RGB',(1400,480*len(chosen)),'white');draw=ImageDraw.Draw(canvas)
        for rr,iid in enumerate(chosen):
            row=t.loc[iid];y0=rr*480
            draw.text((8,y0+4),f'{iid} | {row.port} | truth={row.class_name} | N={row.neighbors} | native scale (no display resizing)',fill='black',font=font)
            scores=cases[cases.instance_id.eq(iid)].set_index('arm')
            for j,arm in enumerate(['whole_raw','whole_unit','split_unit','attention','conditional','conditional_constant','conditional_sham']):
                a=scores.loc[arm]
                draw.text((8,y0+28+j*21),f'{arm}: {int(a.base_prediction)} -> {int(a.prediction)} | {a.kind} | margin {a.margin:+.3f}',fill='black',font=font)
            for col,pol in enumerate(['vv','vh']):
                im=Image.open(OLD/'ports24/native_targets'/f'{iid}_{pol}.png').convert('RGB')
                canvas.paste(im,(8+col*170,y0+224));draw.text((8+col*170,y0+203),'target '+pol.upper(),fill='black',font=font)
            own=edges[edges.instance_id.eq(iid)]
            ranks=effects[effects.instance_id.eq(iid)&effects.arm.eq('whole_unit')].copy()
            ranks['absdrop']=ranks.margin_drop.abs()
            chosen_edges=ranks.sort_values('absdrop',ascending=False).head(3).detection_id.tolist()
            if iid=='pair_0069192':
                chosen_edges=['det_01443']+[a for a in chosen_edges if a!='det_01443'];chosen_edges=chosen_edges[:3]
            for j,did in enumerate(chosen_edges):
                edge=own[own.detection_id.eq(did)].iloc[0];xx=420+j*320
                eff=ranks[ranks.detection_id.eq(did)].iloc[0]
                draw.text((xx,y0+28),f'{did} | delete margin {eff.margin_drop:+.3f}',fill='black',font=font)
                draw.text((xx,y0+50),str(edge.land_action),fill='black',font=font)
                points=np.asarray(json.loads(edge.source_points));window=json.loads(edge.read_window)
                for k,pol in enumerate(['vv','vh']):
                    path=row['raster_'+pol]
                    if path not in datasets:datasets[path]=gdal.Open(path)
                    im=Image.fromarray(datasets[path].ReadAsArray(*window)).convert('RGB')
                    dr=ImageDraw.Draw(im);pts=points-np.asarray(window[:2]);dr.line([tuple(a) for a in np.vstack([pts,pts[0]])],fill='#ffb020',width=1)
                    draw.text((xx,y0+80+k*172),pol.upper(),fill='black',font=font)
                    canvas.paste(im,(xx,y0+103+k*172))
            draw.text((8,y0+449),'0=bulk, 1=fishing, 2=general cargo. Margin uses fixed strongest baseline rival; not a probability. Orange=neighbor OBB.',fill='black',font=font)
        canvas.save(HERE/f'case_review_{page}.png')


if __name__=='__main__':
    main()
