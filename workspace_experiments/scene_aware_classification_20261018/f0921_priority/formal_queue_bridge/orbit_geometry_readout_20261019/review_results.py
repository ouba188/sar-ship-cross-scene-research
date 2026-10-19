"""Outcome-stratified native VV/VH review and exact support evidence traces."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
from threadpoolctl import threadpool_limits
import run_experiment as ev
from spatial_readout import masks_for

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
spec=importlib.util.spec_from_file_location('parent_review',PARENT/'review_cases.py')
review=importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def panel(ids,meta,axes,geo,regions,path):
    canvas=Image.new('RGB',(240*len(ids),780),'white')
    draw=ImageDraw.Draw(canvas)
    colors=np.array([[225,80,60],[240,165,40],[30,160,180],[110,95,205]],float)
    for j,i in enumerate(ids):
        row,a,g=meta.loc[i],axes.loc[i],geo.loc[i]
        x0=j*240
        draw.text((x0+5,5),f'row{i} class{row.label}\nreadout={bool(a.readout_valid)}',fill='black')
        for p,name in enumerate(['png_vv','png_vh']):
            img=Image.open(row[name]).convert('RGB')
            canvas.paste(img,(x0+5,55+p*240))
            d=ImageDraw.Draw(canvas)
            if a.readout_valid:
                c=np.array([x0+5+a.center_x,55+p*240+a.center_y])
                v=np.array([np.cos(g.los_angle_image_rad),np.sin(g.los_angle_image_rad)])
                tip=c+65*v
                d.line([tuple(c),tuple(tip)],fill='cyan',width=2)
                for sign in [-1,1]:
                    back=tip-10*v+sign*5*np.array([-v[1],v[0]])
                    d.line([tuple(tip),tuple(back)],fill='cyan',width=2)
                u=45*np.array([np.cos(a.ship_axis_rad),np.sin(a.ship_axis_rad)])
                d.line([tuple(c-u),tuple(c+u)],fill='lime',width=2)
        if a.readout_valid:
            masks=masks_for(a,regions[i],g.los_angle_image_rad,g.orbit_incidence_deg)
            weights=masks.reshape(4,2,*masks.shape[-2:]).sum(1)
            rgb=np.einsum('thw,tc->hwc',weights,colors)
            image=Image.fromarray(np.rint(rgb).astype(np.uint8)).resize((224,224))
            canvas.paste(image,(x0+5,535))
        draw.text((x0+5,520),'Soft region membership (not saliency)',fill='black')
    draw.text((5,765),'Green unsigned ship axis; cyan arrow points toward satellite. VV top / VH below.',fill='black')
    canvas.save(path)


def main():
    meta=pd.read_csv(PARENT/'cohort.csv',dtype={'date':str,'mmsi':str}).set_index('row',drop=False)
    axes=pd.read_csv(HERE/'axes.csv').set_index('row')
    geo=pd.read_csv(HERE/'orbit_geometry.csv').set_index('row')
    pred=pd.read_csv(HERE/'predictions.csv',dtype={'fold':str})
    supports=pd.read_csv(PARENT/'supports.csv',dtype={'fold':str})
    features=dict(np.load(PARENT/'features.npz'))
    spatial=dict(np.load(HERE/'spatial_features.npz'))
    regions=np.load(PARENT/'masks.npz')['regions']
    tasks=json.loads((PARENT/'episodes.json').read_text())
    y=meta.label.to_numpy()
    selected,paths,contrib=[],[],[]
    with threadpool_limits(limits=1):
        for arm,comp in [('add_orbit','base'),('add_orbit','add_image'),('add_orbit','add_pooled')]:
            out=HERE/'case_review'/f'{arm}_vs_{comp}'
            out.mkdir(parents=True,exist_ok=True)
            candidates,_=review.select_events(pred,meta,arm,comp)
            chosen=[]
            for category in review.CATEGORIES:
                pool=[e for e in candidates if e[0]==category]
                if pool:
                    pool.sort(key=lambda e:(e[2].margin-e[3].margin,e[1]))
                    chosen.append(pool[(len(pool)-1)//2])
            for category,key,own,base in chosen:
                stem=f'B{key[1]}_{category}_{key[0]}_d{key[2]}_r{key[3]}'
                panels=review.actual_panels(key,own,base,supports,meta,arm,comp)
                img,_=review.render_case(category,key,own,base,panels,meta,regions,arm,comp,
                    'Fixed additive channel; same support and query. Geometry is not a physical-cause label.')
                img.save(out/(stem+'.png'))
                ids=list(dict.fromkeys(int(p['row']) for p in panels if p.get('row') is not None and p['row']>=0))
                panel(ids,meta,axes,geo,regions,out/(stem+'_geometry.png'))
                for i in ids[:3]:
                    review.native_query(meta.loc[i],out/(stem+f'_member{i}_native.png'))
                event=dict(category=category,fold=key[0],budget=key[1],draw=key[2],row=key[3],arm=arm,comparator=comp,
                    truth=int(own.truth),old_prediction=int(base.prediction),new_prediction=int(own.prediction),
                    old_margin=float(base.margin),new_margin=float(own.margin),readout_valid=bool(axes.loc[key[3],'readout_valid']),
                    support_rows=json.dumps(ids[1:]),figure=str(out/(stem+'.png')),
                    native=str(out/(stem+f'_member{key[3]}_native.png')),geometry=str(out/(stem+'_geometry.png')))
                selected.append(event)
                task=next(t for t in tasks if t['fold']==key[0] and t['budget']==key[1])
                src,s=np.array(task['source']),np.array(task['supports'][key[2]])
                g=ev.bridge.global_descriptor(features['global'],src);kg=g@g.T
                prior=ev.bridge.prior(kg,src,y,7)
                ks=ev.kernels(features,spatial,s,kg)
                scores,alphas={},{}
                for name in [arm,comp]:
                    scores[name],alphas[name]=ev.bridge.adapted(ks[name],prior,s,y)
                q,truth=key[3],int(own.truth)
                rival_scores=scores[comp][q].copy();rival_scores[truth]=-np.inf
                rival=int(rival_scores.argmax())
                aa=alphas[arm][:,truth]-alphas[arm][:,rival]
                ab=alphas[comp][:,truth]-alphas[comp][:,rival]
                ka,kb=ks[arm][q,s],ks[comp][q,s]
                direct=.5*(ka-kb)*(aa+ab);refit=.5*(ka+kb)*(aa-ab);total=direct+refit
                observed=(scores[arm][q,truth]-scores[arm][q,rival])-(scores[comp][q,truth]-scores[comp][q,rival])
                np.testing.assert_allclose(total.sum(),observed,atol=1e-9)
                tags={k:event[k] for k in ['fold','budget','draw','row','arm','comparator']}
                paths.append(dict(**tags,fixed_rival=rival,direct=direct.sum(),refit=refit.sum(),
                    r50_margin_change=observed,same_class_sum=total[y[s]==truth].sum(),other_class_sum=total[y[s]!=truth].sum()))
                contrib.extend(dict(**tags,support_row=int(i),support_label=int(y[i]),
                    old_similarity=float(kb[j]),new_similarity=float(ka[j]),direct=float(direct[j]),
                    refit=float(refit[j]),total=float(total[j])) for j,i in enumerate(s))
    pd.DataFrame(selected).to_csv(HERE/'selected_cases.csv',index=False)
    pd.DataFrame(paths).to_csv(HERE/'case_paths.csv',index=False)
    pd.DataFrame(contrib).to_csv(HERE/'case_support_contributions.csv',index=False)
    print(pd.DataFrame(selected)[['category','budget','row','comparator','old_prediction','new_prediction','readout_valid']].to_string(index=False))


if __name__=='__main__':
    main()
