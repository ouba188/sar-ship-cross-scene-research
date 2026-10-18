"""Saved prediction review plus exact relation/refit paths for bounded cases."""
import importlib.util
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
from threadpoolctl import threadpool_limits
import run_axis as ev

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
spec=importlib.util.spec_from_file_location('parent_review',PARENT/'review_cases.py')
review=importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def axis_panel(ids,cohort,axes,geometry,destination):
    canvas=Image.new('RGB',(240*len(ids),570),'white')
    draw=ImageDraw.Draw(canvas)
    for j,i in enumerate(ids):
        row,a,g=cohort.loc[i],axes.loc[i],geometry.loc[i]
        x0=240*j
        draw.text((x0+5,5),f"row {i} class {row.label}\naxis reliability {a.axis_reliability:.3f}\nradar valid {bool(g.geometry_valid)}",fill='black')
        for p,path in enumerate([row.png_vv,row.png_vh]):
            image=Image.open(path).convert('RGB')
            overlay=ImageDraw.Draw(image)
            for theta,color,enabled in [(a.ship_axis_rad,'lime',a.axis_reliability>0),
                                        (g.radar_angle_image_rad,'cyan',g.geometry_valid)]:
                if enabled:
                    delta=60*np.array([np.cos(theta),np.sin(theta)])
                    center=np.array([a.center_x,a.center_y])
                    overlay.line([tuple(center-delta),tuple(center+delta)],fill=color,width=2)
            canvas.paste(image,(x0+5,70+240*p))
            draw.text((x0+5,55+240*p),'VV' if p==0 else 'VH',fill='black')
    draw.text((5,550),'Green: unsigned detector axis. Cyan: radar range axis. Neither is bow/heading.',fill='black')
    canvas.save(destination)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--conditional',action='store_true')
    args=parser.parse_args()
    prefix='condition_' if args.conditional else ''
    meta=pd.read_csv(PARENT/'cohort.csv',dtype={'date':str,'mmsi':str}).set_index('row',drop=False)
    axes=pd.read_csv(HERE/'axes.csv').set_index('row',drop=False)
    geom=pd.read_csv(HERE/'geometry.csv').set_index('row',drop=False)
    pred=pd.read_csv(HERE/'predictions.csv',dtype={'fold':str})
    if args.conditional:
        import run_condition as condition
        pred=pd.concat([pred,pd.read_csv(HERE/'condition_predictions.csv',dtype={'fold':str})])
    supports=pd.read_csv(PARENT/'supports.csv',dtype={'fold':str})
    regions=np.load(PARENT/'masks.npz')['regions']
    features=dict(np.load(PARENT/'features.npz'))
    directions=dict(np.load(HERE/'direction_features.npz'))
    tasks=json.loads((PARENT/'episodes.json').read_text())
    labels=meta.label.to_numpy()
    selected,contributions,paths=[],[],[]
    comparisons=([('conditioned_ship','fg_bg_global'),('conditioned_ship','availability_control'),
                  ('conditioned_ship','geometry_only')] if args.conditional else
                 [('ship_radar','fg_bg_global'),('ship_radar','ship_image'),('ship_soft','image_soft')])
    with threadpool_limits(limits=1):
        for arm,comp in comparisons:
            destination=HERE/'case_review'/f'{arm}_vs_{comp}'
            destination.mkdir(parents=True,exist_ok=True)
            events,_=review.select_events(pred,meta,arm,comp)
            # One lower-median representative across the preselected budget cases
            # per outcome type. All event counts remain in paired_events.csv.
            chosen=[]
            for category in review.CATEGORIES:
                pool=[event for event in events if event[0]==category]
                if pool:
                    pool.sort(key=lambda e:(e[2].margin-e[3].margin,e[1]))
                    chosen.append(pool[(len(pool)-1)//2])
            for category,key,own,base in chosen:
                stem=f'B{key[1]}_{category}_{key[0]}_d{key[2]}_r{key[3]}'
                panels=review.actual_panels(key,own,base,supports,meta,arm,comp)
                canvas,_=review.render_case(category,key,own,base,panels,meta,regions,arm,comp,
                    'Bounded lower-median representative; unsigned axis not anatomical parts; radar valid only product-exclusive native windows.')
                figure=destination/(stem+'.png')
                canvas.save(figure)
                native=destination/(stem+'_native.png')
                limits=review.native_query(meta.loc[key[3]],native)
                ids=list(dict.fromkeys(int(p['row']) for p in panels
                                       if p.get('row') is not None and p['row']>=0))
                axisfile=destination/(stem+'_axes.png')
                axis_panel(ids,meta,axes,geom,axisfile)
                # Also replay the actual same and confusing support native windows.
                for i in ids[1:3]:
                    review.native_query(meta.loc[i],destination/(stem+f'_support{i}_native.png'))
                event=dict(category=category,fold=key[0],budget=key[1],draw=key[2],row=key[3],arm=arm,comparator=comp,
                    truth=int(own.truth),old_prediction=int(base.prediction),new_prediction=int(own.prediction),
                    old_margin=float(base.margin),new_margin=float(own.margin),
                    radar_valid=bool(geom.loc[key[3],'geometry_valid']),axis_reliability=float(axes.loc[key[3],'axis_reliability']),
                    figure=str(figure),native=str(native),axes=str(axisfile),native_limits=json.dumps(limits),support_rows=json.dumps(ids[1:]))
                selected.append(event)
                task=next(t for t in tasks if t['fold']==key[0] and t['budget']==key[1])
                source,s=np.array(task['source']),np.array(task['supports'][key[2]])
                g=ev.bridge.global_descriptor(features['global'],source)
                kg=g@g.T
                prior=ev.bridge.prior(kg,source,labels,7)
                grid=ev.bridge.cosine(features['grid2'].reshape(len(meta),2,-1),s).astype(np.float64)
                local=(condition.conditional_local(features,directions,axes,s,condition.geometry_kernels(axes,geom))
                       if args.conditional else ev.kernels(features,directions,axes,s))
                ks={name:.5*kg+.25*grid+.25*local[name] for name in [comp,arm]}
                scores,alpha={},{}
                for name,k in ks.items():
                    scores[name],alpha[name]=ev.bridge.adapted(k,prior,s,labels)
                query,truth=key[3],int(own.truth)
                old=scores[comp][query].copy();old[truth]=-np.inf
                rival=int(old.argmax())
                aa=alpha[arm][:,truth]-alpha[arm][:,rival]
                ab=alpha[comp][:,truth]-alpha[comp][:,rival]
                ka,kb=ks[arm][query,s],ks[comp][query,s]
                direct=.5*(ka-kb)*(aa+ab)
                refit=.5*(ka+kb)*(aa-ab)
                total=direct+refit
                change=(scores[arm][query,truth]-scores[arm][query,rival])-(scores[comp][query,truth]-scores[comp][query,rival])
                np.testing.assert_allclose(total.sum(),change,atol=1e-9)
                paths.append(dict(**{k:event[k] for k in ['fold','budget','draw','row','arm','comparator']},
                    fixed_rival=rival,direct=direct.sum(),refit=refit.sum(),r50_margin_change=change,
                    same_class_sum=total[labels[s]==truth].sum(),other_class_sum=total[labels[s]!=truth].sum()))
                contributions.extend(dict(**{k:event[k] for k in ['fold','budget','draw','row','arm','comparator']},
                    support_row=int(i),support_label=int(labels[i]),old_similarity=float(kb[j]),new_similarity=float(ka[j]),
                    direct=float(direct[j]),refit=float(refit[j]),total=float(total[j])) for j,i in enumerate(s))
    pd.DataFrame(selected).to_csv(HERE/(prefix+'selected_cases.csv'),index=False)
    pd.DataFrame(paths).to_csv(HERE/(prefix+'case_paths.csv'),index=False)
    pd.DataFrame(contributions).to_csv(HERE/(prefix+'case_support_contributions.csv'),index=False)
    print(pd.DataFrame(selected)[['category','budget','row','arm','comparator','old_prediction','new_prediction','radar_valid']].to_string(index=False))


if __name__=='__main__':
    main()
