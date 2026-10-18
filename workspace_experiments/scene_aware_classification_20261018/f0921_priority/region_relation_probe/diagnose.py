"""Paired descriptive diagnostics and native-chip/support case panels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from evaluate import cosine, choose_support

HERE=Path(__file__).resolve().parent


def main(diagnostic=False):
    meta=pd.read_csv(HERE/'cohort.csv')
    pred=pd.read_csv(HERE/'predictions.csv')
    out=HERE
    if diagnostic:
        extra=pd.read_csv(HERE/'diagnostic_predictions.csv')
        extra['arm']='global_region'
        pred=pd.concat([pred[~pred.arm.eq('global_region')],extra],ignore_index=True)
        out=HERE/'diagnostic_review'
        out.mkdir(exist_ok=True)
    metrics=pd.read_csv(HERE/'metrics.csv')
    if diagnostic:
        extra_metrics=pd.read_csv(HERE/'diagnostic_metrics.csv')
        extra_metrics['arm']='global_region'
        metrics=pd.concat([metrics[~metrics.arm.eq('global_region')],extra_metrics],ignore_index=True)
    feat=dict(np.load(HERE/'features.npz'))
    data=np.load(HERE/'inputs.npz')
    metrics.groupby(['fold','arm'])[['oa','ba','map_r']].mean().to_csv(out/'by_acquisition.csv')
    pred.groupby(['arm','truth'])[['correct','ap','relation_margin']].mean().to_csv(out/'by_class.csv')
    # Keep support-draw variability separate from variation between acquisitions.
    metrics.groupby(['arm','draw'])[['oa','ba','map_r']].mean().groupby('arm').agg(['mean','std']).to_csv(out/'draw_variability.csv')
    r=pred[pred.arm.eq('global_region')].set_index(['fold','draw','row'])
    g=pred[pred.arm.eq('global_grid')].set_index(['fold','draw','row'])
    a=pred[pred.arm.eq('global')].set_index(['fold','draw','row'])
    joined=r[['correct','ap','relation_margin']].join(g[['correct','ap','relation_margin']],rsuffix='_grid')
    joined['delta_margin']=joined.relation_margin-joined.relation_margin_grid
    joined=joined.reset_index().merge(meta.reset_index(names='row')[['row','class_raw','image_other_fraction','image_rest_fraction']],on='row')
    joined['neighbor_present']=joined.image_other_fraction.gt(0)
    joined.groupby('neighbor_present')[['correct','correct_grid','ap','ap_grid','delta_margin']].agg(['mean','count']).to_csv(out/'neighbor_strata.csv')
    cases=[]
    groups=[('rescue',joined.correct&~joined.correct_grid,False),
            ('harm',~joined.correct&joined.correct_grid,True),
            ('both_correct',joined.correct&joined.correct_grid,False),
            ('both_wrong',~joined.correct&~joined.correct_grid,True)]
    for category,select,ascending in groups:
        subset=joined[select].sort_values(['delta_margin','row','draw'],ascending=[ascending,True,True])
        if subset.empty:continue
        case=subset.iloc[0];key=(case.fold,int(case.draw),int(case.row))
        rr=r.loc[key];gg=g.loc[key]
        indices=[key[2],int(rr.top_same_row),int(rr.top_wrong_row),int(gg.top_same_row),int(gg.top_wrong_row)]
        roles=['QUERY','REG same','REG confusing','GRID same','GRID confusing']
        canvas=Image.new('RGB',(1120,920),'white');d=ImageDraw.Draw(canvas)
        d.text((10,8),f'{category} | fold {key[0]} draw {key[1]} | true {int(rr.truth)} REG {int(rr.prediction)} GRID {int(gg.prediction)}',fill='black')
        for j,(idx,role) in enumerate(zip(indices,roles)):
            item=meta.iloc[idx];x=10+j*222
            d.text((x,35),f'{role} row {idx}',fill='black')
            d.text((x,52),f'class {item.label} / {item["product"][-4:]}',fill='black')
            d.text((x,69),f'MMSI {item.mmsi}',fill='black')
            for pol,y in [('vv',95),('vh',355)]:
                im=Image.open(item['native_'+pol]).convert('RGB')
                # Uniform nearest enlargement for visibility; native dimensions recorded.
                im.thumbnail((210,220))
                if max(im.size)<110:im=im.resize((im.width*2,im.height*2),Image.Resampling.NEAREST)
                canvas.paste(im,(x,y+20));d.text((x,y),f'{pol.upper()} native {item.native_width}x{item.native_height}',fill='black')
            mask=data['regions'][idx].argmax(0)
            colors=np.array([[230,70,70],[240,170,40],[70,150,220],[140,140,140]],np.uint8)
            canvas.paste(Image.fromarray(colors[mask]).resize((200,200),Image.Resampling.NEAREST),(x,640))
            d.text((x,850),f'other {item.image_other_fraction:.2f}',fill='black')
            d.text((x,867),f'rest {item.image_rest_fraction:.2f}',fill='black')
            cases.append(dict(category=category,role=role,fold=key[0],draw=key[1],query_row=key[2],row=idx,
                              object_id=item.object_id,class_raw=item.class_raw,mmsi=int(item.mmsi),
                              native_vv=item.native_vv,native_vh=item.native_vh,
                              region_prediction=int(rr.prediction),grid_prediction=int(gg.prediction)))
        d.text((10,897),'Masks: red target candidate; orange other candidates; blue map water; gray rest/unknown. Not physical truth.',fill='black')
        canvas.save(out/f'case_{category}.png')
    pd.DataFrame(cases).to_csv(out/'case_index.csv',index=False)
    if diagnostic:return
    area=feat['area'][:,0]
    empties=(area==0).sum(0)
    stat=dict(empty_regions=empties.tolist(),n_observations=len(meta),
              rescue_region_vs_grid=int((joined.correct&~joined.correct_grid).sum()),
              harm_region_vs_grid=int((~joined.correct&joined.correct_grid).sum()),
              repeated_query_events=len(joined),unique_queries=meta.shape[0])
    # Decompose centered descriptor energy: absent regions become -support mean.
    energy=[]
    for product in sorted(meta['product'].unique()):
        q=np.flatnonzero(meta['product'].eq(product));src=np.flatnonzero(~meta['product'].eq(product))
        for seed in range(5):
            s=choose_support(src,meta.label.to_numpy(),seed)
            z=(feat['region']-feat['region'][s].mean(0)).reshape(len(meta),2,4,-1)
            block=np.square(z).sum((1,3))
            empty_share=(block*(area==0)).sum(1)/np.maximum(block.sum(1),1e-12)
            for i in q:energy.append(dict(row=int(i),fold=product[-4:],draw=seed,empty_centered_energy=float(empty_share[i])))
    pd.DataFrame(energy).to_csv(HERE/'empty_region_energy.csv',index=False)
    stat['mean_empty_centered_energy']=float(pd.DataFrame(energy).empty_centered_energy.mean())
    stat['max_empty_centered_energy']=float(pd.DataFrame(energy).empty_centered_energy.max())
    (HERE/'diagnostics.json').write_text(json.dumps(stat,indent=2),encoding='utf-8')
    print(json.dumps(stat,indent=2))
    print(metrics.groupby(['fold','arm'])[['oa','ba','map_r']].mean().to_string())


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--diagnostic',action='store_true')
    main(parser.parse_args().diagnostic)
