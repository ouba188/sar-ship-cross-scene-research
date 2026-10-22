"""Replay saved linear/projection caches and display four native test cases."""
import json
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from run_baseline import HERE, ROOT, load, isolated, weights, metrics

sys.path.insert(0,str(ROOT/'same_vessel_gallery_20261019/local_condition_module'))
from native_input import crop_pad
from prepare import masks_for


def replay_projection(x, model, prefix=''):
    # Match sklearn's float32 transform operation order, including casts.
    x=x.copy()
    x-=model[prefix+'scale_mean'].astype(x.dtype)
    x/=model[prefix+'scale_scale'].astype(x.dtype)
    x=x@model[prefix+'pca_components'].T-model[prefix+'pca_mean'][None]@model[prefix+'pca_components'].T
    x-=model[prefix+'post_mean'].astype(x.dtype)
    x/=model[prefix+'post_scale'].astype(x.dtype)
    return x


def verify(f,raw):
    rows=[]
    for fold in range(3):
        d=dict(np.load(HERE/f'fold{fold}_inputs.npz'))
        np.testing.assert_array_equal(d['instance_ids'],f.instance_id)
        for k,v in d.items():
            if v.dtype.kind in 'fc':
                assert np.isfinite(v).all(),k
        np.testing.assert_allclose(d['q'].sum(1)+d['b'],d['base'],atol=1e-7)
        np.testing.assert_allclose(d['oq'].sum(1)+d['ob'],d['oof'],atol=1e-7)
        np.testing.assert_array_equal(np.flatnonzero(d['oof_valid']),d['train'])
        np.testing.assert_array_equal(d['valid'],raw['valid'])
        np.testing.assert_array_equal(d['regional_valid'],raw['regional_valid'])
        for inner in range(-1,4):
            tag='outer' if inner<0 else f'inner{inner}'
            model=dict(np.load(HERE/'models'/f'fold{fold}_{tag}.npz'))
            fit=model['fit']
            take=np.arange(len(f)) if inner<0 else model['held']
            if inner>=0:
                isolated(f,fit,take)
                np.testing.assert_array_equal(take,np.flatnonzero(d['inner_bin']==inner))
            else:
                np.testing.assert_array_equal(fit,d['train'])
            np.testing.assert_allclose(model['h0'],raw['layer3'][fit].mean(0,dtype=np.float64),atol=1e-12)
            np.testing.assert_allclose(model['h0_weighted'],np.average(raw['layer3'][fit].astype(np.float64),axis=0,weights=weights(f,fit)),atol=1e-12)
            # Scaler statistics identify exactly which source rows were fitted.
            np.testing.assert_allclose(model['scale_mean'],raw['layer3'][fit].mean(0,dtype=np.float64),atol=1e-12)
            np.testing.assert_allclose(model['cond_scale_mean'],raw['stem'][fit].mean(0,dtype=np.float64),atol=1e-12)
            sample=take[np.linspace(0,len(take)-1,31,dtype=int)]
            z=replay_projection(raw['layer3'][sample].astype(np.float64),model)
            score=(z@model['head_coef'].T+model['head_intercept'])*3
            cached=d['base' if inner<0 else 'oof'][sample]
            linear_error=float(np.max(np.abs(score-cached)))
            assert linear_error<1e-7
            cond={k:replay_projection(raw[k][sample],model,'cond_') for k in ('stem','fg','bg')}
            norm=lambda x:x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)*np.sqrt(32.)
            whole=norm(cond['stem']).astype(np.float32)
            fg=norm(cond['fg']).astype(np.float32)
            whole[~raw['valid'][sample].any(1)]=0
            fg[~raw['regional_valid'][sample,:,0].any(1)]=0
            parts=[]
            for j,key in enumerate(('fg','bg')):
                part=norm(cond[key][:,:32]).astype(np.float32)
                part[~raw['regional_valid'][sample,:,j].any(1)]=0
                parts.append(part)
            dual=(np.concatenate(parts,axis=1)/np.sqrt(2.)).astype(np.float32)
            prefix='' if inner<0 else 'oof_'
            projection_error=max(float(np.max(np.abs(a-d[prefix+k][sample]))) for k,a in [('whole64',whole),('fg64',fg),('dual64',dual)])
            assert projection_error<1e-5,projection_error
            rows.append(dict(fold=fold,inner=inner,linear_error=linear_error,projection_error=projection_error,checked_rows=len(sample)))
        missing=~raw['valid'][:,1]&raw['valid'][:,0]
        assert (np.linalg.norm(d['whole64'][missing],axis=1)>1).all()
        assert (np.linalg.norm(d['fg64'][missing],axis=1)>1).all()
    pd.DataFrame(rows).to_csv(HERE/'serialized_replay.csv',index=False)
    return rows


def review(f,raw):
    p=pd.read_csv(HERE/'predictions.csv',dtype={'mmsi':str})
    p=p[p.role.eq('test')]
    snapshots={v['obs_id']:v for v in json.loads((ROOT/'same_vessel_gallery_20261019/local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    names=f.groupby('class_id').class_name.first().to_dict()
    out=HERE/'review'
    out.mkdir(exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
    rows=[]
    for correct,cls in [(True,0),(True,4),(False,3),(False,2)]:
        candidates=p[p.class_id.eq(cls)&p.prediction.eq(p.class_id).eq(correct)]
        row=candidates.sort_values('obs_id').iloc[0]
        i=int(row.row_index)
        scores=row[[f's{c}' for c in range(8)]].to_numpy(dtype=float)
        truth=int(row.class_id)
        other=scores.copy();other[truth]=-np.inf
        margin=float(scores[truth]-other.max())
        status='correct' if correct else 'wrong'
        canvas=Image.new('RGB',(748,254),'white')
        draw=ImageDraw.Draw(canvas)
        draw.text((8,5),f'{status.upper()} | {row.instance_id} | obs {row.obs_id} | fold {row.outer_fold}',font=font,fill='black')
        draw.text((8,23),f'Truth: {names[truth]} | Pred: {names[int(row.prediction)]} | true margin {margin:+.4f}',font=font,fill='black')
        draw.text((8,41),f'{row.port} | {row.time} | Original 160 crop/pad; no resizing',font=font,fill='black')
        for pol,polname in enumerate(('vv','vh')):
            src=row['image_'+polname]
            original=np.asarray(Image.open(src).convert('L'))
            pixels,masks,place=crop_pad(original,masks_for(snapshots[int(row.obs_id)]['images'][polname]),160)
            fg=masks[0]>0
            occupied=torch.from_numpy(masks[:2].sum(0))[None,None]
            dilated=F.max_pool2d(occupied,7,stride=1,padding=3)[0,0].numpy()>0
            bg=(masks.sum(0)>0)&~dilated
            rgb=np.repeat(pixels[...,None],3,-1).astype(float)
            color=np.zeros_like(rgb);color[fg]=[40,255,70];color[bg]=[30,220,255]
            alpha=.25*(fg|bg)[...,None]
            overlay=np.uint8(rgb*(1-alpha)+color*alpha)
            valid=bool(raw['valid'][i,pol])
            x=8+pol*370
            draw.text((x,62),polname.upper()+(' used' if valid else ' NOT USED'),font=font,fill='black' if valid else '#b91c1c')
            draw.text((x+178,62),'Actual pooling mask',font=font,fill='black')
            canvas.paste(Image.fromarray(pixels).convert('RGB'),(x,82))
            canvas.paste(Image.fromarray(overlay),(x+178,82))
            rows.append(dict(status=status,instance_id=row.instance_id,obs_id=int(row.obs_id),fold=int(row.outer_fold),
                class_id=truth,class_name=names[truth],prediction=int(row.prediction),predicted_class=names[int(row.prediction)],
                true_margin=margin,pol=polname,valid=valid,image_path=src,panel_path=str(out/f'{status}_{row.instance_id}.png'),
                **place,**{f's{c}':float(scores[c]) for c in range(8)}))
        canvas.save(out/f'{status}_{row.instance_id}.png')
    pd.DataFrame(rows).to_csv(HERE/'review_cases.csv',index=False)
    return rows


def main():
    start=time.perf_counter()
    f,raw=load()
    replay=verify(f,raw)
    cases=review(f,raw)
    summary=dict(serialized_models_checked=len(replay),panels=len(cases)//2,
        max_linear_error=max(x['linear_error'] for x in replay),
        max_projection_error=max(x['projection_error'] for x in replay),
        seconds=time.perf_counter()-start,passed=True)
    (HERE/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print(pd.DataFrame(cases)[['status','instance_id','class_name','predicted_class','pol','valid','true_margin','panel_path']].to_string(index=False))


if __name__=='__main__':
    main()
