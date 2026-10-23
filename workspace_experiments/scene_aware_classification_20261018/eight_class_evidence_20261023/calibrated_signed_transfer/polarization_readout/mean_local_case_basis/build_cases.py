"""Six prespecified source-validation examples, without fitting or test selection."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'mean_only_control'))
import run_mean as mean_run
strong, parent = mean_run.strong, mean_run.parent
sys.path.insert(0, str(parent.HERE))
import analyze_results as common


def rival(score, truth):
    s = score.copy()
    s[truth] = -np.inf
    return int(s.argmax())


def main():
    torch.set_num_threads(2)
    table = parent.load_table()
    by_id = table.set_index('instance_id')
    cache = dict(np.load(parent.HERE/'signed_features.npz'))
    np.testing.assert_array_equal(cache['instance_ids'], table.instance_id)
    lookup = {v:i for i,v in enumerate(cache['instance_ids'])}
    names = list(cache['reader_names'])
    local = pd.read_csv(strong.HERE/'predictions.csv')
    local = local[local.arm.eq('local') & local.seed.eq(17) & local.role.eq('validation')]
    mean = pd.read_csv(mean_run.HERE/'predictions.csv')
    mean = mean[mean.seed.eq(17) & mean.role.eq('validation')]
    key = ['fold','instance_id']
    local, mean = local.set_index(key).sort_index(), mean.set_index(key).sort_index()
    np.testing.assert_array_equal(local.index, mean.index)
    np.testing.assert_array_equal(local.class_id, mean.class_id)
    np.testing.assert_array_equal(local[[f'b{c}' for c in range(8)]], mean[[f'b{c}' for c in range(8)]])
    choices, groups = [], []
    lc, mc = local.prediction.eq(local.class_id), mean.prediction.eq(mean.class_id)
    for c, short, dis in [(3,'chemical',mc & ~lc),(7,'offshore',lc & ~mc)]:
        for label, cond in [('disagree',dis),('CC',lc & mc),('WW',~lc & ~mc)]:
            h = local[local.class_id.eq(c) & cond].reset_index().sort_values(['instance_id','fold'])
            groups.append(dict(class_id=c,class_name=short,group=label,n=len(h),
                unique_observations=h.instance_id.nunique(),missing=not len(h)))
            if len(h):
                q=h.iloc[0]
                choices.append((short,label,int(q.fold),q.instance_id))
    pd.DataFrame(groups).to_csv(HERE/'selection_groups.csv',index=False)
    snapshots={s['obs_id']:s for s in json.loads((strong.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    (HERE/'panels').mkdir(exist_ok=True)
    cases, inputs, descriptors = [], [], []
    for short,label,fold,ident in choices:
        a,b = local.loc[(fold,ident)],mean.loc[(fold,ident)]
        row=by_id.loc[ident]; i=lookup[ident]; y=int(row.class_id)
        base=a[[f'b{c}' for c in range(8)]].to_numpy(float)
        ls=a[[f's{c}' for c in range(8)]].to_numpy(float)
        ms=b[[f's{c}' for c in range(8)]].to_numpy(float)
        ri=names.index(f'fold{fold}_outer')
        mu,mom,valid=cache['mean_pol'][ri,i],cache['moments_pol'][ri,i],cache['valid'][i].astype(bool)
        r0,rl,rm=[rival(s,y) for s in [base,ls,ms]]
        dlocal,dmean=ls-base,ms-base
        for arm,runner,slots,pred in [('local',strong,mom,ls),('mean_only',mean_run,np.repeat(mu[...,None],2,-1),ms)]:
            ck,model=runner.restore(runner.HERE/'models'/f'fold{fold}_{arm}_seed17.pt')
            x,carrier,_,_=runner.inputs(arm,base[None],slots[None],valid[None],ck['mu'],ck['scale'])
            with torch.inference_mode():
                replay=base+model(x,carrier).numpy()[0]
            assert np.abs(replay-pred).max()<1e-5 and replay.argmax()==pred.argmax()
            for c in range(8):
                descriptors.append(dict(instance_id=ident,fold=fold,arm=arm,class_id=c,
                    **{f'x{k}':float(x[0,c,k]) for k in range(11)}))
        fig,axes=plt.subplots(2,4,figsize=(19,10),layout='constrained')
        dimensions=[]
        for p,pol in enumerate(['vv','vh']):
            view=snapshots[int(row.obs_id)]['images'][pol]
            with Image.open(row[f'image_{pol}']) as im:raw=np.array(im.convert('L'))
            pix,masks,place=common.crop_pad(raw,common.masks_for(view),160)
            dimensions.append(f'{pol}:{raw.shape[1]}x{raw.shape[0]}')
            for col,pixels,title in [(0,raw,f'{pol.upper()} original {raw.shape[1]}x{raw.shape[0]}'),
                                    (1,pix,f'160 crop/pad; valid={valid[p]}; footprint={masks.sum(0).mean():.3f}')]:
                axes[p,col].imshow(pixels,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                axes[p,col].set_title(title,fontsize=10)
                axes[p,col].set_xlabel('pixel x');axes[p,col].set_ylabel('pixel y')
            prior=view['priors'];ti=prior.get('target_index');det=prior.get('detections',[])
            if ti is not None and 0<=ti<len(det):
                points=np.array(det[ti]['corners']);points=np.r_[points,points[:1]]
                axes[p,0].plot(*points.T,color='#13c74e',lw=1)
                points=points-[place['crop_x'],place['crop_y']]+[place['pad_left'],place['pad_top']]
                axes[p,1].plot(*points.T,color='#13c74e',lw=1)
            for value,name,color in [(mu[p],'mean','#1463a3'),(mom[p,:,0],'sigma','#d07d16'),(mom[p,:,1],'kappa','#9b3787')]:
                axes[p,3].plot(range(8),value,marker='o',ms=3,label=name,color=color)
            axes[p,3].axhline(0,color='gray',lw=.5);axes[p,3].set_xticks(range(8))
            axes[p,3].set_title(f'{pol.upper()} cached outer W responses',fontsize=10)
            axes[p,3].legend(fontsize=8)
            for c in range(8):
                inputs.append(dict(instance_id=ident,fold=fold,seed=17,role='validation',group=label,
                    true_class=y,class_id=c,pol=pol,valid=bool(valid[p]),mean=mu[p,c],
                    center_mean=mu[p,c]-mu[p].mean(),sigma=mom[p,c,0],
                    center_sigma=mom[p,c,0]-mom[p,:,0].mean(),kappa=mom[p,c,1],
                    center_kappa=mom[p,c,1]-mom[p,:,1].mean(),
                    mean_slot0=mu[p,c],mean_slot1=mu[p,c],local_slot0=mom[p,c,0],local_slot1=mom[p,c,1],
                    B=base[c],mean_score=ms[c],local_score=ls[c],mean_delta=dmean[c],local_delta=dlocal[c],
                    source_W=str(parent.PARENT/'baseline/models'/f'fold{fold}_outer.npz'),
                    native_image=row[f'image_{pol}']))
        xx=np.arange(8)
        for off,score,name,color in [(-.25,base,'Frozen B','#666666'),(0,ms,'Mean-only','#1463a3'),(.25,ls,'Local','#c26919')]:
            axes[0,2].bar(xx+off,score,width=.24,label=name,color=color)
        axes[0,2].set_xticks(xx);axes[0,2].legend(fontsize=8);axes[0,2].set_title('Actual class scores')
        axes[1,2].bar(xx-.16,dmean,width=.32,label='Mean direct delta',color='#1463a3')
        axes[1,2].bar(xx+.16,dlocal,width=.32,label='Local direct delta',color='#c26919')
        axes[1,2].set_xticks(xx);axes[1,2].axhline(0,color='gray',lw=.5)
        axes[1,2].legend(fontsize=8);axes[1,2].set_title('Direct residuals, not physical contributions')
        fig.suptitle(f'SOURCE validation seed17 / {short} {label} / {ident} / fold{fold} / {row.port}\n'
            f'true={y}; B/mean/local winner={base.argmax()}/{ms.argmax()}/{ls.argmax()}; '
            f'rival={r0}/{rm}/{rl}; epochs mean/local={int(b.epoch)}/{int(a.epoch)}\n'
            f'true margin B={base[y]-base[r0]:+.4f}, mean={ms[y]-ms[rm]:+.4f}, local={ls[y]-ls[rl]:+.4f}; '
            'same outer auxiliary W; no test-selected rule',fontsize=11)
        path=HERE/'panels'/f'{short}_{label}.png'
        fig.savefig(path,dpi=100);plt.close(fig)
        rec=dict(instance_id=ident,fold=fold,seed=17,role='validation',class_id=y,class_name=row.class_name,
            port=row.port,group=label,B_winner=int(base.argmax()),mean_winner=int(ms.argmax()),local_winner=int(ls.argmax()),
            B_rival=r0,mean_rival=rm,local_rival=rl,B_margin=base[y]-base[r0],
            mean_margin=ms[y]-ms[rm],local_margin=ls[y]-ls[rl],dimensions=';'.join(dimensions),
            image_vv=row.image_vv,image_vh=row.image_vh,panel=str(path))
        for name,delta,r in [('mean',dmean,rm),('local',dlocal,rl)]:
            rec.update({f'{name}_delta_true':delta[y],f'{name}_delta_Brival':delta[r0],
                f'{name}_fixed_Brival_margin_delta':delta[y]-delta[r0],
                f'{name}_delta_finalrival':delta[r],
                f'{name}_largest_raised_rival':rival(delta,y)})
        cases.append(rec)
    pd.DataFrame(cases).to_csv(HERE/'cases.csv',index=False)
    pd.DataFrame(inputs).to_csv(HERE/'case_inputs.csv',index=False)
    pd.DataFrame(descriptors).to_csv(HERE/'standardized_slots.csv',index=False)
    (HERE/'summary.json').write_text(json.dumps(dict(cases=len(cases),test_cases=0,training_runs=0,
        missing_groups=[g for g in groups if g['missing']],images_viewed=False,
        selection='seed17 source validation; first instance_id then smallest fold per class/group'),indent=2),encoding='utf-8')
    print(pd.DataFrame(groups).to_string(index=False))
    print(pd.DataFrame(cases).drop(columns=['image_vv','image_vh','panel']).to_string(index=False))


if __name__=='__main__':main()
