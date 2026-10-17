"""Actual SAR pixels, all supports and exact margin paths for changed queries."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

import run_effect as run
from body_context_gate import _coefficients

OUT=run.HERE/'visual_diagnosis'
PIXELS=Path('D:/SAR_completion_cache_20261010/source_inputs128.npy')
MANIFEST=run.NEW/'geometry_input_manifest.csv'
REGION=run.HERE.parent/'region_cue_replication_20261013/ports/frame.csv'
NAMES=['bulk','fishing','general cargo','product/chemical','container','crude tanker','tug','offshore']


def show(ax,image,title):
    ax.imshow(image,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
    ax.set_title(title,fontsize=8); ax.set_xticks([]); ax.set_yticks([])


@torch.no_grad()
def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    events=pd.read_csv(run.HERE/'analysis/changed_cases.csv',dtype={'mmsi':str})
    selected=events.loc[(events.family=='strict')&(events.origin=='cross')&(events.seed==0)].sort_values(['global_row','draw'])
    assert len(selected)==3 and selected.global_row.nunique()==2
    manifest=pd.read_csv(MANIFEST,low_memory=False)
    regions=pd.read_csv(REGION).set_index('sample_id')
    pixels=np.load(PIXELS,mmap_mode='r')
    data=run.load('Singapore',True); run.prepare(data,0); run.set_input(data,'strict','new')
    model=run.model_from(run.HERE,'Singapore','strict',0)
    raw=run.base.raw_descriptors(model,data); d=data['tensors']
    fig,axes=plt.subplots(3,6,figsize=(18,9))
    supports=[]; cases=[]; pieces=[]
    for ci,row in enumerate(selected.itertuples()):
        item=data['cache']['target'][row.task]
        qrows=data['meta'].iloc[item['q'].cpu().numpy()].global_row.to_numpy()
        qi=int(np.flatnonzero(qrows==row.global_row)[0]); sids=item['s'].cpu().numpy()
        prob,qs,h=run.fit.predict(model,data,item,raw=raw)
        np.testing.assert_allclose(prob[qi].cpu().numpy(),[getattr(row,f'p{i}') for i in range(8)],atol=1e-10,rtol=0)
        ns=len(sids); sy=item['sy'].cpu().numpy(); y=int(row.truth)
        rival=int(row.pred if row.harm else row.pred_old)
        oldc=_coefficients(item['ss'],item['prior_s'],item['sy'])
        newc=_coefficients(item['ss']+.75*(h[:ns]@h[:ns].T),item['prior_s'],item['sy'])
        direct=(qs[qi]-item['qs'][qi])*(oldc[:,y]-oldc[:,rival])
        resolve=qs[qi]*((newc[:,y]-newc[:,rival])-(oldc[:,y]-oldc[:,rival]))
        exact=qs[qi]@(newc[:,y]-newc[:,rival])-item['qs'][qi]@(oldc[:,y]-oldc[:,rival])
        torch.testing.assert_close(direct.sum()+resolve.sum(),exact,atol=1e-10,rtol=1e-10)
        total=(direct+resolve).cpu().numpy()
        case=f'{row.mmsi}_draw{row.draw}'
        counts=np.bincount(sy,minlength=8)
        valid=d['valid'][item['s']].cpu().numpy()
        valid_counts=np.bincount(sy[valid],minlength=8)
        cases.append(dict(case=case,global_row=row.global_row,mmsi=row.mmsi,draw=row.draw,task=row.task,
            truth=y,rival=rival,pred_old=row.pred_old,pred_new=row.pred,
            old_margin=float(item['old_prob'][qi,y]-item['old_prob'][qi,rival]),
            new_margin=float(prob[qi,y]-prob[qi,rival]),direct=float(direct.sum()),resolve=float(resolve.sum()),
            expert_margin_delta=float(exact),true_supports=int(counts[y]),valid_true_supports=int(valid_counts[y]),
            rival_supports=int(counts[rival]),valid_rival_supports=int(valid_counts[rival]),valid_supports=int(valid.sum())))
        sign='HARM' if row.harm else 'RESCUE'
        for pol in range(2):
            show(axes[ci,pol],pixels[row.global_row,pol],f'{case} {sign}\nquery {NAMES[y]} {"VV" if pol==0 else "VH"}')
        for block,si,title in [(1,int(np.argmax(total)),'strongest +margin'),(2,int(np.argmin(total)),'strongest -margin')]:
            grow=int(data['meta'].iloc[sids[si]].global_row)
            for pol in range(2):
                show(axes[ci,2*block+pol],pixels[grow,pol],
                     f'{title} {total[si]:+.4f}\n{NAMES[sy[si]]} valid={int(valid[si])} {"VV" if pol==0 else "VH"}')
        allfig,allaxes=plt.subplots(4,8,figsize=(18,10))
        for si,idx in enumerate(sids):
            grow=int(data['meta'].iloc[idx].global_row)
            show(allaxes.flat[si],np.concatenate([pixels[grow,0],pixels[grow,1]],axis=1),
                 f'{si}: {NAMES[sy[si]]} v={int(valid[si])}\ncontribution {total[si]:+.4f}')
            supports.append(dict(case=case,query_row=row.global_row,support_row=grow,
                sample_id=data['meta'].iloc[idx].sample_id,mmsi=data['meta'].iloc[idx].mmsi,
                support_class=int(sy[si]),valid=bool(valid[si]),direct=float(direct[si]),resolve=float(resolve[si]),total=float(total[si])))
            block_kernel=(.75*h[ns+qi].reshape(3,4,16)*h[si].reshape(3,4,16)).sum(-1)
            torch.testing.assert_close(block_kernel.sum(),qs[qi,si]-item['qs'][qi,si],rtol=1e-7,atol=1e-10)
            for ring in range(3):
                for slot in range(4):
                    pieces.append(dict(case=case,support_row=grow,ring=ring,slot=slot,
                        kernel=float(block_kernel[ring,slot]),
                        direct_margin=float(block_kernel[ring,slot]*(oldc[si,y]-oldc[si,rival]))))
        allfig.suptitle(case+' | All 32 supports, VV | VH. Contributions are algebraic, not causal.',fontsize=12)
        allfig.tight_layout(rect=(0,0,1,.96)); allfig.savefig(OUT/(case+'_all_supports.png'),dpi=140); plt.close(allfig)
    fig.suptitle('Actual cached SAR inputs, fixed display scale; support selection by exact margin contribution',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.97)); fig.savefig(OUT/'query_support_cases.png',dpi=140); plt.close(fig)
    stats=[]
    queryrows=sorted(selected.global_row.unique())
    fig,axes=plt.subplots(2,4,figsize=(13,7))
    for i,grow in enumerate(queryrows):
        m=manifest.iloc[grow]; r=regions.loc[m.instance_id]
        yy=(np.arange(56)+.5)/56; xx=yy
        box=np.zeros((56,56),bool)
        for pol in ('vv','vh'):
            box|=((xx[None]>=r['bbox_x0_'+pol]/r['native_w_'+pol])&(xx[None]<r['bbox_x1_'+pol]/r['native_w_'+pol])&
                  (yy[:,None]>=r['bbox_y0_'+pol]/r['native_h_'+pol])&(yy[:,None]<r['bbox_y1_'+pol]/r['native_h_'+pol]))
        expanded=[box]+[binary_dilation(box,iterations=n) for n in (2,8,16)]
        masks=[box]+[expanded[j]&~expanded[j-1] for j in range(1,4)]
        for j,pol in enumerate(('vv','vh')):
            with Image.open(m['image_relpath_'+pol]) as im:
                native=np.asarray(im.convert('L')); native_hw=im.size
            show(axes[i,j],native,f'{m.instance_id}\nnative {pol.upper()} {native_hw}')
            show(axes[i,j+2],pixels[grow,j],f'cached {pol.upper()}: body/touch/near/far boundaries')
            for mask,color in zip(expanded,['cyan','lime','orange','magenta']):
                axes[i,j+2].contour(mask.astype(float),levels=[.5],colors=[color],linewidths=.7,extent=(-.5,127.5,127.5,-.5))
            for ri,mask in enumerate(masks):
                mask128=np.asarray(Image.fromarray(mask).resize((128,128),Image.Resampling.NEAREST),bool)
                v=pixels[grow,j][mask128]
                stats.append(dict(global_row=grow,pol=pol,region=ri,pixels=len(v),mean=float(v.mean()),std=float(v.std()),
                                  display_bright_fraction=float((v>=200).mean()),display_saturated_fraction=float((v==255).mean())))
    fig.tight_layout(); fig.savefig(OUT/'query_regions.png',dpi=160); plt.close(fig)
    pd.DataFrame(cases).to_csv(OUT/'case_accounting.csv',index=False)
    pd.DataFrame(supports).to_csv(OUT/'all_support_contributions.csv',index=False)
    pd.DataFrame(pieces).to_csv(OUT/'region_slot_contributions.csv',index=False)
    pd.DataFrame(stats).to_csv(OUT/'display_region_stats.csv',index=False)
    print(pd.DataFrame(cases).to_string(index=False))
    print(pd.DataFrame(pieces).groupby(['case','ring']).direct_margin.sum().to_string())


if __name__=='__main__':
    main()
