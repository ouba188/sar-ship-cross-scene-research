"""Paired outcomes, bounded raw-chip review and actual strip response plots."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from run import ROOT, PREV, LOCAL, ARMS, LENGTH, StripResidual, prior, old, strip_layout, shuffle


def main():
    torch.set_num_threads(2)
    f=pd.read_csv(prior.ROOT/'cohort.csv',dtype={'mmsi':str}); index={int(v):i for i,v in enumerate(f.obs_id)}
    names=f.groupby('class_id').class_name.first().to_dict()
    pred=pd.read_csv(ROOT/'predictions.csv',dtype={'mmsi':str})
    replay=[]
    for fold in range(3):
        for seed in range(3):
            current=np.load(ROOT/f'scores_f{fold}_frozen_fivepoint_s{seed}.npz')
            previous=np.load(PREV/'mse_aligned'/f'scores_f{fold}_keep_refs_s{seed}.npz')
            np.testing.assert_array_equal(current['rows'],previous['rows'])
            error=float(np.max(np.abs(current['scores']-previous['scores'])))
            assert error < 1e-5, error
            replay.append(dict(fold=fold,seed=seed,max_abs=error))
    pd.DataFrame(replay).to_csv(ROOT/'frozen_score_replay.csv',index=False)
    coverage=pd.read_csv(ROOT/'coverage.csv')
    cov=coverage.pivot(index='obs_id',columns='pol',values=['radar_valid','disabled','length_px'])
    rows=[]; groups=[]; classes=[]
    for (arm,seed),part in pred.groupby(['arm','seed']):
        part=part.sort_values(['fold','obs_id']); scores=[]; yy=[]
        for fold in range(3):
            data=np.load(ROOT/f'scores_f{fold}_{arm}_s{seed}.npz')
            scores.append(data['scores']); yy.extend(f.iloc[data['rows']].class_id.tolist())
        basecorrect=part.base_prediction.eq(part.class_id)
        rows.append(dict(arm=arm,seed=seed,rescued=int((part.correct&~basecorrect).sum()),
            harmed=int((~part.correct&basecorrect).sum()),net=int(part.correct.sum()-basecorrect.sum()),
            **prior.metrics(np.array(yy),np.concatenate(scores))))
        for cls,cc in part.groupby('class_id'):
            classes.append(dict(arm=arm,seed=seed,class_id=cls,name=names[cls],n=len(cc),
                recall=cc.correct.mean(),base_recall=cc.base_prediction.eq(cc.class_id).mean()))
        length=cov.loc[part.obs_id,('length_px','vv')].to_numpy()/4
        good=(cov.loc[part.obs_id,('radar_valid','vh')]&~cov.loc[part.obs_id,('disabled','vh')]).to_numpy(bool)
        flags={'vv_lt4cells':length<4,'vv_4to8cells':(length>=4)&(length<8),'vv_ge8cells':length>=8,
               'both_radar_and_vh_enabled':good,'vh_radar_fallback_or_disabled':~good}
        for label,mask in flags.items():
            cc=part.iloc[np.flatnonzero(mask)]
            base=cc.base_prediction.eq(cc.class_id)
            ba=cc.groupby('class_id').correct.mean().mean()
            bba=cc.assign(base_ok=base).groupby('class_id').base_ok.mean().mean()
            groups.append(dict(arm=arm,seed=seed,group=label,n=len(cc),classes=cc.class_id.nunique(),
                oa=cc.correct.mean(),ba=ba,base_oa=base.mean(),base_ba=bba,
                delta_oa=cc.correct.mean()-base.mean(),delta_ba=ba-bba))
    per=pd.DataFrame(rows); per.to_csv(ROOT/'per_seed.csv',index=False)
    summary=per.groupby('arm')[['oa','ba','brier','net']].agg(['mean','std'])
    summary.to_csv(ROOT/'summary.csv')
    pd.DataFrame(groups).to_csv(ROOT/'groups.csv',index=False)
    pd.DataFrame(classes).to_csv(ROOT/'classes.csv',index=False)
    pairs=[]
    for other in ('frozen_fivepoint','strip_mean','strip_shuffle'):
        for seed in range(3):
            a=pred[pred.arm.eq('strip_conv')&pred.seed.eq(seed)].set_index('obs_id')
            b=pred[pred.arm.eq(other)&pred.seed.eq(seed)].set_index('obs_id').loc[a.index]
            pairs.append(dict(seed=seed,comparison=other,rescued=int((a.correct&~b.correct).sum()),
                harmed=int((~a.correct&b.correct).sum()),net=int(a.correct.sum()-b.correct.sum())))
    pd.DataFrame(pairs).to_csv(ROOT/'paired_conv.csv',index=False)
    # Select representative seed0 cases by score changes, not visual attractiveness.
    a=pred[pred.arm.eq('strip_conv')&pred.seed.eq(0)].set_index('obs_id')
    b=pred[pred.arm.eq('frozen_fivepoint')&pred.seed.eq(0)].set_index('obs_id').loc[a.index]
    info={}
    for fold in range(3):
        d=np.load(ROOT/f'scores_f{fold}_strip_conv_s0.npz')
        for j,i in enumerate(d['rows']):
            y=int(f.iloc[i].class_id); rival=int(d['base'][j].argmax())
            if rival==y: rival=int(np.argmax(np.where(np.arange(8)==y,-np.inf,d['base'][j])))
            before=float(d['base'][j,y]-d['base'][j,rival]); after=float(d['scores'][j,y]-d['scores'][j,rival])
            info[int(f.iloc[i].obs_id)]=(fold,rival,before,after)
    a['change']=[info[int(i)][3]-info[int(i)][2] for i in a.index]
    chosen=[]
    for label,mask,ascending in (('rescue',a.correct&~b.correct,False),('harm',~a.correct&b.correct,True),
                                ('persistent_correct',a.correct&b.correct,False),('persistent_wrong',~a.correct&~b.correct,True)):
        subset=a[mask].sort_values('change',ascending=ascending)
        for oid in subset.head(2 if label in ('rescue','harm') else 1).index: chosen.append((label,int(oid)))
    for oid in (8899,1719):
        if oid in a.index and oid not in [c[1] for c in chosen]: chosen.append(('predefined',oid))
    views={v['obs_id']:v for v in json.loads((LOCAL/'view_snapshot.json').read_text(encoding='utf-8'))}
    cache=np.load(ROOT/'tokens.npy',mmap_mode='r'); masses=np.load(ROOT/'masses.npy',mmap_mode='r')
    perms=np.load(ROOT/'permutations.npy',mmap_mode='r'); records=[]; contributions=[]
    for label,oid in chosen:
        i=index[oid]; row=f.iloc[i]; fold,rival,before,after=info[oid]
        head=torch.load(ROOT/f'head_f{fold}_strip_conv_s0.pt',map_location='cpu',weights_only=False)
        model=StripResidual(True); model.load_state_dict(head['state']); model.eval()
        x=(torch.tensor(np.array(cache[i:i+1]),dtype=torch.float32)-head['mean'])/head['std']
        m=torch.tensor(np.array(masses[i:i+1]),dtype=torch.float32)
        pp=torch.tensor(np.array(perms[i:i+1]),dtype=torch.int64)
        with torch.inference_mode():
            full=model(x,m)[0]; sx,sm=shuffle(x,m,pp); shuffled=model(sx,sm)[0]
            for pol in range(2):
                for ref in range(2):
                    altered=m.clone(); altered[:,pol,:,ref]=0
                    change=full-model(x,altered)[0]
                    contributions.append(dict(obs_id=oid,pol=('vv','vh')[pol],reference=('ship','radar_or_fallback')[ref],
                        rival=rival,margin_term=float(change[int(row.class_id)]-change[rival])))
        fig,axs=plt.subplots(2,3,figsize=(15,9))
        fig.suptitle(f'{label} obs {oid} | {row.class_name} | {row.port}\nfrozen fivepoint {names[int(b.loc[oid,"prediction"])]} -> strip conv {names[int(a.loc[oid,"prediction"])]}, seed0 step{head["step"]}')
        for polidx,pol in enumerate(('vv','vh')):
            view=views[oid]['images'][pol]; raw=np.asarray(Image.open(row['image_'+pol]).convert('L'))
            pix,mm,place=old.crop_pad(raw,old.masks_for(view),160)
            co,valid,_=strip_layout(view,place)
            co=co-np.array([place['pad_left']-place['crop_x'],place['pad_top']-place['crop_y']])
            axs[polidx,0].imshow(raw,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            if mm[0].any(): axs[polidx,0].contour(old.masks_for(view)[0],[.5],colors=['lime'],linewidths=.8)
            enabled=not bool(coverage[coverage.obs_id.eq(oid)&coverage.pol.eq(pol)].iloc[0].disabled)
            if enabled:
                for ref,color in enumerate(('cyan','magenta')):
                    for lane in range(3):
                        points=co[ref,lane,valid[ref].astype(bool)]
                        axs[polidx,0].plot(points[:,0],points[:,1],'.-',color=color,lw=.5,ms=2,alpha=.7 if lane==0 else .35)
            axs[polidx,0].set_xlim(-.5,raw.shape[1]-.5); axs[polidx,0].set_ylim(raw.shape[0]-.5,-.5)
            axs[polidx,0].set_title(f'{pol.upper()} native {raw.shape[1]}x{raw.shape[0]} | '+('enabled' if enabled else 'NEW BRANCH DISABLED'))
            for ref in range(2):
                ax=axs[polidx,ref+1]
                for lane in range(3):
                    response=np.asarray(cache[i,polidx,0,ref,lane],dtype=np.float32).mean(-1)
                    validmass=np.asarray(masses[i,polidx,0,ref,lane]).sum(-1)>.05
                    xx=(np.arange(LENGTH)-LENGTH//2)*4
                    ax.plot(xx,np.where(validmass,response,np.nan),'.-',label=('center','side -','side +')[lane],lw=1,ms=3)
                ax.set_title(('Ship-axis','Radar / image fallback')[ref]+' stem mean, rotation0')
                ax.set_xlabel('Signed offset (native pixels; NOT bow/stern)'); ax.set_ylabel('Mean over 64 frozen channels'); ax.legend(fontsize=8)
        fig.tight_layout(rect=(0,0,1,.93)); fig.savefig(ROOT/f'case_{label}_{oid}.png',dpi=120); plt.close(fig)
        records.append(dict(outcome=label,obs_id=oid,fold=fold,seed=0,step=head['step'],class_name=row.class_name,port=row.port,
            base_prediction=int(b.loc[oid,'prediction']),prediction=int(a.loc[oid,'prediction']),rival=rival,
            before_margin=before,after_margin=after,shuffled_margin=before+float(shuffled[int(row.class_id)]-shuffled[rival]),
            image_vv=row.image_vv,image_vh=row.image_vh))
    pd.DataFrame(records).to_csv(ROOT/'case_review.csv',index=False)
    pd.DataFrame(contributions).to_csv(ROOT/'case_reference_terms.csv',index=False)
    print(summary.to_string()); print('cases',chosen)


if __name__=='__main__': main()
