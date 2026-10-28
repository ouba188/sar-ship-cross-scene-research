"""Trace actual early-M2 writes and paired full/bypass decisions on native chips."""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from run import ROOT, RELEASE, ARMS, Cache, construct, setup, read_manifest, pair_mask
from sar_m123.current.data import prepare_record


def main():
    setup('cuda')
    font_manager.fontManager.addfont('C:/Windows/Fonts/simhei.ttf')
    plt.rcParams.update({'font.family':'SimHei','axes.unicode_minus':False})
    data=np.load(ROOT/'predictions.npz')
    y=data['labels'];scores=data['scores'];base=scores[0,0].argmax(1)
    records=read_manifest(RELEASE/'manifests/development.json')
    refs=read_manifest(RELEASE/'manifests/references.json')
    state=torch.load(RELEASE/'weights/teacher.pt',map_location='cpu',weights_only=True)['state']
    cache=Cache('dev');bank=Cache('refs').get(np.arange(len(refs)))
    output=[]
    panels=ROOT/'update_panels';panels.mkdir(exist_ok=True)
    for ai,arm in enumerate(ARMS):
        if arm=='baseline':continue
        model=construct(state,ARMS[arm],17)
        delta=torch.load(ROOT/'checkpoints'/f'{arm}_17.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(delta['state'],strict=False)
        pred=scores[ai,0].argmax(1)
        categories=dict(rescued=(base!=y)&(pred==y),harmed=(base==y)&(pred!=y),
                        correct=(base==y)&(pred==y),wrong=(base!=y)&(pred!=y))
        for kind,mask in categories.items():
            candidates=np.flatnonzero(mask)
            if not len(candidates):continue
            # Prefer a change in the true-class margin; still descriptive, not validation selection.
            margins=lambda z:z[np.arange(len(y)),y]-np.where(np.eye(8)[y].astype(bool),-np.inf,z).max(1)
            changes=margins(scores[ai,0])-margins(scores[0,0])
            i=int(candidates[np.argmax(np.abs(changes[candidates]))])
            row=records[i];q=cache.get([i]);legal=pair_mask([row],refs,'cuda')
            with torch.no_grad():score,detail=model(q,bank,legal,True)
            assert np.max(abs(score.cpu().numpy()[0]-scores[ai,0,i]))<1e-4
            _,views=prepare_record(row,True)
            fig,axes=plt.subplots(2,3,figsize=(12,8))
            stats=[]
            for p,pol in enumerate(('VV','VH')):
                view=views[p]
                for ax in axes[p]:ax.set_xticks([]);ax.set_yticks([])
                if view is None or not q['available'][0,p]:
                    for ax in axes[p]:ax.text(.5,.5,f'{pol} 缺失',ha='center');ax.axis('off')
                    continue
                axes[p,0].imshow(view['pixels'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                axes[p,0].set_title(f'{pol} 原生裁剪 / 补零（不缩放）')
                for col,stage in enumerate((1,2),1):
                    ax=axes[p,col]
                    if stage not in ARMS[arm]:
                        ax.text(.5,.5,f'未插入 layer{stage}',ha='center');ax.axis('off');continue
                    d=detail['trace'][f'early_{stage}'][p]
                    update=d['update'][0].norm(dim=0).detach().cpu().numpy()
                    im=ax.imshow(update,cmap='magma',interpolation='nearest')
                    ax.set_title(f'{pol} layer{stage} 实际回写 L2\n{update.shape[0]}×{update.shape[1]} 特征网格')
                    fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
                    weights=d['write_weights'][0].detach().cpu().numpy()
                    norm_by_region=(weights*update).sum((1,2))/weights.sum((1,2)).clip(1e-8)
                    stats.append(dict(pol=pol,stage=stage,max_update_l2=float(update.max()),
                                      regional_mean_l2=dict(zip(('body','end','side'),norm_by_region.tolist()))))
            full=int(scores[ai,0,i].argmax());off=int(data['bypasses'][ai,0,i].argmax())
            title=f'{arm} / seed17 / {kind} / ID {row["id"]}: true={y[i]}, baseline={base[i]}, full={full}, early-off={off}'
            fig.suptitle(title,fontsize=12)
            fig.text(.5,.015,'颜色只表示实际回写幅度，不表示对真实类别的正贡献，也不识别物理散射来源。',ha='center',fontsize=10)
            fig.tight_layout(rect=(0,.04,1,.95))
            path=panels/f'{arm}_{kind}_{row["id"]}.png';fig.savefig(path,dpi=130);plt.close(fig)
            output.append(dict(arm=arm,seed=17,kind=kind,id=str(row['id']),true=int(y[i]),baseline=int(base[i]),
                full=full,early_off=off,true_margin_change_vs_baseline=float(changes[i]),updates=stats,panel=str(path)))
        del model
        torch.cuda.empty_cache()
    (ROOT/'UPDATE_EVIDENCE.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'{len(output)} native-grid update panels generated',flush=True)


if __name__=='__main__':main()
