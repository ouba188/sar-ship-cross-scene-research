"""Cases where the same trained classifier changes with M2 feedback."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from experiment import HERE, PILOT, old, setup, load, predict, restore
from analyze import result, margin


@torch.no_grad()
def main():
    setup()
    selected = json.loads((HERE/'source_selected.json').read_text())
    stage, gate = selected['stage'], selected['gate']
    table, arrays = load(stage)
    images = np.load(PILOT/'cache/images.npy',mmap_mode='r')
    pool, counts = [], []
    for seed in old.SEEDS:
        data = result(stage,gate,seed)
        ids,y,full = data['ids'],data['y'],data['score']
        model = restore(stage,gate,seed)
        model.stages = ()
        bypass = predict(model,arrays,ids,stage)
        before,after = bypass.argmax(1)==y, full.argmax(1)==y
        delta = margin(full,y)-margin(bypass,y)
        counts.append(dict(seed=seed,rescued=int((~before & after).sum()),harmed=int((before & ~after).sum())))
        for kind, flag in [('direct_rescue',~before & after),('direct_harm',before & ~after)]:
            candidates = np.flatnonzero(flag)
            if len(candidates):
                k = candidates[np.argmax(abs(delta[candidates]))]
                pool.append(dict(kind=kind,seed=seed,idx=int(ids[k]),y=int(y[k]),before=int(bypass[k].argmax()),
                    after=int(full[k].argmax()),delta=float(abs(delta[k]))))
        del model
    pd.DataFrame(counts).to_csv(HERE/'direct_counts.csv',index=False)
    cases, rows = [], []
    for kind in ('direct_rescue','direct_harm'):
        item = max([r for r in pool if r['kind']==kind],key=lambda r:r['delta'])
        idx,seed,y = item['idx'],item['seed'],item['y']
        model = restore(stage,gate,seed)
        x,masks,valid,av = old.batch(arrays,[idx])
        fig,axes = plt.subplots(2,3,figsize=(9,6.5),constrained_layout=True)
        for p,pol in enumerate(('VV','VH')):
            _,d = model.blocks[f'{p}_{stage}'](x[:,p],masks[:,p],valid[:,p],return_details=True)
            im = images[idx,p,0]
            for a in axes[p]: a.imshow(im,cmap='gray',vmin=0,vmax=255)
            axes[p,0].set_title(pol+' native crop/pad')
            update = d['update'][0].norm(dim=0).cpu().numpy()
            axes[p,1].imshow(update,extent=(0,160,160,0),cmap='magma',vmin=0,alpha=.65,interpolation='nearest')
            axes[p,1].set_title(f'Update norm max {update.max():.3f}')
            m = masks[0,p].cpu().numpy().transpose(1,2,0)
            rgba = np.concatenate((m,np.clip(m.sum(-1,keepdims=True),0,1)*.8),-1)
            axes[p,2].imshow(rgba,extent=(0,160,160,0),interpolation='nearest')
            axes[p,2].set_title('Red body / green ends / blue sides')
        for a in axes.flat:a.axis('off')
        obs = int(table.iloc[idx].obs_id)
        fig.suptitle(f'{kind}: {obs}, seed {seed}, true {y}\nSame trained head: feedback off {item["before"]} -> on {item["after"]}')
        filename = f'{kind}_{obs}_{seed}.png'
        fig.savefig(HERE/'analysis'/filename,dpi=130)
        plt.close(fig)
        for arm in ('full','no_feedback','no_body','no_end','no_side','no_vv','no_vh'):
            mm = masks.clone()
            if arm=='no_feedback':mm.zero_()
            if arm in ('no_body','no_end','no_side'):mm[:,:,('no_body','no_end','no_side').index(arm)]=0
            if arm in ('no_vv','no_vh'):mm[:,('no_vv','no_vh').index(arm)]=0
            score = model.forward_from(x,mm,valid,av,start=stage).cpu().numpy()
            rows.append(dict(obs_id=obs,kind=kind,seed=seed,arm=arm,y=y,pred=int(score.argmax()),margin=float(margin(score,np.array([y]))[0])))
        cases.append(dict(**item,obs_id=obs,panel=filename,image_vv=table.iloc[idx].image_vv,image_vh=table.iloc[idx].image_vh))
        del model
    old.write(HERE/'analysis/direct_cases.json',cases)
    pd.DataFrame(rows).to_csv(HERE/'direct_case_interventions.csv',index=False)
    print(pd.DataFrame(counts).to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
