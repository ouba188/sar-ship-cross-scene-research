"""Fixed-weight M3 diagnostic and targeted class/availability case inspection."""
import gc
import json
import numpy as np
import pandas as pd
import staged as e
from summarize_runs import panel, ZH


@e.torch.no_grad()
def main():
    root=e.ROOT
    selected=json.loads((root/'selected.json').read_text())['scales']
    protocol=json.loads((root/'protocol.json').read_text())
    e.task.current.setup(e.task.DEVICE)
    rows,groups=e.prev.records();ids=groups['evaluation'];rid=groups['references']
    a,c=e.prev.arrays();a['write']=np.load(e.PREVIOUS/'common_init/write.npy',mmap_mode='r')
    av=np.array(a['available'][ids],dtype=bool)
    e.resident(a,c)
    y=np.array([rows[i]['label'] for i in ids]);full=[];without=[];off=[];records=[]
    folder=root/'followup';folder.mkdir(exist_ok=True)
    for seed in (17,29,43):
        run=root/f'full_s{seed}_lr{selected["full"]:g}'
        receipt=json.loads((run/'receipt.json').read_text())
        net=e.Network('full',seed,protocol['pretrain'])
        net.load_state_dict(e.torch.load(receipt['final'],map_location='cpu',weights_only=True))
        e.configure(net,3);net.requires_grad_(False)
        e.freeze_delta(net,a,rows,rid)
        on=e.predict(net,a,c,rows,ids,rid)
        saved=np.load(run/'evaluation.npz')['score']
        error=float(abs(on-saved).max());assert error<1e-4,error
        z=np.concatenate([net(a,c,rows,ids[j:j+12],rid,m3_on=False)[0].cpu().numpy()
                          for j in range(0,len(ids),12)])
        np.savez_compressed(folder/f'm3_switch_s{seed}.npz',ids=ids,on=on,off=z)
        delta=(on.argmax(1)==y).astype(float)-(z.argmax(1)==y)
        records.append(dict(seed=seed,replay_error=error,delta_oa=100*delta.mean(),
            delta_ba=100*np.mean([delta[y==k].mean() for k in range(4)]),
            rescued=int((delta>0).sum()),harmed=int((delta<0).sum()),
            **{f'off_{key}':value for key,value in e.task.metrics(y,z).items()}))
        full.append(on);off.append(z)
        without.append(np.load(root/f'no_m2_s{seed}_lr{selected["no_m2"]:g}'/'evaluation.npz')['score'])
        del net;gc.collect();e.torch.cuda.empty_cache()
    pd.DataFrame(records).to_csv(root/'same_weight_m3.csv',index=False)
    full=np.stack(full);without=np.stack(without);off=np.stack(off)
    correct=full.argmax(2)==y[None]
    rescue=correct & (without.argmax(2)!=y[None])
    direct=correct & (off.argmax(2)!=y[None])
    candidates=[(f'm2_class_{k}',(y==k)&(rescue.sum(0)>=2),'m2') for k in (2,3)]
    candidates.append(('m2_vh_only',(~av[:,0])&av[:,1]&(rescue.sum(0)>=2),'m2'))
    candidates.append(('m3_direct_rescue',direct.any(0),'m3'))
    candidates.append(('m3_direct_harm',((~correct)&(off.argmax(2)==y[None])).any(0),'m3_harm'))
    refs=[rows[i] for i in rid];cases=[]
    for name,mask,kind in candidates:
        hits=np.flatnonzero(mask)
        if not len(hits):continue
        pos=int(hits[0])
        event=rescue[:,pos] if kind=='m2' else direct[:,pos] if kind=='m3' else ((~correct)&(off.argmax(2)==y[None]))[:,pos]
        j=int(np.flatnonzero(event)[0]);seed=(17,29,43)[j]
        before=without[j,pos] if kind=='m2' else off[j,pos]
        item=panel(rows,int(ids[pos]),before,full[j,pos],folder/name,
            f'{name} / seed{seed} / '+('retrained comparison' if kind=='m2' else 'same-weight switch'),refs)
        item.update(file=f'followup/{name}.png',name=name,seed=seed,
            full_predictions=full[:,pos].argmax(1).tolist(),
            comparison_predictions=(without[:,pos] if kind=='m2' else off[:,pos]).argmax(1).tolist())
        cases.append(item)
    e.task.write(root/'followup_cases.json',cases)
    print(pd.DataFrame(records).to_string(index=False),flush=True)
    print([(r['name'],r['id']) for r in cases],flush=True)


if __name__=='__main__':main()
