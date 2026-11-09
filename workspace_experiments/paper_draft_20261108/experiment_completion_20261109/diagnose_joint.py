"""Same-weight M3 switch, separating immediate writeback from refit differences."""
import json
from pathlib import Path
import numpy as np
import common_ablation as e

ROOT=Path(__file__).resolve().parent/'common_init'


@e.torch.no_grad()
def main():
    e.task.current.setup(e.task.DEVICE)
    rows,groups=e.prev.records();a,c=e.prev.arrays()
    a['write']=np.load(ROOT/'write.npy',mmap_mode='r')
    protocol=json.loads((ROOT/'protocol.json').read_text())
    result=[];cases=[]
    for seed in (17,29,43):
        folder=ROOT/f'full_s{seed}_360'
        net=e.Network('full',seed,protocol['pretrain'])
        net.load_state_dict(e.torch.load(folder/'final.pt',map_location='cpu',weights_only=True))
        for split in ('small','evaluation'):
            ids=groups[split];y=np.array([rows[i]['label'] for i in ids]);scores=[]
            for start in range(0,len(ids),12):
                z=net(a,c,rows,ids[start:start+12],groups['references'],m3_on=False)[0]
                scores.append(z.cpu().numpy())
            off=np.concatenate(scores);on=np.load(folder/f'{split}.npz')['score']
            before,after=off.argmax(1)==y,on.argmax(1)==y
            result.append(dict(seed=seed,split=split,off=e.task.metrics(y,off),on=e.task.metrics(y,on),
                               rescued=int((after&~before).sum()),harmed=int((before&~after).sum())))
            np.savez_compressed(folder/f'{split}_m3_off.npz',ids=ids,score=off)
            if seed == 17 and split == 'evaluation':
                from summarize_runs import panel
                refs=[rows[i] for i in groups['references']]
                for name,mask in {'rescue':after&~before,'harm':before&~after}.items():
                    hits=np.flatnonzero(mask)
                    if len(hits):
                        i=int(hits[0]);stem=ROOT/'cases'/f'direct_m3_{name}_s17'
                        item=panel(rows,int(ids[i]),off[i],on[i],stem,f'M3 off -> on / same weights / {name}',refs)
                        item['file']=str(stem.with_suffix('.png'));cases.append(item)
        # Same current path must still reconstruct the saved scores after diagnostic off calls.
        ids=groups['evaluation'][:4]
        replay=net(a,c,rows,ids,groups['references'])[0].cpu().numpy()
        error=float(abs(replay-np.load(folder/'evaluation.npz')['score'][:4]).max())
        assert error<1e-4,error
        result[-1]['replay_error']=error
        del net;e.gc.collect();e.torch.cuda.empty_cache()
    e.task.write(ROOT/'m3_same_weight.json',result)
    e.task.write(ROOT/'m3_direct_cases.json',cases)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
