"""No-refit separation of early feedback propagation vs shallow head readout."""
import json
import numpy as np
import torch
from run import ROOT, RELEASE, Cache, construct, setup, read_manifest, pair_mask, write
from analyze import metrics


@torch.no_grad()
def main():
    setup('cuda')
    state=torch.load(RELEASE/'weights/teacher.pt',map_location='cpu',weights_only=True)['state']
    rows=read_manifest(RELEASE/'manifests/development.json')
    refs=read_manifest(RELEASE/'manifests/references.json')
    cache=Cache('dev');bank=Cache('refs').get(np.arange(len(refs)))
    data=np.load(ROOT/'predictions.npz');y=data['labels'];results=[]
    scores=[]
    for si,seed in enumerate((17,29,43)):
        model=construct(state,(2,),seed)
        model.load_state_dict(torch.load(ROOT/'checkpoints'/f'layer2_{seed}.pt',weights_only=True,map_location='cpu')['state'],strict=False)
        seed_scores=[]
        for route in ('full','backbone_only','readout_only'):
            parts=[]
            for start in range(0,len(rows),16):
                subset=rows[start:start+16]
                parts.append(model(cache.get(np.arange(start,start+len(subset))),bank,
                    pair_mask(subset,refs,'cuda'),route=route).cpu().numpy())
            z=np.concatenate(parts);seed_scores.append(z)
            if route=='full':
                err=float(abs(z-data['scores'][2,si]).max());assert err<1e-4,err
            results.append(dict(seed=seed,route=route,**metrics(z,y)))
        results.append(dict(seed=seed,route='all_early_off',**metrics(data['bypasses'][2,si],y)))
        results.append(dict(seed=seed,route='matched_baseline',**metrics(data['scores'][0,si],y)))
        scores.append(seed_scores)
        del model
        torch.cuda.empty_cache()
    write('path_diagnosis.json',dict(arm='layer2',type='same_weights_no_refit',results=results,
        interpretation='backbone_only restores original layer2 head descriptor but retains modified downstream convolution/value paths; readout_only does the reverse'))
    np.savez(ROOT/'path_predictions.npz',scores=np.array(scores),labels=y,ids=data['ids'])
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
