"""Fixed-weight C pathway dependence for the CPU-selected native cases."""
import gc
import json
import numpy as np
import torch
from run import HERE, RELEASE, ARMS, SEEDS, model, features, setup, read_manifest, pair_mask, write


def margins(score,label,rival):
    value = score.detach().cpu().numpy()[0]
    competing = value.copy(); competing[label] = -np.inf
    return dict(prediction=int(value.argmax()),top1_margin=float(value[label]-competing.max()),
                fixed_margin=float(value[label]-value[rival]))


@torch.inference_mode()
def main():
    setup('cuda')
    rows = read_manifest(HERE/'query_records.json'); lookup = {r['id']:r for r in rows}
    index = {r['id']:i for i,r in enumerate(rows)}
    cases = json.loads((HERE/'analysis/cases/cases.json').read_text(encoding='utf-8'))
    refs = read_manifest(RELEASE/'manifests/references.json')
    predictions = np.load(HERE/'predictions.npz')
    state = torch.load(RELEASE/'weights/teacher.pt',weights_only=True,map_location='cpu')['state']
    output = []
    for seed in SEEDS:
        selected = [c for c in cases if c['representative_seed']==seed]
        if not selected:continue
        net,_ = model(state,'preadd',seed)
        bank = features(net.base,refs)
        for case in selected:
            row = lookup[case['id']]; i = index[case['id']]
            q = features(net.base,[row]); legal = pair_mask([row],refs,'cuda')
            score,details = net(q,bank,legal,details=True)
            expected = predictions['scores'][ARMS.index('preadd'),SEEDS.index(seed),i]
            error = float(np.max(abs(score.cpu().numpy()[0]-expected)))
            assert error<1e-4,(case['id'],seed,error)
            paths = {'full':margins(score,case['label'],case['fixed_rival'])}
            for route in ('backbone_only','readout_only'):
                paths[route] = margins(net(q,bank,legal,route=route),case['label'],case['fixed_rival'])
            for p,pol in enumerate(('VV','VH')):
                net.early[f'2_{p}'].enabled=False
                paths[f'off_{pol}']=margins(net(q,bank,legal),case['label'],case['fixed_rival'])
                net.early[f'2_{p}'].enabled=True
            for r,region in enumerate(('body','end','side')):
                hooks=[module.transforms[r].register_forward_hook(lambda m,args,v:torch.zeros_like(v)) for module in net.early.values()]
                try:paths[f'off_{region}']=margins(net(q,bank,legal),case['label'],case['fixed_rival'])
                finally:
                    for hook in hooks:hook.remove()
            for module in net.early.values():module.enabled=False
            paths['early_off']=margins(net(q,bank,legal),case['label'],case['fixed_rival'])
            for module in net.early.values():module.enabled=True
            updates=[]
            for p,pol in enumerate(('VV','VH')):
                if not q['available'][0,p]:continue
                trace=details['trace']['early_2'][p]
                strength=trace['update'][0].norm(dim=0)
                masks=trace['write_weights'][0]
                stats=(masks*strength).sum((1,2))/masks.sum((1,2)).clamp_min(1e-8)
                updates.append(dict(pol=pol,regional_mean_proposed_update_l2=dict(zip(('body','end','side'),stats.cpu().tolist()))))
            output.append(dict(id=row['id'],state=case['state'],seed=seed,label=case['label'],
                comparison=case.get('comparison'),fixed_rival=case['fixed_rival'],paths=paths,
                updates=updates,replay_error=error))
        del net,bank
        gc.collect();torch.cuda.empty_cache()
    write('case_paths.json',dict(cases=output,
          interpretation='Frozen C; removal affects BOTH query and references; no refit, no physical or pixel attribution. Proposed update precedes final layer2 ReLU.'))
    print('CASE PATHS COMPLETE',flush=True)


if __name__=='__main__':main()
