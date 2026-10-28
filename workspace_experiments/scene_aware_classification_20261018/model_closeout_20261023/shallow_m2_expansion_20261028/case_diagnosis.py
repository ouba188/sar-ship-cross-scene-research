"""No-refit path dependence on outcome-selected native cases, plus bounded timing."""
import json
import time
import numpy as np
import torch
from run import HERE, RELEASE, SEEDS, models, features, setup, read_manifest, pair_mask, write


def margins(score, label, rival):
    value = score.detach().cpu().numpy()[0]
    competing = value.copy()
    competing[label] = -np.inf
    return dict(prediction=int(value.argmax()), top1_margin=float(value[label]-competing.max()),
                fixed_margin=float(value[label]-value[rival]))


@torch.inference_mode()
def main():
    setup('cuda')
    rows = read_manifest(HERE/'query_records.json')
    lookup = {r['id']:r for r in rows}
    cases = json.loads((HERE/'analysis/cases/cases.json').read_text(encoding='utf-8'))
    refs = read_manifest(RELEASE/'manifests/references.json')
    nets = models()
    bank = features(nets[0].base,refs)
    output = []
    for case in cases:
        si = SEEDS.index(case['representative_seed'])
        net = nets[3+si]
        row = lookup[case['id']]
        q = features(nets[0].base,[row])
        legal = pair_mask([row],refs,'cuda')
        score, detail = net(q,bank,legal,details=True)
        expected = case['per_seed'][si]['layer2_logits']
        error = float(np.max(abs(score.cpu().numpy()[0]-expected)))
        assert error < 1e-4, (case['id'],error)
        paths = {'full':margins(score,case['label'],case['fixed_rival'])}
        for route in ('backbone_only','readout_only'):
            paths[route] = margins(net(q,bank,legal,route=route),case['label'],case['fixed_rival'])
        for module in net.early.values():
            module.enabled = False
        paths['early_off'] = margins(net(q,bank,legal),case['label'],case['fixed_rival'])
        for module in net.early.values():
            module.enabled = True
        updates = []
        for p, pol in enumerate(('VV','VH')):
            if not q['available'][0,p]:
                continue
            trace = detail['trace']['early_2'][p]
            strength = trace['update'][0].norm(dim=0)
            masks = trace['write_weights'][0]
            stats = (masks*strength).sum((1,2))/masks.sum((1,2)).clamp_min(1e-8)
            updates.append(dict(pol=pol,regional_mean_update_l2=dict(zip(('body','end','side'),stats.cpu().tolist()))))
        output.append(dict(id=row['id'],state=case['state'],seed=case['representative_seed'],
                           label=case['label'],paths=paths,updates=updates,replay_error=error))
    write('case_paths.json',dict(cases=output,
          interpretation='Same-trained-weight path removal; not a retrained ablation or pixel/physical attribution.'))
    q = features(nets[0].base,rows[:16])
    legal = pair_mask(rows[:16],refs,'cuda')
    selected = (nets[0],nets[3])
    for _ in range(5):
        for net in selected:
            net(q,bank,legal)
    times = [[],[]]
    for repeat in range(12):
        for idx in ((0,1) if repeat%2==0 else (1,0)):
            torch.cuda.synchronize()
            start=time.perf_counter()
            selected[idx](q,bank,legal)
            torch.cuda.synchronize()
            times[idx].append((time.perf_counter()-start)*1000)
    write('timing.json',dict(query_batch=16,references=16,seed=17,repeats=12,
          scope='Post-frozen-prefix forward only, includes live reference recomputation; excludes preprocessing/prefix/disk IO. Not full chip-to-output latency.',
          baseline_ms=times[0],layer2_ms=times[1],
          medians_ms=[float(np.median(x)) for x in times]))
    print('CASE PATHS AND TIMING COMPLETE',flush=True)


if __name__=='__main__':
    main()
