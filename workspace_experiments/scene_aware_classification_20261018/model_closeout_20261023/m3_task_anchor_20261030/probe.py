"""Frozen source relationship and fixed-development comparison, no test data."""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent/'m3_pairwise_repair_20261030'
spec = importlib.util.spec_from_file_location('anchor_previous', PREVIOUS/'experiment.py')
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
torch, np = previous.torch, previous.np
from task_anchor import attach


@torch.no_grad()
def source_relations(net, cache, source, refs, bank):
    rows = []
    for start in range(0, len(source), 16):
        group = source[start:start+16]
        z, details = net.forward_from(**cache.take(group), reference=bank,
            pair_valid=previous.pair_mask(group, refs, 'cuda'), return_details=True)
        same = torch.tensor([[r['label']==s['label'] for s in refs] for r in group], device='cuda')
        for pol, d in enumerate(details['operators']['m3_0']):
            w = d['reference_weights']
            live = w.sum(1)>0
            same_mass = (w*same[...,None]).sum(1)
            entropy = -(w*w.clamp_min(1e-20).log()).sum(1)
            for i,r in enumerate(group):
                if live[i].any():
                    rows.append(dict(id=str(r['id']), label=int(r['label']), pol=pol,
                        same_mass=float(same_mass[i][live[i]].mean()),
                        entropy=float(entropy[i][live[i]].mean())))
    summary = []
    for pol in (0,1):
        active = [r for r in rows if r['pol']==pol]
        classes = sorted({r['label'] for r in active})
        summary.append(dict(pol=pol, count=len(active),
            same_mass=float(np.mean([r['same_mass'] for r in active])),
            balanced_same_mass=float(np.mean([np.mean([r['same_mass'] for r in active if r['label']==c]) for c in classes])),
            entropy=float(np.mean([r['entropy'] for r in active]))))
    return dict(summary=summary, rows=rows)


def main(dataset, seed=17):
    previous.setup()
    cache, (fit, source, queries, refs) = previous.data(dataset)
    bank = cache.take(refs)
    sy = np.array([r['label'] for r in source]); y = np.array([r['label'] for r in queries])
    net = previous.original(dataset,seed)
    before = cache.predict(net,queries,refs,bank)
    source_before = cache.predict(net,source,refs,bank)
    relation_before = source_relations(net,cache,source,refs,bank)
    state = {n:v.detach().cpu().clone() for n,v in net.state_dict().items()}
    attach(net)
    assert all(torch.equal(net.state_dict()[n].cpu(),v) for n,v in state.items())
    source_after = cache.predict(net,source,refs,bank)
    relation_after = source_relations(net,cache,source,refs,bank)
    after = cache.predict(net,queries,refs,bank)
    out = HERE/dataset/'probe'; out.mkdir(parents=True,exist_ok=True)
    np.savez(out/f'frozen_s{seed}.npz',score=after,baseline=before,y=y,ids=np.array([str(r['id']) for r in queries]))
    np.savez(out/f'source_s{seed}.npz',score=source_after,baseline=source_before,y=sy,ids=np.array([str(r['id']) for r in source]))
    previous.write(out/f'source_relations_s{seed}.json',dict(before=relation_before,after=relation_after))
    report = dict(dataset=dataset,seed=seed,source_count=len(source),query_count=len(queries),
        source_before=previous.metrics(source_before,sy),source_after=previous.metrics(source_after,sy),
        before=previous.metrics(before,y),after=previous.metrics(after,y),
        relations_before=relation_before['summary'],relations_after=relation_after['summary'],
        source_changed=int((source_before.argmax(1)!=source_after.argmax(1)).sum()),
        changed=int((before.argmax(1)!=after.argmax(1)).sum()),
        max_logit_change=float(abs(after-before).max()),state_unchanged=True)
    small=queries[:4]
    q=previous.batch([previous.prepare_record(r) for r in small],'cuda')
    r=previous.batch([previous.prepare_record(v) for v in refs],'cuda')
    with torch.no_grad():
        r['prefix']=net.prefix(r['images'],stop=2)
        native=previous.forward_batch(net,q,r,previous.pair_mask(small,refs,'cuda')).cpu().numpy()
    report['native_error']=float(abs(native-after[:4]).max())
    assert report['native_error']<1e-4
    previous.write(out/f'result_s{seed}.json',report)
    print('TASK_ANCHOR',report,flush=True)


if __name__=='__main__':
    p=previous.argparse.ArgumentParser();p.add_argument('--dataset',choices=['private','oss2'],required=True)
    a=p.parse_args();main(a.dataset)
