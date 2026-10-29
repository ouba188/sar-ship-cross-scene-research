"""Same-checkpoint comparison/value swaps, not independent retrained arms."""
from run import *
from model import add_metric
from sar_m123.current.model import load_model


def is_compare(name):
    return name.startswith('m3.') and any(t in name for t in ('selector.', 'branch_bias', 'metric_'))


@torch.no_grad()
def relation_stats(net, arrays, ids, queries, refs, bank):
    rl=torch.tensor([r['label'] for r in refs],device='cuda')
    out=[[],[]]
    for first in range(0,len(ids),16):
        q=queries[first:first+16]
        y=torch.tensor([r['label'] for r in q],device='cuda')
        _,d=refit.score(net,arrays,ids[first:first+16],q,refs,bank,True)
        for pol,t in enumerate(d['operators']['m3_0']):
            live=t['reference_valid']
            same=rl[None].eq(y[:,None]) & live
            active=same.any(1)
            p=t['reference_logits'].masked_fill(~live,-1e4).softmax(1)
            w=t['reference_weights']
            for i in torch.where(active)[0].tolist():
                out[pol].append(dict(id=str(q[i]['id']), top_reference=str(refs[int(p[i].argmax())]['id']),
                    same_mass=float((p[i]*same[i]).sum()),
                    top_correct=float(rl[p[i].argmax()]==y[i]),
                    write_same=float((w[i]*same[i,:,None]).sum(0).mean())))
    return [dict(summary={k:float(np.mean([v[k] for v in a]))
                          for k in ('same_mass','top_correct','write_same')}, records=a) for a in out]


def main():
    setup()
    rows,arrays,indices,bank=refit.load()
    labels=np.array([r['label'] for r in rows['fit']])
    ix=np.concatenate([np.flatnonzero(labels==c)[:8] for c in range(8)])
    report={}
    for seed in SEEDS:
        start=FULL/'training'/f'spatial_s{seed}/model.pt'
        original=load_model(start,'cuda')
        report[f'original_s{seed}']=dict(source=relation_stats(original,arrays,indices['fit'][ix],
            [rows['fit'][i] for i in ix],rows['references'],bank),
            development=relation_stats(original,arrays,indices['development'],rows['development'],rows['references'],bank))
        original_state={k:v.detach().cpu().clone() for k,v in original.state_dict().items()}
        del original
        for arm in ARMS:
            trained=load(HERE/'training'/f'{arm}_s{seed}/model.pt')
            state={k:v.detach().cpu().clone() for k,v in trained.state_dict().items()}
            report[f'{arm}_s{seed}']=dict(source=relation_stats(trained,arrays,indices['fit'][ix],
                [rows['fit'][i] for i in ix],rows['references'],bank),
                development=relation_stats(trained,arrays,indices['development'],rows['development'],rows['references'],bank))
            for path in ('compare_only','value_only'):
                net=fresh(start,arm if path=='compare_only' else 'control')
                replacement={k:(state[k] if (is_compare(k) if path=='compare_only' else k.startswith('m3.') and not is_compare(k)) else original_state[k])
                             for k in net.state_dict()}
                net.load_state_dict(replacement,strict=True)
                _,m=evaluate(net,arrays,indices['development'],rows['development'],rows['references'],bank,
                             f'{path}_{arm}_s{seed}')
                report[f'{arm}_s{seed}'][path]=m
                del net
            del trained,state
            print('SWAP',arm,seed,json.dumps({k:v for k,v in report[f'{arm}_s{seed}'].items()
                  if k.endswith('only')}),flush=True)
        del original_state
        torch.cuda.empty_cache()
    write(HERE/'path_diagnosis.json',report)


if __name__=='__main__':
    main()
