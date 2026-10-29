"""Label-informed reference interventions: diagnostic only, never deployable scores."""
from run import *
from sar_m123.current.model import load_model


@torch.no_grad()
def main():
    setup()
    rows,arrays,indices,bank=refit.load()
    references=rows['references']
    rl=torch.tensor([r['label'] for r in references],device='cuda')
    results={}
    for seed in SEEDS:
        net=load_model(FULL/'training'/f'spatial_s{seed}/model.pt','cuda')
        values={k:[] for k in ('original','same_class_only','uniform_references')}
        for first in range(0,len(rows['development']),16):
            q=rows['development'][first:first+16]
            inputs,extra=refit.get_batch(arrays,indices['development'][first:first+16])
            legal=pair_mask(q,references,'cuda')
            truth=torch.tensor([r['label'] for r in q],device='cuda')
            for mode in values:
                allowed=legal & truth[:,None].eq(rl[None]) if mode=='same_class_only' else legal
                for group in net.m3:
                    for op in group:
                        op.temperature=1e9 if mode=='uniform_references' else .1
                z=net.forward_from(*inputs,**extra,reference=bank,pair_valid=allowed)
                values[mode].append(z.cpu().numpy())
        for mode,parts in values.items():
            z=np.concatenate(parts)
            y=np.array([r['label'] for r in rows['development']])
            p=HERE/'diagnostic_predictions';p.mkdir(exist_ok=True)
            np.savez(p/f'{mode}_s{seed}.npz',score=z,y=y,ids=np.array([str(r['id']) for r in rows['development']]))
            results[f'{mode}_s{seed}']=metrics(z,y)
        with np.load(HERE/'predictions'/f'before_control_s{seed}.npz') as saved:
            error=float(abs(np.concatenate(values['original'])-saved['score']).max())
            assert error<1e-4,error
        print(seed,json.dumps({k:v for k,v in results.items() if k.endswith(f's{seed}')}),flush=True)
        del net
    write(HERE/'reference_use.json',dict(label_informed_diagnostic_not_classifier=True,
        caveat='Same-class references are not an oracle upper bound: frozen writer may not exploit them.',results=results))


if __name__=='__main__':
    main()
