"""Finite development-only shrinkage comparison, no expanded/test inputs."""
from run import *


def main():
    setup()
    rows,arrays,indices,bank = refit.load()
    result = []
    y = np.array([r['label'] for r in rows['development']])
    out = HERE/'adapt_pilot/strength'
    out.mkdir(parents=True,exist_ok=True)
    for seed in (17,29,43):
        original = load_model(ROOT/'spatial_m12_full_20261029/training'/f'spatial_s{seed}/model.pt','cuda')
        baseline_state = {n:p.detach().clone() for n,p in original.named_parameters() if n.startswith('m3.')}
        del original
        for arm in ('total','control'):
            for strength in (.25,.5):
                net = load(HERE/'adapt_pilot/training'/f'{arm}_s{seed}/model.pt')
                with torch.no_grad():
                    for n,p in net.named_parameters():
                        if arm == 'total' and '.extra_output.' in n:
                            p.mul_(strength)
                        elif arm == 'control' and n in baseline_state:
                            p.copy_(torch.lerp(baseline_state[n],p,strength))
                score = refit.predict(net,arrays,indices['development'],rows['development'],rows['references'],bank)
                m = metrics(score,y)
                result.append(dict(arm=arm,seed=seed,strength=strength,**m))
                np.savez(out/f'{arm}_a{strength}_s{seed}.npz',score=score,y=y,ids=np.array([str(r['id']) for r in rows['development']]))
                print(result[-1],flush=True)
                del net
            receipt = json.loads((HERE/'adapt_pilot/training'/f'{arm}_s{seed}/training.json').read_text())
            result.append(dict(arm=arm,seed=seed,strength=0.,**receipt['before']))
            result.append(dict(arm=arm,seed=seed,strength=1.,**receipt['after']))
        del baseline_state
    summary=[]
    for arm in ('total','control'):
        for a in (0.,.25,.5,1.):
            values=[r for r in result if r['arm']==arm and r['strength']==a]
            summary.append(dict(arm=arm,strength=a,**{k:float(np.mean([r[k] for r in values])) for k in ('oa','ba','nll','brier')}))
    write(out/'results.json',dict(records=result,summary=summary,
        selection='Same strength across all orders. Maximize mean of OA and BA on historical development; tie uses lower change. No expanded/test scores.'))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
