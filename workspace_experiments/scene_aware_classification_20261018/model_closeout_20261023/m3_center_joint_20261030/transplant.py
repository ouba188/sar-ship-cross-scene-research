"""Separate M3 changes from jointly adapted LIVE weights on development only."""
from experiment import *


def main(dataset, seeds):
    setup()
    if dataset == 'private':
        cache = PrivateCache()
        queries, refs = cache.rows['development'], cache.rows['references']
        loader = load_model
    else:
        rows = {k: read_manifest(PUB/'manifests'/f'{k}.json') for k in ('validation','references')}
        cache = Cache(rows)
        queries, refs = rows['validation'], rows['references']
        loader = api.load
    bank = cache.take(refs)
    y = np.array([r['label'] for r in queries])
    out = HERE/dataset/'transplant'
    out.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        originalpath = (ROOT/'spatial_m12_full_20261029/training'/f'spatial_s{seed}/model.pt'
                        if dataset == 'private' else PUB/'training'/f'full_s{seed}/adapt/model.pt')
        original = loader(originalpath, 'cuda')
        original_state = {n: v.detach().cpu().clone() for n, v in original.state_dict().items()}
        baseline = cache.predict(original, queries, refs, bank)
        del original
        reports = {}
        for arm in ('raw_joint','center_joint'):
            candidate = HERE/dataset/'training'/f'{arm}_s{seed}.pt'
            for component in ('m3_only','live_only'):
                if component == 'm3_only':
                    net = restore(candidate)
                    state = net.state_dict()
                    # Replace every original non-M3 tensor, including LIVE, with saved full-model values.
                    for n, v in original_state.items():
                        if not n.startswith('m3.'):
                            state[n].copy_(v)
                    net.load_state_dict(state, strict=True)
                else:
                    net = loader(originalpath, 'cuda')
                    new = torch.load(candidate, map_location='cpu', weights_only=True)['state']
                    state = net.state_dict()
                    for n in LIVE:
                        state[n].copy_(new[n])
                    net.load_state_dict(state, strict=True)
                    del new
                score = cache.predict(net, queries, refs, bank)
                name = f'{arm}_{component}'
                np.savez(out/f'{name}_s{seed}.npz', score=score, baseline=baseline, y=y,
                         ids=np.array([str(r['id']) for r in queries]))
                old, pred = baseline.argmax(1), score.argmax(1)
                reports[name] = dict(after=metrics(score,y), rescued=int(((old!=y)&(pred==y)).sum()),
                    harmed=int(((old==y)&(pred!=y)).sum()), changed=int((old!=pred).sum()))
                print('TRANSPLANT',dataset,seed,name,reports[name],flush=True)
                del net,state
        write(out/f'results_s{seed}.json', dict(baseline=metrics(baseline,y), variants=reports,
              evaluation_used=False, role='development component diagnostic, no parameter sweep'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['private','oss2'],required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[17,29,43])
    args=parser.parse_args()
    main(args.dataset,args.seeds)
