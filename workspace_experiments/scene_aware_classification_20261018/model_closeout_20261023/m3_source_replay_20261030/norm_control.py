"""No-training diagnostic: match the parameter-update norm, without label selection."""
from diagnose import *


def main(dataset, seeds):
    setup()
    cache, (_, _, queries, refs) = data(dataset)
    bank = cache.take(refs)
    y = np.array([r['label'] for r in queries])
    for seed in seeds:
        out = HERE/dataset/'training'
        net = original(dataset, seed)
        names = [n for n,p in parameters(net)]
        net.requires_grad_(False)
        control = torch.load(out/f'adapt_only_s{seed}.pt', map_location='cpu', weights_only=True)['state']
        candidate = torch.load(out/f'source_mix_s{seed}.pt', map_location='cpu', weights_only=True)['state']
        original_state = net.state_dict()
        delta = {n:control[n]-original_state[n].cpu() for n in names}
        control_norm = sum(float(d.square().sum()) for d in delta.values())**.5
        mixed_norm = sum(float((candidate[n]-original_state[n].cpu()).square().sum()) for n in names)**.5
        ratio = mixed_norm/control_norm
        with torch.no_grad():
            for n, p in net.named_parameters():
                if n in delta:
                    p.add_(delta[n].to(p)*ratio)
        z = cache.predict(net, queries, refs, bank)
        pred = HERE/dataset/'predictions'
        baseline = np.load(pred/f'adapt_only_s{seed}.npz')['baseline']
        np.savez(pred/f'norm_control_s{seed}.npz', score=z, baseline=baseline, y=y,
                 ids=np.array([str(r['id']) for r in queries]))
        report = dict(dataset=dataset, seed=seed, ratio=ratio, control_norm=control_norm, mixed_norm=mixed_norm,
            after=metrics(z,y), baseline=metrics(baseline,y), no_training=True, evaluation_used=False,
            interpretation='Same global M3 update norm, not identical per-layer updates or proof of causality.')
        write(HERE/dataset/'diagnostics'/f'norm_control_s{seed}.json', report)
        print('NORM_CONTROL', report, flush=True)
        del net, control, candidate


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['private','oss2'], required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[17,29,43])
    a = p.parse_args()
    main(a.dataset,a.seeds)
