"""Measure M3 source-gradient conflict before choosing a replay experiment."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'m3_reference_coverage_20261030'))
from center import *
HERE = Path(__file__).resolve().parent


def data(dataset):
    if dataset == 'private':
        cache = PrivateCache()
        add_adapt(cache.rows, cache.arrays, cache.indices)
        cache.lookup.update({str(r['id']): int(i) for r, i in zip(cache.rows['adapt'], cache.indices['adapt'])})
        groups = [cache.rows[k] for k in ('fit', 'adapt', 'development', 'references')]
    else:
        rows = {k: read_manifest(PUB/'manifests'/f'{k}.json')
                for k in ('source_fit', 'source_adapt', 'validation', 'references')}
        cache = Cache(rows)
        groups = [rows[k] for k in ('source_fit', 'source_adapt', 'validation', 'references')]
    return cache, groups


def original(dataset, seed):
    if dataset == 'private':
        return load_model(ROOT/'spatial_m12_full_20261029/training'/f'spatial_s{seed}/model.pt', 'cuda')
    return api.load(PUB/'training'/f'full_s{seed}/adapt/model.pt', 'cuda')


def parameters(net):
    net.eval().requires_grad_(False)
    selected = []
    for n, p in net.named_parameters():
        if n.startswith('m3.') and ('.project.' in n or '.output.' in n or n in {'m3.0.0.gain', 'm3.0.1.gain'}):
            p.requires_grad_(True)
            selected.append((n, p))
    return selected


def gradient(net, cache, rows, refs, bank, selected, vessel_weight):
    from collections import Counter
    counts = Counter(str(r['mmsi']) for r in rows)
    w = np.array([1/counts[str(r['mmsi'])] if vessel_weight else 1 for r in rows], dtype=np.float64)
    w /= w.sum()
    net.zero_grad(set_to_none=True)
    ce = 0.
    for first in range(0, len(rows), 16):
        q = rows[first:first+16]
        z = cache.score(net, q, refs, bank)
        y = torch.tensor([r['label'] for r in q], device='cuda')
        loss = (F.cross_entropy(z, y, reduction='none')*torch.tensor(w[first:first+16], device='cuda')).sum()
        loss.backward()
        ce += float(loss.detach())
    g = torch.cat([p.grad.detach().flatten().cpu() if p.grad is not None else torch.zeros(p.numel()) for n, p in selected])
    assert torch.isfinite(g).all()
    return g, ce


def cosine(a, b):
    return float(F.cosine_similarity(a[None], b[None], dim=1))


def main(dataset, seed):
    setup()
    cache, (fit, adapt, dev, refs) = data(dataset)
    net = original(dataset, seed)
    selected = parameters(net)
    bank = cache.take(refs)
    gradients, report = {}, []
    for name, rows in [('adapt', adapt), ('fit', fit), ('development', dev)]:
        for c in sorted({r['label'] for r in rows}):
            group = [r for r in rows if r['label'] == c]
            g, ce = gradient(net, cache, group, refs, bank, selected, name == 'fit')
            gradients[f'{name}_{c}'] = g
            report.append(dict(group=name, label=c, observations=len(group), vessels=len({str(r['mmsi']) for r in group}),
                               nll=ce, gradient_norm=float(g.norm())))
            print('GRADIENT', dataset, report[-1], flush=True)
    total = {name: torch.stack([v for k, v in gradients.items() if k.startswith(name+'_')]).mean(0)
             for name in ('adapt', 'fit', 'development')}
    common = sorted({r['label'] for r in dev} & {r['label'] for r in adapt} & {r['label'] for r in fit})
    matched = {name: torch.stack([gradients[f'{name}_{c}'] for c in common]).mean(0)
               for name in ('adapt', 'fit', 'development')}
    angles = []
    for c in sorted({r['label'] for r in adapt}):
        g = gradients[f'adapt_{c}']
        angles.append(dict(label=c, to_adapt=cosine(g, total['adapt']), to_fit=cosine(g, total['fit']),
            to_development=cosine(g, total['development']),
            sameclass_fit=cosine(g, gradients[f'fit_{c}']),
            sameclass_development=cosine(g, gradients[f'development_{c}']) if f'development_{c}' in gradients else None))
    out = HERE/dataset/'diagnostics'
    write(out/f'gradients_s{seed}.json', dict(dataset=dataset, seed=seed, parameters=[n for n,p in selected],
        groups=report, angles=angles, total_cosines={f'{a}_{b}':cosine(total[a],total[b])
        for a,b in [('adapt','fit'),('adapt','development'),('fit','development')]},
        matched_classes=common, matched_class_cosines={f'{a}_{b}':cosine(matched[a],matched[b])
        for a,b in [('adapt','fit'),('adapt','development'),('fit','development')]},
        interpretation='Class-balanced full CE gradients, not actual Adam updates or causal attribution. Development labels diagnose only; no gradient enters training.',
        evaluation_used=False))
    torch.save(gradients, out/f'gradients_s{seed}.pt')
    print('DONE', dataset, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['private','oss2'], required=True)
    p.add_argument('--seed', type=int, default=17)
    a = p.parse_args()
    main(a.dataset, a.seed)
