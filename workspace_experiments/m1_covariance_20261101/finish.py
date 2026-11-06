"""Frozen-choice seed summary and covariance-specific, same-weight checks."""
import importlib.util
from run import *


def main():
    task.current.setup(task.DEVICE)
    rows, train, ids, rid = task.records()
    y = np.array([rows[i]['label'] for i in ids])
    selection = json.loads((ROOT/'selection.json').read_text())
    a, c = task.arrays(), e.caches()
    c.update({p.stem: np.load(p, mmap_mode='r') for p in (ROOT/'cache').glob('*.npy')})
    summary, pairs, scores = [], [], {}
    for seed in (17, 29, 43):
        orders = []
        for kind, sel in selection.items():
            name = f'{sel["position"]}_{kind}_g{int(sel["gain"]*10)}_s{seed}'
            dest = ROOT/name
            receipt = json.loads((dest/'receipt.json').read_text())
            assert receipt['selected']['epoch'] == sel['epoch']
            orders.append(json.loads((dest/'orders.json').read_text()))
            z = np.load(dest/'selected.npz')['score']
            scores[(seed, kind)] = z
            summary.append(dict(seed=seed, kind=kind, run=name, epoch=sel['epoch'], **task.metrics(y, z)))
            if kind == 'class' and not (dest/'covariance_zero.npz').exists():
                torch.manual_seed(seed)
                net = Network(sel['position'], kind, sel['gain'])
                net.restore(torch.load(dest/'best.pt', map_location='cpu', weights_only=True))
                replay = e.predict(net, a, c, ids)
                assert float(abs(replay['score']-z).max()) == 0
                np.savez_compressed(dest/'covariance_zero.npz', **e.predict(net, a, c, ids, 'covariance_zero'))
                del net
            if kind == 'class':
                for mode in ('off', 'relation_off', 'merged', 'spatial_mean', 'covariance_zero', 'vv_only', 'vh_only'):
                    scores[(seed, mode)] = np.load(dest/(mode+'.npz'))['score']
        assert all(o == orders[0] for o in orders)
        full = scores[(seed, 'class')]
        for control in ('mean', 'second', 'off', 'relation_off', 'merged', 'spatial_mean', 'covariance_zero'):
            other = scores[(seed, control)]
            good, old = full.argmax(1) == y, other.argmax(1) == y
            fm, om = task.metrics(y, full), task.metrics(y, other)
            pairs.append(dict(seed=seed, control=control, rescues=int((good & ~old).sum()),
                harms=int((~good & old).sum()), changes=int((full.argmax(1) != other.argmax(1)).sum()),
                delta_oa=fm['oa']-om['oa'], delta_ba=fm['ba']-om['ba'], delta_nll=fm['nll']-om['nll']))
    df, paired = pd.DataFrame(summary), pd.DataFrame(pairs)
    df.to_csv(ROOT/'repeats.csv', index=False)
    paired.to_csv(ROOT/'repeat_pairs.csv', index=False)
    stats = df.groupby('kind')[['oa', 'ba', 'nll']].agg(['mean', 'std'])
    stats.to_csv(ROOT/'repeat_summary.csv')
    class_rows = []
    for (seed, kind), score in scores.items():
        for cls in range(4):
            mask = y == cls
            class_rows.append(dict(seed=seed, kind=kind, label=cls, n=int(mask.sum()),
                                   correct=int((score.argmax(1)[mask] == y[mask]).sum())))
    pd.DataFrame(class_rows).to_csv(ROOT/'repeat_classes.csv', index=False)
    events = []
    for j, idx in enumerate(ids):
        item = dict(index=int(idx), id=rows[idx]['id'], label=int(y[j]), mmsi=rows[idx]['mmsi'])
        for seed in (17, 29, 43):
            for kind in ('class', 'mean', 'second', 'relation_off', 'covariance_zero'):
                z = scores[(seed, kind)][j]
                item[f'{kind}_{seed}'] = int(z.argmax())
                item[f'{kind}_{seed}_margin'] = float(z[y[j]]-np.where(np.arange(4) == y[j], -np.inf, z).max())
        events.append(item)
    pd.DataFrame(events).to_csv(ROOT/'repeat_events.csv', index=False)
    task.write(ROOT/'completion.json', dict(source=len(train), development=len(ids),
        development_ships=len({rows[i]['mmsi'] for i in ids}), development_passes=len({task.orbit(rows[i]) for i in ids}),
        references=len(rid), seeds=[17, 29, 43], fits=18, steps=3240,
        same_initial_carrier=True, historical_development=True,
        mean_paired=paired.groupby('control')[['delta_oa', 'delta_ba', 'delta_nll']].mean().to_dict('index')))
    print(stats.to_string(), flush=True)
    print(paired.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
