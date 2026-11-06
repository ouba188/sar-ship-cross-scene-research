"""Distinguish inactive relations, weak class alignment, and transfer failure."""
from run import *


@torch.no_grad()
def main():
    task.current.setup(task.DEVICE)
    rows, train, hold, rid = task.records()
    a, c = task.arrays(), e.caches()
    c.update({p.stem: np.load(p, mmap_mode='r') for p in (ROOT/'cache').glob('*.npy')})
    selection = json.loads((ROOT/'selection.json').read_text())
    results, samples = [], []
    for kind in ('class', 'second', 'mean'):
        sel = selection[kind]
        torch.manual_seed(17)
        net = Network(sel['position'], kind, sel['gain'])
        net.restore(torch.load(ROOT/sel['run']/'best.pt', map_location='cpu', weights_only=True))
        for split, ids in (('source', train), ('development', hold)):
            probe, auxiliary, predictions, off = [], [], [], []
            available = np.asarray(a['available'][ids])
            for start in range(0, len(ids), 24):
                index = ids[start:start+24]
                net.trace = []
                score, aux, _ = net(a, c, index)
                fulltrace = net.trace
                offscore, _, _ = net(a, c, index, 'relation_off')
                offtrace = net.trace
                predictions.append(score.cpu().numpy())
                off.append(offscore.cpu().numpy())
                auxiliary.append(aux.cpu().numpy())
                mus = []
                for p in range(2):
                    d = fulltrace[p]
                    valid = task.tensor(a['valid'], index)[:, p]
                    occ = grid_occupancy(valid, d['response'].shape[-2:]).cpu().numpy()
                    mass = occ.sum((2, 3)).clip(1e-8)
                    mu = (d['response']*occ).sum((2, 3))/mass
                    mus.append(mu)
                    for j, idx in enumerate(index):
                        if not a['available'][idx, p]:
                            continue
                        update = d['update'][j]
                        removed = offtrace[p]['update'][j]
                        samples.append(dict(kind=kind, split=split, index=int(idx), pol=p,
                            positive_cells=int((occ[j] > 0).sum()), effective_cells=float(occ[j].sum()**2/(occ[j]**2).sum()),
                            relation_to_full_update=float(np.linalg.norm(update-removed)/max(np.linalg.norm(update), 1e-8)),
                            covariance_norm=float(np.linalg.norm(d['covariance'][j])),
                            gate_rms=float(np.sqrt(np.mean(d['gate'][j]**2)))))
                probe.append(np.stack(mus, 1))
            probe = np.concatenate(probe)
            auxiliary = np.concatenate(auxiliary)
            full, removed = np.concatenate(predictions), np.concatenate(off)
            y = np.array([rows[i]['label'] for i in ids])
            metrics = {}
            for label, score in (('full', full), ('relation_off', removed)):
                metrics[label] = task.metrics(y, score)
            good, old = full.argmax(1) == y, removed.argmax(1) == y
            item = dict(kind=kind, split=split, run=sel['run'], metrics=metrics,
                        relation_rescues=int((good & ~old).sum()), relation_harms=int((~good & old).sum()), probe=[])
            for p in range(2):
                usable = available[:, p] > 0
                item['probe'].append(dict(pol=p, n=int(usable.sum()),
                    raw_mu=task.metrics(y[usable], probe[usable, p]),
                    joint_aux=task.metrics(y[usable], auxiliary[usable, p])))
            results.append(item)
            np.savez_compressed(ROOT/(kind+'_'+split+'_diagnostic.npz'), ids=ids, probe=probe,
                                auxiliary=auxiliary, full=full, relation_off=removed)
            print('DIAGNOSTIC', kind, split, item['relation_rescues'], item['relation_harms'],
                  metrics, flush=True)
        del net
        gc.collect()
        torch.cuda.empty_cache()
    task.write(ROOT/'diagnostics.json', results)
    df = pd.DataFrame(samples)
    df.to_csv(ROOT/'diagnostic_samples.csv', index=False)
    df.groupby(['kind', 'split', 'pol'])[['effective_cells', 'relation_to_full_update', 'covariance_norm', 'gate_rms']].agg(['mean', 'median']).to_csv(ROOT/'diagnostic_summary.csv')


if __name__ == '__main__':
    main()
