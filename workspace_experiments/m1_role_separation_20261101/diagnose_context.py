"""Measure information and locality available to the frozen M1 reader."""
from run_pilot import *


def main():
    task.current.setup(task.DEVICE)
    rows, train, hold, rid = task.records()
    c = {p.stem: np.load(p, mmap_mode='r') for p in (ROOT/'cache').glob('*.npy')}
    a = task.arrays()
    model = anchor()
    results = []
    with torch.no_grad():
        for cohort, ids in [('source', train), ('development', hold)]:
            for p in range(2):
                relative, cells, effective, kl = [], [], [], []
                for part in np.array_split(ids, 20):
                    xx = task.tensor(c['x'], part)[:, p]
                    occ = task.tensor(c['occupancy'], part)[:, p]
                    cc = task.tensor(c['context'], part)[:, p]
                    mm = task.tensor(c['mean_context'], part)[:, p]
                    present = task.tensor(a['available'], part)[:, p].bool()
                    _, d = model.base.internal_m1[p](xx, occ, None, True)
                    w = d['weights'].flatten(3)
                    uniform = w[:, 0:1].clamp_min(1e-20)
                    sides = w[:, 1:]
                    k = (sides*(sides.clamp_min(1e-20).log()-uniform.log())).sum(-1).mean((1, 2))
                    diff = (cc-mm).flatten(1).norm(dim=1)/cc.flatten(1).norm(dim=1).clamp_min(1e-8)
                    n = occ.flatten(1)
                    relative.extend(diff[present].cpu().tolist())
                    cells.extend((n>0).sum(1)[present].cpu().tolist())
                    effective.extend((n.sum(1).square()/n.square().sum(1).clamp_min(1e-8))[present].cpu().tolist())
                    kl.extend(k[present].cpu().tolist())
                results.append(dict(cohort=cohort, polarization=('VV','VH')[p], n=len(cells),
                                    context_difference_mean=float(np.mean(relative)),
                                    context_difference_median=float(np.median(relative)),
                                    valid_cells_median=float(np.median(cells)),
                                    occupancy_effective_cells_median=float(np.median(effective)),
                                    response_weight_KL_uniform_mean=float(np.mean(kl))))
    pd.DataFrame(results).to_csv(ROOT/'context_diagnostics.csv', index=False)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == '__main__':
    main()
