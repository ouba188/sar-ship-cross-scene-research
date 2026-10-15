"""Source-only label-informed upper bound, never a deployable classifier."""

from correct_pair_margin import BASE, CACHE, OUT, old, np, pd, torch, save_json
from scipy.optimize import linprog


def oracle_margin(scores, effects, label, bound=.1):
    rivals = np.arange(len(scores)) != label
    gaps = scores[label]-scores[rivals]
    delta = effects[:, label, None]-effects[:, rivals]
    # t <= original_gap + delta^T gain, for every rival class.
    inequalities = np.column_stack([-delta.T, np.ones(len(gaps))])
    objective = np.r_[np.zeros(effects.shape[0]), -1.]
    result = linprog(objective, A_ub=inequalities, b_ub=gaps,
                     bounds=[(-bound, bound)]*effects.shape[0]+[(None, None)], method='highs')
    if not result.success:
        raise RuntimeError(result.message)
    gain = result.x[:-1]
    changed = scores+gain@effects
    measured = float(changed[label]-changed[rivals].max())
    np.testing.assert_allclose(measured, result.x[-1], atol=1e-7, rtol=1e-7)
    return measured, gain


def main():
    torch.set_num_threads(1)
    destination = OUT/'capacity_diagnosis'
    destination.mkdir(exist_ok=False)
    meta = pd.read_csv(BASE/'observations.csv')
    fixed = torch.load(old.OUT/'fixed_classifier.pt', map_location='cpu', weights_only=True)
    with np.load(OUT/'source_split.npz') as z:
        roles = {k: z[k] for k in ('fit', 'test_source')}
    rows = []
    for group, ids in roles.items():
        raw = np.array(np.load(CACHE/'frozen_replay/source_extra.npy', mmap_mode='r')[meta.iloc[ids].manifest_row], np.float32)
        h = (torch.from_numpy(raw)-fixed['center'])/fixed['scale']/np.sqrt(512)
        scores = (h@fixed['weight']+fixed['bias']).numpy().astype(np.float64)
        coordinates = (h@fixed['basis']).numpy().astype(np.float64)
        axes = (fixed['basis'].T@fixed['weight']).numpy().astype(np.float64)
        labels = meta.iloc[ids].class_id.to_numpy()
        for i, (oid, label) in enumerate(zip(ids, labels)):
            effects = coordinates[i, :, None]*axes
            margin, gain = oracle_margin(scores[i], effects, label)
            rivals = np.arange(8) != label
            raw_margin = float(scores[i, label]-scores[i, rivals].max())
            assert margin >= raw_margin-1e-7
            assert np.max(np.abs(gain)) <= .1+1e-7
            row = dict(group=group, obs_id=oid, class_id=label, raw_correct=int(scores[i].argmax() == label),
                       raw_margin=raw_margin, oracle_margin=margin, oracle_correct=int(margin > 1e-7),
                       near_boundary=int(abs(margin) <= 1e-7))
            row['max_absolute_gain'] = float(np.max(np.abs(gain)))
            rows.append(row)
        print('CAPACITY', group, len(ids), flush=True)
    data = pd.DataFrame(rows)
    data.to_csv(destination/'per_sample.csv', index=False)
    summary = []
    for group, g in data.groupby('group'):
        wrong = g.raw_correct == 0
        summary.append(dict(group=group, samples=len(g), baseline_OA=100*g.raw_correct.mean(),
            oracle_OA_ceiling=100*g.oracle_correct.mean(), baseline_BA=100*g.groupby('class_id').raw_correct.mean().mean(),
            oracle_BA_ceiling=100*g.groupby('class_id').oracle_correct.mean().mean(),
            wrong_samples=int(wrong.sum()), correctable_wrong=int(g.loc[wrong, 'oracle_correct'].sum()),
            immovable_wrong_percent=100*(g.loc[wrong, 'oracle_correct'] == 0).mean(), boundary_samples=int(g.near_boundary.sum())))
    table = pd.DataFrame(summary)
    table.to_csv(destination/'summary.csv', index=False)
    data.groupby(['group', 'class_id'])[['raw_correct', 'oracle_correct']].mean().mul(100).reset_index().to_csv(destination/'classes.csv', index=False)
    save_json(destination/'COMPLETE.json', dict(samples=len(data), scope='Source-only independent per-image gains at fixed +-0.1. Original frozen classifier; NOT target support-adapted output or deployable accuracy. Ignores context learnability, so an optimistic upper bound.'))
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
