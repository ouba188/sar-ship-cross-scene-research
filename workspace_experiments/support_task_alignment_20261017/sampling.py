"""Support identity resampling with fixed queries, class quotas and priors."""
import numpy as np
import pandas as pd


def metadata(data):
    frames = []
    for role, indices in [('source_train', 'train'), ('source_val', 'val')]:
        frame = data[role].copy()
        frame['local'] = data[indices]
        frames.append(frame)
    return pd.concat(frames).set_index('local', verify_integrity=True)


def legal_pool(data, episode, kind, meta):
    rows = data['train' if kind == 'fit' else 'val']
    frame = meta.loc[rows]
    forbidden = set(meta.loc[np.r_[episode['q'], episode['prior_rows']], 'mmsi'])
    pool = frame[frame.acquisition.eq(episode['support_acquisition']) & ~frame.mmsi.isin(forbidden)]
    pool = pool.sort_values('sample_id').drop_duplicates('mmsi')
    indices = pool.index.to_numpy(int)
    if not set(episode['s']).issubset(indices):
        raise AssertionError('Original support is not in the fixed-prior legal pool')
    return indices


def draw_like(original, pool, labels, rng):
    result = np.empty_like(original)
    for label in np.unique(labels[original]):
        places = np.flatnonzero(labels[original] == label)
        candidates = pool[labels[pool] == label]
        result[places] = rng.choice(candidates, len(places), replace=False)
    return result


def support_variants(data, episodes, kind, seed, count=5):
    meta, rng = metadata(data), np.random.default_rng(seed)
    banks, records = [], []
    for index, episode in enumerate(episodes):
        pool = legal_pool(data, episode, kind, meta)
        original = episode['s']
        variants = [original.copy()] + [draw_like(original, pool, data['labels'], rng) for _ in range(count-1)]
        query_ships = set(meta.loc[episode['q'], 'mmsi'])
        prior_ships = set(meta.loc[episode['prior_rows'], 'mmsi'])
        quota = np.bincount(data['labels'][original], minlength=8)
        signatures = {tuple(sorted(s)) for s in variants}
        for draw, support in enumerate(variants):
            np.testing.assert_array_equal(np.bincount(data['labels'][support], minlength=8), quota)
            ships = set(meta.loc[support, 'mmsi'])
            assert len(ships) == len(support) == 16
            assert not ships & (query_ships | prior_ships)
            assert set(meta.loc[support, 'acquisition']) == {episode['support_acquisition']}
            records.append(dict(kind=kind, episode=index, origin=episode['origin'], acquisition=episode['acquisition'],
                support_acquisition=episode['support_acquisition'], draw=draw, pool_size=len(pool),
                unique_sets=len(signatures), changed_members=16-len(set(support)&set(original)),
                support_global_rows=data['rows'][support].tolist(), query_global_rows=data['rows'][episode['q']].tolist(),
                prior_global_rows=data['rows'][episode['prior_rows']].tolist(), class_counts=quota.tolist()))
        banks.append(variants)
    return banks, records


def training_schedule(seed, tasks, steps=200, draws=5):
    task_rng = np.random.default_rng(20261017+seed)
    draw_rng = np.random.default_rng(20262017+seed)
    families = np.stack([task_rng.choice(tasks, min(2, tasks), replace=False) for _ in range(steps)])
    choices = draw_rng.integers(draws, size=families.shape)
    return families, choices
