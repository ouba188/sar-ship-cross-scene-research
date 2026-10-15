"""Source-only episode construction and fixed-mass logistic fitting."""

import numpy as np
from sklearn.linear_model import LogisticRegression


def draw_ships(frame, indices, n, rng):
    ships = rng.permutation(np.sort(frame.iloc[indices].mmsi.unique()))[:n]
    ids = frame.iloc[indices].mmsi.to_numpy()
    return np.array([rng.choice(indices[ids == ship]) for ship in ships], dtype=int)


def matched_support(frame, pool, labels, rng):
    counts = np.bincount(labels, minlength=8)
    used, answer = set(), []
    order = sorted(np.flatnonzero(counts), key=lambda c:
                   frame.iloc[pool].loc[lambda z:z.class_id == c].mmsi.nunique() / counts[c])
    for cls in order:
        candidates = pool[(frame.iloc[pool].class_id.to_numpy() == cls) &
                          ~frame.iloc[pool].mmsi.isin(used).to_numpy()]
        if frame.iloc[candidates].mmsi.nunique() < counts[cls]:
            return None
        chosen = draw_ships(frame, candidates, int(counts[cls]), rng)
        answer.extend(chosen)
        used.update(frame.iloc[chosen].mmsi)
    return np.array(answer, dtype=int)


def make_episodes(frame, assignment):
    episodes, availability, skipped = [], [], []
    for fold in range(3):
        indices = np.flatnonzero(assignment == fold)
        ships = np.random.default_rng(202610130+fold).permutation(np.sort(frame.iloc[indices].mmsi.unique()))
        for bucket, query_ships in enumerate(np.array_split(ships,4)):
            inside = frame.iloc[indices].mmsi.isin(query_ships).to_numpy()
            q, pool = indices[inside], indices[~inside]
            for draw in range(3):
                rng = np.random.default_rng(202610200+fold*100+bucket*10+draw)
                s = draw_ships(frame,pool,16,rng)
                episodes.append(dict(kind='expanded',fold=fold,key=f'e{fold}_{bucket}_{draw}',
                    support=s.tolist(),query=q.tolist(),port='mixed',draw=draw))
        for pi, port in enumerate(sorted(frame.port.unique())):
            local = indices[frame.iloc[indices].port.to_numpy() == port]
            nships = frame.iloc[local].mmsi.nunique()
            availability.append(dict(fold=fold,port=port,ships=nships,observations=len(local),eligible=nships>=19))
            if nships < 19:
                continue
            for draw in range(3):
                rng = np.random.default_rng(202611000+fold*1000+pi*10+draw)
                s = draw_ships(frame,local,16,rng)
                q = local[~frame.iloc[local].mmsi.isin(frame.iloc[s].mmsi).to_numpy()]
                pool = indices[(frame.iloc[indices].port.to_numpy()!=port) &
                    ~frame.iloc[indices].mmsi.isin(frame.iloc[q].mmsi).to_numpy()]
                other = matched_support(frame,pool,frame.iloc[s].class_id.to_numpy(int),rng)
                if other is None:
                    skipped.append(dict(fold=fold,port=port,draw=draw,reason='distinct_ship_class_match_unavailable'))
                    continue
                key=f'p{fold}_{pi}_{draw}'
                for kind, support in [('local',s),('matched_mixed',other)]:
                    episodes.append(dict(kind=kind,fold=fold,key=key,port=port,draw=draw,
                        support=support.tolist(),query=q.tolist()))
    for ep in episodes:
        s,q = np.array(ep['support']),np.array(ep['query'])
        assert len(s)==16 and frame.iloc[s].mmsi.nunique()==16
        assert not set(frame.iloc[s].mmsi) & set(frame.iloc[q].mmsi)
        assert np.all(assignment[np.r_[s,q]]==ep['fold'])
    for ep in (e for e in episodes if e['kind']=='local'):
        other = next(e for e in episodes if e['key']==ep['key'] and e['kind']=='matched_mixed')
        np.testing.assert_array_equal(ep['query'],other['query'])
        np.testing.assert_array_equal(np.bincount(frame.iloc[ep['support']].class_id,minlength=8),
            np.bincount(frame.iloc[other['support']].class_id,minlength=8))
    return episodes,availability,skipped


def fit_fixed_mass(x, gain, weights, width, mass):
    keep = gain!=0
    xx, yy, w = x[keep,:width], (gain[keep]>0).astype(int), weights[keep]
    mean = np.average(xx,axis=0,weights=w)
    scale = np.sqrt(np.average((xx-mean)**2,axis=0,weights=w))
    scale[scale<1e-12]=1.
    model = LogisticRegression(C=.1,solver='lbfgs',max_iter=1000,tol=1e-9)
    model.fit((xx-mean)/scale,yy,sample_weight=w*(mass/w.sum()))
    return dict(mean=mean,scale=scale,coef=model.coef_[0],intercept=model.intercept_[0],
        width=width,events=int(keep.sum()),event_weight_mass=mass,iterations=int(model.n_iter_[0]))
