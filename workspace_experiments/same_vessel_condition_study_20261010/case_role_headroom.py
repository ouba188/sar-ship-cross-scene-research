"""Separate support-history and query-history benefit without fitting a model."""

from full_teacher_data import BASE, PORTS, write
from case_relation_headroom import OUT as TEACHER, SEEDS, metric, match_energy, r18_columns
from unified_head_core import coefficients
import numpy as np
import pandas as pd

OUT = BASE/'case_role_headroom_20261012'


def main():
    OUT.mkdir(exist_ok=True)
    write(OUT/'PROTOCOL.json', dict(
        question='Which side of same-vessel history creates actual recognition headroom?',
        arms=['raw', 'support_only', 'query_only', 'both'],
        scales=['native', 'matched'], ridge=8, supports=64,
        prior='Shared original vessel-held source prior, unchanged',
        geometry='Coherent role-specific feature maps for both Kss and Kqs',
        access='History arms are privileged diagnostics, NOT deployable results',
        held='Repeated source-held 823 observations / 302 vessels, not new target evaluation',
        gate='Train only a bounded legal support-completion pilot if support-only history improves BA',
        no_hyperparameter_search=True))
    meta = pd.read_csv(BASE/'observations.csv')
    rows, classes, events = [], [], []
    replay = []
    for port in PORTS:
        dest = OUT/port
        dest.mkdir(exist_ok=True)
        with np.load(TEACHER/port/'maps.npz') as z:
            ids, prior, raw, teacher = (z[k] for k in ('obs_ids','prior','single','equal_r18'))
        index = pd.Index(ids)
        y = meta.iloc[ids].class_id.to_numpy()
        for seed in SEEDS:
            for draw in range(3):
                with np.load(TEACHER/port/f'{seed}_{draw}_single_native.npz') as z:
                    so, qo, baseline = (z[k] for k in ('support_obs_ids','query_obs_ids','scores'))
                s, q = index.get_indexer(so), index.get_indexer(qo)
                assert min(s.min(), q.min()) >= 0
                assert not set(meta.iloc[so].mmsi) & set(meta.iloc[qo].mmsi)
                bp = baseline.argmax(1)
                for scaling in ('native','matched'):
                    taught = teacher if scaling == 'native' else match_energy(raw,teacher,s,r18_columns())[0]
                    for arm in ('raw','support_only','query_only','both'):
                        ps = taught[s] if arm in ('support_only','both') else raw[s]
                        pq = taught[q] if arm in ('query_only','both') else raw[q]
                        kss, kqs = ps@ps.T, pq@ps.T
                        alpha = coefficients(kss, prior[s], y[s], 8.)
                        scores = prior[q]+kqs@alpha
                        if arm in ('raw','both'):
                            expected = baseline
                            if arm == 'both':
                                with np.load(TEACHER/port/f'{seed}_{draw}_equal_r18_{scaling}.npz') as z:
                                    expected = z['scores']
                            error = float(np.max(np.abs(scores-expected)))
                            assert error < 1e-10, error
                            replay.append(error)
                        pred = scores.argmax(1)
                        row = dict(context=port, seed=seed, draw=draw, scaling=scaling, arm=arm)
                        rows.append(dict(**row, **metric(y[q],pred),
                            rescue=int(np.sum((bp!=y[q])&(pred==y[q]))),
                            harm=int(np.sum((bp==y[q])&(pred!=y[q])))))
                        for c in np.unique(y[q]):
                            mask = y[q]==c
                            classes.append(dict(**row, class_id=int(c), n=int(mask.sum()),
                                                recall=float(100*np.mean(pred[mask]==c))))
                        for j in np.flatnonzero((bp==y[q])!=(pred==y[q])):
                            events.append(dict(**row, obs_id=int(qo[j]),
                                mmsi=str(meta.iloc[qo[j]].mmsi), class_id=int(y[q[j]]),
                                before=int(bp[j]),after=int(pred[j]),rescue=bool(pred[j]==y[q[j]])))
                        np.savez_compressed(dest/f'{seed}_{draw}_{arm}_{scaling}.npz',
                            support_obs_ids=so,query_obs_ids=qo,scores=scores)
                print('EVALUATED',port,seed,draw,flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(classes).to_csv(OUT/'classes.csv',index=False)
    pd.DataFrame(events).to_csv(OUT/'events.csv',index=False)
    summary = df.groupby(['scaling','arm'])[['oa','ba','rescue','harm']].mean()
    summary.to_csv(OUT/'summary.csv')
    df.groupby(['context','scaling','arm'])[['oa','ba']].mean().to_csv(OUT/'contexts.csv')
    write(OUT/'REPLAY.json',dict(replays=len(replay),max_error=max(replay)))
    print(summary.to_string(),flush=True)


if __name__ == '__main__':
    main()
