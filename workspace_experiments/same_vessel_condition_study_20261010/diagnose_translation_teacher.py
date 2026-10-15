"""Fixed-reader direction-term removal and event attribution; no new fitting."""

import json
import numpy as np
import pandas as pd
from check_translation_teacher_feasibility import BASE, OUT, ARMS, SEEDS, selection
from check_motion_teacher_feasibility import assess


def main():
    meta = pd.read_csv(BASE/'observations.csv')
    rows, by_class, effects, leave_one_ship, donor_trace = [], [], [], [], []
    for role in ('fit', 'held'):
        for seed in SEEDS:
            with np.load(OUT/f'{role}_{seed}_all_history.npz') as f:
                ids = f['obs_ids']
            mm = meta.iloc[ids].reset_index(drop=True)
            cells = pd.read_csv(OUT/f'{role}_cells_{seed}.csv')
            members = [g.index.to_numpy() for _, g in mm.groupby(['mmsi', 'product'], sort=True)]
            assert len(members) == len(cells)
            for i, ix in enumerate(members):
                assert mm.iloc[ix[0]].mmsi == cells.iloc[i].mmsi
                assert mm.iloc[ix[0]]['product'] == cells.iloc[i]['product']
            with np.load(OUT/f'{role}_quality_{seed}.npz') as f:
                x = f['x']; cs = f['scores']
                full_quality = f['direction']
            if role == 'fit':
                with np.load(BASE/'geometry_train_teacher_20261011'/f'teachers_{seed}.npz') as f:
                    np.testing.assert_array_equal(f['fit_obs_ids'], ids)
                    scores = f['original_oof']
                quality = np.full(len(cells), np.nan)
                for k in range(3):
                    with np.load(OUT/f'fold{k}_direction_{seed}.npz') as f:
                        beta = f['beta']
                    use = cells.fold.to_numpy() == k
                    quality[use] = x[use, :3]@beta[:3]
            else:
                with np.load(BASE/'crossfit_evidence_20261011'/f'full_{seed}_outputs.npz') as f:
                    scores = f['prior'][ids]
                with np.load(OUT/f'full_direction_{seed}.npz') as f:
                    beta = f['beta']
                quality = x[:, :3]@beta[:3]
            for matched in (False, True):
                pool = 'same_port_angle' if matched else 'all_history'
                p, multi, eligible, _ = selection(mm, scores, cells, cs, members,
                                                  {'direction': quality}, matched)
                with np.load(OUT/f'{role}_{seed}_{pool}.npz') as f:
                    original = f['direction_soft']; equal = f['equal']
                    speed = f['speed_soft']; capacity = f['capacity_soft']
                    np.testing.assert_allclose(equal, p['equal'])
                    np.testing.assert_array_equal(eligible, f['eligible'])
                removed = p['direction_soft']
                truth = mm.loc[multi].class_id.to_numpy()
                base_correct = equal[multi].argmax(1) == truth
                for name, pred in (('direction', original), ('direction_terms_removed', removed)):
                    correct = pred[multi].argmax(1) == truth
                    rows.append(dict(role=role, seed=seed, pool=pool, arm=name,
                        **assess(pred[multi], truth, mm.loc[multi]),
                        rescue=int((correct & ~base_correct).sum()), harm=int((~correct & base_correct).sum())))
                if role != 'held' or matched:
                    continue
                # This vessel accounts for all repeated container rescues in the
                # saved direction-term intervention. This is sensitivity, not a new cohort claim.
                remaining = multi & (mm.mmsi.to_numpy() != 563050500)
                for name, pred in (('equal', equal), ('speed', speed), ('capacity', capacity),
                                   ('direction', original), ('direction_removed', removed)):
                    leave_one_ship.append(dict(seed=seed, excluded_mmsi=563050500, arm=name,
                        n=int(remaining.sum()), **assess(pred[remaining],
                        mm.loc[remaining].class_id.to_numpy(), mm.loc[remaining])))
                ship_cells = cells.index[cells.mmsi == 563050500].to_numpy()
                for focal in ship_cells:
                    donor = ship_cells[ship_cells != focal]
                    known = cells.loc[donor, 'known'].to_numpy(bool)
                    if known.sum() < 2:
                        continue
                    weights = []
                    for qq in (quality, full_quality):
                        q = qq[donor[known]]; span = np.ptp(q)
                        local = 1+(q-q.min())/span if span > 1e-12 else np.ones(len(q))
                        w = np.ones(len(donor)); w[known] = local/local.mean()
                        weights.append(w/len(donor))
                    for j, d in enumerate(donor):
                        y = int(cells.loc[d, 'class_id'])
                        rival = cs[d].copy(); rival[y] = -np.inf
                        donor_trace.append(dict(seed=seed, focal_obs_id=int(ids[members[focal][0]]),
                            donor_obs_id=int(ids[members[d][0]]), metadata_known=bool(known[j]),
                            weight_without=float(weights[0][j]), weight_with=float(weights[1][j]),
                            quality_without=float(quality[d]), quality_with=float(full_quality[d]),
                            true_class_margin=float(cs[d, y]-rival.max()), donor_prediction=int(cs[d].argmax())))
                old, now = removed.argmax(1), original.argmax(1)
                for cls in np.unique(truth):
                    mask = multi & (mm.class_id.to_numpy() == cls)
                    y = mm.loc[mask].class_id.to_numpy()
                    by_class.append(dict(seed=seed, class_id=int(cls), n=int(mask.sum()),
                        equal_recall=float(100*np.mean(equal[mask].argmax(1) == y)),
                        removed_recall=float(100*np.mean(old[mask] == y)),
                        direction_recall=float(100*np.mean(now[mask] == y))))
                for i in np.flatnonzero(multi & (old != now)):
                    y = int(mm.iloc[i].class_id)
                    effects.append(dict(seed=seed, obs_id=int(ids[i]), mmsi=int(mm.iloc[i].mmsi),
                        class_id=y, without=int(old[i]), with_direction=int(now[i]),
                        rescued=bool(old[i] != y and now[i] == y),
                        harmed=bool(old[i] == y and now[i] != y)))
            # Remove-only diagnostic: keep learned non-direction coefficients fixed.
            np.savez_compressed(OUT/f'{role}_direction_removal_{seed}.npz',
                                original_quality=full_quality, removed_quality=quality)
    pd.DataFrame(rows).to_csv(OUT/'direction_removal.csv', index=False)
    pd.DataFrame(by_class).to_csv(OUT/'held_direction_by_class.csv', index=False)
    pd.DataFrame(effects).to_csv(OUT/'held_direction_term_events.csv', index=False)
    pd.DataFrame(leave_one_ship).to_csv(OUT/'held_leave_rescue_vessel_out.csv', index=False)
    pd.DataFrame(donor_trace).to_csv(OUT/'container_rescue_donor_trace.csv', index=False)
    (OUT/'DIAGNOSIS_COMPLETE.json').write_text(json.dumps(dict(new_fits=0,
        procedure='Post-fit direction term removal, non-direction coefficients unchanged; diagnostic, not newly validated policy.')),
        encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))
    print('DIRECTION TERMS', pd.DataFrame(effects).groupby('seed')[['rescued', 'harmed']].sum().to_string())
    print('LEAVE SHIP OUT', pd.DataFrame(leave_one_ship)[['seed', 'arm', 'n', 'OA', 'BA']].to_string(index=False))


if __name__ == '__main__':
    main()
