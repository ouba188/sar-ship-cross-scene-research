"""Task margins and post-training feature interventions, not new model selection."""

from run_task_pair_relation import BASE, OUT, PARENT, SEEDS, load, make_episode, apply, write_json
import numpy as np
import pandas as pd
from task_pair_relation import ARMS, AMPLITUDE, observation_coordinates, balanced_ce
from condition_support_matching import metrics


def main():
    meta = pd.read_csv(BASE/'observations.csv')
    e = load(BASE/'condition_evidence_20261011/evidence_features.npz')
    features = np.full((len(meta), 6), np.nan); features[e['obs_ids']] = e['features']
    normalization = load(OUT/'normalization.npz')
    coordinates, _ = observation_coordinates(features, None, normalization)
    angles = meta.incidence_deg.to_numpy(float)
    interventions, events, margins, ports, leverage = [], [], [], [], []
    def bound(ep, seed, draw, split):
        y, base = ep['y'], ep['base']
        old = base.argmax(1)
        other = base.copy(); other[np.arange(len(y)), y] = -np.inf
        rival = other.argmax(1)
        contrast = ep['residual'][:, y].T-ep['residual'][:, rival].T
        upper = AMPLITUDE*np.abs(ep['op']*contrast*ep['valid']).sum(1)
        margin = base[np.arange(len(y)), y]-base[np.arange(len(y)), rival]
        wrong = old != y
        leverage.append(dict(seed=seed, draw=draw, split=split, queries=len(y),
            eligible_queries=int(ep['valid'].any(1).sum()), wrong=int(wrong.sum()),
            eligible_wrong=int((wrong & ep['valid'].any(1)).sum()),
            wrong_with_possible_fixed_rival_flip=int((wrong & (upper > -margin)).sum()),
            correct_with_possible_fixed_rival_harm=int((~wrong & (upper > margin)).sum()),
            mean_optimistic_margin_bound=float(upper.mean()),
            mean_wrong_deficit=float(-margin[wrong].mean())))
    for seed in SEEDS:
        bundle = load(PARENT/f'full_{seed}_inputs.npz')
        models = {arm: load(OUT/f'{seed}_{arm}_model.npz') for arm in ARMS}
        validation = load(BASE/'crossfit_evidence_20261011/validation_episodes.npz')
        for draw in range(3):
            ep = make_episode(meta, bundle, validation[f'val_new_ship_{draw}_support'],
                validation[f'val_new_ship_{draw}_query'], coordinates, angles, angles, normalization['limits'])
            bound(ep, seed, draw, 'validation')
        for draw in range(3):
            saved = load(OUT/f'{seed}_source_{draw}_predictions.npz')
            s, q = saved['support_obs_ids'], saved['query_obs_ids']
            ep = make_episode(meta, bundle, s, q, coordinates, angles, angles, normalization['limits'])
            bound(ep, seed, draw, 'held')
            y = meta.class_id.to_numpy()[q]
            base = ep['base']; old = base.argmax(1)
            competitors = base.copy(); competitors[np.arange(len(y)), y] = -np.inf
            rival = competitors.argmax(1)
            old_margin = base[np.arange(len(y)), y]-base[np.arange(len(y)), rival]
            for arm in ARMS:
                for state in ('selected', 'last'):
                    score = saved[f'{arm}_{state}']; new = score.argmax(1)
                    change = (score-base)[np.arange(len(y)), y]-(score-base)[np.arange(len(y)), rival]
                    for group, use in [('all', np.ones(len(y), bool)), ('wrong', old != y), ('correct', old == y),
                                       ('near_boundary_correct', (old == y) & (old_margin < .05))]:
                        margins.append(dict(seed=seed, draw=draw, arm=arm, state=state, group=group,
                            n=int(use.sum()), margin_change=float(change[use].mean()) if use.any() else np.nan))
                    for j in np.flatnonzero((old == y) != (new == y)):
                        events.append(dict(seed=seed, draw=draw, arm=arm, state=state, obs_id=int(q[j]),
                            mmsi=str(meta.iloc[q[j]].mmsi), port=meta.iloc[q[j]].port, class_id=int(y[j]),
                            event='rescue' if new[j] == y[j] else 'harm', old_margin=float(old_margin[j]),
                            margin_change=float(change[j]), old_prediction=int(old[j]), new_prediction=int(new[j])))
                    for port in np.unique(meta.iloc[q].port):
                        use = meta.iloc[q].port.to_numpy() == port
                        ports.append(dict(seed=seed, draw=draw, arm=arm, state=state, port=port, n=int(use.sum()),
                            baseline_BA=metrics(base[use], y[use])['BA'], **metrics(score[use], y[use])))
            # These interventions retain learned shared coefficients; no refitting or held selection.
            for state in ('selected', 'last'):
                for name, index in [('all_angle_terms_off', slice(4, 8)), ('angle_interactions_off', slice(6, 8))]:
                    model = {k: v.copy() for k, v in models['condition'].items()}
                    model[state][index] = 0
                    score, _ = apply(ep, 'condition', model, state)
                    interventions.append(dict(seed=seed, draw=draw, state=state, intervention=name,
                                              **metrics(score, y), CE=balanced_ce(score, y)))
    for name, data in [('interventions', interventions), ('events', events), ('margins', margins), ('ports', ports),
                       ('leverage_bounds', leverage)]:
        pd.DataFrame(data).to_csv(OUT/f'{name}.csv', index=False)
    summary = pd.DataFrame(interventions).groupby(['state', 'intervention'])[['OA', 'BA', 'CE']].mean()
    summary.to_csv(OUT/'intervention_summary.csv')
    write_json(OUT/'DIAGNOSTICS_COMPLETE.json', dict(refits=0, selection_changed=False,
        feature_interventions='Coefficient removal after training; NOT physical angle intervention or causal proof.',
        leverage='Optimistic per-edge independently chosen +/-amplitude bound against fixed original rival. Necessary not sufficient for correction; not a deployable oracle or achieved accuracy.'))
    print(summary.to_string())


if __name__ == '__main__':
    main()
