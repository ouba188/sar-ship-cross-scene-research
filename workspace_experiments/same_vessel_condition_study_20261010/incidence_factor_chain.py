"""Source-only incidence-to-representation-to-decision diagnostic."""

import itertools
import sys
import time
import decision_condition_consistency as previous
from decision_condition_consistency import BASE, CACHE, PREVIOUS, np, pd, torch
from decision_condition_consistency import FactorNet, EvidenceGate, SEEDS, save_json, standardize

OUT = BASE/'incidence_factor_chain_20261011'


def matched_pairs(meta, ids):
    group = meta.iloc[ids]
    valid = np.isfinite(group.incidence_deg.to_numpy(float))
    valid &= group.pass_name.isin(['Ascending', 'Descending']).to_numpy()
    valid &= group.platform.notna().to_numpy()
    pairs = []
    for _, cell in group.loc[valid].groupby(['mmsi', 'port', 'platform', 'pass_name']):
        cell = cell.sort_values(['acq_time', 'obs_id'])
        for a, b in itertools.combinations(cell.index, 2):
            if meta.loc[a, 'product'] != meta.loc[b, 'product']:
                pairs.append((a, b))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def pair_table(meta, pairs):
    a, b = pairs.T
    relative = meta.relative_signed_deg.to_numpy(float)
    gap = abs((relative[b]-relative[a]+90)%180-90)
    return pd.DataFrame(dict(a=a, b=b, mmsi=meta.iloc[a].mmsi.to_numpy(),
        port=meta.iloc[a].port.to_numpy(), class_id=meta.iloc[a].class_id.to_numpy(),
        delta_angle=meta.incidence_deg.to_numpy(float)[b]-meta.incidence_deg.to_numpy(float)[a],
        relative_gap=gap, relative_matched=gap <= 10,
        scene_a=meta.iloc[a]['product'].to_numpy(), scene_b=meta.iloc[b]['product'].to_numpy()))


def ship_weights(table):
    count = table.mmsi.value_counts()
    w = 1/table.mmsi.map(count).to_numpy(float)
    return w/w.sum()


def fit_slope(values, table):
    x = table.delta_angle.to_numpy()/5.
    delta = values[table.b]-values[table.a]
    w = ship_weights(table)
    energy = float(w@(x*x))
    return ((w*x)@delta)/(1.05*energy+1e-12)


def evaluate_slope(values, table, slope):
    if not len(table):
        return dict(pairs=0, ships=0)
    x = table.delta_angle.to_numpy()/5.
    delta = values[table.b]-values[table.a]
    prediction = x[:, None]*slope
    zero = (delta*delta).mean(1)
    error = ((delta-prediction)**2).mean(1)
    w = ship_weights(table)
    direction = slope/max(np.linalg.norm(slope), 1e-12)
    projected, expected = delta@direction, x*np.linalg.norm(slope)
    result = dict(pairs=len(table), ships=int(table.mmsi.nunique()),
        explained_percent=float(100*(1-w@error/max(w@zero, 1e-12))),
        signed_alignment=float(w@(projected*expected)),
        angle_energy=float(w@(x*x)), slope_norm=float(np.linalg.norm(slope)))
    grouped = pd.DataFrame(dict(ship=table.mmsi.to_numpy(), zero=zero, error=error)).groupby('ship')[['zero', 'error']].mean().to_numpy()
    rng = np.random.default_rng(20261011)
    boot = grouped[rng.integers(0, len(grouped), size=(400, len(grouped)))].mean(1)
    ci = np.quantile(100*(1-boot[:, 1]/np.maximum(boot[:, 0], 1e-12)), [.025, .975])
    result.update(descriptive_low=float(ci[0]), descriptive_high=float(ci[1]))
    return result


def decision_metrics(scores, labels, table):
    pred = scores.argmax(-1)
    good = pred == labels[:, None]
    w = ship_weights(table)
    prob = scores-torch.as_tensor(scores).logsumexp(-1, keepdim=True).numpy()
    probabilities = np.exp(prob)
    mixture = .5*(probabilities[:, 0]+probabilities[:, 1])
    js = .5*(probabilities*(prob-np.log(np.maximum(mixture[:, None], 1e-20)))).sum((1, 2))
    correct = np.take_along_axis(scores, labels[:, None, None].repeat(2, 1), axis=-1).squeeze(-1)
    rival = scores.copy()
    for i, label in enumerate(labels):
        rival[i, :, label] = -np.inf
    margin = correct-rival.max(-1)
    return dict(both_correct=float(100*w@good.all(1)), same_wrong=float(100*w@((pred[:, 0] == pred[:, 1])&~good.any(1))),
        endpoint_accuracy=float(100*w@good.mean(1)), worst_margin=float(w@margin.min(1)), js=float(w@js))


def pair_intervention(features, table, slope):
    ids = table[['a', 'b']].to_numpy()
    value = features[ids].copy()
    shift = .5*(table.delta_angle.to_numpy()/5.)[:, None]*slope
    value[:, 0] += shift
    value[:, 1] -= shift
    return value


def random_slopes(slope):
    rng = np.random.default_rng(20261011)
    samples = rng.normal(size=(20, len(slope)))
    return samples/np.linalg.norm(samples, axis=1, keepdims=True)*np.linalg.norm(slope)


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=False)
    started = time.perf_counter()
    meta = pd.read_csv(BASE/'observations.csv')
    with np.load(PREVIOUS/'source_split.npz') as z:
        split = {k: z[k].copy() for k in z.files}
    fit, test = split['factor_fit'], split['test_source']
    assert not set(meta.iloc[fit].mmsi)&set(meta.iloc[test].mmsi)
    train = pair_table(meta, matched_pairs(meta, fit))
    held = pair_table(meta, matched_pairs(meta, test))
    seen = set(meta.iloc[fit]['product'])
    held['both_scenes_unseen'] = ~held.scene_a.isin(seen)&~held.scene_b.isin(seen)
    train.to_csv(OUT/'fit_pairs.csv', index=False)
    held.to_csv(OUT/'held_pairs.csv', index=False)
    coverage = []
    for name, table in [('fit', train), ('held', held)]:
        for subset, mask in [('all', np.ones(len(table), bool)), ('angle_gap2', table.delta_angle.abs() >= 2),
                             ('direction_matched', table.relative_matched)]:
            t = table.loc[mask]
            coverage.append(dict(group=name, subset=subset, pairs=len(t), ships=int(t.mmsi.nunique()), ports=int(t.port.nunique())))
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv', index=False)
    save_json(OUT/'PROTOCOL.json', dict(
        primary='Source-only paired incidence association and model-space intervention; not identification of physical causal factors.',
        source_fit=len(fit), source_held=len(test), fit_pairs=len(train), held_pairs=len(held),
        match='Same MMSI, port, platform, orbit; distinct products, sorted acquisition time. Missing fields excluded. Relative-angle<=10deg sensitivity reported, not silently treated as matched.',
        estimator='No-intercept delta_vector=(delta_incidence/5)*slope; equal vessel weight; ridge .05 times scalar design energy. No class-specific coefficients or label use in slope fit.',
        validation='Identity-held source pairs, additionally both-scene-unseen subset and leave-held-port-out response estimates. Frozen factor encoder is NOT held-port-out, explicitly a weaker transfer test.',
        intervention='Each real pair mapped to its midpoint incidence by equal/opposite predicted shifts. Paired mechanistic diagnostic, not a deployable single-chip algorithm or new rendered SAR observation.',
        controls='Original no-intervention; 20 predetermined equal-norm random directions; projected angular variation versus full vector error. No best-random selection.',
        uncertainty='400 vessel bootstrap descriptive intervals; shared scenes and repeated development remain. No target evaluation, no target labels, no new backbone training.',
        scope='B anchor condition branch and stable branch frozen at both seeds; original rawR18 h classifier and original B gate. Only side folder writes.'))
    raw = np.array(np.load(CACHE/'frozen_replay/source_extra.npy', mmap_mode='r')[meta.manifest_row], np.float32)
    fixed = torch.load(PREVIOUS/'fixed_classifier.pt', map_location='cpu', weights_only=True)
    h = ((torch.from_numpy(raw)-fixed['center'])/fixed['scale']/np.sqrt(512)).numpy()
    norm = (raw-split['center'])/split['scale']
    spaces, contexts, gates = {('raw', -1): h}, {}, {}
    for seed in SEEDS:
        encoder = FactorNet()
        encoder.load_state_dict(torch.load(PREVIOUS/f'anchor_{seed}_factor.pt', map_location='cpu', weights_only=True)['model'])
        encoder.eval().requires_grad_(False)
        with torch.no_grad():
            stable, condition = [], []
            for start in range(0, len(norm), 256):
                x = torch.from_numpy(norm[start:start+256])
                stable.append(encoder.stable(x))
                condition.append(encoder.condition(x))
            s, c = torch.cat(stable).numpy(), torch.cat(condition).numpy()
        spaces['stable', seed] = standardize(s, fit).astype(np.float32)
        # Reproduce the B gate's original462-sample condition scaling, not the newer1296 scaling.
        contexts[seed] = standardize(c, split['fit']).astype(np.float32)
        spaces['condition', seed] = contexts[seed]
        gate = EvidenceGate(fixed['basis'])
        gate.load_state_dict(torch.load(PREVIOUS/f'anchor_{seed}_gate.pt', map_location='cpu', weights_only=True))
        gates[seed] = gate.eval()
    rows, random_rows, port_rows, slopes, leverage = [], [], [], {}, []
    x = train.delta_angle.to_numpy()/5.
    energy = ship_weights(train)*x*x
    for port in sorted(train.port.unique()):
        leverage.append(dict(port=port, pairs=int((train.port == port).sum()), angle_energy_fraction=float(energy[train.port == port].sum()/energy.sum())))
    pd.DataFrame(leverage).to_csv(OUT/'fit_port_leverage.csv', index=False)
    subsets = {'fit': train, 'held': held, 'held_gap2': held.loc[held.delta_angle.abs() >= 2],
               'held_new_scene': held.loc[held.both_scenes_unseen], 'held_direction_matched': held.loc[held.relative_matched]}
    for (space, seed), values in spaces.items():
        slope = fit_slope(values, train)
        slopes[space, seed] = slope
        np.save(OUT/f'{space}_{seed}_slope.npy', slope)
        for group, table in subsets.items():
            rows.append(dict(space=space, seed=seed, group=group, **evaluate_slope(values, table, slope)))
        for index, control in enumerate(random_slopes(slope)):
            random_rows.append(dict(space=space, seed=seed, control=index, **evaluate_slope(values, held, control)))
        for port, table in held.groupby('port'):
            training = train.loc[train.port != port]
            coefficient = fit_slope(values, training)
            port_rows.append(dict(space=space, seed=seed, port=port, fit_ships=int(training.mmsi.nunique()),
                                  **evaluate_slope(values, table, coefficient)))
    pd.DataFrame(rows).to_csv(OUT/'representation_results.csv', index=False)
    pd.DataFrame(random_rows).to_csv(OUT/'random_representation.csv', index=False)
    pd.DataFrame(port_rows).to_csv(OUT/'port_transfer.csv', index=False)
    weight, bias = fixed['weight'].numpy(), fixed['bias'].numpy()
    decision = []
    ids = held[['a', 'b']].to_numpy()
    y = meta.iloc[ids[:, 0]].class_id.to_numpy()
    np.testing.assert_array_equal(y, meta.iloc[ids[:, 1]].class_id.to_numpy())
    for space, seed in [('raw', -1)]+[('condition', seed) for seed in SEEDS]:
        baseline_values = spaces[space, seed][ids]
        directions = [('original', -1, None), ('incidence', -1, slopes[space, seed])]
        directions += [('random', i, slope) for i, slope in enumerate(random_slopes(slopes[space, seed]))]
        for arm, control, coefficient in directions:
            values = baseline_values if coefficient is None else pair_intervention(spaces[space, seed], held, coefficient)
            with torch.no_grad():
                if space == 'raw':
                    score = values@weight+bias
                else:
                    corrected = gates[seed](torch.from_numpy(h[ids].reshape(-1, 512)),
                                            torch.from_numpy(values.astype(np.float32).reshape(-1, 32)))
                    score = (corrected@fixed['weight']+fixed['bias']).numpy().reshape(-1, 2, 8)
            decision.append(dict(space=space, seed=seed, arm=arm, control=control, **decision_metrics(score, y, held)))
            if arm != 'random':
                np.savez_compressed(OUT/f'{space}_{seed}_{arm}_pair_scores.npz', ids=ids, scores=score)
    pd.DataFrame(decision).to_csv(OUT/'decision_results.csv', index=False)
    save_json(OUT/'COMPLETE.json', dict(seconds=time.perf_counter()-started, spaces=len(spaces),
        held_pairs=len(held), held_ships=int(held.mmsi.nunique()), held_new_scene_pairs=int(held.both_scenes_unseen.sum()),
        strict_direction_matched_pairs=int(held.relative_matched.sum()), neural_training=False, target_evaluated=False, parent_modified=False))
    print('REPRESENTATION', pd.DataFrame(rows).to_string(index=False), flush=True)
    print('DECISION', pd.DataFrame(decision).query('arm != "random"').to_string(index=False), flush=True)


def inventory():
    meta = pd.read_csv(BASE/'observations.csv')
    with np.load(PREVIOUS/'source_split.npz') as split:
        print('SPLIT', split.files)
        for key in ('factor_fit', 'test_source'):
            pairs = matched_pairs(meta, split[key])
            a, b = pairs.T
            angle = meta.incidence_deg.to_numpy(float)
            gap = abs(angle[b]-angle[a])
            relative = meta.relative_signed_deg.to_numpy(float)
            rgap = abs((relative[b]-relative[a]+90)%180-90)
            print(key, dict(pairs=len(pairs), ships=int(meta.iloc[a].mmsi.nunique()),
                ports=int(meta.iloc[a].port.nunique()), gap_quantiles=np.quantile(gap, [0, .25, .5, .75, 1]).tolist(),
                gap2=int((gap >= 2).sum()), gap5=int((gap >= 5).sum()),
                direction_known=int(np.isfinite(rgap).sum()), direction_matched=int((rgap <= 10).sum())), flush=True)
            print(meta.iloc[a].port.value_counts().to_string(), flush=True)


def candidate_inventory():
    meta = pd.read_csv(BASE/'observations.csv')
    target_mask = meta.port.isin(['Los Angeles', 'Port Klang', 'Santos'])
    source = ~target_mask&~meta.mmsi.isin(set(meta.loc[target_mask, 'mmsi']))
    with np.load(PREVIOUS/'source_split.npz') as split:
        seen_ships = set(meta.iloc[np.r_[split['factor_fit'], split['classifier']]].mmsi)
        seen_scenes = set(meta.iloc[split['factor_fit']]['product'])
    tables, rows = [], []
    for group, selected in [('all_allowed_source', source), ('encoder_unseen_ships', source&~meta.mmsi.isin(seen_ships))]:
        table = pair_table(meta, matched_pairs(meta, np.flatnonzero(selected)))
        table['both_scenes_unseen'] = ~table.scene_a.isin(seen_scenes)&~table.scene_b.isin(seen_scenes)
        for subset, mask in [('matched_basic', np.ones(len(table), bool)), ('direction_matched', table.relative_matched),
                             ('direction_matched_gap2', table.relative_matched&(table.delta_angle.abs() >= 2)),
                             ('direction_matched_new_scenes', table.relative_matched&table.both_scenes_unseen)]:
            t = table.loc[mask]
            rows.append(dict(group=group, subset=subset, pairs=len(t), ships=int(t.mmsi.nunique()),
                             ports=int(t.port.nunique()), classes=int(t.class_id.nunique())))
        if group == 'encoder_unseen_ships':
            tables.append(table.loc[table.relative_matched])
    result = pd.DataFrame(rows)
    result.to_csv(OUT/'expanded_source_coverage.csv', index=False)
    pd.concat(tables, ignore_index=True).to_csv(OUT/'expanded_direction_candidates.csv', index=False)
    print(result.to_string(index=False), flush=True)


if __name__ == '__main__':
    if '--inventory' in sys.argv:
        inventory()
    elif '--candidate-inventory' in sys.argv:
        candidate_inventory()
    else:
        main()
