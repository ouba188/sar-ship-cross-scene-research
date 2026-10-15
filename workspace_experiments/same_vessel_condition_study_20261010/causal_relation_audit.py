"""Source-only audit of task-anchored relations, not causal identification."""

import numpy as np
import pandas as pd
import torch
from shared_evidence_distillation import (
    BASE, OUT as PREVIOUS, SEEDS, EvidenceStudent, centered, previous,
    FactorNet, BoundedOperator, encode_two, standardize, load_data, save_json, metrics,
)

OUT = BASE/'causal_relation_audit_20261011'
SCOPES = ('all_other', 'same_port', 'cross_port')
EPS = 1e-8


def scoped_teacher(scores, meta, scope):
    """Exclude the focal entire product; weight eligible products equally."""
    meta = meta.reset_index(drop=True)
    teacher = np.zeros_like(scores)
    count = np.zeros(len(meta), dtype=int)
    consensus = np.zeros_like(scores)
    angle_gap = np.full(len(meta), np.nan)
    orbit_change = np.full(len(meta), np.nan)
    for _, ship in meta.groupby('mmsi'):
        for i, row in ship.iterrows():
            donors = ship[ship['product'] != row['product']]
            if scope == 'same_port':
                donors = donors[donors.port == row.port]
            elif scope == 'cross_port':
                donors = donors[donors.port != row.port]
            products = [g for _, g in donors.groupby('product')]
            if not products:
                continue
            values = np.stack([scores[g.index].mean(0) for g in products])
            teacher[i] = values.mean(0)
            count[i] = len(products)
            consensus[i] = (values[:, int(row.class_id), None]-values > EPS).mean(0)
            if np.isfinite(row.incidence_deg):
                means = np.array([g.incidence_deg.mean() for g in products])
                finite = np.isfinite(means)
                if finite.any():
                    angle_gap[i] = np.abs(means[finite]-row.incidence_deg).mean()
            if pd.notna(row.pass_name):
                values_pass = [g.pass_name.dropna() for g in products]
                change = [float((v != row.pass_name).mean()) for v in values_pass if len(v)]
                if change:
                    orbit_change[i] = np.mean(change)
    return teacher, count, consensus, angle_gap, orbit_change


def relation_components(base, teacher, y):
    bm = base[y]-base
    tm = teacher[y]-teacher
    delta = tm-bm
    rivals = np.arange(8) != y
    benefit = (tm > EPS)&(delta > EPS)&rivals
    harm = (delta < -EPS)&rivals
    partial = (tm <= EPS)&(delta > EPS)&rivals
    offset = teacher-base
    full = float(np.mean((offset-offset.mean())**2))
    anchored = float(np.sum(delta[rivals]**2)/64)
    return bm, tm, dict(
        full_energy=full, true_class_energy=anchored,
        rival_only_energy=max(0., full-anchored),
        beneficial_energy=float(np.sum(delta[benefit]**2)/64),
        harmful_energy=float(np.sum(delta[harm]**2)/64),
        partial_recovery_energy=float(np.sum(delta[partial]**2)/64))


def balanced_ship_mean(frame, column):
    if not len(frame):
        return np.nan
    return float(frame.groupby(['class_id', 'mmsi'])[column].mean().groupby('class_id').mean().mean())


def aggregate(frame):
    columns = ['beneficial', 'partial_recovery', 'margin_harm', 'rescue', 'harm',
               'net_flip', 'teacher_positive', 'donor_positive_fraction', 'multiple_donors',
               'consensus_benefit', 'teacher_delta', 'student_delta']
    result = dict(chips=int(frame.obs_id.nunique()), ships=int(frame.mmsi.nunique()),
                  ports=int(frame.port.nunique()), classes=int(frame.class_id.nunique()))
    result.update({c: balanced_ship_mean(frame, c) for c in columns})
    helpful, harmful = frame[frame.beneficial], frame[frame.margin_harm]
    result['student_gain_on_beneficial'] = balanced_ship_mean(helpful, 'student_delta')
    result['self_gain_on_beneficial'] = balanced_ship_mean(helpful, 'self_delta')
    result['supervised_gain_on_beneficial'] = balanced_ship_mean(helpful, 'supervised_delta')
    result['student_follows_benefit'] = balanced_ship_mean(helpful, 'student_positive')
    result['student_follows_harm'] = balanced_ship_mean(harmful, 'student_negative')
    return result


def export_readout_directions():
    selected = pd.read_csv(OUT/'selected_pairs.csv')
    fixed = torch.load(previous.old.OUT/'fixed_classifier.pt', map_location='cpu', weights_only=True)
    pairs = selected[['class_id', 'rival']].to_numpy()
    weight, bias = fixed['weight'].numpy(), fixed['bias'].numpy()
    directions = np.stack([weight[:, y]-weight[:, k] for y, k in pairs], axis=1)
    offsets = np.array([bias[y]-bias[k] for y, k in pairs])
    basis, _ = np.linalg.qr(directions)
    np.savez_compressed(OUT/'candidate_readout_directions.npz', pairs=pairs,
        directions=directions, bias_offsets=offsets, basis=basis)
    save_json(OUT/'READOUT_SCOPE.json', dict(
        dimension=int(directions.shape[0]), rank=int(np.linalg.matrix_rank(directions)),
        interpretation='Existing classifier readout directions for source-selected class relations; not newly discovered causal vectors.',
        coordinate='h after original frozen B additive condition operator, not raw512 or stable128.',
        identity='class margin = h @ (W[:,y]-W[:,k]) + bias[y]-bias[k]',
        boundary='Removing these directions tests classifier dependence by construction, NOT their causal/invariant purity.'))
    frame = pd.read_csv(OUT/'relations.csv')
    frame = frame[frame.scope.eq('cross_port')].merge(selected, on=['class_id', 'rival'])
    rows = []
    for keys, group in frame.groupby(['role', 'class_id', 'rival', 'port']):
        rows.append(dict(zip(['role', 'class_id', 'rival', 'port'], keys), **aggregate(group)))
    pd.DataFrame(rows).to_csv(OUT/'selected_pair_ports.csv', index=False)
    repeated = frame.groupby(['role', 'class_id', 'rival', 'obs_id']).agg(
        consensus_both_seeds=('consensus_benefit', 'all'), mmsi=('mmsi', 'first'), port=('port', 'first')).reset_index()
    repeated[repeated.consensus_both_seeds].to_csv(OUT/'consensus_candidates.csv', index=False)


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=False)
    save_json(OUT/'PROTOCOL.json', dict(
        purpose='Find task-anchored relational evidence consistent with a causal-content research direction, not certify causal purity.',
        scope='Source factor-fit and identity-held source-test only. No target prediction archives read, no new neural training or target metrics.',
        teachers='Same vessel, exclude focal entire product; equal eligible-product means. All-other, same-port-other, and cross-port-only.',
        relation='True class y versus each rival k: margin p[y]-p[k]. Beneficial=t_margin>0 and t_margin>p_margin. Partial recovery remains <=0; harmful decreases margin. Label-aware categorization is diagnostic, not an inference-time oracle.',
        energy='Centered8-score MSE exactly equals sum of all28 unordered pair-difference squared errors /64. Partition into7 truth-anchored and21 rival-only relations; indirect relations are not assumed useless.',
        aggregation='Within vessel then equal class means; seeds averaged within these groups. Acquisition-level teacher means, no chip-count domination of donors. Counts unique observations/vessels, not seed duplicates.',
        candidates='Select top5 cross-port class pairs using source-fit net sign rescue only, require>=10 distinct ships and>=3 anchor ports. Report same pairs in identity-held source, no held-based reselection. Exploratory reused source diagnostic, not independent significance test.',
        boundary='Stable same-vessel relations may encode identity; cross-port observation is not a randomized intervention. Need other-vessel transfer and controlled latent interventions for stronger causal evidence.'))
    meta, split, roles, source, raw, norm, fixed, h = load_data()
    fit, test = split['factor_fit'], split['test_source']
    xt = torch.from_numpy(norm)
    relations, energies, coverage, teacher_quality = [], [], [], []
    for seed in SEEDS:
        factor = FactorNet()
        factor.load_state_dict(torch.load(previous.old.OUT/f'anchor_{seed}_factor.pt', map_location='cpu', weights_only=True)['model'])
        factor.eval()
        _, condition = encode_two(factor, norm)
        context = torch.from_numpy(standardize(condition, roles['fit']).astype(np.float32))
        operator = BoundedOperator(fixed['basis'], (h[roles['fit']]@fixed['basis']).square().mean(0).sqrt(), 'residual')
        operator.load_state_dict(torch.load(previous.OUT/f'anchor_residual_{seed}_operator.pt', map_location='cpu', weights_only=True))
        with torch.no_grad():
            base_full = operator(h, context)@fixed['weight']+fixed['bias']
        models = {}
        for arm in ('supervised', 'self_teacher', 'other_acquisition_teacher'):
            model = EvidenceStudent(centered(base_full[fit]).square().mean(0).sqrt())
            model.load_state_dict(torch.load(PREVIOUS/f'{arm}_{seed}.pt', map_location='cpu', weights_only=True)['selected'])
            models[arm] = model.eval()
        for role, ids in [('fit', fit), ('held', test)]:
            mm = meta.iloc[ids].reset_index(drop=True)
            base = base_full[ids].numpy()
            predictions = {}
            if role == 'held':
                for arm in ('baseline',)+tuple(models):
                    with np.load(PREVIOUS/f'{arm}_{seed}_source_scores.npz') as z:
                        np.testing.assert_array_equal(z['obs_ids'], ids)
                        predictions[arm] = z['scores']
                np.testing.assert_array_equal(base, predictions['baseline'])
            else:
                with torch.no_grad():
                    predictions = {arm: model(xt[ids], base_full[ids]).numpy() for arm, model in models.items()}
                predictions['baseline'] = base
            for scope in SCOPES:
                teacher, counts, consensus, gaps, orbits = scoped_teacher(base, mm, scope)
                valid = counts > 0
                y = mm.class_id.to_numpy()
                coverage.append(dict(seed=seed, role=role, scope=scope, chips=int(valid.sum()),
                    ships=int(mm.loc[valid, 'mmsi'].nunique()), ports=int(mm.loc[valid, 'port'].nunique()),
                    multiple_donor_chips=int((counts >= 2).sum()), incidence_comparable=int(np.isfinite(gaps).sum()),
                    orbit_comparable=int(np.isfinite(orbits).sum())))
                for name, score in [('base', base), ('teacher', teacher)]:
                    teacher_quality.append(dict(seed=seed, role=role, scope=scope, mode=name,
                        **metrics(score[valid].argmax(1), y[valid])))
                for i in np.flatnonzero(valid):
                    row, yi = mm.iloc[i], int(y[i])
                    bm, tm, energy = relation_components(base[i], teacher[i], yi)
                    descriptor = dict(seed=seed, role=role, scope=scope, obs_id=int(ids[i]), mmsi=row.mmsi,
                        class_id=yi, port=row.port, donor_products=int(counts[i]),
                        teacher_correct=bool(teacher[i].argmax() == yi), base_correct=bool(base[i].argmax() == yi))
                    energies.append(dict(**descriptor, **energy))
                    for rival in range(8):
                        if rival == yi:
                            continue
                        delta = float(tm[rival]-bm[rival])
                        beneficial = bool(tm[rival] > EPS and delta > EPS)
                        rescue = bool(bm[rival] <= 0 and tm[rival] > 0)
                        harm = bool(bm[rival] > 0 and tm[rival] <= 0)
                        gain = {name: float(score[i, yi]-score[i, rival]-bm[rival]) for name, score in predictions.items()}
                        relations.append(dict(**descriptor, rival=rival, base_margin=float(bm[rival]),
                            teacher_margin=float(tm[rival]), teacher_delta=delta, beneficial=beneficial,
                            partial_recovery=bool(tm[rival] <= EPS and delta > EPS), margin_harm=bool(delta < -EPS),
                            rescue=rescue, harm=harm, net_flip=int(rescue)-int(harm), teacher_positive=bool(tm[rival] > EPS),
                            donor_positive_fraction=float(consensus[i, rival]), multiple_donors=bool(counts[i] >= 2),
                            consensus_benefit=bool(beneficial and counts[i] >= 2 and consensus[i, rival] == 1),
                            student_delta=gain['other_acquisition_teacher'], self_delta=gain['self_teacher'],
                            supervised_delta=gain['supervised'], student_positive=bool(gain['other_acquisition_teacher'] > EPS),
                            student_negative=bool(gain['other_acquisition_teacher'] < -EPS),
                            mean_incidence_gap=float(gaps[i]), orbit_change_fraction=float(orbits[i])))
            print('AUDITED', role, seed, flush=True)
    data, energy = pd.DataFrame(relations), pd.DataFrame(energies)
    data.to_csv(OUT/'relations.csv', index=False)
    energy.to_csv(OUT/'energy.csv', index=False)
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv', index=False)
    pd.DataFrame(teacher_quality).to_csv(OUT/'teacher_quality.csv', index=False)
    summary = []
    for keys, group in data.groupby(['role', 'scope']):
        summary.append(dict(zip(['role', 'scope'], keys), **aggregate(group)))
    pd.DataFrame(summary).to_csv(OUT/'summary.csv', index=False)
    pairs = []
    for keys, group in data.groupby(['role', 'scope', 'class_id', 'rival']):
        pairs.append(dict(zip(['role', 'scope', 'class_id', 'rival'], keys), **aggregate(group)))
    pair_frame = pd.DataFrame(pairs)
    pair_frame.to_csv(OUT/'class_pairs.csv', index=False)
    train_pairs = pair_frame.query("role == 'fit' and scope == 'cross_port' and ships >= 10 and ports >= 3")
    candidates = train_pairs.sort_values(['net_flip', 'class_id', 'rival'], ascending=[False, True, True]).head(5)
    selected = candidates[['class_id', 'rival']].assign(selected_on='source_fit_only')
    selected.to_csv(OUT/'selected_pairs.csv', index=False)
    pair_frame.merge(selected, on=['class_id', 'rival']).to_csv(OUT/'selected_pair_transfer.csv', index=False)
    strata = []
    for keys, group in data.groupby(['role', 'scope', 'teacher_correct']):
        strata.append(dict(zip(['role', 'scope', 'teacher_correct'], keys), **aggregate(group)))
    pd.DataFrame(strata).to_csv(OUT/'teacher_correctness_strata.csv', index=False)
    energy_rows = []
    for keys, group in energy.groupby(['role', 'scope']):
        stats = {c: balanced_ship_mean(group, c) for c in ('full_energy', 'true_class_energy',
            'rival_only_energy', 'beneficial_energy', 'harmful_energy', 'partial_recovery_energy')}
        energy_rows.append(dict(zip(['role', 'scope'], keys), **stats,
            **{c+'_fraction': v/stats['full_energy'] for c, v in stats.items() if c != 'full_energy'}))
    pd.DataFrame(energy_rows).to_csv(OUT/'energy_summary.csv', index=False)
    conditions = []
    data['angle_stratum'] = np.where(data.mean_incidence_gap.isna(), 'unknown',
        np.where(data.mean_incidence_gap < 2, 'under_2deg', 'at_least_2deg'))
    data['orbit_stratum'] = np.where(data.orbit_change_fraction.isna(), 'unknown',
        np.where(data.orbit_change_fraction == 0, 'same_observed_pass', 'some_changed_pass'))
    for field in ('angle_stratum', 'orbit_stratum'):
        for keys, group in data.groupby(['role', 'scope', field]):
            conditions.append(dict(role=keys[0], scope=keys[1], condition=field, stratum=keys[2], **aggregate(group)))
    pd.DataFrame(conditions).to_csv(OUT/'condition_strata.csv', index=False)
    export_readout_directions()
    save_json(OUT/'COMPLETE.json', dict(seeds=SEEDS, source_only=True, new_training=0,
        relation_rows=len(data), energy_rows=len(energy), source_prediction_exact_replays=2,
        target_score_files_read=0, causal_identification_claim=False))
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
