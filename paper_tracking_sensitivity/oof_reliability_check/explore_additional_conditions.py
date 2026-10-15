"""Bounded cached-pass response probe. Source associations, not causal effects."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import run_conditioned_vessel_drift as previous


OUT = Path('E:/Hermes/paper_tracking/sensitivity/additional_conditions_20261009')
CACHE = Path('E:/Hermes/paper_tracking/s1_product_incidence.json')
PROXIES = OUT.parent/'latent_clue_cards_20261008/image_proxies.csv'
MODELS = {
    'angle': [0, 1],
    'angle_pass': [0, 1, 4],
    'angle_pass_interaction': [0, 1, 4, 5, 6],
    'angle_position': [0, 1, 2, 3],
    'angle_position_pass': [0, 1, 2, 3, 4],
    'angle_position_pass_interaction': [0, 1, 2, 3, 4, 5, 6],
}
COMPARISONS = {
    'pass_over_angle': ('angle_pass', 'angle'),
    'interaction_over_angle': ('angle_pass_interaction', 'angle'),
    'interaction_over_pass': ('angle_pass_interaction', 'angle_pass'),
    'pass_over_angle_position': ('angle_position_pass', 'angle_position'),
    'interaction_over_angle_position': ('angle_position_pass_interaction', 'angle_position'),
    'interaction_over_pass_position': ('angle_position_pass_interaction', 'angle_position_pass'),
}


def attach_metadata(frame, side, cache):
    frame = frame.merge(side, on=['instance_id', 'product', 'port'], how='left',
                        sort=False, validate='one_to_one')
    frame.loc[frame.status.ne('ok'), ['incidence_deg', 'latitude', 'longitude']] = np.nan
    frame['pass_name'] = frame['product'].map(lambda p: cache.get(p, {}).get('pass'))
    frame['pass_value'] = frame.pass_name.map({'Ascending': 1., 'Descending': 0.})
    return frame


def coverage_table(frame):
    records = []
    for (port, label), group in frame.groupby(['port', 'class_id'], sort=True):
        complete = group[['incidence_deg', 'latitude', 'longitude', 'pass_value']].notna().all(axis=1)
        selected = group.loc[complete]
        ships = selected.loc[selected.mmsi.str.fullmatch(r'[0-9]{9}', na=False)
                             & selected.mmsi.ne('000000000')]
        records.append(dict(port=port, class_id=label, rows=len(group),
            angle_rows=int(group.incidence_deg.notna().sum()),
            pass_rows=int(group.pass_value.notna().sum()), complete_rows=len(selected),
            ascending_rows=int(selected.pass_value.eq(1).sum()),
            descending_rows=int(selected.pass_value.eq(0).sum()),
            complete_products=selected['product'].nunique(), complete_ships=ships.mmsi.nunique(),
            both_pass_ships=int(ships.groupby('mmsi').pass_value.nunique().eq(2).sum())))
    cells = pd.MultiIndex.from_product([sorted(frame.port.unique()), sorted(frame.class_id.unique())],
                                     names=['port', 'class_id'])
    return pd.DataFrame(records).set_index(['port', 'class_id']).reindex(cells, fill_value=0).reset_index()


def load_and_audit():
    frame = pd.read_csv(previous.SOURCE/'manifest.csv', dtype={'mmsi': str},
                        low_memory=False, keep_default_na=False)
    side = pd.read_csv(previous.SIDECAR, usecols=['instance_id', 'product', 'port', 'status',
        'incidence_deg', 'latitude', 'longitude', 'coordinate_kind'])
    cache = json.loads(CACHE.read_text(encoding='utf-8'))
    frame = attach_metadata(frame, side, cache)
    coverage_table(frame).to_csv(OUT/'coverage_by_port_class.csv', index=False)
    complete = frame[['incidence_deg', 'latitude', 'longitude', 'pass_value']].notna().all(axis=1)
    kept = frame.loc[complete]
    valid_identity = frame.mmsi.str.fullmatch(r'[0-9]{9}', na=False) & frame.mmsi.ne('000000000')
    products = frame[['product', 'pass_name', 'pass_value']].drop_duplicates('product').copy()
    products['acq_time'] = pd.to_datetime(products['product'].str.extract(
        r'_([0-9]{8}T[0-9]{6})_', expand=False), format='%Y%m%dT%H%M%S', errors='coerce')
    products.to_csv(OUT/'product_inventory.csv', index=False)
    times = pd.to_datetime(frame['product'].str.extract(r'_([0-9]{8}T[0-9]{6})_', expand=False),
                           format='%Y%m%dT%H%M%S', errors='coerce')
    frame.assign(month=times.dt.to_period('M').astype(str)).groupby(
        ['port', 'class_id', 'month']).size().rename('rows').to_csv(OUT/'month_inventory.csv')
    geometry = [c for c in frame if c.startswith(('width_', 'height_', 'crop_side_',
        'target_long_side_', 'target_occupancy_', 'redetect_model_conf_', 'target_chip_center_'))]
    inventory = [dict(factor=c, rows=int(pd.to_numeric(frame[c], errors='coerce').notna().sum()),
                      scope='manifest', interpretation='pixel/detection clue; not assumed nuisance')
                 for c in geometry]
    if PROXIES.exists():
        proxies = pd.read_csv(PROXIES)
        for name in proxies.columns.difference(['manifest_row', 'port', 'class_id']):
            inventory.append(dict(factor=name, rows=int(proxies[name].notna().sum()),
                scope='existing paired-image proxy cache',
                interpretation='pooled VV/VH image association; not separate polarization or nuisance'))
    pd.DataFrame(inventory).to_csv(OUT/'existing_clue_inventory.csv', index=False)
    summary = dict(rows=len(frame), products=frame['product'].nunique(),
        pass_rows=int(frame.pass_value.notna().sum()), pass_products=int(products.pass_value.notna().sum()),
        ascending_rows=int(frame.pass_value.eq(1).sum()), descending_rows=int(frame.pass_value.eq(0).sum()),
        accepted_angle_rows=int(frame.incidence_deg.notna().sum()), complete_rows=int(complete.sum()),
        complete_products=kept['product'].nunique(), complete_ports=kept.port.nunique(),
        complete_both_pass_port_mmsi=int(frame.loc[complete & valid_identity].groupby(
            ['port', 'mmsi']).pass_value.nunique().eq(2).sum()),
        all_both_pass_port_mmsi=int(frame.loc[valid_identity].groupby(
            ['port', 'mmsi']).pass_value.nunique().eq(2).sum()),
        complete_pass_counts=kept.pass_name.value_counts().to_dict(),
        acquisition_time_rows=int(times.notna().sum()), acquisition_start=str(times.min()),
        acquisition_end=str(times.max()), product_prefix_counts=frame['product'].str[:3].value_counts().to_dict(),
        acquisition_mode_tokens=frame['product'].str.split('_').str[1].value_counts().to_dict(),
        polarization_mode_tokens=frame['product'].str.split('_').str[3].value_counts().to_dict(),
        ais_inventory_previously_inspected=dict(exact_product_files=503, total_mib=693.5514259338379,
            motion_fields_accepted=0, join_executed=False,
            limitation='No accepted TXT motion-field mapping; bounded inventory only this round. Not zero availability.'),
        relative_orbit=dict(derived=False, reason='No verified formula retained; no inferred orbit conditions used.'),
        pass_cache=str(CACHE), pass_producer='E:/Hermes/scripts/fetch_s1_incidence.py',
        pass_source='Exact complete-product AWS annotation/iw-vv.xml pass tag',
        incidence_sidecar=str(previous.SIDECAR), representative_safe_templates_used=False,
        product_mid_swath_incidence_used=False, new_downloads=0, missing_imputed=False)
    previous.write_json(OUT/'metadata_inventory.json', summary)
    print(json.dumps(dict(stage='inventory', **summary)), flush=True)
    return frame


def observations_with_fixed_splits(frame, rows, values):
    """Assign the prior angle-complete split BEFORE filtering missing pass."""
    obs, aggregate = previous.collect_observations(frame, rows, values)
    width = aggregate.shape[1] - 4
    angle_good = np.isfinite(aggregate[:, width:width+3]).all(axis=1)
    obs = obs.loc[angle_good].reset_index(drop=True)
    aggregate = aggregate[angle_good]
    split = previous.split_observations(obs)
    obs['split'] = ''
    for name, mask in split.items():
        obs.loc[mask, 'split'] = name
    # Means propagate missingness. Nonbinary means also reject conflicting pass
    # values among duplicate same-port/MMSI/acquisition observations.
    pass_good = np.isin(aggregate[:, -1], [0., 1.])
    audit = dict(angle_observations=len(obs), complete_observations=int(pass_good.sum()),
                 rejected_pass_observations=int((~pass_good).sum()))
    assignment = obs.copy()
    assignment['pass_complete'] = pass_good
    return obs.loc[pass_good].reset_index(drop=True), aggregate[pass_good], assignment, audit


def design(obs, conditions, pairs):
    x = previous.condition_differences(obs, conditions[:, :3], pairs)
    t = (conditions[:, 0] - 38.) / 5.
    p = conditions[:, 3]
    phi = np.column_stack((p, p*t, p*t*t))
    a, b = pairs.obs_a.to_numpy(int), pairs.obs_b.to_numpy(int)
    return np.column_stack((x, phi[b] - phi[a]))


def switch_coverage(pairs, x):
    changed = np.abs(x[:, 4]) > 0
    selected = pairs.loc[changed]
    return dict(switch_pairs=int(changed.sum()), switch_ships=selected.mmsi.nunique(),
                switch_ports=selected.port.nunique(), switch_classes=selected.class_id.nunique(),
                adequate_switch_coverage=bool(changed.sum() >= 10 and selected.mmsi.nunique() >= 5))


def run_port(folder, frame):
    out = OUT/'ports'/folder.name
    out.mkdir(parents=True, exist_ok=True)
    rows = np.load(folder/'source_rows.npy')
    target = np.load(folder/'target_rows.npy')
    # Target identities/products only serve the existing exclusion contract.
    if set(frame.loc[rows, 'product']) & set(frame.loc[target, 'product']):
        raise ValueError('Outer source/target product overlap')
    with (folder/'geometry.pkl').open('rb') as handle:
        geometry = pickle.load(handle)
    bank = np.load(folder/'source_bank.npy', mmap_mode='r')
    reduced = previous.latent(bank, geometry)
    width = 128 + reduced.shape[1]
    cond = frame.loc[rows, ['incidence_deg', 'latitude', 'longitude', 'pass_value']].to_numpy(float)
    values = np.column_stack((bank[:, :128], reduced, cond))
    obs, aggregate, assignment, obs_audit = observations_with_fixed_splits(frame, rows, values)
    if set(obs.mmsi) & set(frame.loc[target, 'mmsi']):
        raise ValueError('Outer source/target valid identity overlap')
    del bank, reduced, values
    assignment.to_csv(out/'observation_splits.csv', index=False)
    coverage_table(frame.loc[rows]).to_csv(out/'row_coverage_by_port_class.csv', index=False)
    models, records, coverage, group_scores, pair_cells = {}, [], [], [], []
    train_ships = set(obs.loc[obs.split.eq('fit'), 'mmsi'])
    train_products = set(obs.loc[obs.split.eq('fit'), 'product'])
    pass_supported = False
    spaces = {'retained128': slice(0, 128), 'stable_drift': slice(128, width)}
    for split in ('fit', 'new_ship', 'new_month', 'new_ship_month'):
        mask = obs.split.eq(split).to_numpy()
        o, z = obs.loc[mask].reset_index(drop=True), aggregate[mask]
        if split in ('new_ship', 'new_ship_month') and train_ships & set(o.mmsi):
            raise ValueError('Held ship overlap')
        if split in ('new_month', 'new_ship_month') and train_products & set(o['product']):
            raise ValueError('Held month product overlap')
        pairs, delta, weights = previous.make_pairs(o, z[:, :width])
        if pairs.empty:
            coverage.append(dict(target_port=folder.name, split=split, observations=len(o), pairs=0,
                                 status='no_pairs', **obs_audit))
            continue
        x = design(o, z[:, width:], pairs)
        sw = switch_coverage(pairs, x)
        if split == 'fit':
            pass_supported = sw['adequate_switch_coverage']
        a, b = pairs.obs_a.to_numpy(int), pairs.obs_b.to_numpy(int)
        pairs['product_a'] = o.iloc[a]['product'].to_numpy()
        pairs['product_b'] = o.iloc[b]['product'].to_numpy()
        pairs['pass_a'] = z[a, -1]
        pairs['pass_b'] = z[b, -1]
        pairs['switch'] = np.abs(x[:, 4]) > 0
        pairs['weight'] = weights
        for j, name in enumerate(('dt', 'dt2', 'east_10km', 'north_10km', 'dp', 'dpt', 'dpt2')):
            pairs[name] = x[:, j]
        pairs.to_csv(out/(split+'_pairs.csv'), index=False)
        coverage.append(dict(target_port=folder.name, split=split, observations=len(o), pairs=len(pairs),
            ships=pairs.mmsi.nunique(), source_ports=pairs.port.nunique(), classes=pairs.class_id.nunique(),
            ascending_observations=int((z[:, -1] == 1).sum()), descending_observations=int((z[:, -1] == 0).sum()),
            gap_days_median=float(pairs.gap_days.median()), design_rank=int(np.linalg.matrix_rank(x)),
            fit_pass_supported=pass_supported, status='ok', **sw, **obs_audit))
        for (port, label), cell in pairs.groupby(['port', 'class_id']):
            pair_cells.append(dict(target_port=folder.name, split=split, source_port=port, class_id=label,
                pairs=len(cell), ships=cell.mmsi.nunique(), switch_pairs=int(cell['switch'].sum()),
                switch_ships=cell.loc[cell['switch'], 'mmsi'].nunique()))
        groups = [('switch', 'yes', pairs['switch'].to_numpy()),
                  ('switch', 'no', ~pairs['switch'].to_numpy())]
        groups += [('class', str(c), pairs.class_id.eq(c).to_numpy()) for c in sorted(pairs.class_id.unique())]
        groups += [('source_port', p, pairs.port.eq(p).to_numpy()) for p in sorted(pairs.port.unique())]
        for space, columns in spaces.items():
            y = delta[:, columns]
            for model, features in MODELS.items():
                if 'pass' in model and not pass_supported:
                    continue
                key = space+'__'+model
                xx = x[:, features]
                if split == 'fit':
                    models[key] = previous.fit_response(xx, y, weights)
                if key not in models:
                    continue
                pred = xx @ models[key]
                common = dict(target_port=folder.name, split=split, space=space, model=model)
                records.append(dict(**common, **previous.response_metrics(y, pred, weights)))
                for kind, name, use in groups:
                    if use.any():
                        ww = weights[use]/weights[use].sum()
                        group_scores.append(dict(**common, group_kind=kind, group=name, pairs=int(use.sum()),
                            **previous.response_metrics(y[use], pred[use], ww)))
    np.savez_compressed(out/'response_models.npz', **models)
    previous.write_json(out/'COMPLETE.json', dict(**obs_audit, models=len(models),
        source_rows=len(rows), fit_pass_supported=pass_supported, target_query_labels_used=0))
    print(json.dumps(dict(port=folder.name, coverage=coverage,
        held=[r for r in records if r['split'] == 'new_ship_month' and r['space'] == 'stable_drift'])), flush=True)
    return records, coverage, group_scores, pair_cells


def summarize(scores, coverage):
    means = scores.groupby(['space', 'split', 'model']).agg(
        mean_skill=('skill', 'mean'), positive_folds=('skill', lambda x: int((x > 0).sum())),
        folds=('skill', 'size')).reset_index()
    means.to_csv(OUT/'response_means.csv', index=False)
    pivot = scores.pivot(index=['target_port', 'space', 'split'], columns='model', values='skill')
    increments = []
    for name, (new, control) in COMPARISONS.items():
        if new in pivot and control in pivot:
            table = (pivot[new] - pivot[control]).dropna().rename('increment').reset_index()
            table['comparison'] = name
            increments.append(table)
    inc = pd.concat(increments, ignore_index=True) if increments else pd.DataFrame()
    inc.to_csv(OUT/'incremental_skill.csv', index=False)
    overview = inc.groupby(['space', 'split', 'comparison']).agg(
        mean_increment=('increment', 'mean'), positive_folds=('increment', lambda x: int((x > 0).sum())),
        negative_folds=('increment', lambda x: int((x < 0).sum())), folds=('increment', 'size')).reset_index()
    overview.to_csv(OUT/'increment_means.csv', index=False)
    meta = json.loads((OUT/'metadata_inventory.json').read_text(encoding='utf-8'))
    lines = ['# Cached-pass source response probe', '',
        'Only source response models were fitted; no classifier changes or target-query outcomes were used.',
        'These are conditional association replications in existing source-fitted Qs/Qd and retained128 geometry, not nested OOF representation tests or causal effects.', '',
        f"Pass: {meta['pass_rows']}/{meta['rows']} rows, {meta['pass_products']}/{meta['products']} products. Complete angle/position/pass: {meta['complete_rows']} rows, {meta['complete_products']} products, {meta['complete_ports']} ports, {meta['complete_both_pass_port_mmsi']} port-MMSI groups with both passes.",
        'Exact-product cache only; no representative SAFE metadata, mid-swath angle substitution, or missing-value imputation.', '',
        '## Fixed comparison', '',
        'Same-port, same-MMSI adjacent observations; duplicate times averaged. Source-label-conflicting identities excluded. The original angle-complete identity split (seed20261009; one third held) is assigned BEFORE filtering pass. Months3/6/9/12 held. All six arms use identical observations/pairs/weights within each split.',
        'Angle inputs are differences of t=(angle-38)/5 and t^2. Pass is Ascending=1, Descending=0; interaction adds differences of p*t and p*t^2, not products of differences. Position is existing east/north displacement divided by10km. Zero-intercept weighted ridge=.01; no tuning.',
        'Weights balance class, port within class, ship within class-port, and pairs within ship. Pass arms require at least10 training switch pairs from5 ships; held switch sufficiency is reported by the same count rule. This is a coverage rule, not a significance test.',
        'skill=1-weighted residual energy/weighted observed change energy. Increment is a difference of skills, not classification accuracy. Source folds overlap heavily;23 positive folds are not23 independent confirmations.', '',
        '## Held source results', '',
        '| Space | Split | Arm | Mean skill (%) | Positive folds |',
        '|---|---|---|---:|---:|']
    for r in means.itertuples():
        if r.split != 'fit':
            lines.append(f'| {r.space} | {r.split} | {r.model} | {100*r.mean_skill:.4f} | {r.positive_folds}/{r.folds} |')
    lines += ['', '## Added information on matched cases', '',
        '| Space | Split | Comparison | Mean increment (percentage points) | Positive/negative folds |',
        '|---|---|---|---:|---:|']
    for r in overview.itertuples():
        if r.split != 'fit':
            lines.append(f'| {r.space} | {r.split} | {r.comparison} | {100*r.mean_increment:.4f} | {r.positive_folds}/{r.negative_folds} |')
    lines += ['', '## Coverage and limits', '',
        '| Split | Pairs per fold | Switch pairs | Switch ships | Adequate switch coverage |',
        '|---|---:|---:|---:|---:|']
    for split, c in coverage.groupby('split'):
        lines.append(f"| {split} | {c.pairs.min()}-{c.pairs.max()} | {c.switch_pairs.min()}-{c.switch_pairs.max()} | {c.switch_ships.min()}-{c.switch_ships.max()} | {int(c.adequate_switch_coverage.sum())}/{len(c)} |")
    lines += ['',
        'Port/class row coverage, observation split assignments, pair tables, per-class/per-source-port and switch/non-switch metrics retain all negative results. Missing metadata is structured by port and complete-case findings do not describe uncovered ports.',
        'Pass can co-vary with orbit, acquisition time, location, ship state and image content. First-order position control cannot remove these ambiguities. Chip angles include accepted pair-reference locations; VV/VH co-registration is not established here.', '',
        '## Existing additional inventory', '',
        'Manifest crop geometry covers25,656 rows; the existing pooled VV/VH image-proxy cache has3,718 rows. Exact numeric column coverage is in existing_clue_inventory.csv. These clues can contain vessel/class information; no nuisance correction was fitted from them.',
        'Product acquisition times are parsed by the existing helper convention; month_inventory.csv records observed month/port/class coverage. Product prefixes S1A32184/S1C5907 are inventoried only; the main branch owns the platform probe. All38091 rows have IW and1SDV tokens, so these mode tokens have no variation. Paired VV/VH image evidence remains. Relative orbit was not derived without a retained verified formula.',
        'The previously inspected AIS export index has503 exact-product files (693.55MiB). No MMSI-motion join was run and no heading/course/speed field was accepted. This is unfinished metadata validation, not evidence that AIS motion is absent. No further AIS search or download was performed.']
    (OUT/'FINDINGS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return means, overview


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Preserve completed additional-condition results')
    previous.write_json(OUT/'PROTOCOL.json', dict(date='2026-10-09', source_only=True,
        models=MODELS, ridge=.01, threads=2, device='CPU',
        interpreter='D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe',
        splits='Prior angle-complete identity/month assignments, then filter missing pass',
        held_months=[3, 6, 9, 12], vessel_seed=20261009,
        pass_encoding='Ascending=1; Descending=0', pass_min_fit_switch_pairs=10,
        pass_min_fit_switch_ships=5, representation='Existing outer-source geometry, conditional replication only',
        target_query_labels_used=0, classifier_experiments=0, parameter_search=False,
        metadata_scope='Cached exact-product pass plus accepted incidence sidecar only; other factors inventory only'))
    frame = load_and_audit()
    if args.audit_only:
        return
    folders = [f for f in sorted((previous.PARENT/'ports').iterdir()) if (f/'policies.npz').exists()]
    collected = [[], [], [], []]
    with threadpool_limits(limits=2):
        for folder in folders:
            for target, new in zip(collected, run_port(folder, frame)):
                target.extend(new)
    names = ('response_metrics', 'pair_coverage', 'group_metrics', 'pair_coverage_by_port_class')
    tables = [pd.DataFrame(rows) for rows in collected]
    for name, table in zip(names, tables):
        table.to_csv(OUT/(name+'.csv'), index=False)
    summarize(tables[0], tables[1])
    previous.write_json(OUT/'COMPLETE.json', dict(folds=len(folders), records=len(tables[0]),
        target_query_labels_used=0, classifier_fits=0, source_response_fits=int((tables[0]['split']=='fit').sum())))
    print(json.dumps(dict(stage='complete', folds=len(folders), output=str(OUT))), flush=True)


if __name__ == '__main__':
    main()
