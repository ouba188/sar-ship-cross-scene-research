"""Source-only observational diagnostics for the learned c7/r25 decomposition.

Run only after the four full-model exports are ready. Fold exports are not read.
The only fitted objects here are the existing independent linear class probes.
"""

import argparse
import itertools
import json
import os
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import pandas as pd

from paired_evidence_factors import acquisition_table, paired_covariance, row_weights, fit_probe, score_probe


BASE = Path(__file__).resolve().parent
RUN = BASE/'change_decomposition_20261011'
SEEDS = (31415, 27182)
METHODS = ('class_only', 'paired')
DIMENSIONS = dict(content=7, residual=25, z=32,
                  content_component=32, residual_component=32)
ENERGIES = ('original_delta_energy', 'residual_delta_mse',
            'content_delta_energy', 'residual_delta_energy', 'full_delta_mse',
            'acquisition_input_residual_delta_mse', 'acquisition_input_content_delta_energy',
            'acquisition_input_full_delta_mse', 'residual_jensen_delta_energy',
            'residual_jensen_endpoint_energy', 'content_residual_delta_inner_product')
RATIOS = ('residual_delta_mse_ratio', 'residual_explained_fraction',
          'content_delta_energy_ratio', 'residual_delta_energy_ratio',
          'full_reconstruction_delta_ratio', 'acquisition_input_residual_delta_mse_ratio',
          'acquisition_input_residual_explained_fraction', 'acquisition_input_content_delta_energy_ratio',
          'acquisition_input_full_reconstruction_delta_ratio', 'residual_jensen_delta_energy_ratio',
          'residual_jensen_delta_rms_ratio', 'residual_jensen_endpoint_rms_ratio',
          'partition_identity_gap', 'content_residual_delta_inner_product_ratio')
MATCHES = BASE/'paired_evidence_factors_20261011/identity_controls/matches.csv'


def ratio(numerator, denominator):
    return float(numerator/denominator) if denominator > 0 else np.nan


def delta_energies(z, content, residual):
    """Coordinate-mean squared pair differences, without centering the deltas."""
    return np.column_stack([np.square(v).mean(1) for v in
                            (z, residual-z, content, residual, content+residual-z)])


def normalized_delta(energies):
    original = energies['original_delta_energy']
    residual_error = ratio(energies['residual_delta_mse'], original)
    result = dict(residual_delta_mse_ratio=residual_error,
                residual_explained_fraction=1-residual_error,
                content_delta_energy_ratio=ratio(energies['content_delta_energy'], original),
                residual_delta_energy_ratio=ratio(energies['residual_delta_energy'], original),
                full_reconstruction_delta_ratio=ratio(energies['full_delta_mse'], original))
    acquisition_error = ratio(energies.get('acquisition_input_residual_delta_mse', np.nan), original)
    gap_delta = ratio(energies.get('residual_jensen_delta_energy', np.nan), original)
    result.update(
        acquisition_input_residual_delta_mse_ratio=acquisition_error,
        acquisition_input_residual_explained_fraction=1-acquisition_error,
        acquisition_input_content_delta_energy_ratio=ratio(energies.get('acquisition_input_content_delta_energy', np.nan), original),
        acquisition_input_full_reconstruction_delta_ratio=ratio(energies.get('acquisition_input_full_delta_mse', np.nan), original),
        residual_jensen_delta_energy_ratio=gap_delta,
        residual_jensen_delta_rms_ratio=float(np.sqrt(gap_delta)),
        residual_jensen_endpoint_rms_ratio=float(np.sqrt(ratio(energies.get('residual_jensen_endpoint_energy', np.nan), original))),
        partition_identity_gap=residual_error-result['content_delta_energy_ratio'],
        content_residual_delta_inner_product_ratio=ratio(energies.get('content_residual_delta_inner_product', np.nan), original))
    return result


def paired_rows(values, meta, scope):
    """Aggregate chips first, then equal pairs/vessel, vessels/class, classes."""
    names = ['z', 'content_component', 'residual_component']
    has_acquisition_input = 'acquisition_input_residual' in values
    if has_acquisition_input:
        names += ['acquisition_input_content', 'acquisition_input_residual']
    combined = np.concatenate([values[k] for k in names], axis=1)
    means, acquisitions = acquisition_table(combined, meta)
    records = []
    for ship, group in acquisitions.groupby('mmsi', sort=True):
        if group.class_id.nunique() != 1:
            raise ValueError('A vessel has conflicting source class labels')
        pairs = [(a, b) for a, b in itertools.combinations(group.index, 2)
                 if acquisitions.loc[a, 'product'] != acquisitions.loc[b, 'product']
                 and (scope == 'all' or acquisitions.loc[a, 'port'] != acquisitions.loc[b, 'port'])]
        if not pairs:
            continue
        a, b = np.asarray(pairs).T
        dz, dc, dr = np.split((means[a]-means[b])[:, :96], 3, axis=1)
        energies = dict(zip(ENERGIES[:5], delta_energies(dz, dc, dr).mean(0)))
        energies.update(dict.fromkeys(ENERGIES[5:], np.nan))
        energies['content_residual_delta_inner_product'] = float((dc*dr).mean())
        if has_acquisition_input:
            ac, ar = np.split((means[a]-means[b])[:, 96:], 2, axis=1)
            gap = means[:, 128:160]-means[:, 64:96]
            energies.update(
                acquisition_input_residual_delta_mse=float(np.square(ar-dz).mean()),
                acquisition_input_content_delta_energy=float(np.square(ac).mean()),
                acquisition_input_full_delta_mse=float(np.square(ac+ar-dz).mean()),
                residual_jensen_delta_energy=float(np.square(ar-dr).mean()),
                residual_jensen_endpoint_energy=float((np.square(gap[a])+np.square(gap[b])).mean()/2))
        used = np.unique(np.r_[a, b])
        records.append(dict(mmsi=ship, class_id=int(group.class_id.iloc[0]),
                            pairs=len(pairs), covered_acquisitions=len(used),
                            covered_chips=int(acquisitions.loc[used, 'chips'].sum()),
                            **energies))
    ships = pd.DataFrame(records, columns=['mmsi', 'class_id', 'pairs',
                                         'covered_acquisitions', 'covered_chips', *ENERGIES])
    rows = []
    for label in (-1, *range(8)):
        selected = ships if label == -1 else ships[ships.class_id == label]
        chips = meta if label == -1 else meta[meta.class_id == label]
        cells = acquisitions if label == -1 else acquisitions[acquisitions.class_id == label]
        if len(selected):
            energy = selected.groupby('class_id')[list(ENERGIES)].mean().mean().to_dict()
        else:
            energy = dict.fromkeys(ENERGIES, np.nan)
        rows.append(dict(scope=scope, class_id=label, chips=len(chips),
                         ships=int(chips.mmsi.nunique()), acquisitions=len(cells),
                         classes=int(chips.class_id.nunique()),
                         covered_classes=int(selected.class_id.nunique()),
                         covered_ships=len(selected), pairs=int(selected.pairs.sum()),
                         covered_acquisitions=int(selected.covered_acquisitions.sum()),
                         covered_chips=int(selected.covered_chips.sum()),
                         **energy, **normalized_delta(energy)))
    return rows


def reconstruction_row(values, meta):
    weights = row_weights(meta)
    z = values['z']
    error = values['content_component']+values['residual_component']-z
    energy = float(weights@np.square(z).mean(1))
    mse = float(weights@np.square(error).mean(1))
    return dict(chips=len(meta), ships=int(meta.mmsi.nunique()),
                original_energy=energy, reconstruction_mse=mse,
                full_reconstruction_ratio=ratio(mse, energy))


def class_probes(fit_values, held_values, fm, hm):
    weights = row_weights(fm)
    labels, truth = fm.class_id.to_numpy(int), hm.class_id.to_numpy(int)
    rows, classes = [], []
    for space in ('content', 'residual', 'z'):
        coefficient = fit_probe(fit_values[space], labels, weights)
        score = score_probe(held_values[space], coefficient)
        predicted = score.argmax(1)
        recalls = []
        for label in range(8):
            selected = truth == label
            recall = float(100*np.mean(predicted[selected] == label)) if selected.any() else np.nan
            if selected.any():
                recalls.append(recall)
            classes.append(dict(space=space, class_id=label,
                                fit_chips=int((labels == label).sum()),
                                fit_ships=int(fm.loc[labels == label, 'mmsi'].nunique()),
                                chips=int(selected.sum()),
                                ships=int(hm.loc[selected, 'mmsi'].nunique()), recall=recall))
        rows.append(dict(space=space, dimension=fit_values[space].shape[1],
                         fit_chips=len(fm), fit_ships=int(fm.mmsi.nunique()),
                         chips=len(hm), ships=int(hm.mmsi.nunique()),
                         OA=float(100*np.mean(predicted == truth)), BA=float(np.mean(recalls)),
                         centered_score_rms=float(np.sqrt(np.square(score-score.mean(1, keepdims=True)).mean()))))
    return rows, classes


def read_source_values(path, fit, held):
    """obs_ids are global metadata row positions; slice before any diagnostics."""
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(np.sort(archive['fit_obs_ids']), np.sort(fit),
                                      err_msg=f'{path.name}: full fit cohort mismatch')
        index = pd.Index(archive['obs_ids'])
        if not index.is_unique:
            raise ValueError(f'{path.name}: duplicate obs_ids')
        positions = index.get_indexer(np.r_[fit, held])
        if (positions < 0).any():
            raise ValueError(f'{path.name}: missing source observations')
        values = {}
        for name, dimension in DIMENSIONS.items():
            exported = archive[name]
            if exported.shape != (len(index), dimension):
                raise ValueError(f'{path.name}: invalid {name} shape {exported.shape}')
            values[name] = np.asarray(exported[positions], dtype=np.float64)
            if not np.isfinite(values[name]).all():
                raise ValueError(f'{path.name}: nonfinite source {name}')
    return ({k: v[:len(fit)] for k, v in values.items()},
            {k: v[len(fit):] for k, v in values.items()})


def acquisition_input_values(model, values, meta):
    """Evaluate selected model on mean source z; repeat results for aggregation."""
    import torch
    means, acquisitions = acquisition_table(values['z'], meta)
    keys = ['mmsi', 'product', 'port']
    positions = pd.MultiIndex.from_frame(acquisitions[keys]).get_indexer(pd.MultiIndex.from_frame(meta[keys]))
    with torch.no_grad():
        output = model(torch.tensor(means, dtype=torch.float32))
    values['acquisition_input_content'] = output['content_component'].numpy().astype(np.float64)[positions]
    values['acquisition_input_residual'] = output['residual_component'].numpy().astype(np.float64)[positions]


def selected_readout(model, content, meta):
    import torch
    with torch.no_grad():
        scores = model.head(torch.tensor(content, dtype=torch.float32)).numpy()
    truth = meta.class_id.to_numpy(int)
    predicted = scores.argmax(1)
    classes = []
    for label in range(8):
        selected = truth == label
        classes.append(dict(class_id=label, chips=int(selected.sum()),
                            recall=float(100*np.mean(predicted[selected] == label)) if selected.any() else np.nan))
    return dict(OA=float(100*np.mean(predicted == truth)),
                BA=float(np.mean([row['recall'] for row in classes if row['chips']]))), classes


def matched_cohort(held, meta):
    """Reuse the existing independent metadata cohort; never rebuild or tune it."""
    if not MATCHES.exists():
        return []
    frame = pd.read_csv(MATCHES)
    index = pd.Index(held)
    result = []
    for row in frame.itertuples(index=False):
        anchor = index.get_indexer([row.anchor_obs_id])
        same = index.get_indexer(json.loads(row.same_obs_ids))
        other = index.get_indexer(json.loads(row.other_obs_ids))
        if (np.r_[anchor, same, other] < 0).any():
            raise ValueError('Existing matched cohort is not confined to this test_source')
        a, s, o = meta.iloc[anchor[0]], meta.iloc[same], meta.iloc[other]
        if not (len(same) and len(other) and (s.mmsi == a.mmsi).all()
                and (o.mmsi != a.mmsi).all() and (s.class_id == a.class_id).all()
                and (o.class_id == a.class_id).all() and (s['product'] != a['product']).all()
                and (s.port != a.port).all() and s['product'].nunique() == 1
                and s.port.nunique() == 1 and (o['product'] == s['product'].iloc[0]).all()
                and (o.port == s.port.iloc[0]).all()):
            raise ValueError('Existing matched cohort does not satisfy same-class cross-port control')
        result.append((int(anchor[0]), same, other))
    return result


def matched_rows(values, meta, cohort):
    if not cohort:
        return []
    rows = []
    for space in DIMENSIONS:
        x = values[space]
        records = []
        for anchor, same, other in cohort:
            records.append(dict(anchor=anchor,
                                same_distance=float(np.square(x[anchor]-x[same].mean(0)).sum()),
                                other_distance=float(np.square(x[anchor]-x[other].mean(0)).sum())))
        anchors = pd.DataFrame(records).groupby('anchor').mean()
        covered = meta.iloc[anchors.index].reset_index(drop=True)
        weights = row_weights(covered)
        same, other = weights@anchors[['same_distance', 'other_distance']].to_numpy()
        rows.append(dict(space=space, anchors=len(anchors),
                         ships=int(covered.mmsi.nunique()), classes=int(covered.class_id.nunique()),
                         matched_pairs=len(cohort), same_distance=float(same),
                         other_distance=float(other), same_other_ratio=ratio(same, other)))
    return rows


def seed_summary(frame, keys, metrics):
    """Count defined seeds so an undefined ratio is not presented as a two-seed mean."""
    grouped = frame.groupby(keys, dropna=False, sort=False)
    means = grouped[metrics].mean()
    counts = grouped[metrics].count().rename(columns=lambda key: key+'_defined_seeds')
    return means.join(counts).join(grouped.seed.nunique().rename('seed_count')).reset_index()


def content_scale_rows(content, meta):
    """Normalize pair changes by each export's full held-source c7 variance."""
    weights = row_weights(meta)
    centered = np.asarray(content, dtype=np.float64)-weights@content
    covariance = centered.T@(weights[:, None]*centered)
    variance = float(np.trace(covariance))
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
    positive = eigenvalues[eigenvalues > 0]
    probability = positive/positive.sum() if len(positive) else positive
    effective_rank = float(np.exp(-np.sum(probability*np.log(probability)))) if len(positive) else np.nan
    rows = []
    for scope in ('all', 'cross_port'):
        moment, coverage = paired_covariance(content, meta, scope=scope)
        delta_energy = float(np.trace(moment))
        rows.append(dict(role='test_source', scope=scope, chips=len(meta),
                         ships=int(meta.mmsi.nunique()), classes=int(meta.class_id.nunique()),
                         pairs=int(coverage.pairs.sum()) if len(coverage) else 0,
                         covered_ships=len(coverage),
                         covered_classes=int(coverage.class_id.nunique()) if len(coverage) else 0,
                         content_weighted_variance=variance, content_effective_rank=effective_rank,
                         content_pair_delta_energy=delta_energy,
                         content_delta_over_own_variance=ratio(delta_energy, variance)))
    return rows


def write_content_scale(rows, out):
    frame = pd.DataFrame(rows)
    metrics = ['content_weighted_variance', 'content_effective_rank',
               'content_pair_delta_energy', 'content_delta_over_own_variance']
    summary = seed_summary(frame, ['method', 'scope'], metrics)
    frame.to_csv(out/'content_scale.csv', index=False)
    summary.to_csv(out/'content_scale_held_summary.csv', index=False)
    lines = ['# Held Content Scale Diagnostic', '',
             'Each export uses its own weighted centered covariance of all 823 held-source c7 rows. '
             'Weights are equal classes, vessels/class, acquisitions/vessel, chips/acquisition. '
             'Weighted variance V=trace(Cov_w(c)); effective rank=exp(-sum(p*log(p))) '
             'for covariance eigenvalues normalized to sum one (maximum 7).', '',
             'Pair numerator is E_pair[||delta_c||^2] after acquisition aggregation, with equal '
             'covered classes, vessels/class, pairs/vessel. Divide by V, without a factor of 2 '
             'or a dimension divisor. The same full-held covariance denominator is used for all '
             'and cross_port; their pair coverage differs. This controls overall content scale, '
             'not class composition, identity effects, or causal factors.', '',
             'Two-seed arithmetic means of per-seed metrics (not a ratio of pooled means):', '',
             summary[['method', 'scope', *metrics, 'seed_count']].to_string(index=False), '',
             'Individual seeds:', '', frame.to_string(index=False), '',
             'Lower original-z-normalized pair energy alone does not establish greater stability. '
             'Compare the own-variance ratio before attributing the reduction to preferential '
             'suppression of within-vessel changes. No probe fitting or model training in this supplement.']
    (out/'CONTENT_SCALE.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return summary


def run_content_scale(input_dir):
    with np.load(BASE/'grouped_condition_anchor_20261011/source_split.npz') as split:
        held = split['test_source'].copy()
    hm = pd.read_csv(BASE/'observations.csv').iloc[held].reset_index(drop=True)
    rows = []
    for seed in SEEDS:
        for method in METHODS:
            with np.load(input_dir/f'full_{seed}_{method}_decomposition.npz') as archive:
                positions = pd.Index(archive['obs_ids']).get_indexer(held)
                if (positions < 0).any():
                    raise ValueError('Export is missing held-source content rows')
                content = np.asarray(archive['content'][positions], dtype=np.float64)
            rows.extend(dict(model='full', seed=seed, method=method, **row)
                        for row in content_scale_rows(content, hm))
    out = RUN/'mechanism'
    out.mkdir(parents=True, exist_ok=True)
    summary = write_content_scale(rows, out)
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=RUN,
                        help='Directory containing the four ready full decomposition exports')
    parser.add_argument('--content-scale-only', action='store_true',
                        help='Add held content variance and normalized drift without rerunning probes')
    args = parser.parse_args()
    expected = {f'full_{seed}_{method}_decomposition.npz'
                for seed in SEEDS for method in METHODS}
    found = {path.name for path in args.input_dir.glob('full_*_decomposition.npz')}
    if found != expected:
        raise FileNotFoundError(f'Expected exactly full 2 seeds x 2 methods; '
                                f'missing={sorted(expected-found)}, unexpected={sorted(found-expected)}')
    if args.content_scale_only:
        run_content_scale(args.input_dir)
        return

    # The established loader necessarily loads global metadata/features. All
    # subsequent label access and arithmetic use only these two source cohorts.
    from train_explicit_candidate import load_data
    from change_decomposition import EvidenceDecomposition
    import torch
    torch.set_num_threads(1)
    loaded = load_data()
    meta, split = loaded[:2]
    fit, held = split['factor_fit'].copy(), split['test_source'].copy()
    fm, hm = [meta.iloc[ids].reset_index(drop=True).copy() for ids in (fit, held)]
    del loaded, meta, split
    if set(fm.mmsi) & set(hm.mmsi):
        raise ValueError('Source-fit and held-source vessels overlap')
    cohort = matched_cohort(held, hm)
    probes, classes, pairs, reconstruction, controls, readouts, readout_classes = [], [], [], [], [], [], []
    content_scales = []
    for seed in SEEDS:
        reference = None
        for method in METHODS:
            path = args.input_dir/f'full_{seed}_{method}_decomposition.npz'
            fv, hv = read_source_values(path, fit, held)
            if reference is None:
                reference = (fv['z'], hv['z'])
            else:
                for original, exported in zip(reference, (fv['z'], hv['z'])):
                    np.testing.assert_allclose(exported, original, rtol=1e-6, atol=1e-7,
                                               err_msg='Methods must use the same baseprior32 z')
            tags = dict(model='full', seed=seed, method=method)
            content_scales.extend(dict(**tags, **row) for row in content_scale_rows(hv['content'], hm))
            state = torch.load(args.input_dir/f'full_{seed}_{method}.pt', map_location='cpu', weights_only=True)
            model = EvidenceDecomposition()
            model.load_state_dict(state['selected'])
            model.eval()
            acquisition_input_values(model, fv, fm)
            acquisition_input_values(model, hv, hm)
            readout, readout_by_class = selected_readout(model, hv['content'], hm)
            readouts.append(dict(**tags, role='test_source', **readout))
            readout_classes.extend(dict(**tags, role='test_source', **row) for row in readout_by_class)
            probe, by_class = class_probes(fv, hv, fm, hm)
            probes.extend(dict(**tags, role='test_source', **row) for row in probe)
            classes.extend(dict(**tags, role='test_source', **row) for row in by_class)
            for role, values, frame in [('sourcefit', fv, fm), ('test_source', hv, hm)]:
                reconstruction.append(dict(**tags, role=role, **reconstruction_row(values, frame)))
                for scope in ('all', 'cross_port'):
                    pairs.extend(dict(**tags, role=role, **row) for row in paired_rows(values, frame, scope))
            controls.extend(dict(**tags, role='test_source', **row)
                            for row in matched_rows(hv, hm, cohort))
            print('DIAGNOSED', path.name, flush=True)

    tables = dict(class_probes=pd.DataFrame(probes), class_recall=pd.DataFrame(classes),
                  paired_delta=pd.DataFrame(pairs), reconstruction=pd.DataFrame(reconstruction),
                  selected_readout=pd.DataFrame(readouts), selected_readout_classes=pd.DataFrame(readout_classes))
    coverage_keys = ['role', 'scope', 'class_id', 'chips', 'ships', 'acquisitions', 'classes',
                     'covered_classes', 'covered_ships', 'pairs', 'covered_acquisitions', 'covered_chips']
    tables['coverage'] = tables['paired_delta'].loc[:, coverage_keys].drop_duplicates()
    summary_specs = {
        'class_probes': (['method', 'space'], ['OA', 'BA', 'centered_score_rms']),
        'class_recall': (['method', 'space', 'class_id'], ['recall']),
        'paired_delta': (['method', 'scope', 'class_id'], [*ENERGIES, *RATIOS]),
        'reconstruction': (['method'], ['original_energy', 'reconstruction_mse', 'full_reconstruction_ratio']),
        'selected_readout': (['method'], ['OA', 'BA']),
        'selected_readout_classes': (['method', 'class_id'], ['recall']),
    }
    if controls:
        tables['matched_controls'] = pd.DataFrame(controls)
        summary_specs['matched_controls'] = (['method', 'space'],
                                             ['same_distance', 'other_distance', 'same_other_ratio'])
    for name, (keys, metrics) in summary_specs.items():
        frame = tables[name]
        tables[name+'_held_summary'] = seed_summary(frame[frame.role == 'test_source'], keys, metrics)

    out = RUN/'mechanism'
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out/f'{name}.csv', index=False)
    content_scale_summary = write_content_scale(content_scales, out)
    no_cross = tables['coverage'][(tables['coverage'].scope == 'cross_port')
                                  & (tables['coverage'].class_id >= 0)
                                  & (tables['coverage'].covered_ships == 0)]
    protocol = dict(
        artifacts=sorted(expected), seeds=list(SEEDS), methods=list(METHODS), model='full',
        roles=['sourcefit', 'test_source'], dimensions=DIMENSIONS,
        sourcefit_chips=len(fm), sourcefit_ships=int(fm.mmsi.nunique()),
        held_chips=len(hm), held_ships=int(hm.mmsi.nunique()),
        probes='Existing fit_probe ridge=1 and score_probe on supplied coordinates, without rescaling. '
               'Shared fit row_weights: equal classes, vessels/class, acquisitions/vessel, chips/acquisition. '
               'Held OA and class recall are chip means; BA is mean recall of present classes. No checkpoint selection.',
        pairs='Mean chips within MMSI/product/port first; only distinct-product same-vessel pairs. '
              'all includes same-port and cross-port; cross_port requires different ports. '
              'Equal pairs/vessel, vessels/class, then covered classes. class_id=-1 is the aggregate.',
        energies='Uncentered coordinate-mean squared deltas in common source-whitened baseprior32. '
                 'residual_delta_mse=E||delta_residual_component-delta_z||^2/32. '
                 'Every paired ratio divides weighted mean numerator by actual weighted E||delta_z||^2/32 '
                 'on the SAME cohort; no per-pair ratio averaging or denominator from another space. '
                 'Explained fraction=1-residual_delta_mse_ratio, may be negative. '
                 'content_delta_energy_ratio=E||delta_content_component||^2/E||delta_z||^2. '
                 'full_reconstruction_delta_ratio uses delta_content+delta_residual-delta_z.',
        reconstruction='Full reconstruction ratio=weighted E||content_component+residual_component-z||^2/'
                       'weighted E||z||^2, using role-local row_weights and uncentered z energy.',
        acquisition_input='Load exported selected model without training; run model(mean acquisition z). '
                          'Compare with acquisition mean of exported model(chip z) components. '
                          'acquisition_input_* metrics use the former, ordinary metrics the latter. '
                          'Both use identical actual original z deltas and class/ship/pair weights. '
                          'Jensen gap=model(mean z).residual-mean(model(z).residual). Delta RMS ratio '
                          'is sqrt(E||gap_a-gap_b||^2/E||delta_z||^2). Endpoint RMS ratio is '
                          'sqrt(E[(||gap_a||^2+||gap_b||^2)/2]/E||delta_z||^2), on the same covered pairs.',
        mechanical_identity='Orthogonal decoder components impose delta_content dot delta_residual=0. '
                            'Let e=delta_content+delta_residual-delta_z. Residual delta error energy '
                            '=content delta energy+||e||^2-2 delta_content dot e. Exact reconstruction '
                            'therefore forces residual explained fraction=1-content delta energy ratio. '
                            'These are not independent evidence of separation. partition_identity_gap '
                            'is residual error ratio minus content energy ratio; approximate reconstruction '
                            'can change it. No fixed physical axes or causal factors are identified.',
        selected_readout='Selected checkpoint content head applied to exported c7; compare its held '
                         'OA/BA with the independent sourcefit ridge probe. No additional model selection.',
        summary='Arithmetic mean of two full-model seed metrics on test_source only. '
                'Defined-seed counts expose zero-denominator or absent-class results; coverage counts are '
                'unique cohort counts, never summed across seeds or methods.',
        matched_control='Reused existing held-source class-matched other-vessel exact donor product/port cohort. '
                        'Chip anchor versus acquisition-mean donors; equal matched donors/anchor then '
                        'row_weights over covered anchors. Ratio of weighted mean squared same/other distances.',
        matched_cohort_path=str(MATCHES), matched_pairs=len(cohort),
        matched_anchors=len({row[0] for row in cohort}),
        absent_cross_port_coverage=no_cross[['role', 'class_id', 'chips', 'ships']].to_dict('records'),
        limitations='Observational separation diagnostics, not causal identification. Residual class readability '
                    'is not expected to be zero. Coverage is class dependent; zero coverage is not zero drift. '
                    'No validation or target metrics, architecture comparisons, or neural training.',
    )
    (out/'PROTOCOL.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    report = ['# Source Change Decomposition Diagnostics', '',
              'Two-seed full-model held-source means. Observational, not causal.', '',
              '## Independent Class Probes', '',
              tables['class_probes_held_summary'][['method', 'space', 'OA', 'BA', 'seed_count']].to_string(index=False),
              '', '## Same-Vessel Distinct-Product Deltas', '']
    overall = tables['paired_delta_held_summary']
    report.append(overall.loc[overall.class_id == -1, ['method', 'scope', *RATIOS[:5]]].to_string(index=False))
    report.extend(['', '## Content Own Scale', '',
                   content_scale_summary[['method', 'scope', 'content_weighted_variance',
                       'content_effective_rank', 'content_delta_over_own_variance']].to_string(index=False),
                   '', 'See CONTENT_SCALE.md for covariance and pair weighting definitions. '
                   'Original-z-normalized pair energy reduction alone is not evidence of greater stability.'])
    report.extend(['', '## Acquisition Input Versus Chip Aggregation', '',
                   overall.loc[overall.class_id == -1, ['method', 'scope',
                       'residual_explained_fraction', 'acquisition_input_residual_explained_fraction',
                       'residual_jensen_delta_rms_ratio', 'residual_jensen_endpoint_rms_ratio']].to_string(index=False),
                   '', protocol['acquisition_input'], '', '## Mechanical Partition Identity', '',
                   protocol['mechanical_identity'], '',
                   overall.loc[overall.class_id == -1, ['method', 'scope', 'partition_identity_gap',
                       'full_reconstruction_delta_ratio', 'content_residual_delta_inner_product_ratio']].to_string(index=False),
                   '', '## Selected Content Readout', '',
                   tables['selected_readout_held_summary'].to_string(index=False)])
    report.extend(['', '## Actual Coverage', '', tables['coverage'].to_string(index=False),
                   '', 'Classes without cross-port pair coverage:', '', no_cross.to_string(index=False),
                   '', '## Full Reconstruction', '',
                   tables['reconstruction_held_summary'].to_string(index=False)])
    if controls:
        report.extend(['', '## Independent Matched Identity Control', '',
                       tables['matched_controls_held_summary'][['method', 'space', 'same_other_ratio', 'seed_count']].to_string(index=False),
                       '', f'Actual held cohort: {len(cohort)} matched pairs; {protocol["matched_anchors"]} anchors.'])
    else:
        report.extend(['', 'Existing matched cohort unavailable or empty; optional control omitted.'])
    report.extend(['', protocol['limitations'], '',
                   'Class-wise recalls are in class_recall_held_summary.csv. '
                   'A missing ratio remains undefined; explained fractions are not clipped. '
                   'Decoder components are orthogonal, but their energy ratios relative to original z '
                   'need not add to one because reconstruction is approximate.'])
    (out/'REPORT.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    (out/'COMPLETE.json').write_text(json.dumps(dict(artifacts=4, probe_fits=12,
        new_neural_fits=0, validation_metrics=False, target_metrics=False,
        held_seed_count=2, matched_pairs=len(cohort)), indent=2), encoding='utf-8')
    print(tables['class_probes_held_summary'][['method', 'space', 'OA', 'BA']].to_string(index=False), flush=True)
    print('OUTPUT', out, flush=True)


if __name__ == '__main__':
    main()
