"""Frozen four-class M1/M2/M3 transfer; no target fitting or selection."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '2'
import sys
sys.dont_write_bytecode = True
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / 'oss1_m123_replication_20261026'
sys.path.insert(0, str(CODE))
import extract_frontend as extract
import train_frontend as front
import train_m3 as m3
from sar_m123.m12 import attention_raw

ARMS = ('old_both', 'B_fixed', 'spatial', 'frequency')
SUPERVISED = ('fit_train', 'fit_validation', 'utility_train', 'selection')
FEATURES = ('x', 'mean', 'mass', 'joint', 'e_image', 'coverage')
LABEL_COLUMNS = {'class_id', 'original_class_id', 'class_name', 'ais_type', 'ais_ship_type', 'y', 'notes'}
PASS = re.compile(r'(S1[AB])_(\d{6})_([0-9A-F]{6})(?:\b|_)')
PRODUCT = re.compile(r'(S1[AB])_[A-Z0-9_]+?_\d{8}T\d{6}_\d{8}T\d{6}_(\d{6})_([0-9A-F]{6})_')


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def read_table(path, labels=False):
    def use(name):
        return labels or (name not in LABEL_COLUMNS and not name.startswith('label'))
    table = pd.read_csv(path, dtype={'mmsi': str, 'sample_id': str}, usecols=use)
    if table.empty or not table.obs_id.is_unique:
        raise ValueError(f'Expected nonempty unique obs_id table: {path}')
    return table.reset_index(drop=True)


def canonical_pass(table):
    """Recover physical acquisition, independent of dataset-specific role names."""
    result = []
    for row in table.to_dict('records'):
        found = None
        for key in ('scene_id', 'sample_id', 'overpass', 'pass_group'):
            value = str(row.get(key, '')).upper()
            hit = PRODUCT.search(value) or PASS.search(value)
            if hit:
                found = '_'.join(hit.groups())
                break
        if found is None:
            raise ValueError(f'No real Sentinel acquisition key for obs_id={row["obs_id"]}')
        result.append(found)
    return np.array(result)


def select_target(source, target, views):
    """Uses identities and availability only; target class labels are absent."""
    known = set(source.role)-set(SUPERVISED)-{'evaluation'}
    if known:
        raise ValueError(f'Unknown source roles, cannot establish supervision scope: {known}')
    supervised = source[source.role.isin(SUPERVISED)]
    if set(supervised.role) != set(SUPERVISED):
        raise ValueError('Source must identify all four supervised roles.')
    if supervised.mmsi.isna().any() or target.mmsi.isna().any():
        raise ValueError('Missing MMSI prevents ship-disjoint filtering.')
    reasons = pd.DataFrame({'obs_id': target.obs_id})
    reasons['source_mmsi_overlap'] = target.mmsi.isin(supervised.mmsi).to_numpy()
    reasons['source_pass_overlap'] = np.isin(canonical_pass(target), canonical_pass(supervised))
    view = views.copy()
    # prepare_view derives geometry/axis from accepted OBB corners even when the
    # separate axis-valid flag is false (for example nearly square detections).
    usable = view['exists'].eq(True) & view.detected.eq(True) & view.confidence.ge(.25)
    if 'valid_target' in view:
        usable &= view.valid_target.eq(True)
    usable_ids = view.loc[usable, 'obs_id']
    reasons['no_usable_view'] = ~target.obs_id.isin(usable_ids).to_numpy()
    reasons['included'] = ~reasons.iloc[:, 1:].any(axis=1)
    selected = target.loc[reasons.included].copy().reset_index(drop=True)
    selected['physical_pass_group'] = canonical_pass(selected)
    if selected.empty:
        raise ValueError('No nonoverlapping usable target observations.')
    return selected, reasons


def aligned(path, table):
    z = load_npz(path)
    index = pd.Index(z['obs_id']).get_indexer(table.obs_id)
    if (index < 0).any():
        raise ValueError(f'Missing requested observations in {path}')
    return {k: v[index] for k, v in z.items()
            if v.ndim and len(v) == len(z['obs_id']) and k != 'slot_region'}


def ensure_raw_cache(table, cache, out, device, weights):
    names = ('frontend_raw.npz', 'm3_raw.npz')
    if not all((cache / name).is_file() for name in names):
        destination = out / 'raw_cache'
        extract.extract_raw(table, destination, device, weights)
        return destination
    missing = np.zeros(len(table), bool)
    for name in names:
        with np.load(cache / name) as z:
            index = pd.Index(z['obs_id']).get_indexer(table.obs_id)
            present = index >= 0
            if 'sample_id' in z:
                np.testing.assert_array_equal(z['sample_id'][index[present]].astype(str),
                                               table.loc[present, 'sample_id'].astype(str))
            missing |= ~present
    if not missing.any():
        return cache
    destination = out / 'raw_cache'
    fresh = destination / 'new_observations'
    print('Additional raw observations:', int(missing.sum()), flush=True)
    extract.extract_raw(table.loc[missing].reset_index(drop=True), fresh, device, weights)
    for name in names:
        old, new = load_npz(cache / name), load_npz(fresh / name)
        old_ix = pd.Index(old['obs_id']).get_indexer(table.loc[~missing, 'obs_id'])
        order = pd.Index(np.r_[table.loc[~missing, 'obs_id'], table.loc[missing, 'obs_id']]).get_indexer(table.obs_id)
        merged = {}
        for key, value in old.items():
            if key != 'slot_region' and value.ndim and len(value) == len(old['obs_id']):
                merged[key] = np.concatenate((value[old_ix], new[key]))[order]
            else:
                merged[key] = value
        np.savez_compressed(destination / name, **merged)
    write_json(destination / 'cache_provenance.json', dict(reused_from=str(cache.resolve()),
               requested=len(table), newly_extracted=int(missing.sum()),
               new_obs_id=table.loc[missing, 'obs_id'].tolist(), reader_dependent=False))
    return destination


def frozen_model(model, checkpoint, device='cpu'):
    model.load_state_dict(checkpoint['state'], strict=True)
    return model.to(device).eval().requires_grad_(False)


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def frontend_predict(raw, summaries, models, seed):
    """Replay original dtype/batching and all five frozen residual stages."""
    ridge = load_npz(models / 'axis_ridge.npz')
    x = (raw['axis']-ridge['mean'])/ridge['std']/np.sqrt(raw['axis'].shape[1])
    axis = (x @ ridge['coef'].T+ridge['intercept']).astype(float)

    def residual(name, model, values, scale=True):
        ck = load_checkpoint(models / f'{name}_seed{seed}.pt')
        if scale:
            values = (values-ck['mu'])/ck['scale']
        return front.upstream.predict(frozen_model(model, ck), values.astype(np.float32))

    a = axis+residual('five', front.Five(), raw['five'])
    sc = load_npz(models / 'neighborhood_scaler.npz')
    feat, level = np.zeros_like(raw['centered']), np.zeros_like(raw['means'])
    for p in range(2):
        feat[:, p] = ((raw['centered'][:, p]-sc['mean'][p])/sc['std'][p]
                      *raw['usable'][:, p, None])
        level[:, p] = ((raw['means'][:, p]-sc['level_mean'][p])/sc['level_std'][p]
                       *raw['usable'][:, p])
    xd = np.concatenate((feat.reshape(len(a), -1), level, raw['usable'].astype(np.float32)), 1)
    u = a+residual('neighborhood', front.Neighborhood(), xd, scale=False)
    b = u+residual('calibration', front.Calibration(), front.upstream.center(u))
    x1 = attention_raw(b, summaries['mean3'], summaries['hb3'], summaries['valid'])
    s1 = b+residual('attention', front.upstream.AttentionHead(), x1)
    x2 = front.regional_raw(s1, summaries['summary1'], summaries['quality'], summaries['geometry'])
    s12 = s1+residual('full', front.upstream.RegionHead(), x2)
    return dict(axis=axis, A=a, U=u, frontend_base=b, m1=s1, s12=s12,
                delta_m1=s1-b, delta_m2=s12-s1)


def project_raw(raw, projection):
    mass = raw['mass'].astype(np.float32)
    mean = ((raw['layer1']-projection['mean']) @ projection['components'].T).astype(np.float32)
    x = m3.tokens.unit(mean).astype(np.float32)
    x[mass <= 0] = 0
    joint = (raw['joint_raw'] @ projection['components'].T).reshape(len(mass), 2, 20, 96)
    return dict(x=x, mean=mean, mass=mass, joint=joint.astype(np.float32),
                e_image=raw['e_image'], coverage=raw['coverage'])


def source_bank(root, source):
    references = load_npz(root / 'references.npz')
    rows = references['bank']
    if rows.shape != (4, 8) or len(np.unique(rows)) != 32:
        raise ValueError('Expected the frozen 4 x 8 source bank.')
    bank_table = source.iloc[rows.ravel()]
    if not bank_table.role.eq('fit_train').all() or bank_table.mmsi.nunique() != 32:
        raise ValueError('Source references must be 32 distinct fit_train ships.')
    np.testing.assert_array_equal(bank_table.class_id, np.repeat(np.arange(4), 8))
    local = load_npz(root / 'local.npz')
    if len(local['x']) != len(source):
        raise ValueError('Source local.npz does not align with source M3 table.')
    bank = {k: local[k][rows.ravel()] for k in FEATURES}
    covariance = {k: references[k] for k in ('mu', 'scale', 'active')}
    return bank, rows.ravel(), covariance


def retrieve(query, bank, query_mmsi, bank_mmsi):
    body = m3.tokens.body_descriptor(query['x'], query['mass'])
    ref_body = m3.tokens.body_descriptor(bank['x'], bank['mass']).reshape(4, 8, -1)
    score = np.einsum('nd,ckd->nck', body, ref_body)
    same = np.asarray(query_mmsi)[:, None, None] == np.asarray(bank_mmsi).reshape(1, 4, 8)
    score[same] = -np.inf
    pick = np.argsort(-score, axis=-1)[..., :2]
    return np.arange(4)[None, :, None]*8+pick


@torch.inference_mode()
def m3_predict(query, bank, refs, covariance, base, seed_root, device='cpu', batch=48):
    scalers = load_checkpoint(seed_root / 'scalers.pt')
    metric = frozen_model(m3.PairMetric(), load_checkpoint(seed_root / 'both_metric.pt'), device)
    # Reuse the exact matcher with a compact source-bank prefix. Only query rows run.
    features = {k: np.concatenate((bank[k], query[k])) for k in FEATURES}
    all_refs = np.concatenate((np.zeros((32, 4, 2), np.int64), refs))
    matcher = m3.Matcher(features, all_refs, scalers['metric_scaler'], device)
    local = matcher.predict(np.arange(32, 32+len(base)), metric, batch=batch)
    raw = m3.local_raw(local, base)
    joint = frozen_model(m3.JointHead(), load_checkpoint(seed_root / 'joint_head.pt'), device)
    old_both = base+joint(torch.as_tensor(m3.transform(raw, scalers['head_scaler']),
                                        device=device)).cpu().numpy()
    xp = torch.as_tensor(m3.transform(raw, scalers['polar_scaler']), device=device)
    delta = []
    for pol, name in enumerate(('vv', 'vh_centered')):
        ck = load_checkpoint(seed_root / f'{name}.pt')
        model = frozen_model(m3.PolarHead(pol, centered=bool(pol), null=ck['state'].get('null')), ck, device)
        delta.append(model(xp).cpu().numpy())
    dv, dh = delta
    qr = m3.regional_descriptors(query, covariance)
    br = m3.regional_descriptors(bank, covariance)
    gr, available = m3.gate_raw(qr, br, refs, base, dv, dh)
    alpha = float(read_json(seed_root / 'selection.json')['alpha'])
    fixed_weight = np.array([alpha, 1-alpha], np.float32)*available
    result = dict(old_both=old_both, B_fixed=base+fixed_weight[:, :1]*dv+fixed_weight[:, 1:]*dh,
                  base=base, delta_vv=dv, delta_vh=dh, local=local, gate_raw=gr,
                  available=available, weight_B_fixed=fixed_weight)
    for arm, dim in (('spatial', 41), ('frequency', 185)):
        ck = load_checkpoint(seed_root / f'{arm}.pt')
        if float(ck['alpha']) != alpha:
            raise ValueError('Gate and B_fixed source alpha disagree.')
        gate = frozen_model(m3.Gate(dim, ck['alpha']), ck)
        weights = gate(torch.tensor(m3.transform(gr[..., :dim], ck['scaler'])),
                       torch.tensor(available)).numpy()
        result[arm] = base+weights[:, :1]*dv+weights[:, 1:]*dh
        result[f'weight_{arm}'] = weights
    return result


def compare_replay(actual, expected, keys, tolerance=2e-5):
    report = {}
    for key in keys:
        a, b = actual[key], expected[key]
        np.testing.assert_allclose(a, b, rtol=0, atol=tolerance, err_msg=key)
        report[key] = dict(max_abs_error=float(np.max(np.abs(a-b))),
                           bitwise_equal=bool(np.array_equal(a, b)))
        if key in (*ARMS, 'base', 'm1', 's12', 'frontend_base'):
            np.testing.assert_array_equal(a.argmax(1), b.argmax(1), err_msg=f'{key} predictions')
    return report


def mapped_tables(source, target, source_root, target_root):
    parts = []
    for side, root, table in (('source', source_root, source), ('target', target_root, target)):
        part = table.copy()
        part['dataset'] = root.name
        part['dataset_side'] = side
        part['original_obs_id'] = part.obs_id
        part['dataset_row'] = np.arange(len(part))
        part['original_role'] = part.get('role', 'unspecified')
        if side == 'target':
            part['role'] = 'evaluation'
        parts.append(part)
    merged = pd.concat(parts, ignore_index=True)
    merged['row_index'] = np.arange(len(merged))
    merged['obs_id'] = merged.row_index
    return merged


def evaluate_saved(out, label_path, target, seeds):
    """Only invoked after predictions_saved.json exists for ALL requested seeds."""
    if not (out / 'predictions_saved.json').exists():
        raise RuntimeError('Predictions must be committed to disk before reading target labels.')
    labels = pd.read_csv(label_path, usecols=['obs_id', 'class_id']).set_index('obs_id')
    y = labels.loc[target.obs_id, 'class_id'].to_numpy(int)
    if not np.isin(y, np.arange(4)).all():
        raise ValueError('Target cohort must already use the four-class mapping 0/2/3/4 -> 0..3.')
    rows = []
    for seed in seeds:
        z = load_npz(out / f'seed{seed}' / 'predictions.npz')
        for arm in (*ARMS, 'frontend_base', 'm1', 's12'):
            rows.append(dict(seed=seed, arm=arm, **m3.metrics(y, z[arm])))
    pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
    pd.DataFrame(rows).groupby('arm')[['OA', 'BA', 'NLL', 'Brier']].agg(['mean', 'std']).to_csv(out / 'summary.csv')
    annotated = target.copy()
    annotated['class_id'] = y
    annotated.to_csv(out / 'target_table.csv', index=False)
    merged = pd.read_csv(out / 'table.csv', dtype={'mmsi': str})
    merged.loc[merged.dataset_side.eq('target'), 'class_id'] = y
    merged.to_csv(out / 'table.csv', index=False)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--source-m3', type=Path, required=True)
    p.add_argument('--target-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source-frontend', type=Path)
    p.add_argument('--target-cache', type=Path)
    p.add_argument('--full-table', type=Path, help='Prepared full four-class target cohort; replaces stack_table.')
    p.add_argument('--view-inventory', type=Path, required=True)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--weights', type=Path, default=extract.WEIGHTS)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--seeds', nargs='+', type=int, default=(17, 29, 43))
    p.add_argument('--batch', type=int, default=48)
    p.add_argument('--validate-source-replay', action='store_true')
    return p


def main():
    args = parser().parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(HERE) or out == HERE:
        raise ValueError('Use a new output subdirectory inside this branch.')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty output directory to keep predictions and provenance unambiguous.')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    sf = args.source_frontend or args.source_root / 'frontend'
    source = read_table(args.source_m3 / 'table.csv', labels=True)
    source_all = read_table(args.source_root / 'stack_table.csv', labels=True)
    np.testing.assert_array_equal(source.obs_id, source_all.obs_id)
    for key in ('sample_id', 'mmsi', 'role', 'class_id'):
        np.testing.assert_array_equal(source[key], source_all[key], err_msg=f'source {key}')
    config = read_json(sf / 'config.json')
    if config['class_map'] != {'0': 0, '2': 1, '3': 2, '4': 3}:
        raise ValueError('Source class map differs from the declared four-class protocol.')
    np.testing.assert_array_equal(config['obs_id'], source.obs_id)
    label_path = args.full_table or args.target_root / 'stack_table.csv'
    target = read_table(label_path)
    extract.VIEW_TABLE = args.view_inventory.resolve()
    extract.RADAR_TABLE = args.geometry.resolve()
    views = pd.read_csv(extract.VIEW_TABLE)
    if not extract.RADAR_TABLE.is_file():
        raise FileNotFoundError(extract.RADAR_TABLE)
    if args.validate_source_replay:
        if args.target_root.resolve() != args.source_root.resolve() or args.full_table:
            raise ValueError('Replay requires target-root=source-root and the original stack table.')
        np.testing.assert_array_equal(target.obs_id, source.obs_id)
        target['physical_pass_group'] = canonical_pass(target)
        mode = 'source_replay_not_transfer'
    else:
        if args.target_root.resolve() == args.source_root.resolve():
            raise ValueError('Same-root inference requires --validate-source-replay.')
        target, exclusions = select_target(source_all, target, views)
        exclusions.to_csv(out / 'target_eligibility.csv', index=False)
        mode = 'frozen_cross_dataset'
    tau_record = read_json(sf / 'scales.json')
    cache = args.target_cache or args.target_root / 'frontend_cache'
    # Raw caches have no class-reader dependency. Missing/full-cohort caches are
    # extracted locally, while reader-dependent summaries ALWAYS come from raw images.
    if not args.validate_source_replay:
        cache = ensure_raw_cache(target, cache, out, args.device, args.weights)
    raw = aligned(cache / 'frontend_raw.npz', target)
    mraw = aligned(cache / 'm3_raw.npz', target)
    for data in (raw, mraw):
        if 'sample_id' in data:
            np.testing.assert_array_equal(data['sample_id'].astype(str), target.sample_id.astype(str))
    if args.validate_source_replay:
        sums = aligned(sf / 'summaries.npz', target)
    else:
        sums = extract.extract_summaries(target, np.array([], dtype=int), cache, sf / 'models',
                                         out / 'summaries', args.device, args.weights,
                                         frozen_tau=tau_record['tau'])
    query = project_raw(mraw, load_npz(args.source_m3 / 'projection.npz'))
    bank, bank_rows, covariance = source_bank(args.source_m3, source)
    refs = retrieve(query, bank, target.mmsi, source.iloc[bank_rows].mmsi)
    merged = mapped_tables(source, target, args.source_root, args.target_root)
    merged.to_csv(out / 'table.csv', index=False)
    query_rows = np.arange(len(source), len(source)+len(target))
    target['query_row'] = np.arange(len(target))
    target['combined_row'] = query_rows
    target.to_csv(out / 'target_table.csv', index=False)
    bank_map = source.iloc[bank_rows].copy()
    bank_map['bank_index'] = np.arange(32)
    bank_map['combined_row'] = bank_rows
    bank_map.to_csv(out / 'source_bank.csv', index=False)
    protocol = dict(mode=mode, source_root=str(args.source_root.resolve()),
                    source_m3=str(args.source_m3.resolve()), source_frontend=str(sf.resolve()),
                    target_root=str(args.target_root.resolve()), target_table=str(label_path.resolve()),
                    target_cache=str(cache.resolve()), view_inventory=str(extract.VIEW_TABLE),
                    geometry=str(extract.RADAR_TABLE), weights=str(args.weights),
                    source_scales=str(sf / 'scales.json'), tau=tau_record['tau'],
                    source_tau_fit_obs_id=tau_record['fit_obs_id'], seeds=args.seeds,
                    n_source=len(source), n_target=len(target), n_references=32,
                    source_supervised_roles=list(SUPERVISED), target_fit=False,
                    target_labels_read=False, target_cohort='caller-declared four classes; all roles',
                    source_replay_exclusions_bypassed=args.validate_source_replay,
                    summaries_from_source_cache=args.validate_source_replay,
                    prediction_alignment='target_table rows; query_rows and refs index combined table.csv',
                    raw_chip_gallery_pending=True)
    write_json(out / 'protocol.json', protocol)
    replay = {}
    if args.validate_source_replay:
        replay['summaries_from_existing_source_cache'] = True
        expected_local = load_npz(args.source_m3 / 'local.npz')
        replay['projection'] = compare_replay(query, expected_local, FEATURES)
        expected_refs = load_npz(args.source_m3 / 'references.npz')['refs']
        np.testing.assert_array_equal(bank_rows[refs], expected_refs)
        replay['references_exact'] = True
    for seed in args.seeds:
        f = frontend_predict(raw, sums, sf / 'models', seed)
        prediction = m3_predict(query, bank, refs, covariance, f['s12'].astype(np.float32),
                                args.source_m3 / f'seed{seed}', args.device, args.batch)
        prediction.update(f)
        prediction.update(refs=bank_rows[refs], refs_bank=refs, query_rows=query_rows,
                          obs_id=target.obs_id.to_numpy(), sample_id=target.sample_id.to_numpy(str))
        if args.validate_source_replay:
            expected_front = aligned(sf / f'scores_seed{seed}.npz', target)
            expected_front['frontend_base'] = expected_front['base']
            report = compare_replay(f, expected_front, ('frontend_base', 'm1', 's12'))
            expected = load_npz(args.source_m3 / f'seed{seed}' / 'predictions.npz')
            report.update(compare_replay(prediction, expected,
                (*ARMS, 'base', 'delta_vv', 'delta_vh', 'local', 'gate_raw', 'available',
                 'weight_B_fixed', 'weight_spatial', 'weight_frequency')))
            replay[str(seed)] = report
        for key in (*ARMS, 'base', 'm1', 's12'):
            if not np.isfinite(prediction[key]).all():
                raise ValueError(f'Nonfinite {key} scores, seed {seed}')
        dest = out / f'seed{seed}'
        dest.mkdir()
        np.savez_compressed(dest / 'predictions.npz', **prediction)
        print('Saved frozen predictions', seed, len(target), flush=True)
    write_json(out / 'predictions_saved.json', dict(seeds=args.seeds, n=len(target),
               all_predictions_saved=True, target_labels_read=False))
    if args.validate_source_replay:
        write_json(out / 'source_replay.json', dict(passed=True, checks=replay,
                   exact_prediction_and_reference_match=True, score_absolute_tolerance=2e-5,
                   raw_summary_reextraction_tested=False, no_transfer_results=True))
    else:
        evaluate_saved(out, label_path, target, args.seeds)
    write_json(out / 'complete.json', dict(mode=mode, seeds=args.seeds, n=len(target),
               trained=False, target_fit=False, evaluated=not args.validate_source_replay,
               raw_chip_gallery_pending=not args.validate_source_replay))


if __name__ == '__main__':
    main()
