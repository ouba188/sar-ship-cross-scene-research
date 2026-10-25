"""Source-constant D mask, frozen full-system replay, no target fitting."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import transfer as tr

SEEDS = (17, 29, 43)
STAGES = ('axis', 'A', 'U', 'frontend_base', 'm1', 's12', *tr.ARMS)


def source_mask(root):
    table = tr.read_table(root / 'stack_table.csv')
    raw = tr.aligned(root / 'frontend_cache/frontend_raw.npz', table)
    sc = tr.load_npz(root / 'frontend/models/neighborhood_scaler.npz')
    fit = table.role.eq('fit_train').to_numpy()
    np.testing.assert_array_equal(table.loc[fit, 'obs_id'], sc['fit_obs_id'])
    active = np.ones(sc['mean'].shape, bool)
    rows = []
    for p in range(2):
        x = raw['centered'][fit & raw['usable'][:, p].astype(bool), p]
        active[p] = np.ptp(x, axis=0) > 0
        for c in np.flatnonzero(~active[p]):
            rows.append(dict(polarization=('VV', 'VH')[p], channel=int(c),
                source_min=float(x[:, c].min()), source_max=float(x[:, c].max()),
                mean=float(sc['mean'][p, c]), std=float(sc['std'][p, c])))
    return active, rows


def frozen_prediction(raw, summaries, models, source_m3, query, bank, refs, covariance, seed, active):
    f = tr.frontend_predict(raw, summaries, models, seed, neighborhood_active=active)
    z = tr.m3_predict(query, bank, refs, covariance, f['s12'].astype(np.float32),
                      source_m3 / f'seed{seed}', device='cuda')
    z.update(f)
    for k in STAGES:
        if not np.isfinite(z[k]).all():
            raise ValueError(f'Nonfinite repaired {k}, seed {seed}')
    return z


def source_checks(root, source_m3, active):
    """Recompute all source roles; fit invariance is mandatory, held shifts reported."""
    out = HERE / ('source_' + ('oss1' if 'oss1_' in root.name else 'oss2'))
    out.mkdir(exist_ok=True)
    table = tr.read_table(root / 'stack_table.csv', labels=True)
    raw = tr.aligned(root / 'frontend_cache/frontend_raw.npz', table)
    sums = tr.aligned(root / 'frontend/summaries.npz', table)
    query = tr.load_npz(source_m3 / 'local.npz')
    query = {k: query[k] for k in tr.FEATURES}
    bank, bank_rows, covariance = tr.source_bank(source_m3, table)
    refs = tr.retrieve(query, bank, table.mmsi, table.iloc[bank_rows].mmsi)
    expected_refs = tr.load_npz(source_m3 / 'references.npz')['refs']
    np.testing.assert_array_equal(bank_rows[refs], expected_refs)
    fit = table.role.eq('fit_train').to_numpy()
    records, validation = [], []
    for seed in SEEDS:
        cached_f = tr.aligned(root / f'frontend/scores_seed{seed}.npz', table)
        cached_f['frontend_base'] = cached_f['base']
        old_f = tr.frontend_predict(raw, sums, root / 'frontend/models', seed)
        tr.compare_replay(old_f, cached_f, ('frontend_base', 'm1', 's12'))
        old_z = tr.load_npz(source_m3 / f'seed{seed}/predictions.npz')
        z = frozen_prediction(raw, sums, root / 'frontend/models', source_m3,
                               query, bank, refs, covariance, seed, active)
        dest = out / f'seed{seed}'
        dest.mkdir(exist_ok=True)
        np.savez_compressed(dest / 'predictions.npz', **z)
        for stage in STAGES:
            before = old_z[stage] if stage in tr.ARMS else old_f[stage]
            error = float(np.abs(z[stage][fit] - before[fit]).max())
            np.testing.assert_allclose(z[stage][fit], before[fit], rtol=0, atol=2e-5,
                                       err_msg=f'source fit changed {seed}/{stage}')
            np.testing.assert_array_equal(z[stage][fit].argmax(1), before[fit].argmax(1))
            validation.append(dict(seed=seed, stage=stage, source_fit_max_error=error))
            for role, ix in table.groupby('role').groups.items():
                ix = np.array(ix)
                for mode, scores in (('original', before), ('repaired', z[stage])):
                    records.append(dict(seed=seed, stage=stage, role=role, mode=mode,
                        **tr.m3.metrics(table.iloc[ix].class_id.to_numpy(int), scores[ix])))
    pd.DataFrame(records).to_csv(out / 'role_metrics.csv', index=False)
    pd.DataFrame(validation).to_csv(out / 'fit_invariance.csv', index=False)
    tr.write_json(out / 'complete.json', dict(seeds=SEEDS, n=len(table), trained=False,
                  source_fit_prediction_invariance=True, downstream_reselected=False))
    print('Source checks complete', root.name, flush=True)


def transfer_run(direction, active):
    original = HERE.parent / direction
    protocol = tr.read_json(original / 'protocol.json')
    out = HERE / direction
    if out.exists():
        raise ValueError(f'Use a new output folder, not overwrite: {out}')
    out.mkdir()
    source_root, source_m3 = Path(protocol['source_root']), Path(protocol['source_m3'])
    source = tr.read_table(source_root / 'stack_table.csv', labels=True)
    target = tr.read_table(original / 'target_table.csv')
    raw = tr.aligned(Path(protocol['target_cache']) / 'frontend_raw.npz', target)
    mraw = tr.aligned(Path(protocol['target_cache']) / 'm3_raw.npz', target)
    sums = tr.aligned(original / 'summaries/summaries.npz', target)
    query = tr.project_raw(mraw, tr.load_npz(source_m3 / 'projection.npz'))
    bank, bank_rows, covariance = tr.source_bank(source_m3, source)
    refs = tr.retrieve(query, bank, target.mmsi, source.iloc[bank_rows].mmsi)
    query_rows = target.combined_row.to_numpy(int)
    tr.mapped_tables(source, target, source_root, Path(protocol['target_root'])).to_csv(out / 'table.csv', index=False)
    target.to_csv(out / 'target_table.csv', index=False)
    protocol.update(repair='source_fit_constant_neighborhood_mask', original_result=str(original),
        fitted_on_target=False, thresholds_tuned=False, trained=False,
        mask_rule='ptp(source fit usable centered features)==0',
        source_inactive_indices=[np.flatnonzero(~a).tolist() for a in active],
        raw_chip_gallery_pending=True)
    tr.write_json(out / 'protocol.json', protocol)
    checks = []
    for seed in SEEDS:
        old = tr.load_npz(original / f'seed{seed}/predictions.npz')
        unchanged = tr.frontend_predict(raw, sums, Path(protocol['source_frontend']) / 'models', seed)
        tr.compare_replay(unchanged, old, ('axis', 'A', 'U', 'frontend_base', 'm1', 's12'))
        z = frozen_prediction(raw, sums, Path(protocol['source_frontend']) / 'models', source_m3,
                               query, bank, refs, covariance, seed, active)
        z.update(refs=bank_rows[refs], refs_bank=refs, query_rows=query_rows,
                 obs_id=target.obs_id.to_numpy(), sample_id=target.sample_id.to_numpy(str))
        np.testing.assert_array_equal(z['refs'], old['refs'])
        np.testing.assert_allclose(z['local'], old['local'], rtol=0, atol=2e-5)
        for key in ('axis', 'A', 'available', 'weight_B_fixed'):
            np.testing.assert_allclose(z[key], old[key], rtol=0, atol=2e-5)
        if np.array_equal(z['s12'], old['s12']):
            tr.compare_replay(z, old, (*tr.ARMS, 'delta_vv', 'delta_vh', 'gate_raw',
                                     'weight_spatial', 'weight_frequency'))
        checks.append(dict(seed=seed, references_same=True, local_matching_same=True,
            raw_baseline_max_difference=float(np.abs(z['frontend_base']-old['frontend_base']).max()),
            s12_max_difference=float(np.abs(z['s12']-old['s12']).max())))
        dest = out / f'seed{seed}'
        dest.mkdir()
        np.savez_compressed(dest / 'predictions.npz', **z)
        print('Saved repaired complete-system', direction, seed, len(target), flush=True)
    tr.write_json(out / 'predictions_saved.json', dict(seeds=SEEDS, n=len(target),
                  all_predictions_saved=True, target_labels_read=False))
    tr.write_json(out / 'integration_checks.json', checks)
    tr.evaluate_saved(out, Path(protocol['target_table']), target, SEEDS)
    tr.write_json(out / 'complete.json', dict(mode='frozen_cross_dataset', seeds=SEEDS,
                  n=len(target), trained=False, target_fit=False, evaluated=True,
                  raw_chip_gallery_pending=True))


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    masks = {}
    for direction in ('oss1-to-oss2', 'oss2-to-oss1'):
        p = tr.read_json(HERE.parent / direction / 'protocol.json')
        root, source_m3 = Path(p['source_root']), Path(p['source_m3'])
        active, rows = source_mask(root)
        masks[direction] = active
        name = 'oss1' if direction.startswith('oss1') else 'oss2'
        pd.DataFrame(rows).to_csv(HERE / f'{name}_source_constant_dimensions.csv', index=False)
        source_checks(root, source_m3, active)
    for direction, active in masks.items():
        transfer_run(direction, active)


if __name__ == '__main__':
    main()
