"""Fixed conditional M1/M2 head comparisons using corrected frozen inputs."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRANSFER = ROOT / 'public_m123_cross_dataset_20261026'
REPAIR = TRANSFER / 'normalization_repair_20261026'
sys.path.insert(0, str(ROOT / 'oss1_m123_replication_20261026'))
import train_frontend as front

SEEDS = (17, 29, 43)
ARMS = ('m1_both_refit', 'm1_mean_only', 'm2_identity_refit', 'm2_summary_merged')
PAIRS = (('m1_mean_only', 'm1_both_refit'), ('m2_summary_merged', 'm2_identity_refit'),
         ('m1_both_refit', 'original_m1'), ('m2_identity_refit', 'original_s12'))
META = ['obs_id', 'sample_id', 'mmsi', 'pass_group', 'role']


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def merged_summaries(sums):
    """Summary-level merge; quality coverage/neff are proxies, not pixel recomputes."""
    q = sums['quality']
    usable = (q[..., 3] > 0) & (q[..., 0] > 0) & sums['valid'][..., None]
    area = np.where(usable, q[..., 0], 0)
    total = area.sum(2)
    w = np.divide(area, total[..., None], out=np.zeros_like(area), where=total[..., None] > 0)
    merged = (sums['summary1'] * w[..., None, None]).sum(2)
    quality = np.stack((total, (q[..., 1] * w).sum(2), (q[..., 2] * w).sum(2),
                        usable.any(2).astype(float)), -1)
    return {**sums, 'summary1': np.repeat(merged[:, :, None], 3, axis=2),
            'quality': np.repeat(quality[:, :, None], 3, axis=2)}


def raw_features(arm, base, sums):
    if arm.startswith('m1_'):
        hb = np.zeros_like(sums['hb3']) if arm == 'm1_mean_only' else sums['hb3']
        return front.attention_raw(base, sums['mean3'], hb, sums['valid'])
    s = merged_summaries(sums) if arm == 'm2_summary_merged' else sums
    return front.regional_raw(base, s['summary1'], s['quality'], s['geometry'])


def fit_scaler(raw, tr):
    train = raw[tr]
    active = (np.ptp(train.reshape(-1, train.shape[-1]), axis=0) > 0)[None, None]
    mu = train.mean((0, 1), keepdims=True)
    scale = np.maximum(train.std((0, 1), keepdims=True), 1e-8)
    return dict(mu=mu, scale=scale, active=active)


def transform(raw, scaler):
    x = np.where(scaler['active'], (raw - scaler['mu']) / scaler['scale'], 0).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite transformed summaries')
    return x


def corrected(name, seed, table, source=False):
    path = REPAIR / name / f'seed{seed}/predictions.npz'
    with np.load(path) as z:
        if source:
            # Repair source caches have no IDs; generation uses stack_table order.
            root = ROOT / f'{name.removeprefix("source_")}_m123_replication_20261026'
            with np.load(root / f'frontend/scores_seed{seed}.npz') as original:
                np.testing.assert_array_equal(original['obs_id'], table.obs_id)
        else:
            np.testing.assert_array_equal(z['obs_id'], table.obs_id)
        out = {k: z[k].copy() for k in ('frontend_base', 'm1', 's12')}
    for a in out.values():
        if a.shape != (len(table), 4) or not np.isfinite(a).all():
            raise ValueError(f'Invalid corrected cache: {path}')
    return out


def data_receipt(table, sums):
    rows = []
    for role, idx in table.groupby('role', sort=False).groups.items():
        part = table.loc[idx]
        rows.append(dict(role=role, n=len(part), ships=part.mmsi.nunique(), passes=part.pass_group.nunique(),
                         valid_polar=sums['valid'][idx].sum(0).tolist(),
                         valid_regions=(sums['quality'][idx, ..., 3] > 0).sum(0).tolist()))
    return rows


def run_training():
    torch.set_num_threads(2)
    protocol = dict(seeds=SEEDS, arms=ARMS, steps=200, optimizer='AdamW lr=.003 weight_decay=.01',
        loss='.5 observation CE + .5 class-balanced CE', selection='fit_validation exact (OA+BA)/2; earliest strict maximum; every 5 steps including zero',
        fit_role='fit_train', validation_role='fit_validation', source_corrected=str(REPAIR),
        source_summaries='oss{1,2}_m123_replication_20261026/frontend/summaries.npz',
        cross_summaries=str(TRANSFER / '{direction}/summaries/summaries.npz'),
        scaler='source fit only; axes observations/classes; strict ptp==0 -> zero; std floor 1e-8',
        M1='both branches unchanged; mean-only sets hb3=0 before raw feature construction; 266 parameters',
        M2='both freeze original corrected m1; unchanged 69->12->1; 853 parameters',
        merge='valid-region area weighted summary; copy to 3 slots; quality total area, area-weighted coverage/min_neff, any-valid; geometry unchanged',
        scope='conditional matched head comparison; not independent physical causality; not complete M3',
        total_fits=24, target_selection=False, original_cached_controls=True)
    write_json(HERE / 'protocol.json', protocol)
    receipts = {}
    for source, target in (('oss1', 'oss2'), ('oss2', 'oss1')):
        direction = f'{source}-to-{target}'
        root = ROOT / f'{source}_m123_replication_20261026'
        table, tr, va = front.load_table(root / 'stack_table.csv')
        table = table[META + ['class_id']].copy()
        cross = pd.read_csv(TRANSFER / direction / 'target_table.csv', usecols=META,
                            dtype={'mmsi': str, 'sample_id': str})
        sums = front.aligned(root / 'frontend/summaries.npz', table)
        csums = front.aligned(TRANSFER / direction / 'summaries/summaries.npz', cross)
        source_ids = set(table.loc[table.role.ne('evaluation'), 'mmsi'])
        source_pass = set(table.loc[table.role.ne('evaluation'), 'pass_group'])
        if source_ids & set(cross.mmsi) or source_pass & set(cross.pass_group):
            raise ValueError('Cross cohort overlaps source supervised ships/passes')
        receipts[direction] = dict(source=data_receipt(table, sums), cross=data_receipt(cross, csums),
            source_summary_order='obs_id aligned', corrected_source_order='repair generation stack_table order; original scores obs_id verified',
            fit_class_counts=table.iloc[tr].class_id.value_counts().sort_index().to_dict(),
            validation_class_counts=table.iloc[va].class_id.value_counts().sort_index().to_dict())
        out = HERE / direction
        out.mkdir(exist_ok=True)
        (out / 'models').mkdir(exist_ok=True)
        table.to_csv(out / 'source_roles_masked.csv', index=False)
        cross.to_csv(out / 'cross_metadata_no_labels.csv', index=False)
        for seed in SEEDS:
            src = corrected('source_' + source, seed, table, source=True)
            dst = corrected(direction, seed, cross)
            sources = dict(frontend_base=src['frontend_base'], original_m1=src['m1'], original_s12=src['s12'])
            crosses = dict(frontend_base=dst['frontend_base'], original_m1=dst['m1'], original_s12=dst['s12'])
            for arm in ARMS:
                base = src['frontend_base' if arm.startswith('m1_') else 'm1']
                crossbase = dst['frontend_base' if arm.startswith('m1_') else 'm1']
                raw, craw = raw_features(arm, base, sums), raw_features(arm, crossbase, csums)
                scaler = fit_scaler(raw, tr)
                x, cx = transform(raw, scaler), transform(craw, scaler)
                torch.manual_seed(seed)
                model = front.upstream.AttentionHead() if arm.startswith('m1_') else front.upstream.RegionHead()
                sources[arm] = front.train_head(out, arm, seed, model, x, base, table, tr, va,
                                                scaler['mu'], scaler['scale'])
                crosses[arm] = crossbase + front.upstream.predict(model, cx)
                np.savez_compressed(out / 'models' / f'{arm}_seed{seed}_scaler.npz', **scaler)
                checkpoint = torch.load(out / 'models' / f'{arm}_seed{seed}.pt', map_location='cpu', weights_only=False)
                checkpoint['active'] = scaler['active']
                checkpoint['conditional_base'] = 'corrected frontend_base' if arm.startswith('m1_') else 'frozen original corrected m1'
                torch.save(checkpoint, out / 'models' / f'{arm}_seed{seed}.pt')
                receipt = json.loads((out / f'selection_{arm}_{seed}.json').read_text())
                receipt.update(inactive_columns=np.flatnonzero(~scaler['active'].ravel()).tolist(),
                    source_input_absmax=float(abs(x).max()), cross_input_absmax=float(abs(cx).max()),
                    stage='s1' if arm.startswith('m1_') else 's12', corrected_base=True,
                    head_class=type(model).__name__, target_labels_used=False)
                write_json(out / f'selection_{arm}_{seed}.json', receipt)
            dest = out / f'seed{seed}'
            dest.mkdir(exist_ok=True)
            for side, values, tab in (('source', sources, table), ('cross', crosses, cross)):
                if not all(np.isfinite(z).all() for z in values.values()):
                    raise ValueError('Nonfinite saved prediction')
                np.savez_compressed(dest / f'{side}_predictions.npz', **values,
                    obs_id=tab.obs_id.to_numpy(), role=tab.role.to_numpy(str),
                    class_ids=np.arange(4), seed=seed)
            print('SAVED', direction, seed, flush=True)
    write_json(HERE / 'data_receipt.json', receipts)
    write_json(HERE / 'predictions_saved.json', dict(all_24_fits_complete=True,
        all_source_cross_predictions_saved=True, downstream_labels_used=False))


def evaluate():
    if not (HERE / 'predictions_saved.json').exists():
        raise ValueError('Save all predictions before evaluation')
    records, selections, paired, strata = [], [], [], []
    for source, target in (('oss1', 'oss2'), ('oss2', 'oss1')):
        direction = f'{source}-to-{target}'
        out = HERE / direction
        for seed in SEEDS:
            for arm in ARMS:
                s = json.loads((out / f'selection_{arm}_{seed}.json').read_text())
                selections.append(dict(direction=direction, arm=arm, seed=seed, epoch=s['epoch'],
                    parameters=s['parameters'], validation_J=s['fit_validation']['J'],
                    inactive_columns=str(s['inactive_columns']), seconds=s['seconds']))
            for side in ('source', 'cross'):
                path = (ROOT / f'{source}_m123_replication_20261026/stack_table.csv' if side == 'source'
                        else TRANSFER / direction / 'target_table.csv')
                table = pd.read_csv(path, dtype={'mmsi': str, 'sample_id': str})
                with np.load(out / f'seed{seed}/{side}_predictions.npz') as z:
                    np.testing.assert_array_equal(z['obs_id'], table.obs_id)
                    predictions = {k: z[k].copy() for k in ('frontend_base', 'original_m1', 'original_s12', *ARMS)}
                for role, idx in table.groupby('role').groups.items():
                    idx = np.asarray(idx)
                    y = table.iloc[idx].class_id.to_numpy(int)
                    for arm, score in predictions.items():
                        records.append(dict(direction=direction, side=side, role=role, arm=arm, seed=seed,
                            ships=table.iloc[idx].mmsi.nunique(), passes=table.iloc[idx].pass_group.nunique(),
                            stage='base' if arm == 'frontend_base' else ('s1' if arm.startswith('m1_') or arm == 'original_m1' else 's12'),
                            **front.metrics(y, score[idx])))
                    for candidate, reference in PAIRS:
                        a, b = predictions[candidate][idx].argmax(1), predictions[reference][idx].argmax(1)
                        paired.append(dict(direction=direction, side=side, role=role, seed=seed, candidate=candidate,
                            reference=reference, n=len(idx), rescued=int(((a == y) & (b != y)).sum()),
                            harmed=int(((a != y) & (b == y)).sum()), changed=int((a != b).sum())))
                        for field in ('class_id', 'pass_group'):
                            for label, positions in table.iloc[idx].groupby(field).groups.items():
                                positions = np.asarray(positions)
                                yy = table.iloc[positions].class_id.to_numpy(int)
                                aa, bb = predictions[candidate][positions].argmax(1), predictions[reference][positions].argmax(1)
                                strata.append(dict(direction=direction, side=side, role=role, seed=seed,
                                    candidate=candidate, reference=reference, field=field, group=str(label), n=len(positions),
                                    ships=table.iloc[positions].mmsi.nunique(), rescued=int(((aa == yy) & (bb != yy)).sum()),
                                    harmed=int(((aa != yy) & (bb == yy)).sum())))
    frame = pd.DataFrame(records)
    frame.to_csv(HERE / 'metrics.csv', index=False)
    summary = frame.groupby(['direction', 'side', 'role', 'stage', 'arm'], as_index=False).agg(
        OA=('OA', 'mean'), BA=('BA', 'mean'), OA_std=('OA', 'std'), BA_std=('BA', 'std'),
        NLL=('NLL', 'mean'), Brier=('Brier', 'mean'), n=('n', 'first'))
    summary.to_csv(HERE / 'metrics_summary.csv', index=False)
    pd.DataFrame(selections).to_csv(HERE / 'selections.csv', index=False)
    pd.DataFrame(paired).to_csv(HERE / 'paired_changes.csv', index=False)
    pd.DataFrame(strata).to_csv(HERE / 'stratified_changes.csv', index=False)
    print(summary.loc[summary.role.eq('evaluation')].to_string(index=False), flush=True)
    write_json(HERE / 'complete.json', dict(fits=24, steps=200, seeds=SEEDS, source_cross_saved_before_evaluation=True,
        conditional_matched_head_comparison=True, complete_m3_comparison=False, raw_chip_review_pending=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('all', 'train', 'evaluate'), nargs='?', default='all')
    args = parser.parse_args()
    if args.stage in ('all', 'train'):
        run_training()
    if args.stage in ('all', 'evaluate'):
        evaluate()


if __name__ == '__main__':
    main()
