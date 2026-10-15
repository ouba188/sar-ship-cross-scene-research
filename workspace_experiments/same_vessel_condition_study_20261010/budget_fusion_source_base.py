"""Exact frozen base components and vessel-excluded source linear priors.

prepare(context) returns arrays in case_relation_headroom maps.npz order.
Only the balanced linear heads are refitted; frequency weights stay frozen.
Build caches in the existing ClearSAR runtime to reproduce saved PCA arithmetic;
frequency inference uses the existing GPU runtime. Cached loading needs neither fit.
"""

from full_teacher_data import BASE, ROOT, SOURCE, PORTS, write
from prepare_learned_frequency import (
    PARENT, REFERENCE, C4, DIAGNOSIS, base_phi, replace_r50,
    normalize_direction, rank_conditions, load_pickle,
)
from direct_incidence_conditioning import extend_transforms
from protected_drift_activation import fit_source_heads
from run_protected_drift_activation import make_transforms
from unified_head_expert import _conditions
from unified_head_run import base_kernel
from unified_head_core import score

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


OUT = BASE/'support_budget_fusion_select_20261012/base_cache'
FREQUENCY = ROOT/'learned_frequency_residual_20261009'
GPU_PYTHON = ROOT.parent/'envs/msfa-openmmlab-py310-rtx50-cu132/Scripts/python.exe'
KEYS = ('obs_ids', 'phi', 'mean', 'conditions', 'embedding',
        'prior_held', 'prior_oof', 'eta')


def _read(path, keys):
    with np.load(path) as z:
        return {key: z[key] for key in keys}


def _features(rows, bank, geometry, averaged, incidence, conditions):
    replaced = replace_r50(np.asarray(averaged[rows], float), bank, geometry)
    return dict(phi=base_phi(replaced, geometry, incidence[rows]),
                mean=normalize_direction((replaced[:, :128]-geometry['base_center'])
                                         /geometry['base_scale']),
                conditions=conditions[rows]), np.c_[replaced, incidence[rows]]


def _encode(folder, rows, conditions, manifest):
    # Saved PCA uses ClearSAR/NumPy2; saved bf16 convolutions use this GPU runtime.
    dest = OUT/folder.name
    dest.mkdir(parents=True, exist_ok=True)
    request, result = dest/'frequency_request.npz', dest/'frequency_result.npy'
    np.savez_compressed(request, rows=rows, conditions=conditions,
                        ids=manifest.iloc[rows].instance_id.to_numpy(str))
    subprocess.run([str(GPU_PYTHON), '-B', str(Path(__file__).resolve()),
                    '--encode', folder.name], check=True)
    answer = np.load(result)
    request.unlink()
    result.unlink()
    return answer


def _encode_local(folder, rows, conditions, ids):
    """The deployed bf16 inference recipe on only the requested cached maps."""
    import torch
    from learned_frequency_model import FrequencyResidual

    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    maps_root = FREQUENCY/'maps/self24ports'
    np.testing.assert_array_equal(np.load(maps_root/'sample_ids.npy')[rows], ids)
    maps = np.load(maps_root/'maps.npy', mmap_mode='r')
    checkpoint = torch.load(folder/'selective.pt', map_location='cpu', weights_only=True)
    model = FrequencyResidual('selective').cuda().eval()
    model.load_state_dict(checkpoint['state'])
    model.requires_grad_(False)
    answer = np.empty((len(rows), 256), dtype=np.float32)
    with torch.inference_mode():
        for lo in range(0, len(rows), 128):
            ix = rows[lo:lo+128]
            x = torch.from_numpy(np.asarray(maps[ix], dtype=np.float32)).cuda()
            c = torch.as_tensor(conditions[lo:lo+len(ix)], dtype=torch.float32, device='cuda')
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                z, _ = model(x, c)
            answer[lo:lo+len(ix)] = z.cpu().numpy()
    return answer


def _replay(context, folder, geometry, averaged, incidence, conditions,
            manifest, original_head):
    """Deployment-only verification; this head never supplies source priors."""
    target = _read(folder/'target.npz',
                   ('rows', 'phi', 'prior', 'mean', 'conditions', 'support', 'support_labels'))
    rows = target['rows']
    np.testing.assert_array_equal(rows, np.load(PARENT/context/'target_rows.npy'))
    bank = np.load(PARENT/context/'target_bank.npy', mmap_mode='r')
    data, augmented = _features(rows, bank, geometry, averaged, incidence, conditions)
    prior = augmented@original_head['coef']+original_head['intercept']
    errors = {}
    for key, actual in dict(**data, prior=prior).items():
        np.testing.assert_allclose(actual, target[key], atol=1e-10, rtol=0,
                                   err_msg=f'{context} deployed {key}')
        errors[key] = float(np.max(np.abs(actual-target[key])))
    saved = _read(folder/'selective_embedding.npz', ('rows', 'embedding'))
    np.testing.assert_array_equal(saved['rows'], rows)
    # Preserve the original 128-row inference batch to detect precision drift.
    n = min(128, len(rows))
    inferred = _encode(folder, rows[:n], data['conditions'][:n], manifest)
    np.testing.assert_allclose(inferred, saved['embedding'][:n], atol=1e-7, rtol=0,
                               err_msg=f'{context} selective encoder replay')
    errors['embedding'] = float(np.max(np.abs(inferred-saved['embedding'][:n])))
    eta = json.loads((folder/'PREPARED.json').read_text())['eta']
    support = target['support'][0]
    embedding = saved['embedding'].astype(float)
    scores = score(base_kernel(data, embedding, support, eta), prior,
                   support, target['support_labels'][0])[0]
    expected = _read(BASE/'support_completion_target_20261012'/context/'predictions.npz',
                     ('rows', 'base'))
    np.testing.assert_array_equal(expected['rows'], rows)
    np.testing.assert_allclose(scores, expected['base'][0], atol=1e-10, rtol=0)
    np.testing.assert_array_equal(scores.argmax(1), expected['base'][0].argmax(1))
    errors['base_solver'] = float(np.max(np.abs(scores-expected['base'][0])))
    return dict(max_absolute_errors=errors, component_rows=len(rows),
                neural_inference_rows=n, solver_draw=0, support_labels_used=64,
                target_query_labels_used=False, target_parameter_selection=False,
                independent_of_new_source_heads=True)


def prepare(context: str) -> dict:
    """Load/build exact base arrays; prior_oof covers the first 1,296 rows only."""
    if context not in PORTS:
        raise ValueError(f'Unsupported bounded context: {context}')
    dest = OUT/context
    cache = dest/'base.npz'
    ids = _read(BASE/'case_relation_headroom_20261012'/context/'maps.npz', ('obs_ids',))['obs_ids']
    if cache.exists() and (dest/'RECEIPT.json').exists():
        data = _read(cache, KEYS)
        np.testing.assert_array_equal(data['obs_ids'], ids)
        data['eta'] = float(data['eta'])
        return data

    started = time.perf_counter()
    folder = FREQUENCY/'tasks/self24ports'/context
    folds = _read(BASE/'crossfit_evidence_20261011/source_folds.npz', ('fit_obs_ids', 'assignment'))
    nfit = len(folds['fit_obs_ids'])
    np.testing.assert_array_equal(ids[:nfit], folds['fit_obs_ids'])
    meta = pd.read_csv(BASE/'observations.csv', usecols=['obs_id', 'manifest_row', 'mmsi'])
    mm = meta.set_index('obs_id').loc[ids]
    rows = mm.manifest_row.to_numpy(int)
    manifest = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id', 'port', 'mmsi'])
    np.testing.assert_array_equal(manifest.iloc[rows].mmsi.to_numpy(), mm.mmsi.to_numpy())
    descriptors = pd.read_csv(DIAGNOSIS/'descriptors/self24ports.csv')
    np.testing.assert_array_equal(descriptors.instance_id, manifest.instance_id)
    conditions = np.empty((len(manifest), 2))
    # Port populations are fixed before support draws and require no labels.
    for _, members in manifest.groupby('port', sort=False).groups.items():
        conditions[members] = rank_conditions(descriptors.loc[members])
    incidence = _conditions()
    averaged = np.load(C4/'pooled_c4.npy', mmap_mode='r')
    geometry = load_pickle(PARENT/context/'geometry.pkl')
    source_rows = np.load(PARENT/context/'source_rows.npy')
    positions = pd.Index(source_rows).get_indexer(rows)
    if np.any(positions < 0):
        raise ValueError('Observation manifest rows must belong to the context source bank')
    source_bank = np.load(PARENT/context/'source_bank.npy', mmap_mode='r')
    augmented = np.c_[source_bank, incidence[source_rows]]
    transform = extend_transforms({'joint_base': make_transforms(geometry)['joint_base']})
    allowed = set(source_rows.tolist())
    # Do not read the excluded target-port query labels, even for head replay.
    labels = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id'],
                         skiprows=lambda line: line > 0 and line-1 not in allowed).class_id.to_numpy()
    labels = labels[pd.Index(np.sort(source_rows)).get_indexer(source_rows)]

    def fit(mask):
        return fit_source_heads(augmented[mask], labels[mask], transform)['joint_base']

    with threadpool_limits(limits=2):
        # Verify the original training recipe separately before any exclusions.
        original = fit(np.ones(len(source_rows), dtype=bool))
        saved_head = load_pickle(REFERENCE/context/'direct_heads.pkl')['joint_base']
        for key in ('coef', 'intercept'):
            np.testing.assert_allclose(original[key], saved_head[key], atol=1e-10, rtol=0)
        replay = _replay(context, folder, geometry, averaged, incidence,
                         conditions, manifest, original)
        print('BASE DEPLOYMENT REPLAY PASS', context, replay['max_absolute_errors'], flush=True)
        data, observed = _features(rows, source_bank[positions], geometry,
                                   averaged, incidence, conditions)
        held_ships = mm.iloc[nfit:].mmsi.unique()
        fit_ships = mm.iloc[:nfit].mmsi.to_numpy()
        source_ships = manifest.iloc[source_rows].mmsi.to_numpy()
        if np.intersect1d(held_ships, fit_ships).size:
            raise ValueError('Fit and held cohorts share vessels')
        train = ~np.isin(source_ships, held_ships)
        head = fit(train)
        data['prior_held'] = observed@head['coef']+head['intercept']
        data['prior_oof'] = np.empty((nfit, 8))
        audits = []
        for fold in np.unique(folds['assignment']):
            val = folds['assignment'] == fold
            excluded = np.unique(fit_ships[val])
            if np.isin(fit_ships[~val], excluded).any():
                raise ValueError('Source folds split a vessel')
            use = train & ~np.isin(source_ships, excluded)
            reader = fit(use)
            data['prior_oof'][val] = observed[:nfit][val]@reader['coef']+reader['intercept']
            audits.append(dict(fold=int(fold), fit_rows=int(use.sum()),
                               excluded_fold_vessels=len(excluded), score_rows=int(val.sum()),
                               held_vessel_overlap=int(np.isin(source_ships[use], held_ships).sum()),
                               fold_vessel_overlap=int(np.isin(source_ships[use], excluded).sum()),
                               class_counts=np.bincount(labels[use], minlength=8).tolist()))
        data['embedding'] = _encode(folder, rows, data['conditions'], manifest)

    data.update(obs_ids=ids, eta=float(json.loads((folder/'PREPARED.json').read_text())['eta']))
    for key in KEYS:
        if not np.isfinite(data[key]).all():
            raise ValueError(f'Nonfinite source base component: {key}')
    receipt = dict(context=context, observation_rows=len(ids), fit_rows=nfit,
        held_rows=len(ids)-nfit, order='case_relation_headroom maps.npz obs_ids',
        shapes={key: list(np.shape(data[key])) for key in KEYS}, eta=data['eta'],
        head_training='Original unrotated source_bank + direct incidence; joint_base transform; balanced ridge1',
        prediction_features='C4 replaced R50, exact protected drift/stable phi, normalized mean, incidence',
        conditions='Within each actual port, complete SOURCE manifest population; average ranks, missing=.5; fixed across B/folds/draws',
        rank_reference_counts=manifest.groupby('port').size().to_dict(),
        rank_reference_uses_labels=False, embedding_conditions_identical_to_kernel=True,
        frequency_checkpoint=str(folder/'selective.pt'),
        geometry_runtime=sys.executable, frequency_runtime=str(GPU_PYTHON),
        frequency_input=str(FREQUENCY/'maps/self24ports/maps.npy'),
        source_neural_inference_rows=len(rows), neural_fits=0, source_head_fits=1+len(audits),
        deployment_replay_head_fits=1, deployment_replay=replay,
        held_head_fit_rows=int(train.sum()), held_excluded_vessels=len(held_ships),
        held_head_class_counts=np.bincount(labels[train], minlength=8).tolist(), folds=audits,
        target_query_labels_used=False, cached_original_prior_used_for_source=False,
        source_neural_exposure=True, geometry_refit=False,
        limitations=['Historical neural weights and geometry exposed to source vessels; only linear heads are vessel-excluded.',
                     'Unlabelled full-port descriptor ranks include fit and held rows; ranks are not fold-refitted.',
                     'Reused source development cohorts are not independent confirmation.',
                     'Source episodes span actual ports; each row retains its fixed actual-port ranks.'],
        seconds=time.perf_counter()-started)
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **data)
    write(dest/'RECEIPT.json', receipt)
    print('BASE CACHE READY', context, f'{receipt["seconds"]:.1f}s', flush=True)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contexts', nargs='+', default=list(PORTS), choices=PORTS)
    parser.add_argument('--encode', choices=PORTS)
    args = parser.parse_args()
    if args.encode:
        dest = OUT/args.encode
        request = _read(dest/'frequency_request.npz', ('rows', 'conditions', 'ids'))
        values = _encode_local(FREQUENCY/'tasks/self24ports'/args.encode,
                               request['rows'], request['conditions'], request['ids'])
        np.save(dest/'frequency_result.npy', values)
    else:
        for name in args.contexts:
            prepare(name)
