"""CPU-only common readout for the matched general/domain SSL pilot.

Importing this module performs no fitting, extraction or writes. Source and target
features must come from the new matched RGB=[VV,VH,mean] encoders. Large artifacts
and numeric reports live under CACHE/readout, independently of the main trainer.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from unified_head_core import coefficients


ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
OUT = ROOT / 'general_domain_ssl_20261011'
CACHE = Path('C:/Users/25791/AppData/Local/Temp/sar_general_domain_20261011')
SOURCE_MANIFEST = Path('E:/\u4e34\u65f6\u4f1a\u8bdd/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv')
TARGETS = {
    'f0921': dict(design=ROOT / 'learned_frequency_residual_20261009/tasks/f0921/target.npz',
                 manifest=ROOT / 'f0921_detector_recrop_20261007/target_manifest.csv'),
    'oss2': dict(design=ROOT / 'learned_frequency_residual_20261009/tasks/oss2/target.npz',
                manifest=ROOT / 'opensarship2_transfer_20261007/target_manifest.csv'),
}
FEATURE_DIMS = dict(G50=2048, D50=2048, G18=512, D18=512)
PRIMARY_MODES = {
    'A_G50': ('G50',), 'B_D50': ('D50',),
    'C_G50_G18': ('G50', 'G18'), 'D_G50_D18': ('G50', 'D18'),
    'E_D50_D18': ('D50', 'D18'),
}
DIAGNOSTIC_MODES = {'diag_G18': ('G18',), 'diag_D18': ('D18',)}
LATE_MODES = {
    'late_G50_G18': ('A_G50', 'diag_G18'),
    'late_G50_D18': ('A_G50', 'diag_D18'),
    'late_D50_D18': ('B_D50', 'diag_D18'),
}
DESIGN_KEYS = ('rows', 'ids', 'support', 'support_labels', 'eval_masks')
CLASSES = 8
BLOCK = 512


def open_features(features):
    """Accept one path/array or a sequence of branch paths/arrays; mmap paths."""
    if isinstance(features, (str, Path, np.ndarray)):
        features = (features,)
    arrays = tuple(np.load(x, mmap_mode='r', allow_pickle=False)
                   if isinstance(x, (str, Path)) else x for x in features)
    if len(arrays) not in (1, 2):
        raise ValueError('Expected one or two feature branches')
    if any(x.ndim != 2 or len(x) != len(arrays[0]) for x in arrays):
        raise ValueError('Feature branches must share the full manifest row axis')
    return arrays


def design_block(features, rows):
    """Normalize only selected rows in float64; zero embeddings stay zero."""
    arrays = open_features(features)
    parts = []
    for array in arrays:
        raw = np.array(array[rows], dtype=np.float64, copy=True)
        if not np.isfinite(raw).all():
            raise ValueError('Nonfinite features on requested rows; encoding is incomplete')
        raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
        parts.append(raw * np.sqrt(32. / len(arrays)))
    return np.concatenate((*parts, np.ones((len(parts[0]), 1))), axis=1)


@threadpool_limits.wrap(limits=2, user_api='blas')
def fit_head(features, labels, rows, *, alpha=1., block_size=BLOCK, n_classes=CLASSES):
    """Return float64 [sum(branch widths)+1, classes] source coefficients.

    Labels and rows use the full source-manifest axis, including for memmaps with
    NaN outside the permitted pool. Fit balanced 2*onehot-1 ridge with an
    unpenalized final constant. No population-sized normalized array is formed.
    """
    arrays = open_features(features)
    rows = np.asarray(rows, dtype=np.int64)
    labels = np.asarray(labels)
    selected = labels[rows]
    counts = np.bincount(selected, minlength=n_classes)
    if len(counts) != n_classes or (counts == 0).any():
        raise ValueError('Source head requires every declared class')
    weights = len(rows) / (n_classes * counts)
    width = sum(x.shape[1] for x in arrays) + 1
    gram = np.zeros((width, width), dtype=np.float64)
    rhs = np.zeros((width, n_classes), dtype=np.float64)
    desired = 2 * np.eye(n_classes) - 1.
    for lo in range(0, len(rows), block_size):
        batch = rows[lo:lo + block_size]
        phi = design_block(arrays, batch)
        weighted = phi * weights[labels[batch], None]
        gram += phi.T @ weighted
        rhs += weighted.T @ desired[labels[batch]]
    gram.flat[::width + 1] += np.r_[np.full(width - 1, alpha), 0.]
    return cho_solve(cho_factor(gram, lower=True, check_finite=False), rhs,
                     check_finite=False)


def load_episodes(episodes):
    """Accept the main trainer's NPZ path/mapping or a list of episode dicts."""
    if isinstance(episodes, (str, Path)):
        with np.load(episodes, allow_pickle=False) as saved:
            return load_episodes({k: saved[k] for k in saved.files})
    if isinstance(episodes, dict) or isinstance(episodes, np.lib.npyio.NpzFile):
        ports = episodes['ports']
        return [dict(port=str(port), support=np.asarray(episodes[f'{i}_support']),
                     query=np.asarray(episodes[f'{i}_query']))
                for i, port in enumerate(ports)]
    return list(episodes)


def metrics(truth, prediction):
    classes = np.unique(truth)
    return dict(oa=100. * float(np.mean(truth == prediction)),
                ba=100. * float(np.mean([np.mean(prediction[truth == c] == c)
                                        for c in classes])))


@threadpool_limits.wrap(limits=2, user_api='blas')
def evaluate_source(feature_path, train_rows, episodes, labels, *, block_size=BLOCK):
    """Single sqrt32 head; return (mean episode BA percent, episode records).

    A temporary checkpoint feature path is accepted directly. This function
    writes nothing and never reads target features or labels. The caller owns
    positive-epoch selection and freezing; this function never selects/refits SSL.
    """
    features = open_features(feature_path)
    if len(features) != 1:
        raise ValueError('Checkpoint selection uses a single-branch head')
    labels = np.asarray(labels)
    train_rows = np.asarray(train_rows, dtype=np.int64)
    episodes = load_episodes(episodes)
    if not episodes:
        raise ValueError('No source validation episodes')
    for episode in episodes:
        support, query = episode['support'], episode['query']
        if np.intersect1d(train_rows, np.r_[support, query]).size:
            raise ValueError('Source head fitting includes validation rows')
        if np.intersect1d(support, query).size:
            raise ValueError('Source support/query overlap')
    coef = fit_head(features, labels, train_rows, block_size=block_size)
    records = []
    for i, episode in enumerate(episodes):
        support = np.asarray(episode['support'], dtype=np.int64)
        query = np.asarray(episode['query'], dtype=np.int64)
        sp = design_block(features, support)
        alpha = coefficients(sp @ sp.T, sp @ coef, labels[support], 8.)
        prediction = np.empty(len(query), dtype=np.int64)
        for lo in range(0, len(query), block_size):
            qp = design_block(features, query[lo:lo + block_size])
            score = qp @ coef + (qp @ sp.T) @ alpha
            prediction[lo:lo + len(qp)] = score.argmax(axis=1)
        records.append(dict(episode=i, port=str(episode['port']),
                            draw=int(episode.get('draw', sum(r['port'] == str(episode['port']) for r in records))),
                            support_n=len(support), query_n=len(query),
                            **metrics(labels[query], prediction)))
    return float(np.mean([r['ba'] for r in records])), records


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def _target_design(dataset):
    paths = TARGETS[dataset]
    with np.load(paths['design'], allow_pickle=False) as saved:
        data = {key: saved[key] for key in DESIGN_KEYS}
    # Only identifiers are read here. Query truth is read in the reporting phase.
    ids = pd.read_csv(paths['manifest'], usecols=['sample_id']).sample_id.to_numpy(str)
    np.testing.assert_array_equal(ids[data['rows']], data['ids'])
    n = len(data['ids'])
    if data['support'].shape != (5, 64) or data['support_labels'].shape != (5, 64):
        raise ValueError('Expected the original five64 support design')
    if data['eval_masks'].shape != (5, n) or data['eval_masks'].dtype != np.bool_:
        raise ValueError('Expected boolean five64 query masks')
    for support, mask in zip(data['support'], data['eval_masks']):
        if (support < 0).any() or (support >= n).any() or len(np.unique(support)) != 64:
            raise ValueError('Invalid target-local support indices')
        expected = np.ones(n, dtype=bool)
        expected[support] = False
        if not np.array_equal(mask, expected):
            raise ValueError('Original target query mask must exclude exactly its support')
    return data, len(ids)


def _encode_predictions(dataset, cache, root, heads, branches, modes, source_rows,
                        secondary, block_size):
    data, full_n = _target_design(dataset)
    target = {}
    for architecture, width in FEATURE_DIMS.items():
        target[architecture] = np.load(cache / 'features' / f'{dataset}_{architecture}.npy', mmap_mode='r')
        if target[architecture].shape != (full_n, width):
            raise ValueError(f'{dataset}/{architecture}: expected full target-manifest feature rows')
    directory = root / dataset
    directory.mkdir(parents=True, exist_ok=True)
    n = len(data['ids'])
    np.savez(directory / 'design.npz', **data, source_rows=source_rows)
    scores = np.lib.format.open_memmap(directory / 'scores.npy', mode='w+', dtype=np.float64,
                                     shape=(5, len(modes), n, CLASSES))
    predictions = np.lib.format.open_memmap(directory / 'predictions.npy', mode='w+', dtype=np.int8,
                                          shape=(5, len(modes), n))
    priors = np.lib.format.open_memmap(directory / 'priors.npy', mode='w+', dtype=np.float64,
                                     shape=(len(branches), n, CLASSES))
    support_state = {}
    for mi, (mode, architectures) in enumerate(branches.items()):
        features = tuple(target[a] for a in architectures)
        sp = np.stack([design_block(features, data['rows'][support]) for support in data['support']])
        ps = sp @ heads[mode]
        alpha = np.stack([coefficients(sp[d] @ sp[d].T, ps[d], data['support_labels'][d], 8.)
                          for d in range(5)])
        support_state[f'{mode}_phi'] = sp
        support_state[f'{mode}_prior'] = ps
        support_state[f'{mode}_coef'] = alpha
        for lo in range(0, n, block_size):
            hi = min(lo + block_size, n)
            phi = design_block(features, data['rows'][lo:hi])
            prior = phi @ heads[mode]
            priors[mi, lo:hi] = prior
            for draw in range(5):
                value = prior + (phi @ sp[draw].T) @ alpha[draw]
                scores[draw, mi, lo:hi] = value
                predictions[draw, mi, lo:hi] = value.argmax(axis=1)
    if secondary:
        for mode, (left, right) in LATE_MODES.items():
            mi, li, ri = modes.index(mode), modes.index(left), modes.index(right)
            for draw in range(5):
                for lo in range(0, n, block_size):
                    hi = min(lo + block_size, n)
                    value = .5 * (softmax(scores[draw, li, lo:hi], axis=1)
                                  + softmax(scores[draw, ri, lo:hi], axis=1))
                    scores[draw, mi, lo:hi] = value
                    predictions[draw, mi, lo:hi] = value.argmax(axis=1)
    for array in (scores, predictions, priors):
        array.flush()
    del scores, predictions, priors
    np.savez(directory / 'support_state.npz', **support_state)
    provenance = dict(dataset=dataset, n=n, source_n=len(source_rows), modes=modes,
                      prior_modes=list(branches), primary_modes=list(PRIMARY_MODES),
                      original_target_design=str(TARGETS[dataset]['design']),
                      target_manifest=str(TARGETS[dataset]['manifest']),
                      features={a: str(cache / 'features' / f'{dataset}_{a}.npy') for a in FEATURE_DIMS},
                      score_axes=['draw', 'mode', 'target_local_row', 'class'],
                      prior_axes=['prior_mode', 'target_local_row', 'class'],
                      score_types={m: 'equal_softmax_probability' if m in LATE_MODES else 'ridge_score' for m in modes},
                      query_labels_read=False)
    _write_json(directory / 'PROVENANCE.json', provenance)
    return provenance


@threadpool_limits.wrap(limits=2, user_api='blas')
def run_target_predictions(source_rows=None, *, out=OUT, cache=CACHE,
                           datasets=('f0921', 'oss2'), secondary=True, block_size=BLOCK):
    """Fit source heads, freeze ALL requested targets, then return numeric reports.

    Defaults to OUT/source_split.npz['allowed'] (shared 28,856 rows). Explicit
    source_rows may be a subset of allowed. Historical target source_rows are
    intentionally not asserted equal to this new pool. Nothing calls the trainer.
    """
    out, cache = Path(out), Path(cache)
    root = cache / 'readout'
    root.mkdir(parents=True, exist_ok=True)
    frozen_path = root / 'PREDICTIONS_FROZEN.json'
    if frozen_path.exists():
        raise FileExistsError(f'Completed predictions already exist: {frozen_path}')
    with np.load(out / 'source_split.npz', allow_pickle=False) as saved:
        allowed = saved['allowed']
    source_rows = np.asarray(allowed if source_rows is None else source_rows, dtype=np.int64)
    if len(np.unique(source_rows)) != len(source_rows) or np.setdiff1d(source_rows, allowed).size:
        raise ValueError('Source rows must be unique and belong to the new shared allowed pool')
    frame = pd.read_csv(SOURCE_MANIFEST, usecols=['instance_id', 'class_id'])
    labels = frame.class_id.to_numpy(np.int64)
    source = {}
    for architecture, width in FEATURE_DIMS.items():
        path = cache / 'features' / f'source_{architecture}.npy'
        source[architecture] = np.load(path, mmap_mode='r')
        if source[architecture].shape != (len(frame), width):
            raise ValueError(f'{architecture}: expected full source-manifest feature rows')
    branches = dict(PRIMARY_MODES)
    if secondary:
        branches.update(DIAGNOSTIC_MODES)
    modes = list(branches) + (list(LATE_MODES) if secondary else [])
    heads = {mode: fit_head(tuple(source[a] for a in architectures), labels, source_rows,
                            block_size=block_size) for mode, architectures in branches.items()}
    np.savez(root / 'source_heads.npz', **{f'coef_{m}': c for m, c in heads.items()},
             source_rows=source_rows, source_ids=frame.instance_id.to_numpy(str)[source_rows],
             source_class_counts=np.bincount(labels[source_rows], minlength=CLASSES),
             modes=np.asarray(list(branches)))
    protocol = dict(source_split=str(out / 'source_split.npz'), source_manifest=str(SOURCE_MANIFEST),
                    allowed_n=len(allowed), source_n=len(source_rows), primary_modes=list(PRIMARY_MODES),
                    secondary_modes=[m for m in modes if m not in PRIMARY_MODES],
                    branches={m: list(a) for m, a in branches.items()},
                    source_features={a: str(cache / 'features' / f'source_{a}.npy') for a in FEATURE_DIMS},
                    input='new matched RGB=[VV,VH,mean], identical input across architectures',
                    geometry='float64 unit branches, sqrt32 single or sqrt16 each dual, one constant',
                    source_head='balanced ridge alpha1, unpenalized constant, desired 2*onehot-1',
                    source_coef_axes=['geometry_coordinate', 'class'],
                    support_head='unified_head_core.coefficients, balanced ridge8, residual to source prior',
                    late='equal softmax of independently source-fitted and support-adapted single scores, temperature1',
                    comparison='new frozen common readout; no historical feature/head or legacy-score substitution',
                    block_size=block_size, support_draws_are_not_neural_seeds=True)
    _write_json(root / 'PROTOCOL.json', protocol)
    datasets = tuple(datasets)
    receipts = [_encode_predictions(d, cache, root, heads, branches, modes, source_rows,
                                    secondary, block_size) for d in datasets]
    _write_json(frozen_path, dict(datasets=list(datasets), query_labels_read=False,
                                 frozen_utc=datetime.now(timezone.utc).isoformat(),
                                 targets=receipts))
    return report_target_outcomes(cache=cache)


def report_target_outcomes(*, cache=CACHE):
    """Read query truth only after every target prediction has been persisted."""
    root = Path(cache) / 'readout'
    frozen = json.loads((root / 'PREDICTIONS_FROZEN.json').read_text(encoding='utf-8'))
    records, class_records = [], []
    for receipt in frozen['targets']:
        dataset, modes = receipt['dataset'], receipt['modes']
        directory = root / dataset
        with np.load(directory / 'design.npz', allow_pickle=False) as saved:
            rows, ids, masks = saved['rows'], saved['ids'], saved['eval_masks']
        truth_frame = pd.read_csv(receipt['target_manifest'], usecols=['sample_id', 'class_id']).iloc[rows]
        np.testing.assert_array_equal(truth_frame.sample_id.to_numpy(str), ids)
        truth = truth_frame.class_id.to_numpy(np.int64)
        prediction = np.load(directory / 'predictions.npy', mmap_mode='r')
        for draw, mask in enumerate(masks):
            y = truth[mask]
            for mi, mode in enumerate(modes):
                p = prediction[draw, mi, mask]
                primary = mode in PRIMARY_MODES
                records.append(dict(dataset=dataset, draw=draw, mode=mode, primary=primary,
                                    n=len(y), **metrics(y, p)))
                for c in np.unique(y):
                    subset = y == c
                    class_records.append(dict(dataset=dataset, draw=draw, mode=mode,
                                              class_id=int(c), n=int(subset.sum()),
                                              recall=100. * float(np.mean(p[subset] == c))))
    table = pd.DataFrame(records)
    table.to_csv(root / 'metrics.csv', index=False)
    pd.DataFrame(class_records).to_csv(root / 'per_class.csv', index=False)
    summary = table.groupby(['dataset', 'mode', 'primary'], sort=False)[['oa', 'ba']].mean().reset_index()
    summary.to_csv(root / 'summary.csv', index=False)
    result = dict(artifact_root=str(root), summary=summary.to_dict(orient='records'),
                  frozen_receipt=str(root / 'PREDICTIONS_FROZEN.json'),
                  metrics_path=str(root / 'metrics.csv'), query_truth_read_after_all_predictions=True)
    _write_json(root / 'COMPLETE.json', result)
    return result
