"""Tiny CPU checks; never use the real population or any GPU."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

import general_domain_readout as readout
from unified_head_core import class_weights, coefficients


def test_source_weights_and_unpenalized_constant(tmp_path):
    rng = np.random.default_rng(71)
    raw = rng.normal(size=(23, 5)).astype(np.float32)
    rows = np.arange(1, 21)
    labels = np.array([0] + [0] * 13 + [1] * 5 + [2] * 2 + [0] * 2)
    raw[[0, 21, 22]] = np.nan
    path = tmp_path / 'features.npy'
    np.save(path, raw)
    coef = readout.fit_head(path, labels, rows, block_size=4, n_classes=3)
    x = raw[rows].astype(np.float64)
    phi = np.c_[np.sqrt(32) * x / np.linalg.norm(x, axis=1, keepdims=True), np.ones(len(rows))]
    counts = np.bincount(labels[rows])
    weights = len(rows) / (3 * counts[labels[rows]])
    target = 2 * np.eye(3)[labels[rows]] - 1
    penalty = np.diag(np.r_[np.ones(5), 0.])
    # Independent augmented least squares includes the unpenalized intercept.
    expected = np.linalg.lstsq(np.r_[phi * np.sqrt(weights[:, None]), penalty],
                              np.r_[target * np.sqrt(weights[:, None]), np.zeros((6, 3))], rcond=None)[0]
    np.testing.assert_allclose(coef, expected, atol=1e-11)
    for c in range(3):
        assert weights[labels[rows] == c].sum() == pytest.approx(len(rows) / 3)
    zeros = np.zeros((len(labels), 5))
    constant = readout.fit_head(zeros, labels, rows, n_classes=3)
    np.testing.assert_allclose(constant[-1], np.full(3, -1 / 3), atol=1e-12)
    np.testing.assert_allclose(constant[:-1], 0., atol=1e-12)


def test_support_primal_dual_and_present_class_weights():
    rng = np.random.default_rng(19)
    phi = np.c_[rng.normal(size=(13, 7)), np.ones(13)]
    support = phi[:9]
    labels = np.array([0] * 6 + [2] * 2 + [3])
    weights = np.array([.5] * 6 + [1.5] * 2 + [3.])
    np.testing.assert_allclose(class_weights(labels), weights)
    head = rng.normal(size=(8, 4))
    prior = phi @ head
    target = 2 * np.eye(4)[labels] - 1 - prior[:9]
    beta = np.linalg.solve(support.T @ (weights[:, None] * support) + 8 * np.eye(8),
                           support.T @ (weights[:, None] * target))
    alpha = coefficients(support @ support.T, prior[:9], labels, 8.)
    np.testing.assert_allclose(prior + phi @ beta,
                               prior + (phi @ support.T) @ alpha, atol=1e-12)


def test_duplicated_branch_same_source_and_residual():
    rng = np.random.default_rng(23)
    raw = rng.normal(size=(40, 6))
    labels = np.arange(40) % 8
    train = np.arange(24)
    support = np.arange(24, 34)
    query = np.arange(34, 40)
    results = []
    for features in (raw, (raw, raw)):
        coef = readout.fit_head(features, labels, train, block_size=5)
        ps = readout.design_block(features, support)
        pq = readout.design_block(features, query)
        alpha = coefficients(ps @ ps.T, ps @ coef, labels[support], 8.)
        results.append((pq @ coef, pq @ coef + (pq @ ps.T) @ alpha))
    np.testing.assert_allclose(results[0], results[1], atol=1e-11)


def test_source_validation_npz_and_no_writes(tmp_path):
    rng = np.random.default_rng(41)
    path = tmp_path / 'temporary_checkpoint.npy'
    np.save(path, rng.normal(size=(56, 5)))
    episodes = tmp_path / 'validation_episodes.npz'
    np.savez(episodes, ports=np.array(['Port A', 'Port A']),
             **{'0_support': np.arange(24, 32), '0_query': np.arange(32, 40),
                '1_support': np.arange(40, 48), '1_query': np.arange(48, 56)})
    before = set(tmp_path.iterdir())
    mean, records = readout.evaluate_source(path, np.arange(24), episodes, np.arange(56) % 8,
                                            block_size=3)
    assert set(tmp_path.iterdir()) == before
    assert len(records) == 2 and [r['draw'] for r in records] == [0, 1]
    assert mean == pytest.approx(np.mean([r['ba'] for r in records]))
    with pytest.raises(ValueError, match='includes validation'):
        readout.evaluate_source(path, np.arange(25), episodes, np.arange(56) % 8)


def test_nonfinite_selected_rows_rejected():
    raw = np.ones((16, 3))
    raw[2] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        readout.fit_head(raw, np.arange(16) % 8, np.arange(16))


def test_callback_global_rows_list_episodes_and_json(tmp_path):
    rng = np.random.default_rng(43)
    train = np.arange(0, 48, 2)
    support, query = np.arange(64, 128), np.arange(128, 256)
    rows = np.r_[train, support, query]
    raw = np.full((320, 5), np.nan, dtype=np.float32)
    raw[rows] = rng.normal(size=(len(rows), 5))
    labels = (np.arange(320) // 2) % 8
    path = tmp_path / 'checkpoint.npy'
    np.save(path, raw)
    episodes = [dict(port=np.str_('Port A'), draw=np.int64(3), support=support, query=query)]
    before = set(tmp_path.iterdir())
    mean, records = readout.evaluate_source(path, train, episodes, labels)
    json.dumps((mean, records), allow_nan=False)
    assert type(mean) is float
    assert records[0]['draw'] == 3
    assert records[0]['support_n'] == 64 and records[0]['query_n'] == 128
    assert set(tmp_path.iterdir()) == before

    # Independently solve both weighted primal systems to verify global indexing.
    x = raw[rows].astype(np.float64)
    phi = np.c_[np.sqrt(32) * x / np.linalg.norm(x, axis=1, keepdims=True), np.ones(len(rows))]
    tr, sp, qp = phi[:24], phi[24:88], phi[88:]
    ty, sy = labels[train], labels[support]
    tw = len(train) / (8 * np.bincount(ty)[ty])
    source_coef = np.linalg.solve(tr.T @ (tw[:, None] * tr) + np.diag([1.] * 5 + [0.]),
                                  tr.T @ (tw[:, None] * (2 * np.eye(8)[ty] - 1)))
    sw = len(support) / (8 * np.bincount(sy)[sy])
    correction = np.linalg.solve(sp.T @ (sw[:, None] * sp) + 8 * np.eye(6),
                                 sp.T @ (sw[:, None] * (2 * np.eye(8)[sy] - 1 - sp @ source_coef)))
    prediction = (qp @ (source_coef + correction)).argmax(axis=1)
    truth = labels[query]
    expected_ba = 100 * np.mean([np.mean(prediction[truth == c] == c) for c in np.unique(truth)])
    assert mean == pytest.approx(expected_ba)
    assert records[0]['oa'] == pytest.approx(100 * np.mean(prediction == truth))


def test_tiny_target_pipeline_freezes_both_before_truth(tmp_path, monkeypatch):
    rng = np.random.default_rng(83)
    out, cache = tmp_path / 'out', tmp_path / 'cache'
    out.mkdir()
    features = cache / 'features'
    features.mkdir(parents=True)
    allowed = np.arange(24)
    np.savez(out / 'source_split.npz', allowed=allowed, inner_train=np.arange(16), inner_val=np.arange(16, 24))
    manifest = tmp_path / 'source.csv'
    pd.DataFrame(dict(instance_id=[f's{i}' for i in range(30)], class_id=np.arange(30) % 8)).to_csv(manifest, index=False)
    monkeypatch.setattr(readout, 'SOURCE_MANIFEST', manifest)
    dims = dict(G50=5, D50=5, G18=3, D18=3)
    monkeypatch.setattr(readout, 'FEATURE_DIMS', dims)
    for architecture, width in dims.items():
        raw = rng.normal(size=(30, width)).astype(np.float32)
        raw[24:] = np.nan
        np.save(features / f'source_{architecture}.npy', raw)
    targets = {}
    for dataset, n in [('f0921', 70), ('oss2', 73)]:
        path = tmp_path / f'{dataset}.npz'
        manifest = tmp_path / f'{dataset}.csv'
        ids = np.array([f'{dataset}_{i * 3}' for i in range(n)])
        labels = np.arange(n) % 8
        support = np.stack([rng.choice(n, 64, replace=False) for _ in range(5)])
        masks = np.ones((5, n), dtype=bool)
        masks[np.arange(5)[:, None], support] = False
        np.savez(path, rows=np.arange(n), ids=ids, support=support, support_labels=labels[support],
                 eval_masks=masks, source_rows=np.arange(29))
        pd.DataFrame(dict(sample_id=ids, class_id=labels)).to_csv(manifest, index=False)
        targets[dataset] = dict(design=path, manifest=manifest)
        for architecture, width in dims.items():
            np.save(features / f'{dataset}_{architecture}.npy', rng.normal(size=(n, width)).astype(np.float32))
    monkeypatch.setattr(readout, 'TARGETS', targets)
    original_read_csv = pd.read_csv
    truth_reads = []

    def tracked_read_csv(path, *args, **kwargs):
        if str(path) in [str(t['manifest']) for t in targets.values()] and 'class_id' in kwargs.get('usecols', []):
            receipt = json.loads((cache / 'readout/PREDICTIONS_FROZEN.json').read_text())
            assert receipt['datasets'] == ['f0921', 'oss2']
            for dataset in targets:
                assert (cache / 'readout' / dataset / 'predictions.npy').exists()
                assert (cache / 'readout' / dataset / 'support_state.npz').exists()
            truth_reads.append(path)
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, 'read_csv', tracked_read_csv)
    result = readout.run_target_predictions(out=out, cache=cache, block_size=11)
    assert len(truth_reads) == 2
    assert len(result['summary']) == 20
    assert sum(r['primary'] for r in result['summary']) == 10
    with np.load(cache / 'readout/source_heads.npz') as z:
        np.testing.assert_array_equal(z['source_rows'], allowed)
    for dataset in targets:
        directory = cache / 'readout' / dataset
        scores = np.load(directory / 'scores.npy', mmap_mode='r')
        predictions = np.load(directory / 'predictions.npy', mmap_mode='r')
        provenance = json.loads((directory / 'PROVENANCE.json').read_text())
        modes = provenance['modes']
        for mode, (left, right) in readout.LATE_MODES.items():
            expected = .5 * (softmax(scores[:, modes.index(left)], axis=-1)
                              + softmax(scores[:, modes.index(right)], axis=-1))
            np.testing.assert_allclose(scores[:, modes.index(mode)], expected, atol=1e-12)
        np.testing.assert_array_equal(predictions, scores.argmax(axis=-1))
        # Saved prior, source coef and support dual coef replay the common head.
        with np.load(directory / 'design.npz') as d, np.load(directory / 'support_state.npz') as state:
            q = readout.design_block(features / f'{dataset}_G50.npy', d['rows'][:3])
            prior = np.load(directory / 'priors.npy', mmap_mode='r')[0, :3]
            replay = prior + (q @ state['A_G50_phi'][0].T) @ state['A_G50_coef'][0]
            np.testing.assert_allclose(replay, scores[0, 0, :3], atol=1e-12)
            np.testing.assert_array_equal(d['source_rows'], allowed)
    with pytest.raises(FileExistsError, match='Completed predictions'):
        readout.run_target_predictions(out=out, cache=cache)
