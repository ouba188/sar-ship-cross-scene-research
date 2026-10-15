"""Generic M2 checks: class count, one-bias kernel, fixed base and aggregation."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from generic_joint_completion import (
    DOMAINS, MEASURES, cache_resized_rgb, episode_rows, freeze_pair,
    load_rgb_metadata, rgb_device_batch, summarize,
)


@pytest.mark.parametrize('classes', [7, 65])
def test_generic_pair_matches_weighted_primal_and_preserves_base(classes):
    rng = np.random.default_rng(20261008)
    n = 167
    support = rng.choice(n, 64, replace=False)
    labels = rng.integers(0, min(classes, 13), 64)
    retained = rng.normal(size=(n, 13))/np.sqrt(13)
    discarded = rng.normal(size=(n, 29))/np.sqrt(29)
    prior = rng.normal(size=(n, classes))
    base = rng.normal(size=(n, classes))
    original_base = base.copy()
    gate = dict(center=np.zeros(4), scale=np.ones(4), lr=np.array([.1, -.2, .3, .1, 2.]))
    result = freeze_pair(base, prior, retained@retained[support].T,
                         discarded@discarded[support].T, support, labels, gate, np.arange(n))
    count = np.bincount(labels, minlength=classes)
    weight = 64/(np.count_nonzero(count)*count[labels])
    desired = 2*np.eye(classes)[labels]-1-prior[support]
    for arm in ('reference', 'complement_only'):
        coords = retained if arm == 'reference' else np.column_stack((retained, discarded))/np.sqrt(2)
        phi = np.column_stack((coords, np.ones(n)))
        fitted = Ridge(alpha=8., fit_intercept=False).fit(phi[support], desired, sample_weight=weight)
        expected = prior+fitted.predict(phi)
        np.testing.assert_allclose(result[arm]['expert_scores'], expected, atol=1e-12)
        selected = result[arm]['selected']
        assert not selected[support].any()
        assert selected.sum() <= 5
        assert result[arm]['cap'] == 5
        np.testing.assert_array_equal(result[arm]['predictions'][~selected], base.argmax(1)[~selected])
        assert np.isin(expected.argmax(1)[selected], labels).all()
    np.testing.assert_array_equal(base, original_base)


def test_summary_uses_equal_domain_weights_and_paired_draws():
    rows = []
    for domain, n, original, delta in [('large', 5000, .8, -.1), ('small', 100, .4, .2)]:
        for arm in ('reference', 'complement_only'):
            for draw in range(5):
                value = original+(delta if arm == 'complement_only' else 0)+draw*.001
                rows.append(dict(dataset='example', domain=domain, arm=arm, draw=draw,
                    **{key: value for key in MEASURES}, selected=5, rescued=3, harmed=2, query_rows=n))
    summary = summarize(rows)
    ref, joint = summary['domain_macro_means']
    assert ref['acc'] == pytest.approx(.602)
    assert joint['acc'] == pytest.approx(.652)
    assert joint['delta_vs_reference_pp']['ba'] == pytest.approx(5.)
    assert joint['positive_ba_domains'] == 1
    assert summary['M3'] == summary['joint'] == 'not_run'


def test_rgb_episode_excludes_outer_target_from_heads_and_support_query():
    domains = np.repeat(['a', 'b', 'c', 'target'], 100)
    source = np.flatnonzero(domains != 'target')
    train, held, support, query = episode_rows(source, domains, 'b', 20261008)
    assert len(support) == 64 and len(query) == 36
    assert set(domains[train]) == {'a', 'c'}
    assert set(domains[held]) == {'b'}
    assert not np.intersect1d(support, query).size
    np.testing.assert_array_equal(np.sort(np.r_[support, query]), np.arange(len(held)))
    assert not np.isin(np.r_[train, held], np.flatnonzero(domains == 'target')).any()


def test_rgb_metadata_source_labels_do_not_change_with_target_truth(tmp_path):
    import json
    from generic_transfer_data import DOMAINS as original_domains

    assert DOMAINS == original_domains
    directory = tmp_path/'rgb_images'/'pacs'
    directory.mkdir(parents=True)
    domains = np.repeat(DOMAINS['pacs'], 80)
    labels = np.arange(len(domains)) % 7
    payload = dict(labels=labels.tolist(), domains=domains.tolist(), n_classes=7)
    path = directory/'COMPLETE.json'
    path.write_text(json.dumps(payload))
    initial = load_rgb_metadata('pacs', 'art_painting', tmp_path)
    labels[domains == 'art_painting'] = 100000
    payload['labels'] = labels.tolist()
    path.write_text(json.dumps(payload))
    changed = load_rgb_metadata('pacs', 'art_painting', tmp_path)
    np.testing.assert_array_equal(initial['source_rows'], changed['source_rows'])
    source = changed['source_rows']
    np.testing.assert_array_equal(initial['labels'][source], changed['labels'][source])
    for domain in DOMAINS['pacs'][1:]:
        train, held, support, query = episode_rows(source, changed['domains'], domain, 0)
        assert np.all(changed['labels'][train] < 7)
        assert np.all(changed['labels'][held[np.r_[support, query]]] < 7)


def test_device_batch_normalization_and_forced_flip(monkeypatch):
    import torch

    cache = np.random.default_rng(9).integers(0, 256, (4, 9, 11, 3), dtype=np.uint8)
    rows = np.array([3, 1])
    x = cache[rows].transpose(0, 3, 1, 2).astype(np.float32)/255.
    mean = np.array([.485, .456, .406], dtype=np.float32)[None, :, None, None]
    scale = np.array([.229, .224, .225], dtype=np.float32)[None, :, None, None]
    expected = (x-mean)/scale
    actual = rgb_device_batch(cache, rows, torch.device('cpu'), augment=False)
    np.testing.assert_allclose(actual.numpy(), expected, atol=3e-7, rtol=0)
    assert actual.is_contiguous(memory_format=torch.channels_last)
    monkeypatch.setattr(torch, 'rand', lambda n, device: torch.zeros(n, device=device))
    flipped = rgb_device_batch(cache, rows, torch.device('cpu'), augment=True)
    np.testing.assert_allclose(flipped.numpy(), expected[:, :, :, ::-1], atol=3e-7, rtol=0)


@pytest.mark.parametrize('dataset,shape', [('pacs', (227, 227)), ('officehome', (319, 241))])
def test_uint8_cache_matches_original_deterministic_resize(dataset, shape, tmp_path):
    import json
    from PIL import Image
    from torchvision.transforms import Resize

    directory = tmp_path/'rgb_images'/dataset
    directory.mkdir(parents=True)
    image = Image.fromarray(np.random.default_rng(10).integers(0, 256, (*shape, 3), dtype=np.uint8))
    image.save(directory/'00000000.image', format='PNG')
    (directory/'COMPLETE.json').write_text(json.dumps(dict(rows=1)))
    cache_resized_rgb(dataset, tmp_path)
    actual = np.load(directory/'rgb224_uint8.npy', mmap_mode='r')
    expected = np.asarray(Resize(224 if dataset == 'pacs' else (224, 224))(image))
    assert actual.shape == (1, 224, 224, 3)
    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual[0], expected)
