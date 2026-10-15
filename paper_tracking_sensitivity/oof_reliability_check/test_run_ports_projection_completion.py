"""Check reusable pair geometry against the actual SAR episode helper."""

import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier

from run_ports_projection_completion import fit_episode_cache, materialize_episode, matched_views, summarize_matched, evaluate_matched_ports
from sar_joint_completion import fit_episode


def test_pair_cache_matches_episode_helper_and_rejects_changed_source(tmp_path):
    rng = np.random.default_rng(71)
    base = rng.normal(size=(420, 160)).astype(np.float32)
    extra = rng.normal(size=(420, 80)).astype(np.float32)
    labels = np.arange(420) % 8
    train, held = np.arange(320), np.arange(320, 420)
    path = tmp_path/'pair.npz'
    with threadpool_limits(limits=2):
        geometry, hit = fit_episode_cache(path, base, extra, labels, train)
        assert not hit
        actual = materialize_episode(geometry, base, labels, held)
        expected = fit_episode(base, extra, labels, train, held)
        assert actual.keys() == expected.keys()
        for key in actual:
            np.testing.assert_allclose(actual[key], expected[key], atol=2e-6, rtol=1e-6, err_msg=key)
        reused, hit = fit_episode_cache(path, base, extra, labels, train)
        assert hit
        np.testing.assert_array_equal(reused['expert_coef'], geometry['expert_coef'])
        with pytest.raises(AssertionError):
            fit_episode_cache(path, base, extra, labels, train[1:])


def test_matched_views_keep_cached_base_and_joint_kernel():
    rng = np.random.default_rng(19)
    xs = rng.normal(size=(90, 128)).astype(np.float32)
    extra = rng.normal(size=(90, 80)).astype(np.float32)
    bd = rng.normal(size=(90, 160))
    pca = PCA(n_components=64).fit(extra)
    z = pca.transform(extra)
    raw = np.column_stack((xs, z))
    center, scale = raw.mean(0), raw.std(0)+1e-6
    classifier = RidgeClassifier(alpha=1.).fit((raw-center)/scale, np.arange(90) % 8)
    model = dict(pca_ssl=pca, expert_center=center, expert_scale=scale, expert=classifier)
    stats = dict(discarded_center=np.zeros(240), discarded_scale=np.ones(240))
    prior, retained, discarded = matched_views(xs, bd, extra, model, stats, True)
    # No R50 PCA exists in this model: recomputing the fixed retained branch fails.
    expected = np.column_stack((xs, z))
    expected = (expected-center)/scale
    np.testing.assert_allclose(prior, classifier.decision_function(expected), atol=1e-6)
    np.testing.assert_allclose(retained, expected/np.sqrt(192), atol=1e-7)
    residual = (extra-pca.mean_)-((extra-pca.mean_)@pca.components_.T)@pca.components_
    np.testing.assert_allclose(discarded[:, :160], bd/np.sqrt(240), atol=1e-7)
    np.testing.assert_allclose(discarded[:, 160:], residual/np.sqrt(240), atol=1e-6)
    phi = np.column_stack((retained/np.sqrt(2), discarded/np.sqrt(2), np.ones(90)))
    np.testing.assert_allclose(phi@phi.T, .5*(retained@retained.T)+.5*(discarded@discarded.T)+1., atol=1e-12)


def test_summary_pairs_draws_and_weights_ports_equally():
    rows = []
    shifts = dict(reference=0., complement_only=.01, aligned_only=.02, joint=.03)
    for port, baseline in [('a', .3), ('b', .7)]:
        for draw in range(5):
            for arm, shift in shifts.items():
                row = dict(port=port, draw=draw, arm=arm)
                row.update({key: baseline+shift for key in ('acc', 'ba', 'expert_acc', 'expert_ba', 'base_acc', 'base_ba')})
                row.update(selected=4, rescued=2, harmed=1)
                rows.append(row)
    summary = summarize_matched(list(reversed(rows)))
    assert summary['n_ports'] == 2
    contrast = next(c for c in summary['contrasts'] if c['candidate'] == 'joint' and c['control'] == 'reference')
    assert contrast['means_pp']['ba'] == pytest.approx(3.)
    assert contrast['positive_ports']['ba'] == 2


def test_query_readout_waits_for_all_outer_features(tmp_path):
    inventory = [dict(port='a'), dict(port='b')]
    folder = tmp_path/'a/matched_joint'
    folder.mkdir(parents=True)
    (folder/'FEATURES_COMPLETE.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='Freeze all23 outer checkpoints/features'):
        evaluate_matched_ports(tmp_path, inventory, ['a'])
