"""Focused checks for base-only formula, routing and saved-row contracts."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from base_only_preserved_expert import (
    FACTORS, base_views, fit_episode_gate, gate_features, predict_bases, verify_target_alignment)
from evidence_aware_handoff import fit_gate, predict_gate, route_scores


def fixture():
    rng = np.random.default_rng(7)
    bank = rng.normal(size=(80, 197))
    g = dict(stable_rank=3, drift_rank=2, drift_to_xs=rng.normal(size=(2, 128)),
             base_center=rng.normal(size=128), base_scale=rng.uniform(.5, 2, 128))
    heads = {arm + '_base': dict(coef=rng.normal(size=(197, 8)), intercept=rng.normal(size=8))
             for arm in FACTORS}
    return bank, g, heads


def test_base_views_match_explicit_formula():
    bank, g, _ = fixture()
    actual = base_views(bank, g)
    for arm, (suppress, activate) in FACTORS.items():
        x = bank[:, :128] - .5 * bank[:, 195:] @ g['drift_to_xs'] if suppress else bank[:, :128].astype('float32')
        body = (x-g['base_center'])/g['base_scale']/np.sqrt(128)
        if activate:
            body = np.column_stack((body, bank[:, 192:195]/np.sqrt(3)))
        np.testing.assert_allclose(actual[arm], np.column_stack((body, np.ones(80))), atol=1e-14)


def test_adaptation_matches_independent_primal_ridge():
    bank, g, heads = fixture()
    support = np.arange(64)
    y = np.arange(64) % 8
    views = base_views(bank, g)
    result = predict_bases(bank, views, heads, support, y)
    for arm, phi in views.items():
        head = heads[arm + '_base']
        prior = bank @ head['coef'] + head['intercept']
        regression = Ridge(alpha=8., fit_intercept=False).fit(phi[support], 2*np.eye(8)[y]-1-prior[support])
        np.testing.assert_allclose(result[arm], prior + regression.predict(phi), atol=2e-12)


def test_episode_gate_keeps_original_binary_objective():
    rng = np.random.default_rng(19)
    b, e = rng.normal(size=(100, 8)), rng.normal(size=(100, 8))
    y = np.arange(100) % 8
    episode = dict(base=b, expert=e, labels=y, features=gate_features(b,e),
                   eligible=b.argmax(1) != e.argmax(1), weights=np.ones(100))
    actual = fit_episode_gate([episode])
    expected = fit_gate(episode['features'], b, e, y, episode['eligible'], episode['weights'])
    np.testing.assert_array_equal(predict_gate(actual, episode['features']),
                                  predict_gate(expected, episode['features']))


def test_routing_does_not_mutate_expert_or_select_support():
    rng = np.random.default_rng(11)
    b, e = rng.normal(size=(200, 8)), rng.normal(size=(200, 8))
    before = e.copy()
    query = np.arange(200) >= 64
    routed = route_scores(b, e, np.ones(200), np.arange(8), query, np.arange(200))
    np.testing.assert_array_equal(e, before)
    assert routed['cap'] == 6
    assert routed['selected'].sum() == 6
    assert not routed['selected'][:64].any()
    np.testing.assert_array_equal(routed['predictions'][routed['selected']], e.argmax(1)[routed['selected']])


def test_alignment_rejects_shifted_rows_and_support_in_query():
    rows = np.arange(100)
    evaluation = np.tile(rows >= 64, (5,1))
    old = dict(query_rows=rows, source_rows=np.arange(101, 201),
        support_rows=np.tile(np.arange(64), (5,1)), eval_mask=evaluation, cap=np.ones(5, dtype=int))
    cached = {k: v.copy() for k, v in old.items()}
    verify_target_alignment(old, cached, rows)
    cached['query_rows'][0] = 1000
    with pytest.raises(AssertionError):
        verify_target_alignment(old, cached, rows)
    cached['query_rows'] = rows.copy()
    old['eval_mask'][0,0] = True
    cached['eval_mask'] = old['eval_mask'].copy()
    with pytest.raises(ValueError, match='Support/query'):
        verify_target_alignment(old, cached, rows)
