import numpy as np
from scipy.special import softmax

from full_ssl_readout import kernels, fit_support, scores


def example():
    rng = np.random.default_rng(71)
    widths = dict(phi=5, raw_phi=5, mean=4, embedding=6, conditions=2,
                  single_expert=9, dual_expert=13, prior=8, raw_prior=8,
                  single_prior=8, dual_prior=8)
    comp = {k: rng.normal(size=(20, width)) for k, width in widths.items()}
    positions = np.array([[0, 2, 4, 6], [1, 3, 7, 8]])
    design = dict(support_labels=np.array([[0, 0, 1, 2], [2, 3, 3, 3]]))
    return comp, positions, design


def test_primary_and_unified_follow_declared_balanced_system():
    comp, positions, design = example()
    k = kernels(comp, comp)
    alpha = fit_support(comp, design, positions)
    for draw, ix in enumerate(positions):
        result = scores(comp, k, positions, alpha, draw)
        y = design['support_labels'][draw]
        classes, counts = np.unique(y, return_counts=True)
        weights = np.array([len(y)/(len(classes)*counts[np.flatnonzero(classes == c)[0]]) for c in y])
        for arch in ('single', 'dual'):
            kernel = .5*(k['full']+k[arch])
            prior = .5*(comp['prior']+comp[f'{arch}_prior'])
            desired = 2*np.eye(8)[y]-1
            independent = np.linalg.solve(kernel[np.ix_(ix, ix)]+np.diag(8/weights), desired-prior[ix])
            expected = prior+kernel[:, ix]@independent
            np.testing.assert_allclose(result[f'unified_{arch}'], expected, atol=1e-12)
            expected_late = .5*(softmax(result['base_full'], axis=1)+softmax(result[f'expert_{arch}'], axis=1))
            np.testing.assert_allclose(result[f'late_{arch}'], expected_late, atol=1e-15)


def test_single_decisions_do_not_consult_dual_components():
    comp, positions, design = example()
    before_k = kernels(comp, comp)
    before = scores(comp, before_k, positions, fit_support(comp, design, positions), 0)
    changed = dict(comp, dual_expert=comp['dual_expert']*7+5, dual_prior=comp['dual_prior']*9-2)
    after_k = kernels(changed, changed)
    after = scores(changed, after_k, positions, fit_support(changed, design, positions), 0)
    for mode in ('base_full', 'expert_single', 'late_single', 'unified_single'):
        np.testing.assert_array_equal(before[mode], after[mode])


def test_zero_frequency_restores_conditional_base_and_fusion():
    comp, positions, design = example()
    comp['embedding'][:] = 0
    k = kernels(comp, comp)
    result = scores(comp, k, positions, fit_support(comp, design, positions), 0)
    np.testing.assert_array_equal(result['base_full'], result['base_condition'])
    for arch in ('single', 'dual'):
        np.testing.assert_array_equal(result[f'late_{arch}'], result[f'late_{arch}_no_frequency'])
