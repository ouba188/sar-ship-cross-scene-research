"""Focused checks for incidence interpolation and protected drift integration."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import numpy as np
from threadpoolctl import threadpool_limits
from external_knowledge_metadata import OUT, ROOT, interpolate, write
from external_knowledge_run import unpickle, load, maps, coordinates, expert_drift_moment
from compact_fusion_source import SOURCE, EXTERNAL


def main():
    result = {}
    grid = np.array([[0, 0, 30], [0, 1, 32], [1, 0, 33], [1, 1, 35]])
    values = interpolate(grid, [[.25, .25], [2, 2]])
    np.testing.assert_allclose(values[0], 31.25)
    assert np.isnan(values[1])
    result['incidence_affine_interpolation_and_no_extrapolation'] = True
    for dataset, (_, joint_dir) in EXTERNAL.items():
        saved = unpickle(OUT/dataset/'source_fit.pkl')
        geo = saved['geometry']
        rows = saved['source_rows'][:80]
        raw = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')[rows].astype(float)
        z = (raw-geo['center'])/geo['scale']
        zd = z@geo['qd']
        changed = raw-.5*(zd@geo['qd'].T)*geo['scale']
        z_changed = (changed-geo['center'])/geo['scale']
        np.testing.assert_allclose(z_changed@geo['qs'], z@geo['qs'], atol=1e-10)
        np.testing.assert_allclose(z_changed@geo['qd'], .5*zd, atol=1e-10)
        model = unpickle(ROOT/joint_dir/'source_model.pkl')
        stats = load(ROOT/joint_dir/'deployment_complement.npz')
        extra = np.zeros((len(raw), 512), np.float32)
        ce = coordinates(raw, model, geo, extra)
        dx, ar, ac = maps(model, stats, geo)
        moment = expert_drift_moment(raw, extra, model, stats, ce, ar, ac)
        centered = raw-model['pca'].mean_
        pc = model['pca'].components_.astype(float)
        residual_r = centered-(centered@pc.T)@pc
        sx = extra-model['pca_ssl'].mean_
        sp = model['pca_ssl'].components_.astype(float)
        residual_e = sx-(sx@sp.T)@sp
        residual = (np.c_[residual_r, residual_e]-stats['discarded_center'])/stats['discarded_scale']/np.sqrt(2560)
        retained = ce['x']/np.sqrt(192)
        k = .5*(retained@retained.T+residual@residual.T)+1
        norm = .5*(ar@ar.T+ac@ac.T)
        formula = k-.5*(zd@moment.T+moment@zd.T)+.25*zd@norm@zd.T
        r1, c1 = retained-.5*zd@ar, residual-.5*zd@ac
        explicit = .5*(r1@r1.T+c1@c1.T)+1
        np.testing.assert_allclose(formula, explicit, atol=1e-10)
        x0 = (model['pca'].transform(raw)-model['center'])/model['scale']
        x1 = (model['pca'].transform(changed)-model['center'])/model['scale']
        np.testing.assert_allclose(x1, x0-.5*zd@dx, atol=1e-10)
        result[dataset] = dict(protected_coordinates_unchanged=True, drift_amplitude_halved=True,
            raw_projection_matches_pca_delta=True, kernel_formula_matches_explicit_features=True,
            max_kernel_error=float(np.max(abs(formula-explicit))))
    write(OUT/'verification.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
