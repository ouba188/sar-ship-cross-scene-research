"""Label-free regional query/reference descriptors from existing native caches."""
import common as run
import numpy as np
from scipy.special import softmax

REGIONS = (np.arange(8), np.arange(8, 16), np.arange(16, 20))


def pool(x, mass):
    return np.stack([(x[:, :, r] * mass[:, :, r, None]).sum(2)
                     / np.maximum(mass[:, :, r].sum(2)[..., None], 1e-8) for r in REGIONS], 2)


def pair_stats(v, valid):
    count = valid.sum(2)
    den = np.maximum(count, 1)
    mean = (v * valid).sum(2) / den
    var = (np.square(v - mean[:, :, None]) * valid).sum(2) / den
    return mean, np.sqrt(var)


def score_features(s, available):
    base, vv, vh = (s[a] for a in ('base', 'prior_vv', 'center_vh'))
    values = [base - base.mean(1, keepdims=True), vv-base, vh-base]
    for score in (base, vv, vh):
        other = np.stack([np.delete(score, c, axis=1).max(1) for c in range(8)], 1)
        values.append(score-other)
    values.extend(softmax(score, axis=1) for score in (base, vv, vh))
    values.extend(np.broadcast_to(available[:, p, None], base.shape) for p in range(2))
    return np.stack(values, -1).astype(np.float32)


def main():
    (run.HERE / 'results').mkdir(exist_ok=True)
    data = run.Data()
    with np.load(run.LOCAL / 'spatial_frequency_joint_pooling/features.npz') as z:
        lookup = {int(v): j for j, v in enumerate(z['ids'])}
        order = np.array([lookup[int(v)] for v in data.up])
        mass, content, joint = (z[key][order].copy() for key in ('mass', 'mean', 'joint'))
    with np.load(run.LOCAL / 'frequency_conditioned_matching/features.npz') as z:
        lookup = {int(v): j for j, v in enumerate(z['ids'])}
        order = np.array([lookup[int(v)] for v in data.up])
        # Mean over directions; never feed absolute image-angle identities.
        energy = z['e_image'][order, ..., :12].reshape(len(order), 2, 20, 3, 4).mean(-1)
        coverage = z['coverage'][order].copy()
    rawrefs = np.load(run.LOCAL / 'scores_layer1.npz')['refs'][data.up[:data.n]]
    inverse = {int(v): j for j, v in enumerate(data.up)}
    refs = np.array([inverse[int(v)] for v in rawrefs.ravel()]).reshape(rawrefs.shape)
    assert (data.table.iloc[np.unique(refs)].role == 'strategy_fit').all()
    mass_region = np.stack([mass[:, :, r].sum(2) for r in REGIONS], -1)
    qvalid = mass_region[:data.n] > 0
    rvalid = mass_region[refs] > 0
    valid = qvalid[:, None, None] & rvalid
    has = valid.any(2)
    x = run.unit(pool(content, mass))
    q, s = x[:data.n, None, None], x[refs]
    mean, std = pair_stats((q*s).sum(-1), valid)
    agreement = (s[:, :, 0] * s[:, :, 1]).sum(-1) * valid.all(2)
    spatial = np.stack([mean, std, agreement, valid.mean(2),
        np.broadcast_to(qvalid[:, None], has.shape)], -1).reshape(data.n, 8, -1)

    # Standardize covariance channels using only historical source-training data.
    values = joint[data.tr][mass[data.tr] > 0]
    mu, sd = values.mean(0), values.std(0)
    joint = np.where(sd > 1e-6, (joint-mu)/np.maximum(sd, 1e-6), 0)
    joint[mass <= 0] = 0
    cov = run.unit(pool(joint, mass).reshape(len(mass), 2, 3, 3, 32))
    e, cover = pool(energy, mass), pool(coverage, mass)
    bandvalid = np.broadcast_to(valid[..., None], valid.shape+(3,))
    covmean, covstd = pair_stats((cov[:data.n, None, None]*cov[refs]).sum(-1), bandvalid)
    emean, estd = pair_stats(e[refs], bandvalid)
    ediff, _ = pair_stats(abs(e[:data.n, None, None]-e[refs]), bandvalid)
    cmean, _ = pair_stats(cover[refs], bandvalid)
    qe = np.broadcast_to(e[:data.n, None], emean.shape)
    qc = np.broadcast_to(cover[:data.n, None], emean.shape)
    frequency = np.stack([qe, emean, ediff, estd, covmean, covstd, qc, cmean], -1)
    frequency *= has[..., None, None]
    frequency = frequency.reshape(data.n, 8, -1)
    available = qvalid.any(-1)
    records = []
    for seed in run.SEEDS:
        ss = run.scores(data, seed)
        score = score_features(ss, available)
        raw = np.concatenate((score, spatial, frequency), -1).astype(np.float32)
        assert raw.shape == (data.n, 8, 185) and np.isfinite(raw).all()
        np.savez_compressed(run.HERE / 'results' / f'features_{seed}.npz', raw=raw,
            available=available, up_index=data.up[:data.n], refs=refs,
            base=ss['base'], delta_vv=ss['prior_vv']-ss['base'], delta_vh=ss['center_vh']-ss['base'])
        assert np.max(abs((ss['center_vh']-ss['base'])[~available[:, 1]])) == 0
        records.append(dict(seed=seed, dimensions=dict(score=11, spatial=30, frequency=144),
            source_covariance_scaler_n=len(data.tr), query_n=data.n,
            query_vh_missing=int((~available[:, 1]).sum())))
    run.write_json(run.HERE / 'extraction.json', dict(records=records,
        regions=['body', 'sides', 'ends'], wavelengths_native_pixels=[4, 8, 16],
        spatial_fields=['query_reference_cos_mean', 'cos_std', 'reference_agreement', 'valid_reference_fraction', 'query_valid'],
        frequency_fields=['query_log_energy', 'reference_log_energy', 'energy_abs_difference',
                          'reference_energy_std', 'covariance_cos_mean', 'covariance_cos_std',
                          'query_interior_coverage', 'reference_interior_coverage'],
        frequency_boundary='Native stretched-PNG Gabor descriptors and content covariance; not calibrated RCS or new physical causality.',
        reference_rule='Actual fixed two source references per candidate class; no query true class used.'))
    print(records, flush=True)


if __name__ == '__main__':
    main()
