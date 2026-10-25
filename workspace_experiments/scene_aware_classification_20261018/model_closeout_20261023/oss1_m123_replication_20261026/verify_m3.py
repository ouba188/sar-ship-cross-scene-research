"""Check generic class-count operators against the exported implementation.

Historical assets are used only as a numerical test fixture here, never by
OSS1 training. This detects class-axis reshaping and metric/OT drift.
"""
from pathlib import Path
import json
import numpy as np
import torch
import train_m3 as new
from sar_m123 import m3 as old


def main():
    root = Path('E:/临时会话/sar_m123_standalone/assets')
    model = old.M3(root, device='cpu')
    ck = model._load(17)
    features = model.bank
    ids = np.arange(3)
    q = {k: v[ids] for k, v in features.items()}
    refs = model.retrieve(q)
    reference_local = model.match(q, refs, ck)
    all_refs = np.zeros((len(features['mass']), 8, 2), dtype=int)
    all_refs[ids] = refs
    runner = new.Matcher(features, all_refs, ck['ck']['metric_scaler'], 'cpu')
    local = runner.predict(ids, ck['metric'])
    error = float(abs(local-reference_local).max())
    assert error < 3e-6, error
    rng = np.random.default_rng(17)
    base, dv, dh = (rng.normal(size=(3, 8)).astype(np.float32) for _ in range(3))
    new.C = 8
    raw_error = float(abs(new.local_raw(local, base)-old.local_features(local, base)).max())
    qr = old.regional_descriptors(q, model.covariance_scaler)
    br = old.regional_descriptors(features, model.covariance_scaler)
    a, av = new.gate_raw(qr, br, refs, base, dv, dh)
    b, bv = old.gate_features(qr, br, refs, base, dv, dh)
    gate_error = float(abs(a-b).max())
    assert raw_error < 1e-6 and gate_error < 1e-6
    np.testing.assert_array_equal(av, bv)
    new.C = 4
    g, available = new.gate_raw(qr, br, refs[:, :4], base[:, :4], dv[:, :4], dh[:, :4])
    assert g.shape == (3, 4, 185)
    gate = new.Gate(185, .6)
    weights = gate(torch.tensor(g), torch.tensor(available))
    np.testing.assert_allclose(weights.detach().numpy(), np.array([.6, .4])*available, atol=1e-7)
    # A covariance is linear under a fixed projection; its centering term drops out.
    z = rng.normal(size=(160, 256))
    w = rng.uniform(size=(20, 160))
    ew = rng.uniform(size=(20, 3, 160))*w[:, None]
    p = rng.normal(size=(256, 32))
    mean = rng.normal(size=(256,))
    cov, mu = new.tokens.grid_moments(z, w, ew)
    cp, mp = new.tokens.grid_moments((z-mean)@p, w, ew)
    projection_error = float(abs(cp-cov@p).max())
    mean_error = float(abs(mp-(mu-mean)@p).max())
    assert max(projection_error, mean_error) < 1e-10
    output = dict(match_max_error=error, readout_max_error=raw_error,
                  gate_input_max_error=gate_error, four_class_shape=list(g.shape),
                  projection_covariance_error=projection_error, projection_mean_error=mean_error,
                  historical_assets_only_numerical_fixture=True, oss1_training_assets_not_used=True)
    new.write_json(new.HERE/'m3_operator_checks.json', output)
    print(json.dumps(output))


if __name__ == '__main__':
    main()
