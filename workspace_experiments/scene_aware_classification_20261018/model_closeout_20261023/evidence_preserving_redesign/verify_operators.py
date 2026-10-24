"""Targeted algebra/interface checks, including actual SOURCE response maps.

No classifier fitting, checkpoint selection, evaluation accuracy, or new labels.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from operators import AxisPairConv, TypedPairMetric, native_bands, moment_pool, decode_moments

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent/'m2_operator_validation'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    torch.set_num_threads(2)
    torch.manual_seed(24)
    rng = np.random.default_rng(24)
    checks = {}
    im = rng.uniform(size=(41, 37))
    b = native_bands(im)
    checks['band_reconstruction_max'] = float(abs(b.sum(0)-im).max())
    checks['constant_detail_max'] = float(abs(native_bands(np.ones_like(im))[:2]).max())
    assert checks['band_reconstruction_max'] < 1e-12
    assert checks['constant_detail_max'] < 1e-12
    png = np.zeros((9, 9), np.uint8)
    png[4, 4] = 255
    png_bands = native_bands(png)
    checks['uint8_signed_detail_min'] = float(png_bands[0].min())
    checks['uint8_float_replay'] = float(abs(png_bands-native_bands(png.astype(float)/255)).max())
    checks['uint8_reconstruction_max'] = float(abs(png_bands.sum(0)-png/255.).max())
    assert checks['uint8_signed_detail_min'] < 0
    assert checks['uint8_float_replay'] == 0 and checks['uint8_reconstruction_max'] < 1e-12

    conv = AxisPairConv(3).double()
    x = torch.randn(2, 3, 19, 21, dtype=torch.float64)
    role = torch.softmax(torch.randn(2, 3, 19, 21, dtype=torch.float64), 1)
    valid = torch.ones(2, 1, 19, 21, dtype=torch.float64)
    theta = torch.tensor([.32, -.8], dtype=torch.float64)
    types = torch.tensor([0, 4])
    out = conv(x, role, valid, theta, types)
    checks['conv_initial_identity'] = float((out-x).abs().max().detach())
    (out*torch.randn_like(out)).sum().backward()
    checks['conv_output_initial_gradient'] = float(conv.output.grad.abs().max())
    assert checks['conv_initial_identity'] == 0 and checks['conv_output_initial_gradient'] > 1e-8
    with torch.no_grad():
        conv.output.normal_(0, .2)
        out = conv(x, role, valid, theta, types)
        reversed_out = conv(x, role, valid, theta+torch.pi, types)
        checks['conv_axis_pi_max'] = float((out-reversed_out).abs().max())
        flat = torch.ones_like(x)
        checks['conv_constant_residual_max'] = float((conv(flat, role, valid, theta, types)-flat).abs().max())
        checks['conv_active_delta_max'] = float((out-x).abs().max())
    assert checks['conv_axis_pi_max'] < 1e-10
    assert checks['conv_constant_residual_max'] < 1e-10
    assert checks['conv_active_delta_max'] > 1e-5

    old = module('old_m2_operators', PREVIOUS/'operators.py')
    response = torch.randn(2, 8, 19, dtype=torch.float64)
    a = torch.softmax(torch.randn(2, 5, 19, dtype=torch.float64), 1)
    tau = torch.ones(2, dtype=torch.float64)
    stat = moment_pool(response, a, tau)
    reference = old.class_summary(response, a.sum(1, keepdim=True), tau)
    checks['synthetic_summary_replay'] = float((decode_moments(stat, True)-reference).abs().max())
    assert checks['synthetic_summary_replay'] < 1e-12
    # Changing grouping must affect slot descriptors without erasing the global summary.
    a2 = a.roll(3, -1)
    checks['slot_descriptor_changes'] = float((decode_moments(stat)-decode_moments(moment_pool(response, a2, tau))).abs().max())
    checks['empty_region_zero'] = float(decode_moments(moment_pool(response, a*0, tau)).abs().max())
    assert checks['slot_descriptor_changes'] > 1e-3 and checks['empty_region_zero'] == 0
    peaked = torch.tensor([[[0.,100.], [100.,0.]]], dtype=torch.float64)
    singletons = torch.eye(2, dtype=torch.float64)[None]
    decoded = decode_moments(moment_pool(peaked, singletons, tau[:1]))
    checks['weak_slot_span_max'] = float(decoded[...,1].abs().max())
    checks['weak_slot_bias_error'] = float((decoded[...,2].abs()-50).abs().max())
    assert checks['weak_slot_span_max'] == 0 and checks['weak_slot_bias_error'] == 0
    differentiable = response.clone().requires_grad_()
    empty = a.clone()
    empty[:, -1] = 0
    mixed = moment_pool(differentiable, empty, tau)
    grad = torch.autograd.grad(decode_moments(mixed).square().sum()+decode_moments(mixed, True).square().sum(), differentiable)[0]
    checks['pool_gradient_finite_with_empty_slot'] = bool(torch.isfinite(grad).all())
    assert checks['pool_gradient_finite_with_empty_slot']

    table = pd.read_csv(PREVIOUS/'table.csv')
    tr = table[table.role.eq('utility_train')]
    idx = tr.groupby('class_id', sort=True).head(3).index.to_numpy()
    arrays = {key: np.load(PREVIOUS/(key+'.npy'), mmap_mode='r') for key in ('response', 'region', 'a0')}
    taus = json.loads((HERE.parent/'regional_utility_pilot/upstream/scales.json').read_text(encoding='utf-8'))['tau']['layer1']
    rows = []
    for i in idx:
        for p in range(2):
            for rotation in range(4):
                r = torch.tensor(np.array(arrays['response'][i,p,rotation]), dtype=torch.float64).flatten(1)[None]
                aa = torch.tensor(np.array(arrays['a0'][i,p,rotation]), dtype=torch.float64).flatten(1)[None]
                rr = torch.tensor(np.array(arrays['region'][i,p,rotation]), dtype=torch.float64).flatten(1)[None]
                t = torch.tensor([taus[p]], dtype=torch.float64)
                direct = old.class_summary(r, rr, t)
                compressed = old.pooled_summary(r, aa, t)
                for k, slots in enumerate(old.SLOT_GROUPS):
                    if rr[0,k].sum() == 0:
                        continue
                    merged = decode_moments(moment_pool(r, aa[:, slots], t), True)[0,0]
                    error = float((merged-direct[0,k]).abs().max())
                    h = direct[0,k,:,1]
                    good = h > 1e-5
                    rows.append(dict(obs_id=int(table.loc[i,'obs_id']), polarization=('VV','VH')[p],
                                     region=('body','end','side')[k], rotation=rotation,
                                     max_summary_error=error,
                                     old_mean_first_h_ratio=float((compressed[0,k,good,1]/h[good]).median()) if good.any() else None,
                                     new_moment_h_ratio=float((merged[good,1]/h[good]).median()) if good.any() else None))
    checks['source_observations'] = len(idx)
    checks['source_summary_replay_max'] = max(r['max_summary_error'] for r in rows)
    assert checks['source_summary_replay_max'] < 2e-5
    pd.DataFrame(rows).to_csv(HERE/'source_pool_replay.csv', index=False)

    matching = module('old_match', HERE.parent/'local_evidence_transport/match_readout.py')
    metric = TypedPairMetric(7).double()
    q = torch.randn(2, 4, 7, dtype=torch.float64)
    s = torch.randn(2, 5, 7, dtype=torch.float64)
    tid = torch.tensor([1, 15])
    with torch.no_grad():
        metric.output.weight.normal_(0, .05)
    k = metric(q, s, tid)
    checks['metric_symmetric_max'] = float((k-metric(s,q,tid).transpose(-1,-2)).abs().max().detach())
    checks['metric_amplitude_sensitive'] = float((k-metric(q*2,s,tid)).abs().max().detach())
    assert checks['metric_symmetric_max'] < 1e-12 and checks['metric_amplitude_sensitive'] > .01
    qa, sa = torch.ones(2,4,dtype=q.dtype), torch.ones(2,5,dtype=q.dtype)
    transport, _, _, _ = matching.sinkhorn(k, qa, sa)
    value = (transport*k).sum()
    full_grad = torch.autograd.grad(value, metric.output.weight, retain_graph=True)[0]
    k_grad = torch.autograd.grad((transport.detach()*k).sum(), metric.output.weight)[0]
    checks['both_full_gradient_max'] = float(full_grad.abs().max())
    checks['both_transport_gradient_max'] = float((full_grad-k_grad).abs().max())
    checks['real_match_mass_max'] = float(transport.sum((-1,-2)).max().detach())
    assert checks['both_transport_gradient_max'] > 1e-8
    assert checks['real_match_mass_max'] <= 1+1e-8
    checks['new_classifier_training'] = False
    checks['current_model_integrated'] = False
    (HERE/'operator_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == '__main__':
    main()
