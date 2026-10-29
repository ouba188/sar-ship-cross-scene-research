"""CPU-only independent M3 probes; no datasets, checkpoints, or optimizer runs."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import csv
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'sar_m123_standalone'))
import torch
from torch import nn
from torch.nn import functional as F
from sar_m123.selective_matching import SelectiveReferenceMatch
from sar_m123.m3 import sinkhorn
from sar_m123.current.model import training_policy
from sar_m123.current.training import fitting_loss

torch.set_num_threads(2)
torch.manual_seed(1029)
DTYPE = torch.float64


def relation_loss(z, valid, labels, ref_labels):
    scores = torch.zeros(len(labels), 8, dtype=z.dtype)
    _, terms = fitting_loss(scores, dict(aux_logits=[scores], relations=[(z, valid)]),
                            labels, torch.ones(len(labels), dtype=z.dtype), ref_labels)
    return terms['relation']


def gradient_norms(loss, module):
    names, params = zip(*[(n, p) for n, p in module.named_parameters() if p.requires_grad])
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return {n: None if g is None else float(g.norm()) for n, g in zip(names, grads)}


def inputs(n):
    masks = torch.rand(n, 20, 4, 4, dtype=DTYPE)
    return dict(features=torch.randn(n, 8, 4, 4, dtype=DTYPE),
                anchor=torch.rand(n, 1024, 4, 4, dtype=DTYPE),
                local=torch.rand(n, 512, 4, 4, dtype=DTYPE),
                slots=masks, local_slots=masks.clone(), valid=torch.ones(n, 1, 4, 4, dtype=DTYPE),
                energy=masks[:, :, None]*torch.rand(n, 20, 3, 4, 4, dtype=DTYPE),
                local_energy=masks[:, :, None]*torch.rand(n, 20, 3, 4, 4, dtype=DTYPE))


def invoke(m, q, r, legal):
    return m(q['features'], q['anchor'], q['local'], q['slots'], q['local_slots'],
             q['valid'], q['energy'], q['local_energy'], r, legal, return_details=True)


def module_probe():
    m = SelectiveReferenceMatch(channels=8).double()
    q, r = inputs(2), inputs(4)
    q['anchor'].requires_grad_()
    r['features'].requires_grad_()
    legal = torch.ones(2, 4, dtype=torch.bool)
    labels, ref_labels = torch.tensor([0, 1]), torch.tensor([0, 0, 1, 1])
    head = torch.randn(8, 2, dtype=DTYPE)
    out = {}
    for state in ('initial', 'opened'):
        if state == 'opened':
            # A finite nonzero state checks accessibility, without training any model.
            with torch.no_grad():
                m.output.weight.normal_(std=.02)
                m.selector[-1].weight.normal_(std=.02)
        y, d = invoke(m, q, r, legal)
        ce = F.cross_entropy(y.mean((-2, -1)) @ head, labels)
        rel = relation_loss(d['reference_logits'], d['reference_valid'], labels, ref_labels)
        out[state] = dict(ce=gradient_norms(ce, m), relation=gradient_norms(rel, m),
                          loss=float(rel), update_norm=float(d['update'].norm()))
        anchor_grad, ref_grad = torch.autograd.grad(ce, (q['anchor'], r['features']),
                                                   allow_unused=True, retain_graph=True)
        out[state]['anchor_ce_grad'] = None if anchor_grad is None else float(anchor_grad.norm())
        out[state]['reference_value_ce_grad'] = None if ref_grad is None else float(ref_grad.norm())
        params = (m.branch_bias, m.selector[-1].bias)
        gb, gs = torch.autograd.grad(rel, params, retain_graph=True)
        out[state]['bias_gradient_max_difference'] = float((gb-gs).abs().max())
    assert out['initial']['ce']['output.weight'] > 0
    assert out['initial']['ce']['branch_bias'] == 0
    assert out['initial']['relation']['branch_bias'] > 0
    assert out['initial']['relation']['selector.2.weight'] > 0
    assert out['initial']['relation']['selector.0.weight'] == 0
    assert out['opened']['ce']['selector.0.weight'] > 0
    assert out['opened']['relation']['selector.0.weight'] > 0
    assert out['opened']['relation']['project.weight'] is None
    assert out['opened']['anchor_ce_grad'] is None
    assert out['opened']['reference_value_ce_grad'] > 0
    assert out['opened']['bias_gradient_max_difference'] < 1e-12

    y, d = invoke(m, q, r, legal)
    permutation = torch.tensor([2, 0, 3, 1])
    rp = {k: v[permutation] for k, v in r.items()}
    yp, dp = invoke(m, q, rp, legal[:, permutation])
    out['reference_permutation_output_error'] = float((y-yp).abs().max())
    out['reference_permutation_logits_error'] = float((d['reference_logits'][:, permutation]-dp['reference_logits']).abs().max())
    assert out['reference_permutation_output_error'] < 1e-12
    legal[0] = False
    y, d = invoke(m, q, r, legal)
    rel = relation_loss(d['reference_logits'], d['reference_valid'], labels, ref_labels)
    out['all_invalid_query_update_norm'] = float(d['update'][0].norm())
    out['partial_missing_relation_loss'] = float(rel)
    assert torch.isfinite(rel) and out['all_invalid_query_update_norm'] == 0
    holder = nn.Module()
    holder.m3 = nn.ModuleList([nn.ModuleList([m])])
    holder.backbone, holder.anchor_last = nn.Linear(1, 1), nn.Linear(1, 1)
    training_policy(holder, 'fit')
    out['policy_fit'] = {n: p.requires_grad for n, p in m.named_parameters()}
    training_policy(holder, 'adapt')
    out['policy_adapt_any_m3_trainable'] = any(p.requires_grad for p in m.parameters())
    assert out['policy_fit']['selector.0.weight'] and out['policy_fit']['branch_bias']
    assert not out['policy_fit']['reader2.gain']
    assert not out['policy_adapt_any_m3_trainable']
    return out


def sinkhorn_probe():
    sim = torch.tensor([[[[.2, .7, .1], [.8, .1, .4]]]], dtype=DTYPE, requires_grad=True)
    qa = torch.tensor([[[.4, .6]]], dtype=DTYPE)
    ra = torch.tensor([[[.2, .3, .5]]], dtype=DTYPE)
    probe = torch.tensor([[[[1., -2., .5], [-.7, 1.1, .3]]]], dtype=DTYPE)
    value = (sinkhorn(sim, qa, ra)*probe).sum()
    g, = torch.autograd.grad(value, sim)
    numeric = torch.zeros_like(sim)
    h = 1e-5
    for index in range(sim.numel()):
        plus, minus = sim.detach().clone(), sim.detach().clone()
        plus.flatten()[index] += h
        minus.flatten()[index] -= h
        numeric.flatten()[index] = ((sinkhorn(plus, qa, ra)-sinkhorn(minus, qa, ra))*probe).sum()/(2*h)
    error = float((numeric-g).abs().max())
    assert error < 1e-7 and g.norm() > 0
    one = torch.ones(1, 1, 1, dtype=DTYPE)
    masses = {str(s): float(sinkhorn(torch.full((1, 1, 1, 1), s, dtype=DTYPE), one, one))
              for s in (-.8, 0., .8)}
    return dict(gradient_norm=float(g.norm()), finite_difference_max_error=error,
                one_slot_real_transport_mass=masses)


def same_class_probe():
    z = torch.tensor([[3., 0., 1., 1.]], dtype=DTYPE, requires_grad=True)
    valid = torch.ones_like(z, dtype=torch.bool)
    labels, refs = torch.tensor([0]), torch.tensor([0, 0, 1, 1])
    uniform = relation_loss(z, valid, labels, refs)
    positive_mass = z.logsumexp(1)-z[:, :2].logsumexp(1)
    gu, = torch.autograd.grad(uniform, z, retain_graph=True)
    gm, = torch.autograd.grad(positive_mass.sum(), z)
    assert gu[0, 0] > 0 and gm[0, 0] < 0
    # The decomposition isolates the extra uniform-within-positive penalty.
    within = z[:, :2].softmax(1)
    kl = (torch.full_like(within, .5)*(math.log(.5)-within.log())).sum()
    assert abs(float(uniform-positive_mass.sum()-math.log(2)-kl)) < 1e-12
    one_positive = valid.clone()
    one_positive[:, 1] = False
    one_loss = relation_loss(z, one_positive, labels, refs)
    masked = z.masked_fill(~one_positive, -1e4)
    assert abs(float(one_loss-(masked.logsumexp(1)-z[:, 0]).sum())) < 1e-12
    absent = relation_loss(z, valid, torch.tensor([7]), refs)
    gabs, = torch.autograd.grad(absent, z)
    assert float(absent) == 0 and float(gabs.norm()) == 0
    return dict(probability=z.softmax(1).tolist(), uniform_loss=float(uniform),
                positive_mass_loss=float(positive_mass.sum()), uniform_gradient=gu.tolist(),
                positive_mass_gradient=gm.tolist(), within_positive_KL=float(kl),
                no_positive_loss=float(absent), no_positive_gradient_norm=float(gabs.norm()))


def aggregation_probe():
    # Three actual regions, with cosine-realizable scores constant within each region.
    # Each region has one active slot here; Sinkhorn is still the production function.
    sim = torch.tensor([[1., .1], [.4, .6], [.4, .6]], dtype=DTYPE)
    qa = torch.ones(3, 1, 1, dtype=DTYPE)
    t = sinkhorn(sim[:, :, None, None], qa, torch.ones(1, 2, 1, dtype=DTYPE))
    row_mass = t.sum(-1)
    score = (t*sim[:, :, None, None]).sum(-1)/row_mass
    weights = (score/.1).softmax(1).squeeze(-1)
    relation_prob = (score.squeeze(-1).mean(0)/.1).softmax(0)
    average_attention = weights.mean(0)
    effective_value_coeff = (weights*t[..., 0, 0]).mean(0)
    assert relation_prob[0] > relation_prob[1]
    assert average_attention[0] < average_attention[1]
    return dict(region_scores=sim.tolist(), relation_probability=relation_prob.tolist(),
                per_region_attention=weights.tolist(), mean_region_attention=average_attention.tolist(),
                mean_effective_value_coefficient=effective_value_coeff.tolist(),
                note='Different readouts, not a prediction of the real classifier or prevalence.')


def evidence_span_probe():
    q = torch.rand(2, 3, 4, 1, dtype=DTYPE)
    delta = torch.randn(2, 3, 4, 5, dtype=DTYPE)
    rows = []
    for scale in (0., .001, .1):
        evidence = q+scale*delta
        bias = torch.tensor([.5, .25, 1/12, 1/12, 1/12], dtype=DTYPE).log().requires_grad_()
        weights = bias.softmax(-1)
        sim = (weights*evidence).sum(-1)
        g, = torch.autograd.grad(sim.sum(), bias)
        rows.append(dict(branch_difference_scale=scale, gradient_norm=float(g.norm())))
    assert rows[0]['gradient_norm'] < 1e-12
    assert abs(rows[2]['gradient_norm']/rows[1]['gradient_norm']-100) < 1e-8
    return rows


def historical_evidence():
    old = EXPERIMENTS/'selective_m123_20261027'
    refit = EXPERIMENTS/'spatial_m12_refit_20261029'
    curves = {}
    for path in sorted((refit/'training').glob('*/training.json')):
        d = json.loads(path.read_text(encoding='utf-8'))
        curves[path.parent.name] = dict(ce_first=d['curve'][0]['ce'], ce_last=d['curve'][-1]['ce'],
            relation_first=d['curve'][0]['relation'], relation_last=d['curve'][-1]['relation'],
            relation_min=min(r['relation'] for r in d['curve']), relation_max=max(r['relation'] for r in d['curve']))
    checks = json.loads((refit/'checks.json').read_text(encoding='utf-8'))
    gradients = {arm: {n: g for n, g in row['gradients'].items() if n.startswith('m3.')}
                 for arm, row in checks['arms'].items()}
    selected_layout = json.loads((old/'selection.json').read_text(encoding='utf-8'))['layout']
    selected = json.loads((old/'training'/f'{selected_layout}_17'/'gradient_norms.json').read_text(encoding='utf-8'))
    rows = list(csv.DictReader((old/'analysis/m3_same_other_weights_17.csv').open(encoding='utf-8')))
    groups = {}
    for row in rows:
        if row['scope'] == 'query' and row['reference_class_group'] == 'same':
            key = row['polarization']+'_'+row['region']
            groups.setdefault(key, []).append(float(row['class_weight_mass_mean']))
    summaries = {}
    for key, values in groups.items():
        live = [v for v in values if math.isfinite(v)]
        summaries[key] = dict(n_total=len(values), n_valid=len(live), n_missing=len(values)-len(live),
                              mean=sum(live)/len(live), min=min(live), max=max(live))
    return dict(current_refit_curves=curves, current_refit_initial_gradients=gradients,
                old_selected17_gradients={n: v for n, v in selected.items() if n.startswith('m3.')},
                old_selected17_same_class_attention=summaries,
                caveat='Historical diagnostic subset, repeated regions/polarizations are not independent ships.')


if __name__ == '__main__':
    result = dict(device='cpu', dtype=str(DTYPE), seed=1029, training_run=False, torch_version=torch.__version__)
    for name, fn in [('module', module_probe), ('sinkhorn', sinkhorn_probe),
                     ('same_class', same_class_probe), ('aggregation', aggregation_probe),
                     ('evidence_span', evidence_span_probe), ('historical', historical_evidence)]:
        result[name] = fn()
        print(name+' PASS', flush=True)
    (HERE/'cpu_results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print('All CPU probes passed; cpu_results.json written.', flush=True)
