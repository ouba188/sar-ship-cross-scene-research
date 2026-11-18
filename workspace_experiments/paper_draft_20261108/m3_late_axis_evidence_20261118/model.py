"""Compare reference-local axis hypotheses before aggregating class evidence."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    'late_axis_content_base', ROOT.parent/'m3_content_scale_20261118/model.py')
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
torch, prepare, Inputs = base.torch, base.prepare, base.Inputs


def compare_frames(query, reference, plans, orientation, metric):
    """Return per-reference expected distance; do not merge values before comparison."""
    mass = plans.sum(-1)
    values = torch.einsum('brhij,rjd->brhid', plans, reference)
    values = values/mass[..., None].clamp_min(1e-20)
    weight = orientation[..., None]*mass
    weight = weight/weight.sum(2, keepdim=True).clamp_min(1e-20)
    distance = ((query[:, None, None]-values).square()*metric[:, :, None, None]).sum(-1)
    expected = (weight*distance).sum(2)
    mean = (weight[..., None]*values).sum(2)
    mean_distance = ((query[:, None]-mean).square()*metric[:, :, None]).sum(-1)
    mean_distance = mean_distance*(mass.sum(2) > 0)
    return expected, mean_distance, dict(hypothesis_weight=weight,
        hypothesis_distance=distance, axis_variance=expected-mean_distance)


class LateAxisMatch(base.MixedMatch):
    def __init__(self, classes, comparison='late', weighting='content', geometry='axis'):
        super().__init__(classes, geometry)
        if comparison not in ('prototype', 'pair_mean', 'late'):
            raise ValueError(comparison)
        if weighting not in ('content', 'equal'):
            raise ValueError(weighting)
        self.comparison, self.weighting = comparison, weighting

    def orientation_weights(self, scores):
        if self.weighting == 'equal':
            return torch.ones_like(scores)/scores.shape[-1]
        return super().orientation_weights(scores)

    def comparison_distance(self, query, reference, plans, orientation, class_beta,
                            metric, ref_labels, aligned, prototype):
        if self.comparison == 'prototype':
            return super().comparison_distance(query, reference, plans, orientation,
                class_beta, metric, ref_labels, aligned, prototype)
        if self.coupling != 'paired':
            raise ValueError('Late comparison requires full paired transport.')
        expected, early, trace = compare_frames(query, reference, plans, orientation,
                                                metric[:, ref_labels])
        # Hypothesis labels belong to each reference, not to the whole class bank.
        pair_distance = expected if self.comparison == 'late' else early
        distance = torch.einsum('brci,bri->bci', class_beta, pair_distance)
        trace['class_axis_variance'] = torch.einsum('brci,bri->bci', class_beta, expected-early)
        return distance, trace


class Model(base.Model):
    def __init__(self, selection_mode='auto', comparison='late', weighting='content'):
        super().__init__(selection_mode)
        previous = self.net.matcher
        parameter = next(previous.parameters())
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            current = LateAxisMatch(previous.classes, comparison, weighting)
        current.to(device=parameter.device, dtype=parameter.dtype)
        current.load_state_dict(previous.state_dict(), strict=True)
        flags = dict(previous.named_parameters())
        for name, value in current.named_parameters():
            value.requires_grad_(flags[name].requires_grad)
        current.train(previous.training)
        self.net.matcher = current
