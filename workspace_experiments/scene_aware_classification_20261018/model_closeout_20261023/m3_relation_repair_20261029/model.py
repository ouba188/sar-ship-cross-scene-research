"""Bounded diagonal metric candidate; old M3 remains the matched control."""
import torch
from torch import nn
from torch.nn import functional as F

from sar_m123.selective_matching import SelectiveReferenceMatch
from sar_m123.current.spatial_m12 import SpatialM12ResNet50
from sar_m123.current.model import load_model


class MetricReferenceMatch(SelectiveReferenceMatch):
    def __init__(self):
        super().__init__()
        self.metric_deep = nn.Parameter(torch.zeros(1024))
        self.metric_local = nn.Parameter(torch.zeros(512))

    def descriptors(self, *args):
        streams, mass = super().descriptors(*args)
        # The frozen image descriptors remain anchors. Only their shared
        # query/reference channel metric is learned, outside the no_grad readout.
        scales = [self.metric_deep.tanh().exp()] + [self.metric_local.tanh().exp()]*4
        return [F.normalize(x*s, dim=-1) for x,s in zip(streams,scales)], mass


def add_metric(net):
    device = next(net.parameters()).device
    for block in net.m3:
        for pol, old in enumerate(block):
            new = MetricReferenceMatch().to(device)
            missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
            assert set(missing) == {'metric_deep', 'metric_local'} and not unexpected
            block[pol] = new
    return net


def fresh(path, arm):
    net = load_model(path, 'cuda')
    return add_metric(net) if arm == 'metric' else net


def save(net, path, arm):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(arm=arm, state={k:v.detach().cpu() for k,v in net.state_dict().items()}),path)


def load(path, device='cuda'):
    blob = torch.load(path, map_location='cpu', weights_only=True)
    net = SpatialM12ResNet50(None, 'pre3x3', 'local', 8)
    if blob['arm'] == 'metric':
        net = add_metric(net)
    net.load_state_dict(blob['state'], strict=True)
    return net.to(device).eval().requires_grad_(False)


def configure(net, arm):
    net.eval().requires_grad_(False)
    compare, value = [], []
    for name, parameter in net.named_parameters():
        if not name.startswith('m3.') or '.reader' in name:
            continue
        parameter.requires_grad_(True)
        target = compare if any(s in name for s in ('selector.', 'branch_bias', 'metric_')) else value
        target.append(parameter)
    return [dict(params=compare,lr=.001 if arm=='control' else .01),
            dict(params=value,lr=.001)]
