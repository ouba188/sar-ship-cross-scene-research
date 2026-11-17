"""Trainable Full/PCEA from the selected trained Full, with an optional anchor."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('startpoint_joint_model',
    ROOT.parent / 'pcea_joint_full_20261117/model.py')
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
torch, np, full = old.torch, old.np, old.full


class Model(old.Model):
    def __init__(self, seed, arm):
        assert arm in ('full', 'cross_mean')
        super().__init__(seed, arm)
        self.start_checkpoint = full.ROOT / f'selected_s{seed}/best.pt'
        state = torch.load(self.start_checkpoint, map_location='cpu', weights_only=True)
        result = self.load_state_dict(state, strict=False)
        assert not result.unexpected_keys, result
        assert all(k.startswith(('ccrel.', 'exchange.')) for k in result.missing_keys), result
        if arm != 'full':
            # The new reader starts in the trained Full's current feature coordinates.
            self.ccrel.value.load_state_dict(self.net.guide.value.state_dict(), strict=True)
        self.eval()


class StartingPoint:
    def __init__(self, net):
        self.params = [(n, p) for n, p in net.named_parameters()
                       if n.startswith('net.') and p.requires_grad]
        self.origin = [p.detach().clone() for _, p in self.params]

    def penalty(self):
        return .5 * sum((p - q).square().sum()
                        for (_, p), q in zip(self.params, self.origin))

    @torch.no_grad()
    def drift(self):
        out = {}
        for (name, p), q in zip(self.params, self.origin):
            group = 'backbone' if name.startswith('net.base.backbone.') else 'old_modules'
            v = out.setdefault(group, dict(squared_distance=0., origin_squared_norm=0.))
            v['squared_distance'] += float((p-q).square().sum())
            v['origin_squared_norm'] += float(q.square().sum())
        for v in out.values():
            v['l2_distance'] = v['squared_distance'] ** .5
            v['relative_l2_distance'] = (v['squared_distance'] /
                max(v['origin_squared_norm'], 1e-12)) ** .5
        return out

