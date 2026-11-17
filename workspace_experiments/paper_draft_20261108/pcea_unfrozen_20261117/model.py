"""Matched late-backbone adaptation, preserving the frozen source-reader prefix."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    'unfrozen_local_previous', ROOT.parent/'pcea_local_cross_20261117/model.py')
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
torch, nn, F, np, full = old.torch, old.nn, old.F, old.np, old.full
ARMS = ('full', 'cross_mean', 'cross_local')


class Model(old.Model):
    def __init__(self, seed, arm):
        assert arm in ARMS
        super().__init__(seed, 'cross_mean' if arm == 'full' else arm)
        self.arm = arm
        self.requires_grad_(False)
        bb = self.net.base.backbone.model
        bb.layer3.requires_grad_(True)
        bb.layer4.requires_grad_(True)
        self.net.guide.head.requires_grad_(True)
        if arm != 'full':
            self.ccrel.requires_grad_(True)
            self.exchange.requires_grad_(True)
        else:
            self.exchange.disabled = True

    def _exchange(self, module, args):
        if self.arm == 'full':
            return args
        return super()._exchange(module, args)


calibrate = old.calibrate
