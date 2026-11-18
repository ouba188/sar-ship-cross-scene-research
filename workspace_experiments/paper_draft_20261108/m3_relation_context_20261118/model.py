"""Current chip-to-class model with relation-weighted M3 competition context."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


old = load('relation_context_previous', ROOT.parent/'pcea_class_local_20261117/model.py')
op = load('relation_context_operator', ROOT/'relation_operator.py')
torch, np, full, Inputs = old.torch, old.np, old.full, old.Inputs
START = ROOT.parent/'pcea_routed_correction_20261118/fits/routed_s17/best.pt'
prepare = old.old.joint.full.cost.prepare


class Model(old.Model):
    def __init__(self, mode='relation'):
        super().__init__('generic')
        self.load_state_dict(torch.load(START, map_location='cuda', weights_only=True), strict=True)
        self.start_checkpoint = START
        self.context_mode = mode
        op.install(self.net, mode=mode)
        self.eval()
