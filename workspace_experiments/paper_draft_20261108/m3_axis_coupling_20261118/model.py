"""Full-model experimental entry; leaves the formal checkpoint unchanged."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('axis_previous', ROOT.parent/'m3_relation_context_20261118/model.py')
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
import axis_operator as op

torch, prepare, START = old.torch, old.prepare, old.START


class Model(old.Model):
    def __init__(self, geometry='axis', coupling='paired'):
        super().__init__('relation')
        op.install(self.net, geometry=geometry, coupling=coupling)
        self.eval()
