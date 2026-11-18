"""Keep fine M3 slots; learn a bounded coarse-context supplement before encoding."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


old = load('coarse_plus_previous', ROOT.parent/'pcea_class_local_20261117/model.py')
geometry = load('coarse_plus_groups', ROOT.parent/'size_adaptive_slots_20261118/run.py')
torch, nn, np = old.torch, old.nn, old.np
full = old.full
ARMS = ('continue', 'pointwise', 'coarse_plus')
START = ROOT.parent/'pcea_routed_correction_20261118/fits/routed_s17/best.pt'


def adjacency(rows):
    matrix = np.zeros((len(rows), 2, 20, 20), np.float32)
    for j, row in enumerate(rows):
        for p, pol in enumerate(('vv', 'vh')):
            if row.get(pol) is None:
                continue
            axis = row[pol]['ship_axis']
            for group in geometry.groups_for(axis['length_px'], axis['width_px'], 'adaptive'):
                matrix[j, p][np.ix_(group, group)] = 1
    return torch.as_tensor(matrix, device='cuda')


class Inputs(old.Inputs):
    def data(self, ids, split, code=0):
        result = super().data(ids, split, code)
        result['coarse_adjacency'] = adjacency([self.rows[i] for i in ids])
        return result


class Supplement(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode, self.disabled = mode, False
        self.norm1 = nn.ModuleList(nn.LayerNorm(256) for _ in range(2))
        self.norm2 = nn.ModuleList(nn.LayerNorm(512) for _ in range(2))
        self.down = nn.ModuleList(nn.Linear(768, 32, bias=False) for _ in range(2))
        self.up = nn.ModuleList(nn.Linear(32, 768, bias=False) for _ in range(2))
        for layer in self.up:
            nn.init.zeros_(layer.weight)

    def forward(self, x, mass, groups):
        if self.disabled:
            return x
        if self.mode == 'coarse_plus':
            weights = groups*mass[:, :, None, :]
            weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-20)
            context = torch.einsum('bpij,bpjd->bpid', weights, x)
        else:
            context = x
        result = []
        for p in range(2):
            h = torch.cat((self.norm1[p](context[:, p, :, :256]),
                           self.norm2[p](context[:, p, :, 256:])), -1)
            update = self.up[p](torch.nn.functional.gelu(self.down[p](h))).tanh()
            # Scale each backbone level separately without replacing its fine detail.
            pieces = []
            for start, end in ((0, 256), (256, 768)):
                original = x[:, p, :, start:end]
                scale = original.square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
                pieces.append(original+.1*scale*update[:, :, start:end])
            result.append(torch.cat(pieces, -1)*(mass[:, p, :, None] > 0))
        return torch.stack(result, 1)


class Model(old.Model):
    def __init__(self, arm, seed=17):
        assert arm in ARMS
        super().__init__('generic')
        self.load_state_dict(torch.load(START, map_location='cuda', weights_only=True), strict=True)
        self.start_checkpoint = START
        self.experiment_arm = arm
        if arm != 'continue':
            torch.manual_seed(seed+20261118)
            self.supplement = Supplement(arm).to('cuda')
        self.eval()

    def extract(self, data, descriptors):
        a, c = super().extract(data, descriptors)
        if descriptors and hasattr(self, 'supplement'):
            a['desc'] = self.supplement(a['desc'], a['mass'], data['coarse_adjacency'])
        return a, c
