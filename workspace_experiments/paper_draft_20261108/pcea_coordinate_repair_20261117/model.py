"""Coordinate-consistent source normalization and query-specific regional writeback."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('joint_previous', ROOT.parent/'ccrel_pcea_joint_20261116/model.py')
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
full, torch, nn, F, np = old.full, old.torch, old.nn, old.F, old.np
ARMS = ('coordinated', 'centered', 'common')


class CCREL(old.CCREL):
    def __init__(self, reader):
        super().__init__(reader, False)
        self.raw = None

    def project_raw(self, raw):
        return torch.stack([F.linear(raw[:, p], self.value[p].weight[:, :, 0, 0])
                            for p in range(2)], 1)

    def forward(self, shallow, fine, masks, responses, valid, fine_valid):
        tokens, effective = [], []
        for p in range(2):
            x = shallow[:, p]
            value = F.layer_norm(x.permute(0, 2, 3, 1), (512,)).permute(0, 3, 1, 2)
            value, mask, response, occupancy = (t.flatten(2) for t in
                (value, masks[:, p], responses[:, p], valid[:, p]))
            uniform = mask/mask.sum(-1, keepdim=True).clamp_min(1e-8)
            classes_count = response.shape[1]
            relative = (classes_count*response-response.sum(1, keepdim=True))/(classes_count-1)
            vw = occupancy/occupancy.sum(-1, keepdim=True).clamp_min(1e-8)
            centered = relative-(relative*vw).sum(-1, keepdim=True)
            score = 2*torch.tanh(centered/((centered.square()*vw).sum(-1, keepdim=True)+.01).sqrt())
            logits = score[:, None]+mask.clamp_min(1e-12).log()[:, :, None]
            weight = logits.masked_fill(mask[:, :, None] <= 0, -1e4).softmax(-1)
            weight = weight*(mask[:, :, None] > 0)
            weight = weight/weight.sum(-1, keepdim=True).clamp_min(1e-8)
            mean = torch.einsum('bru,bdu->brd', uniform, value)
            classes = torch.einsum('brku,bdu->brkd', weight, value)
            tokens.append(torch.cat((mean[:, :, None], classes), 2))
            effective.append(mask.sum(-1).square()/mask.square().sum(-1).clamp_min(1e-8))
        self.raw = torch.stack(tokens, 1)
        self.trace = dict(fine_residual_rms=torch.zeros(shallow.shape[:2], device=shallow.device))
        return self.project_raw(self.raw), torch.stack(effective, 1)


class PCEA(old.PCEA):
    def __init__(self, arm, reader):
        super().__init__('joint', reader)
        self.mode = arm
        self.source_raw = None
        self.source_weight = None
        self.reader = None
        self.force_common = False
        self.force_zero = False

    def standardized(self, evidence):
        source = self.reader.project_raw(self.source_raw)
        weight = self.source_weight[:, None, :, None, None]
        count = weight.sum(0).clamp_min(1e-8)
        mean = (source*weight).sum(0)/count
        variance = ((source-mean).square()*weight).sum(0)/count
        moment2 = variance+mean.square()
        floor = (.1*moment2.mean(-1, keepdim=True).sqrt()).clamp_min(.01)
        scale = torch.maximum(variance.clamp_min(1e-12).sqrt(), floor)
        with torch.no_grad():
            self.mean.copy_(mean)
            self.scale.copy_(scale)
        return (evidence-mean)/scale, (source-mean)/scale

    def contents(self, t):
        u, v = (self.project[p](t[:, p]).tanh() for p in range(2))
        h, alpha, beta = self.interaction(u, v)
        return [self.output[p](h.flatten(2)).tanh() for p in range(2)], alpha, beta

    def forward(self, x, writer_masks, evidence, effective, reader_masks, available):
        if self.source_raw is None:
            self.trace = dict(active=available.bool().all(1)[:, None] &
                              (reader_masks.sum((-2, -1)) > 0).all(1) &
                              (writer_masks.sum((-2, -1)) > 0).all(1))
            return x
        t, source = self.standardized(evidence)
        local, alpha, beta = self.contents(t)
        common, _, _ = self.contents(source)
        weights = self.source_weight[..., None]
        common = [(c*weights).sum(0)/weights.sum(0).clamp_min(1e-8) for c in common]
        active = available.bool().all(1)[:, None]
        active = active & (reader_masks.sum((-2, -1)) > 0).all(1)
        active = active & (writer_masks.sum((-2, -1)) > 0).all(1)
        writer = writer_masks/writer_masks.sum(2, keepdim=True).clamp_min(1)
        updates = []
        for p in range(2):
            content = local[p]
            if self.force_common or self.mode == 'common':
                content = common[p][None].expand_as(content)
            elif self.mode == 'centered':
                content = content-common[p][None]
            if self.force_zero:
                content = torch.zeros_like(content)
            mask = writer_masks[:, p].flatten(2)
            w = mask/mask.sum(-1, keepdim=True).clamp_min(1e-8)
            region = torch.einsum('bru,bcu->brc', w, x[:, p].flatten(2))
            scale = region.square().mean(-1, keepdim=True).sqrt().detach()
            delta = .1*scale*content*active[..., None]
            updates.append(torch.einsum('brc,brhw->bchw', delta, writer[:, p]))
        delta = torch.stack(updates, 1)
        self.trace = dict(tokens=evidence.detach(), normalized=t.detach(), effective_cells=effective.detach(),
            active=active.detach(), delta=delta.detach(),
            input_rms=x.detach().square().mean((2, 3, 4)).sqrt(),
            update_rms=delta.detach().square().mean((2, 3, 4)).sqrt(),
            alpha=alpha.detach(), beta=beta.detach(),
            common_content=torch.stack(common).detach())
        return x if self.disabled else x+delta


class Model(old.Model):
    def __init__(self, seed, arm):
        assert arm in ARMS
        super().__init__(seed, 'joint')
        self.arm = arm
        torch.manual_seed(seed+20261116)
        self.exchange = PCEA(arm, self.net.guide.value).to('cuda')
        self.ccrel = CCREL(self.net.guide.value).to('cuda')
        # Frozen source observations, before the trainable linear readout.
        self.register_buffer('source_raw', torch.zeros(719, 2, 3, 5, 512, device='cuda'))
        self.register_buffer('source_weight', torch.zeros(719, 3, device='cuda'))
        self.register_buffer('source_ready', torch.tensor(False, device='cuda'))

    def _exchange(self, module, args):
        if bool(self.source_ready):
            self.exchange.source_raw = self.source_raw
            self.exchange.source_weight = self.source_weight
            # Do not register a second copy of the same reader module.
            object.__setattr__(self.exchange, 'reader', self.ccrel)
        return super()._exchange(module, args)


@torch.no_grad()
def calibrate(net, inputs, seed):
    path = ROOT/f'source_raw_s{seed}.pt'
    if path.exists():
        stats = torch.load(path, map_location='cuda', weights_only=True)
    else:
        net.eval()
        ids = inputs.groups['small']
        assert len(ids) == 719
        reference = net.reference(inputs.data(inputs.groups['references'], 'small'))
        raw, weight = [], []
        for first in range(0, len(ids), 12):
            part = ids[first:first+12]
            net(inputs.data(part, 'small'), [inputs.rows[i] for i in part], reference, inputs.reference_rows)
            raw.append(net.ccrel.raw.detach().clone())
            weight.append(torch.as_tensor(inputs.weights[part], device='cuda')[:, None]*net.exchange.trace['active'])
        stats = dict(raw=torch.cat(raw), weight=torch.cat(weight), ids=torch.as_tensor(ids, device='cuda'))
        torch.save({k: v.cpu() for k, v in stats.items()}, path)
    assert np.array_equal(stats['ids'].cpu().numpy(), inputs.groups['small'])
    assert bool((stats['weight'].sum(0) > 0).all())
    net.source_raw.copy_(stats['raw'])
    net.source_weight.copy_(stats['weight'])
    net.source_ready.fill_(True)
    return stats
