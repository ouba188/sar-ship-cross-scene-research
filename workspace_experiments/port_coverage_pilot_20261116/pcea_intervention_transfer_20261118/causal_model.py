"""Current unified candidate with one common, pre-readout feature split."""
from contextlib import contextmanager
import torch
from unified_model import Model as UnifiedModel, Inputs, RATES
from causal_split import PolarChannelSplit, combine


class Model(UnifiedModel):
    def __init__(self, keep_ratio=.5):
        super().__init__()
        self.disentangler = PolarChannelSplit(256, keep_ratio).to('cuda')
        self.register_buffer('source_prior', torch.ones(self.net.matcher.classes,
            device='cuda')/self.net.matcher.classes)
        self._feature_mode = 'causal'
        self._donor = None
        self._cut_data = None
        self.split_trace = None
        self.capture_split = False
        # All shallow reads and deeper residual paths start after this output.
        self.net.base.backbone.model.layer1.register_forward_hook(self._split_output)

    @contextmanager
    def feature_mode(self, mode, donor=None):
        if mode not in ('causal', 'bias', 'swap', 'raw'):
            raise ValueError(mode)
        old = self._feature_mode, self._donor
        self._feature_mode, self._donor = mode, donor
        try:
            yield
        finally:
            self._feature_mode, self._donor = old

    def _split_output(self, module, args, output):
        if self._cut_data is None:
            raise RuntimeError('Feature split must run through extract(data)')
        data = self._cut_data
        raw = output.reshape(len(data['images']), 2, 256, *output.shape[-2:])
        causal, bias = self.disentangler(raw)
        if self._feature_mode == 'raw':
            result = raw
        elif self._feature_mode == 'causal':
            result = causal
        elif self._feature_mode == 'bias':
            result = bias
        else:
            if self._donor is None:
                raise ValueError('swap requires donor indices')
            result = combine(causal, bias, self._donor, data['valid'], data['available'])
        self.split_trace = dict(mode=self._feature_mode, shape=list(raw.shape),
            retained=self.disentangler.keep)
        if self.capture_split:
            self.split_trace.update(raw=raw.detach().cpu(), causal=causal.detach().cpu(),
                bias=bias.detach().cpu(), output=result.detach().cpu(),
                mask=self.disentangler.mask().detach().cpu())
        return result.flatten(0, 1)

    def extract(self, data, descriptors):
        previous = self._cut_data
        self._cut_data = data
        try:
            return super().extract(data, descriptors)
        finally:
            self._cut_data = previous

    def reference(self, data):
        # M3 values AND PCEA source_raw must use current purified reference images.
        with self.feature_mode('causal'):
            return super().reference(data)

    def forward_variant(self, data, rows, reference, reference_rows, mode='causal', donor=None):
        with self.feature_mode(mode, donor):
            return self(data, rows, reference, reference_rows)

    def optimizer_groups(self):
        rates = dict(RATES, disentangler=1e-3)
        buckets = {key: [] for key in rates}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith('net.base.backbone.'):
                group = 'backbone'
            elif name.startswith('net.'):
                group = 'modules'
            else:
                group = dict(ccrel='reader', exchange='exchange', selector='selector',
                    disentangler='disentangler')[name.split('.')[0]]
            buckets[group].append(parameter)
        return [dict(name=k, params=v, lr=rates[k], base_lr=rates[k]) for k,v in buckets.items() if v]
