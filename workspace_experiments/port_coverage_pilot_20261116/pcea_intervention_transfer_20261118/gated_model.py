"""GDOS-inspired gates at the common image-feature cut of the unified model."""
from contextlib import contextmanager
import torch
from unified_model import Model as UnifiedModel, Inputs, RATES
from gated_split import PolarGatedSplit


class Model(UnifiedModel):
    def __init__(self, initial_gate=.95):
        super().__init__()
        self.disentangler = PolarGatedSplit(initial_gate=initial_gate).to('cuda')
        self._feature_mode, self._donor, self._cut_data = 'content', None, None
        self.capture_split, self.split_trace = False, None
        self.net.base.backbone.model.layer1.register_forward_hook(self._split_output)

    @contextmanager
    def feature_mode(self, mode, donor=None):
        if mode not in ('content', 'swap', 'remainder', 'raw'):
            raise ValueError(mode)
        old = self._feature_mode, self._donor
        self._feature_mode, self._donor = mode, donor
        try:
            yield
        finally:
            self._feature_mode, self._donor = old

    def _split_output(self, module, args, output):
        data = self._cut_data
        if data is None:
            raise RuntimeError('Gate must run through extract(data)')
        raw = output.reshape(len(data['images']), 2, 256, *output.shape[-2:])
        content, remainder, gate = self.disentangler(raw, data['valid'], data['available'])
        if self._feature_mode == 'raw':
            result = raw
        elif self._feature_mode == 'content':
            result = content
        elif self._feature_mode == 'remainder':
            result = remainder
        else:
            if self._donor is None:
                raise ValueError('swap requires donor indices')
            result = self.disentangler.recombine(content, remainder, self._donor,
                                                data['valid'], data['available'])
        if self.capture_split:
            self.split_trace = {k: v.detach().cpu() for k, v in dict(raw=raw, content=content,
                                remainder=remainder, gate=gate, output=result).items()}
        return result.flatten(0, 1)

    def extract(self, data, descriptors):
        previous = self._cut_data
        self._cut_data = data
        try:
            return super().extract(data, descriptors)
        finally:
            self._cut_data = previous

    def reference(self, data):
        # Both M3 descriptors and PCEA source_raw refresh from the same gated path.
        with self.feature_mode('content'):
            return super().reference(data)

    def forward_variant(self, data, rows, reference, reference_rows, mode='content', donor=None):
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
