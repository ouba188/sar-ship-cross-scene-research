"""One M2 inside the last layer2 bottleneck; preserve frozen M3 anchors."""
import torch
from torch import nn

from .bottleneck_evidence import RegionPoolWriteback
from .shallow_writeback import ShallowM2


class Layer2BlockM2(ShallowM2):
    def __init__(self, state, position='pre3x3'):
        super().__init__(state, ())
        if position not in ('none', 'stage', 'pre3x3', 'post3x3'):
            raise ValueError(position)
        self.position = position
        channels = 512 if position == 'stage' else 128
        self.stages = () if position == 'none' else (2,)
        self.early = nn.ModuleDict()
        for p in range(2) if self.stages else ():
            torch.manual_seed(923+p)
            self.early[f'2_{p}'] = RegionPoolWriteback(channels)
        self.eval()

    def feedback(self, value, data, trace):
        b = len(data['raw2'])
        shape = value.shape
        value = value.reshape(b, 2, *shape[1:])
        value = self.early_feedback(value, data, 2, trace)
        return value.flatten(0, 1)

    def stage2(self, data, trace):
        if self.position == 'none':
            return data['raw2']
        if self.position == 'stage':
            return self.early_feedback(data['raw2'], data, 2, trace)
        if 'prefix2' in data:
            prefix = data['prefix2'].flatten(0, 1)
        else:
            # This prefix precedes the only new layer2 adapter and is frozen.
            with torch.no_grad():
                prefix = data['stage1'].flatten(0, 1)
                for block in list(self.base.backbone.model.layer2.children())[:-1]:
                    prefix = block(prefix)
        block = self.base.backbone.model.layer2[-1]
        identity = prefix
        value = block.relu(block.bn1(block.conv1(prefix)))
        if self.position == 'pre3x3':
            value = self.feedback(value, data, trace)
        value = block.relu(block.bn2(block.conv2(value)))
        if self.position == 'post3x3':
            value = self.feedback(value, data, trace)
        value = block.bn3(block.conv3(value))
        if block.downsample is not None:
            identity = block.downsample(identity)
        value = block.relu(value+identity)
        return value.reshape(len(data['raw2']), 2, *value.shape[1:])

    def encode(self, data, trace, aux, propagate=True):
        x2 = self.stage2(data, trace)
        x = (x2 if propagate else data['raw2']).flatten(0, 1)
        for block in list(self.base.backbone.model.layer3.children())[:-1]:
            x = block(x)
        x3 = self.base.live_last(x, data['valid'], data['available'], data['field3'],
                                data['slot_masks'], data['frequency'], aux, trace)
        return x2, x3
