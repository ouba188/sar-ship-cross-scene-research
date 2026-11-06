"""Current spatial M1/M2/M3 with M2 before layer2[2]'s residual addition."""
import torch
from torch import nn

from ..bottleneck_evidence import RegionPoolWriteback
from .spatial_m12 import SpatialEvidencePool, SpatialM123ResNet50


class SpatialLayer2M123ResNet50(SpatialM123ResNet50):
    architecture = 'native160_layer2block2_preadd_spatial_m123'
    cache_start = 1

    def __init__(self, weight_path=None, classes=8):
        super().__init__(weight_path, position='pre3x3', policy='local', classes=classes)
        self.layer2_m2 = nn.ModuleList(RegionPoolWriteback(512) for _ in range(2))
        for writer in self.layer2_m2:
            writer.pool = SpatialEvidencePool(512)

    def prefix(self, images, stop=1):
        if stop != self.cache_start:
            raise ValueError('Cache before the layer2 adapter: use stop=1.')
        return super().prefix(images, stop=stop)

    def stage2_paths(self, prefix, valid, slots, energy):
        if tuple(prefix.shape[2:]) != (256, 40, 40):
            raise ValueError('Expected native160 layer1 features, not a legacy layer2 cache.')
        batch = len(prefix)
        blocks = self.backbone.model.layer2
        with torch.no_grad():
            entry = prefix.flatten(0, 1)
            for block in list(blocks.children())[:2]:
                entry = block(entry)
            raw2 = blocks[3](blocks[2](entry)).reshape(batch, 2, 512, 20, 20)

        block = blocks[2]
        value = block.relu(block.bn1(block.conv1(entry)))
        value = block.relu(block.bn2(block.conv2(value)))
        value = block.bn3(block.conv3(value)).reshape(batch, 2, 512, 20, 20)
        output, details = [], []
        for pol in range(2):
            regions, bands = self.region_integrals(slots[:, pol], energy[:, pol])
            updated, detail = self.layer2_m2[pol](value[:, pol], regions, valid[:, pol], bands)
            output.append(updated)
            details.append(detail)
        value = torch.stack(output, 1).flatten(0, 1)
        identity = entry if block.downsample is None else block.downsample(entry)
        # Frozen downstream convolutions still transmit gradients to the new M2.
        live2 = blocks[3](block.relu(value + identity)).reshape_as(raw2)
        return live2, raw2, details

    def stage3_paths(self, prefix, valid, available, field2, field3, reference,
                     rv, slot_masks, frequency, trace, aux,
                     local_slot_masks=None, local_frequency=None):
        live2, raw2, details = self.stage2_paths(prefix, valid, local_slot_masks, local_frequency)
        ref2, rr2, ref_details = self.stage2_paths(
            reference['prefix'], rv, reference['local_slot_masks'], reference['local_frequency'])
        trace['layer2_block2'] = details
        trace['reference_layer2_block2'] = ref_details

        with torch.no_grad():
            q, r = raw2.flatten(0, 1), rr2.flatten(0, 1)
            for block in list(self.backbone.model.layer3.children())[:-1]:
                q, r = block(q), block(r)
            raw3 = self.anchor_last(q).reshape(len(prefix), 2, 1024, 10, 10)
            rr3 = self.anchor_last(r).reshape(len(rv), 2, 1024, 10, 10)

        q, r = live2.flatten(0, 1), ref2.flatten(0, 1)
        for block in list(self.backbone.model.layer3.children())[:-1]:
            q, r = block(q), block(r)
        live3 = self.live_last(q, valid, available, field3, slot_masks, frequency, aux, trace)
        ref3 = self.live_last(r, rv, reference['available'], reference['field3'],
                              reference['slot_masks'], reference['frequency'], [], {})
        return live2, live3, ref3, raw3, rr3, raw2, rr2


def from_spatial(model, layer2_state=None):
    """Copy a spatial model; optionally import the retained middle-block M2 weights."""
    if type(model) is not SpatialM123ResNet50 or model.position != 'pre3x3':
        raise ValueError('Expected the pre3x3 spatial M1/M2/M3 architecture.')
    result = SpatialLayer2M123ResNet50(classes=model.internal_m1[0].classes)
    state = result.state_dict()
    state.update(model.state_dict())
    result.load_state_dict(state, strict=True)
    if layer2_state is not None:
        for pol, writer in enumerate(result.layer2_m2):
            writer.load_state_dict({name: layer2_state[f'early.2_{pol}.{name}']
                                    for name in writer.state_dict()}, strict=True)
    return result.eval()
