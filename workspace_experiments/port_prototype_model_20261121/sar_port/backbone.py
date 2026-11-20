"""Native160 dual-polarization R50, with the current spatial M1/M2 readout.

No checkpoint or pretrained weights are loaded implicitly. The optional
pretrained_state is an ImageNet-compatible torchvision ResNet50 state dict
or its explicit local path. Full model checkpoints belong to the caller.
"""
from pathlib import Path

import numpy as np  # Import before torch on the supported Windows environment.
import torch
from torch import nn
from torchvision.models import resnet50

from .region_ops import (RelationFeedback, SpatialEvidencePool,
                         RegionPoolWriteback, RegionalGuidance,
                         grid_occupancy, regional_mean, region_integrals)


class ShipBackbone(nn.Module):
    """Split the real classifier into early, pre-M3, and post-M3 execution.

    BatchNorm running statistics stay fixed in both train and eval mode, as
    in the previous full-backbone protocol. BN affine parameters and all
    selected convolution/module parameters remain differentiable. This also
    keeps separately encoded source references independent of batch order.
    """

    def __init__(self, classes=8, pretrained_state=None, *, use_m1=True,
                 use_m2=True, use_guidance=True, m1_strength=.3, m2_strength=1.0):
        super().__init__()
        if classes < 2:
            raise ValueError('Class-relative M1 requires at least two classes.')
        self.classes = classes
        self.use_m1, self.use_m2, self.use_guidance = use_m1, use_m2, use_guidance
        self.resnet = resnet50(weights=None)
        # The ImageNet fc is not part of the SAR classifier.
        self.resnet.fc = nn.Identity()
        if pretrained_state is not None:
            self.load_pretrained(pretrained_state)
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])
        self.m1 = nn.ModuleList(RelationFeedback(128, 'class', m1_strength, classes) for _ in range(2))
        self.m2 = nn.ModuleList(SpatialEvidencePool(512) for _ in range(2))
        self.internal_m2 = nn.ModuleList(RegionPoolWriteback(256, strength=m2_strength) for _ in range(2))
        self.deep_pool = nn.ModuleList(SpatialEvidencePool(1024) for _ in range(2))
        self.deep_output = nn.ModuleList(nn.Linear(1024, 512, bias=False) for _ in range(2))
        for layer in self.deep_output:
            nn.init.zeros_(layer.weight)
        self.guide = RegionalGuidance('class', classes)
        self.train(True)

    def load_pretrained(self, state_or_path):
        """Load an explicit torchvision R50 state, excluding only ImageNet fc.

        Every remaining key must match strictly. Full SAR checkpoints must
        instead be loaded through the outer model's full state_dict API.
        The experiment protocol owns the source/target provenance of this
        explicitly supplied pretraining file.
        """
        if isinstance(state_or_path, (str, Path)):
            state_or_path = torch.load(state_or_path, map_location='cpu', weights_only=True)
        state = dict(state_or_path)
        state.pop('fc.weight', None)
        state.pop('fc.bias', None)
        self.resnet.load_state_dict(state, strict=True)

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    @staticmethod
    def _present(value, available):
        return value * available.to(value)[(...,) + (None,) * (value.ndim - 2)]

    def early(self, images, valid, available, local_slots, feature_transform=None):
        """Return native layer1/layer2 features and M1/M2 source evidence.

        Input images are 0..1 display pixels, with shape [B,2,1,160,160].
        Missing polarization is explicitly zero in every returned feature.
        The returned occupancy is useful for CCREL, but is not an image label.
        """
        if images.shape[1:] != (2, 1, 160, 160):
            raise ValueError('Expected native160 images [B,2,1,160,160].')
        n = len(images)
        av = available.to(images)
        valid = valid.to(images) * av[..., None, None, None]
        images = images * av[..., None, None, None]
        x = images.flatten(0, 1).expand(-1, 3, -1, -1)
        x = (x - self.mean) / self.std
        bb = self.resnet
        z = bb.relu(bb.bn1(bb.conv1(x)))
        first = bb.layer1(bb.maxpool(z)).reshape(n, 2, 256, 40, 40)
        first = self._present(first, av)
        valid40 = grid_occupancy(valid.flatten(0, 1), (40, 40)).reshape(n, 2, 1, 40, 40)
        if feature_transform is not None:
            first = self._present(feature_transform(first, valid40, av), av)
        z = first.flatten(0, 1)
        for layer in list(bb.layer2.children())[:3]:
            z = layer(z)
        identity, block = z, bb.layer2[3]
        x = block.relu(block.bn1(block.conv1(z)))
        x = block.relu(block.bn2(block.conv2(x))).reshape(n, 2, 128, 20, 20)
        values, responses, auxiliary = [], [], []
        occupancy = grid_occupancy(valid.flatten(0, 1), (20, 20)).reshape(n, 2, 1, 20, 20)
        for p in range(2):
            if self.use_m1:
                result, trace = self.m1[p](x[:, p], occupancy[:, p], details=True)
                values.append(result[0])
                auxiliary.append(result[1])
                responses.append(trace['response'])
            else:
                # Disabling M1 removes feedback, not the independent class-map
                # readout needed by the prior student. Its auxiliary CE still
                # trains the reader; no ground-truth label enters this path.
                local = x[:, p].masked_fill(occupancy[:, p] == 0, 0)
                response = self.m1[p].reader(local)
                mass = occupancy[:, p]
                auxiliary.append((response*mass).sum((-2, -1))/mass.sum((-2, -1)).clamp_min(1e-8))
                responses.append(response)
                values.append(x[:, p])
        z = block.relu(block.bn3(block.conv3(torch.stack(values, 1).flatten(0, 1))) + identity)
        second = self._present(z.reshape(n, 2, 512, 20, 20), av)
        low, masks = [], []
        for p in range(2):
            region = region_integrals(local_slots[:, p])
            low.append(self.m2[p](second[:, p], region, valid[:, p])['pooled']
                       if self.use_m2 else second.new_zeros(n, 3, 512))
            masks.append(torch.minimum(region, occupancy[:, p]))
        return dict(f1=first, f2=second,
                    responses=self._present(torch.stack(responses, 1), av),
                    auxiliary=self._present(torch.stack(auxiliary, 1), av),
                    low=self._present(torch.stack(low, 1), av),
                    masks=self._present(torch.stack(masks, 1), av),
                    occupancy=occupancy, valid20=occupancy, valid40=valid40, available=av)

    def middle(self, early, data, exchange=None):
        """Run layer3; exchange(x256, early, data) follows internal M2.

        Return the 1024-channel pre-M3 layer3 output. The caller adds the
        reference-matching residual here, before calling finish().
        """
        av = data['available']
        valid = data['valid'] * av[..., None, None, None]
        n, bb = len(av), self.resnet
        z = early['f2'].flatten(0, 1)
        for layer in list(bb.layer3.children())[:5]:
            z = layer(z)
        identity, block = z, bb.layer3[5]
        x = block.relu(block.bn1(block.conv1(z))).reshape(n, 2, 256, 10, 10)
        values = []
        for p in range(2):
            regions = region_integrals(data['slots'][:, p])
            values.append(self.internal_m2[p](x[:, p], regions, valid[:, p])[0]
                          if self.use_m2 else x[:, p])
        x = self._present(torch.stack(values, 1), av)
        if exchange is not None:
            x = self._present(exchange(x, early, data), av)
        x = block.relu(block.bn2(block.conv2(x.flatten(0, 1))))
        x = block.relu(block.bn3(block.conv3(x)) + identity)
        return self._present(x.reshape(n, 2, 1024, 10, 10), av)

    def finish(self, layer3, early, data):
        """Run layer4, deep M2 readout, and the full 7170-dimensional guide."""
        av = data['available'].to(layer3)
        valid = data['valid'].to(layer3) * av[..., None, None, None]
        layer3 = self._present(layer3, av)
        n = len(layer3)
        deep = self.resnet.layer4(layer3.flatten(0, 1))
        glob = regional_mean(deep, valid.flatten(0, 1), valid.flatten(0, 1))[0][:, 0]
        glob = glob.reshape(n, 2, 2048)
        pooled = []
        for p in range(2):
            region = region_integrals(data['slots'][:, p])
            if self.use_m2:
                extra = self.deep_pool[p](layer3[:, p], region, valid[:, p])['pooled']
                pooled.append((early['low'][:, p] + self.deep_output[p](extra)).flatten(1))
            else:
                pooled.append(layer3.new_zeros(n, 3*512))
        features = self._present(torch.cat((glob, torch.stack(pooled, 1)), -1), av)
        batch = dict(features=torch.cat((features.flatten(1), av), 1),
                     x=early['f2'], mask=early['masks'], response=early['responses'],
                     available=av, valid=early['occupancy'])
        return self.guide(batch) if self.use_guidance else self.guide.head(batch['features'])

    def forward(self, data, exchange=None, residual=None, feature_transform=None):
        early = self.early(data['images'], data['valid'], data['available'], data['local_slots'], feature_transform)
        layer3 = self.middle(early, data, exchange)
        if residual is not None:
            layer3 = layer3 + residual
        return self.finish(layer3, early, data), early['auxiliary']
