"""A matched layer4 fine-tuning pilot, not a claimed new attention mechanism."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class ComplementTail(nn.Module):
    def __init__(self, backbone_state):
        super().__init__()
        net = resnet18(weights=None)
        missing, unexpected = net.load_state_dict(backbone_state, strict=False)
        if set(missing) != {'fc.weight', 'fc.bias'} or unexpected:
            raise ValueError('expected e206 backbone without classification head')
        self.layer4 = net.layer4
        self.classifier = nn.Linear(512, 8)
        self.utility = nn.Sequential(nn.Linear(528, 64), nn.ReLU(), nn.Linear(64, 3))

    def train(self, mode=True):
        super().train(mode)
        # Both arms keep pretrained running statistics; the cached prefix is fixed.
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def features(self, spatial):
        return self.layer4(spatial).mean((2, 3))

    def forward(self, spatial, base_scores):
        features = self.features(spatial)
        logits = self.classifier(features)
        utility = self.utility(torch.cat((features, base_scores, logits.detach()), dim=1))
        return logits, utility


def pilot_loss(logits, utility_logits, labels, base_scores, eligible, class_weights, *, candidate):
    per_row = F.cross_entropy(logits.float(), labels, reduction='none')
    weights = class_weights[labels]
    classification = (per_row*weights).sum()/weights.sum()
    zero = logits.sum()*0.
    utility_loss, tail_loss = zero, zero
    selected_count = 0
    if candidate and eligible.any():
        indices = torch.where(eligible)[0]
        # Targets use source labels only. The utility head differentiates recovery,
        # harm and neutral outcomes of the current neural expert against OOF base.
        delta = logits.detach().argmax(1).eq(labels).long()-base_scores.argmax(1).eq(labels).long()
        utility_per_row = F.cross_entropy(utility_logits[indices].float(), delta[indices]+1, reduction='none')
        utility_loss = (utility_per_row*weights[indices]).sum()/weights[indices].sum()
        probability = utility_logits[indices].detach().float().softmax(1)
        score = probability[:, 2]-probability[:, 0]
        selected_count = max(1, math.ceil(.05*len(indices)))
        selected = indices[score.topk(selected_count).indices]
        tail_loss = (per_row[selected]*weights[selected]).sum()/weights[selected].sum()
    return classification+utility_loss+tail_loss, dict(
        classification=classification.detach(), utility=utility_loss.detach(),
        tail=tail_loss.detach(), selected_count=selected_count)
