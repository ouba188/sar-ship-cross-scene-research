"""One frozen R50 reused by native readouts and isolated neighborhood encoding."""
import torch
from torch import nn
from torchvision.models import resnet50


class Backbone(nn.Module):
    def __init__(self, weight_path, device='cpu'):
        super().__init__()
        self.model = resnet50(weights=None)
        self.model.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True))
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])
        self.to(device).eval().requires_grad_(False)

    @property
    def device(self):
        return self.mean.device

    def stem(self, x):
        z = (x.expand(-1, 3, -1, -1)-self.mean)/self.std
        return self.model.relu(self.model.bn1(self.model.conv1(z)))

    @torch.inference_mode()
    def forward(self, prepared):
        result = []
        for view in prepared:
            if view is None:
                result.append(None)
                continue
            x = torch.tensor(view['pixels'].copy(), device=self.device).float()[None, None]/255
            maps = dict(stem=[], layer1=[], layer3=[])
            for k in range(4):
                stem = self.stem(x.rot90(k, (-2, -1)))
                layer1 = self.model.layer1(self.model.maxpool(stem))
                layer3 = self.model.layer3(self.model.layer2(layer1))
                for key, value in (('stem', stem), ('layer1', layer1), ('layer3', layer3)):
                    maps[key].append(value)
            result.append({key: torch.cat(values) for key, values in maps.items()})
        return result
