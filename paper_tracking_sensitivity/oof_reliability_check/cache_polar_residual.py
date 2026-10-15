"""Add only the missing VV spatial prefix; reuse the existing VH cache."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.models import resnet18

from train_neural_complement import PIXELS, cache_prefix
from extract_opensarship2_transfer import WS, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    protocol = json.loads((out/'protocol.json').read_text())
    if (out/'VV_CACHE_COMPLETE.json').exists():
        raise ValueError('VV cache already completed')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    checkpoint = torch.load(WS/'e206_out/ssl_resnet18.pt', map_location='cpu', weights_only=True)
    backbone = resnet18(weights=None)
    missing, unexpected = backbone.load_state_dict(checkpoint['backbone'], strict=False)
    if set(missing) != {'fc.weight', 'fc.bias'} or unexpected:
        raise ValueError('unexpected e206 state')
    prefix = torch.nn.Sequential(*list(backbone.children())[:-3]).to(device).eval().requires_grad_(False)
    source = np.load(PIXELS, mmap_mode='r')
    target = np.load(Path(protocol['target'])/'inputs128.npy', mmap_mode='r')
    cache_prefix(out, prefix, source, 'source_vv', device, channel=0)
    cache_prefix(out, prefix, target, 'target_vv', device, channel=0)
    write_json(out/'VV_CACHE_COMPLETE.json', dict(source_rows=len(source), target_rows=len(target),
        channel_index=0, prefix='e206 through layer3, RGB repeated VV/255',
        source_vh=str(Path(protocol['original'])/'source_layer3.npy'),
        target_vh=str(Path(protocol['original'])/'target_layer3.npy')))


if __name__ == '__main__':
    main()
