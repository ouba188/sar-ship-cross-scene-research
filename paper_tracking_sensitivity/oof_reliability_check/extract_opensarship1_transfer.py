"""Extract current paired-R50/VH-SSL features, not the old OSS1 RGB cache."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.models import resnet18

from extract_opensarship2_transfer import WS, build_encoder, extract, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    if (out/'features.npz').exists():
        raise ValueError('completed features must not be overwritten')
    target = pd.read_csv(out/'target_manifest.csv', keep_default_na=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    base = build_encoder(device)
    ck = torch.load(WS/'e206_out/ssl_resnet18.pt', map_location='cpu', weights_only=True)
    net = resnet18(weights=None)
    missing, unexpected = net.load_state_dict(ck['backbone'], strict=False)
    if set(missing) != {'fc.weight', 'fc.bias'} or unexpected:
        raise ValueError('unexpected SSL checkpoint')
    ssl = torch.nn.Sequential(*list(net.children())[:-1]).to(device).eval()
    paths = list(zip(target.chip_path_vv, target.chip_path_vh))
    b, e = extract(paths, base, ssl, device)
    np.savez_compressed(out/'features.npz', paired_r50=b, vh_ssl=e,
                        sample_ids=target.sample_id.to_numpy(dtype=str))
    write_json(out/'feature_recipe.json', dict(rows=len(target),
        base='same two-channel paired VV/VH R50 V2: bilinear128, (x/255-.5)/.25, spatial-f16 pooling',
        extra='same frozen e206 VH-RGB R18: Resize128 preserving aspect, ToTensor, float32',
        old_external_feats_reused=False, input='native Patch_Uint8, no added crop/quality selection',
        torch_version=torch.__version__))


if __name__ == '__main__':
    main()
