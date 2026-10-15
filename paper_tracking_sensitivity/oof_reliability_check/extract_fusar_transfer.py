"""Single-image input adaptation for the frozen current source encoders."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.models import resnet18

from extract_opensarship2_transfer import WS, build_encoder, extract, write_json
from fusar_labels import map_fusar_labels, summarize_fusar_mapping

COHORT = WS/'artifacts/bdh_fusar_preparation_20261003/experiment_cohort.csv'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'features.npz').exists():
        raise ValueError('target features already exist; do not overwrite')
    source = pd.read_csv(COHORT, dtype={'mmsi': str}, keep_default_na=False)
    mapped = map_fusar_labels(source)
    mapped.to_csv(args.out/'label_mapping.csv', index=False)
    summary = summarize_fusar_mapping(mapped)
    write_json(args.out/'label_mapping_summary.json', summary)
    target = mapped.loc[mapped.label_included].copy()
    target.to_csv(args.out/'target_manifest.csv', index=False)
    print(f'COHORT {len(target)}/{len(source)}', flush=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    base = build_encoder(device)
    checkpoint = WS/'e206_out/ssl_resnet18.pt'
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    backbone = resnet18(weights=None)
    missing, unexpected = backbone.load_state_dict(ck['backbone'], strict=False)
    if set(missing) != {'fc.weight', 'fc.bias'} or unexpected:
        raise ValueError(f'unexpected SSL checkpoint keys: {missing}, {unexpected}')
    ssl = torch.nn.Sequential(*list(backbone.children())[:-1]).to(device).eval()
    paths = [(p, p) for p in target.image_path]
    r50, extra = extract(paths, base, ssl, device)
    if not np.isfinite(r50).all() or not np.isfinite(extra).all():
        raise ValueError('nonfinite target features')
    # Shared runner keys denote encoder branches, NOT actual FUSAR VV/VH bands.
    np.savez_compressed(args.out/'features.npz', sample_ids=target.sample_id.to_numpy(dtype=str),
                        paired_r50=r50, vh_ssl=extra)
    write_json(args.out/'feature_recipe.json', dict(rows=len(target),
        target_input='single-image dual encoder: image duplicated into two base input channels',
        no_physical_vv_vh_pair=True, checkpoint=str(checkpoint), ssl_epoch=int(ck['epoch']),
        base='unchanged two-channel ImageNet R50 V2; L bilinear128; (x/255-.5)/.25; bfloat16/spatial-fp16 pool',
        extra='same image, RGB Resize(128)/ToTensor -> e206 SSL R18 float32',
        cache_keys='paired_r50 and vh_ssl are shared-runner branch names, NOT FUSAR polarization claims',
        previous_feature_cache='external_os2_fusar/feats_fusar.npz is not reused: different base and resize/normalization recipe',
        source_encoder_replay='same extractor/environment/checkpoint as immediately preceding opensarship2_transfer_20261007/encoder_replay.json',
        torch_version=torch.__version__, quality_filter_added=False))
    print('FUSAR FEATURES SAVED', flush=True)


if __name__ == '__main__':
    main()
