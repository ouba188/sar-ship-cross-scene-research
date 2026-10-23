"""Native OSS2 GRDH: preserve all PNG pixels, pad to 224, frozen R50/C4."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '2')
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet50

HERE = Path(__file__).resolve().parent
EIGHT = HERE.parent / 'eight_class_evidence_20261023'
sys.path.insert(0, str(EIGHT))
from prepare_features import WEIGHTS
SIDE = 224


def native_image(path):
    with Image.open(path) as im:
        assert im.mode == 'L'
        raw = np.array(im)
    h, w = raw.shape
    assert max(h, w) <= SIDE
    top, left = (SIDE-h)//2, (SIDE-w)//2
    image = np.zeros((SIDE, SIDE), np.float32)
    footprint = np.zeros_like(image)
    image[top:top+h, left:left+w] = raw.astype(np.float32)/255
    footprint[top:top+h, left:left+w] = 1
    np.testing.assert_array_equal(np.rint(image[top:top+h, left:left+w]*255).astype(np.uint8), raw)
    return image[None], footprint[None]


def load_batch(table, first, stop):
    images, masks = [], []
    for _, row in table.iloc[first:stop].iterrows():
        for pol in ['vv', 'vh']:
            image, mask = native_image(row[f'chip_path_{pol}'])
            images.append(image)
            masks.append(mask)
    return torch.tensor(np.stack(images), device='cuda'), torch.tensor(np.stack(masks), device='cuda')


def network():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = resnet50(weights=None)
    model.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True))
    return model.cuda().eval().requires_grad_(False)


def feature_map(model, x, rotation):
    mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
    std = x.new_tensor([.229, .224, .225])[None, :, None, None]
    x = x.rot90(rotation, (-2, -1)).expand(-1, 3, -1, -1)
    z = model.relu(model.bn1(model.conv1((x-mean)/std)))
    return model.layer3(model.layer2(model.layer1(model.maxpool(z))))


def spatial_weights(mask, z, rotation):
    return F.interpolate(mask.rot90(rotation, (-2, -1)), size=z.shape[-2:], mode='area')


def main():
    started = time.perf_counter()
    table = pd.read_csv(HERE / 'cohort.csv', dtype={'mmsi': str})
    assert len(table) == 2746 and table.sample_id.is_unique
    model = network()
    features = np.zeros((len(table), 2, 1024), np.float32)
    last = time.perf_counter()
    with torch.inference_mode():
        for first in range(0, len(table), 16):
            stop = min(first+16, len(table))
            x, mask = load_batch(table, first, stop)
            pieces = []
            for rotation in range(4):
                z = feature_map(model, x, rotation)
                w = spatial_weights(mask, z, rotation)
                pieces.append((z*w).sum((-2, -1))/w.sum((-2, -1)))
            features[first:stop] = torch.stack(pieces).mean(0).cpu().numpy().reshape(stop-first, 2, 1024)
            if time.perf_counter()-last > 25:
                print('native GAP', stop, '/', len(table), 'seconds', round(time.perf_counter()-started, 1), flush=True)
                last = time.perf_counter()
    assert np.isfinite(features).all()
    np.savez_compressed(HERE / 'features.npz', sample_ids=table.sample_id.to_numpy(str),
        layer3=features.reshape(len(table), 2048), valid=np.ones((len(table), 2), bool))
    info = dict(n=len(table), side=SIDE, resized=0, cropped=0, pixel_padding_replay=True,
        backbone='frozen ImageNet R50 layer3', rotations=4, valid_both=len(table),
        seconds=time.perf_counter()-started, gpu=torch.cuda.get_device_name(0),
        peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (HERE / 'extraction.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
