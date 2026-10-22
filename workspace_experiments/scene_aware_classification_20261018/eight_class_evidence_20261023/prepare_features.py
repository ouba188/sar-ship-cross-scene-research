"""Native eight-class shared backbone features; no classifier or label updates."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '2')
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.models import resnet50

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[1] / 'same_vessel_gallery_20261019'
sys.path.insert(0, str(OLD / 'local_condition_module'))
from native_input import crop_pad
from prepare import masks_for, WEIGHTS


def pool(z, footprint, maximum=True):
    w = F.interpolate(footprint, size=z.shape[-2:], mode='area')
    mass = w.sum((-2, -1))
    mean = (z * w).sum((-2, -1)) / mass.clamp_min(1e-8)
    if not maximum:
        return mean
    peak = z.masked_fill(w <= 0, -torch.inf).amax((-2, -1))
    peak = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak))
    return torch.cat((mean, peak), 1)


@torch.inference_mode()
def encode(model, images, regions):
    mean = images.new_tensor([.485, .456, .406])[None, :, None, None]
    std = images.new_tensor([.229, .224, .225])[None, :, None, None]
    result = {k: [] for k in ('stem', 'fg', 'bg', 'layer3')}
    for k in range(4):
        x = images.rot90(k, (-2, -1)).expand(-1, 3, -1, -1)
        r = regions.rot90(k, (-2, -1))
        z = model.relu(model.bn1(model.conv1((x - mean) / std)))
        for name, j in [('stem', 0), ('fg', 1), ('bg', 2)]:
            result[name].append(pool(z, r[:, j:j+1]))
        z = model.layer3(model.layer2(model.layer1(model.maxpool(z))))
        result['layer3'].append(pool(z, r[:, 0:1], maximum=False))
    return {k: torch.stack(v).mean(0).cpu().numpy() for k, v in result.items()}


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    frame = pd.read_csv(OLD / 'conditional_evidence_pilot/cohort.csv', dtype={'mmsi': str})
    snapshots = json.loads((OLD / 'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))
    assert frame.obs_id.tolist() == [a['obs_id'] for a in snapshots]
    aligned = pd.read_csv(OLD / 'local_condition_module/crossview_alignment.csv').set_index('obs_id')
    disabled = aligned.loc[frame.obs_id, 'disable_new_vh'].to_numpy(bool)
    keep = frame.obs_id.ne(31237).to_numpy()
    frame = frame.loc[keep].reset_index(drop=True)
    snapshots = [s for s, ok in zip(snapshots, keep) if ok]
    disabled = disabled[keep]
    model = resnet50(weights=None)
    model.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True))
    model = model.cuda().eval().requires_grad_(False)
    arrays = {k: np.zeros((len(frame), 2, d), np.float32)
              for k, d in [('stem', 128), ('fg', 128), ('bg', 128), ('layer3', 1024)]}
    valid = np.zeros((len(frame), 2), bool)
    regional = np.zeros((len(frame), 2, 2), bool)
    coverage, checks = [], []
    for first in range(0, len(frame), 16):
        images, regions, addresses = [], [], []
        for i in range(first, min(first+16, len(frame))):
            for p, pol in enumerate(('vv', 'vh')):
                row, view = frame.iloc[i], snapshots[i]['images'][pol]
                with Image.open(row['image_'+pol]) as im:
                    raw = np.asarray(im.convert('L'))
                assert raw.shape == (view['h'], view['w'])
                pixels, masks, place = crop_pad(raw, masks_for(view), 160)
                assert place['native_target_pixels'] == place['retained_target_pixels']
                is_valid = bool(masks[0].any() and not (p == 1 and disabled[i]))
                coverage.append(dict(instance_id=row.instance_id, class_id=int(row.class_id), pol=pol,
                                     valid=is_valid, disabled_alignment=bool(p == 1 and disabled[i]), **place))
                if not is_valid:
                    continue
                support = masks.sum(0)
                foreground = masks[0]
                all_boxes = torch.from_numpy(masks[:2].sum(0))[None, None]
                dilated = F.max_pool2d(all_boxes, 7, stride=1, padding=3)[0, 0].numpy() > 0
                background = (support > 0) & ~dilated
                reg = np.stack((support, foreground, background)).astype(np.float32)
                valid[i, p] = True
                regional[i, p] = [foreground.any(), background.any()]
                images.append(pixels[None].astype(np.float32)/255)
                regions.append(reg)
                addresses.append((i, p))
        x = torch.tensor(np.stack(images), device='cuda')
        r = torch.tensor(np.stack(regions), device='cuda')
        out = encode(model, x, r)
        if first == 0:
            single = encode(model, x[:1], r[:1])
            rotated = encode(model, x[:1].rot90(1, (-2, -1)), r[:1].rot90(1, (-2, -1)))
            for name in arrays:
                err = float(np.max(np.abs(single[name][0]-out[name][0])))
                rotation_err = float(np.max(np.abs(single[name]-rotated[name])))
                assert np.allclose(single[name][0], out[name][0], atol=1e-4, rtol=1e-4), (name, err)
                assert np.allclose(single[name], rotated[name], atol=1e-5, rtol=1e-5)
                checks.append(dict(feature=name, single_batch_max_error=err, c4_max_error=rotation_err))
        for b, (i, p) in enumerate(addresses):
            for name in arrays:
                arrays[name][i, p] = out[name][b]
        if first % 800 == 0:
            print(first, '/', len(frame), 'elapsed', round(time.perf_counter()-started, 1), flush=True)
    assert valid[:, 0].all()
    assert all(np.isfinite(v).all() and not v[~valid].any() for v in arrays.values())
    np.savez_compressed(HERE/'features.npz', **{k: v.reshape(len(frame), -1) for k, v in arrays.items()},
                        valid=valid, regional_valid=regional, instance_ids=frame.instance_id.to_numpy())
    frame.to_csv(HERE/'cohort.csv', index=False)
    pd.DataFrame(coverage).to_csv(HERE/'coverage.csv', index=False)
    pd.DataFrame(checks).to_csv(HERE/'extraction_checks.csv', index=False)
    summary = dict(rows=len(frame), mmsi=int(frame.mmsi.nunique()), classes=frame.class_name.value_counts().to_dict(),
                   products=int(frame['product'].nunique()), vv_valid=int(valid[:,0].sum()), vh_valid=int(valid[:,1].sum()),
                   invalid_region_counts=(~regional).sum((0,1)).tolist(), seconds=time.perf_counter()-started,
                   no_resize=True, canvas=160, old_model_unchanged=True,
                   note='Old eight-class development cohort; frozen ImageNet R50, shared stem/layer3; valid-area pooling after conv, RF may mix padding/boundaries. No new classification result.')
    (HERE/'EXTRACTION.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    from review_inputs import main as review_inputs
    review_inputs()


if __name__ == '__main__':
    main()
