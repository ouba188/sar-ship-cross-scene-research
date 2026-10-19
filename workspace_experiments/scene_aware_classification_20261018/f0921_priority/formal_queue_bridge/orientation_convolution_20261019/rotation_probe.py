"""Supplement-only coordinate-rotation diagnostic; not a new SAR observation."""
import json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torchvision.models import resnet50
from PIL import Image
from threadpoolctl import threadpool_limits
from run_orientation import HERE, Study, OrientationReader


def main():
    torch.set_num_threads(2)
    study = Study()
    tasks = [t for t in study.tasks['tasks'] if t['line'] == 'cross_pass' and t['budget'] == 16]
    ids = sorted(set(i for t in tasks for i in t['query']+t['supports'][0]))
    model = resnet50(weights=None).cuda().eval()
    model.load_state_dict(torch.load('C:/Users/25791/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth', map_location='cpu', weights_only=True))
    study.rotation_inputs = torch.zeros_like(study.inputs[0])
    with torch.inference_mode():
        for start in range(0, len(ids), 8):
            batch = ids[start:start+8]
            a = np.stack([[np.asarray(Image.open(study.meta.iloc[i][col])) for col in ['png_vv', 'png_vh']] for i in batch])
            x = torch.tensor(a, dtype=torch.float32, device='cuda')/255
            x = torch.rot90(x, 1, (-2, -1))
            x = F.interpolate(x, (128, 128), mode='bilinear', align_corners=False, antialias=True)
            x = x.flatten(0, 1)[:, None].expand(-1, 3, -1, -1)
            x = (x-x.new_tensor([.485, .456, .406])[None, :, None, None])/x.new_tensor([.229, .224, .225])[None, :, None, None]
            f = model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(x)))))
            study.rotation_inputs[batch] = (f.reshape(len(batch), 2, 256, 32, 32)-study.mean)/study.std
    del model
    rows = []
    for arm in ['A', 'B', 'C', 'D']:
        reader = OrientationReader(arm).cuda().eval()
        reader.load_state_dict(torch.load(HERE / f'{arm}_seed0.pt', weights_only=True))
        for task in tasks:
            ct = study.context(task['source'], task['supports'][0], task['query'])
            with torch.no_grad():
                original = study.predict(reader, ct)[0].cpu().numpy()
                rotated = study.predict(reader, ct, True)[0].cpu().numpy()
                raw0, _ = study.embedding(reader, ct)
                raw1, _ = study.embedding(reader, ct, True)
                kernel_diff = float(((raw0@raw0.T)-(raw1@raw1.T)).abs().mean())
            rows.append(dict(arm=arm, task=task['name'], seed=0, draw=0, queries=len(ct['q']),
                             probability_mean_abs=float(abs(original-rotated).mean()),
                             probability_max_abs=float(abs(original-rotated).max()),
                             changed_predictions=int((original.argmax(1) != rotated.argmax(1)).sum()),
                             new_kernel_mean_abs=kernel_diff))
    pd.DataFrame(rows).to_csv(HERE / 'rotation_diagnostic.csv', index=False)
    (HERE / 'rotation_scope.json').write_text(json.dumps(dict(
        coordinate_rotation_degrees=90, rotated_raw_png_before_backbone=True,
        masks_and_ship_and_los_rotated=True, old_kernel_prior_expert_frozen_at_original=True,
        scope='Supplement-path sensitivity only; not complete-system rotational invariance or new SAR aspect',
        seeds=[0], draws=[0], source_normalization='original fit-only frozen'), indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
