"""Measure the actual auxiliary pixel-to-readout cost, not just its tiny head."""
import json
import time
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet50
from run_joint import HERE, parent, OLD, inputs, restore
sys.path.insert(0, str(parent.HERE))
from extract_moments import weighted_moments
sys.path.insert(0, str(parent.PARENT))
from prepare_features import crop_pad, masks_for, WEIGHTS


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    table = parent.load_table().set_index('instance_id')
    pred = pd.read_csv(HERE / 'predictions.csv')
    pred = pred[pred.arm.eq('additive') & pred.seed.eq(17) & pred.fold.eq(0)
                & pred.role.eq('test') & pred.valid_vv & pred.valid_vh]
    pred = pred.sort_values('instance_id').head(32)
    assert len(pred) == 32
    snapshots = {s['obs_id']: s for s in json.loads(
        (OLD / 'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    with np.load(parent.PARENT / 'baseline/models/fold0_outer.npz') as z:
        w = torch.tensor(z['W'].reshape(8, 2, 1024).transpose(1, 0, 2),
            device='cuda', dtype=torch.float64)
    ck, head = restore(HERE / 'models/fold0_additive_seed17.pt')
    model = resnet50(weights=None)
    model.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True))
    model = model.cuda().eval().requires_grad_(False)
    norm_mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
    norm_std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]

    def load(n):
        pixels, masks = [], []
        for ident in pred.instance_id.iloc[:n]:
            row = table.loc[ident]
            for pol in ['vv', 'vh']:
                view = snapshots[int(row.obs_id)]['images'][pol]
                with Image.open(row[f'image_{pol}']) as im:
                    raw = np.array(im.convert('L'))
                pix, mask, _ = crop_pad(raw, masks_for(view), 160)
                pixels.append(pix[None].astype(np.float32)/255)
                masks.append(mask.sum(0)[None].astype(np.float32))
        return np.stack(pixels), np.stack(masks)

    @torch.inference_mode()
    def forward(pixels, masks, n, distribution):
        x = torch.as_tensor(pixels, device='cuda')
        footprint = torch.as_tensor(masks, device='cuda')
        wp = w[torch.arange(2*n, device='cuda') % 2]
        means, moments = [], []
        for rotation in range(4):
            xx = x.rot90(rotation, (-2, -1)).expand(-1, 3, -1, -1)
            mm = footprint.rot90(rotation, (-2, -1))
            z = model.relu(model.bn1(model.conv1((xx-norm_mean)/norm_std)))
            z = model.layer3(model.layer2(model.layer1(model.maxpool(z))))
            weight = F.interpolate(mm, size=z.shape[-2:], mode='area').flatten(2).double()
            response = torch.einsum('bck,bku->bcu', wp, z.flatten(2).double())
            if distribution:
                moment, mu, _ = weighted_moments(response, weight)
                moments.append(moment)
            else:
                mu = (response * weight / weight.sum(-1, keepdim=True)).sum(-1)
            means.append(mu)
        mean = torch.stack(means).mean(0).cpu().numpy().reshape(n, 2, 8)
        if not distribution:
            return mean, None, None
        moment = torch.stack(moments).mean(0).cpu().numpy().reshape(n, 2, 8, 2)
        base = pred.iloc[:n][[f'b{c}' for c in range(8)]].to_numpy()
        xx, _, _ = inputs(base, mean, moment, np.ones((n, 2), bool), ck['mu'], ck['scale'])
        return mean, moment, base + head(xx).numpy().astype(float)

    with np.load(parent.HERE / 'signed_features.npz') as z:
        reader = list(z['reader_names']).index('fold0_outer')
        lookup = {s: i for i, s in enumerate(z['instance_ids'])}
        ix = [lookup[s] for s in pred.instance_id]
        expected_mean, expected_mom = z['mean_pol'][reader, ix], z['moments_pol'][reader, ix]
    pix, mask = load(32)
    mean, moment, scores = forward(pix, mask, 32, True)
    errors = dict(mean_max=float(np.max(abs(mean-expected_mean))),
        moment_max=float(np.max(abs(moment-expected_mom))),
        score_max=float(np.max(abs(scores-pred[[f's{c}' for c in range(8)]].to_numpy()))))
    assert errors['mean_max'] < 1e-5 and errors['moment_max'] < 1e-5
    assert errors['score_max'] < 1e-5
    np.testing.assert_array_equal(scores.argmax(1), pred.prediction)
    rows = []
    for n in [1, 16, 32]:
        pix, mask = load(n)
        for name, include_io, distribution in [
                ('auxiliary_mean_pixels', False, False),
                ('auxiliary_full_pixels', False, True),
                ('auxiliary_full_warm_io', True, True)]:
            for _ in range(3):
                forward(pix, mask, n, distribution)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(10):
                torch.cuda.synchronize()
                start = time.perf_counter()
                if include_io:
                    pix, mask = load(n)
                forward(pix, mask, n, distribution)
                torch.cuda.synchronize()
                times.append((time.perf_counter()-start)*1000)
            rows.append(dict(stage=name, batch=n, repeats=10,
                median_batch_ms=float(np.median(times)),
                median_ms_per_chip=float(np.median(times)/n),
                p10_batch_ms=float(np.quantile(times, .1)),
                p90_batch_ms=float(np.quantile(times, .9)),
                peak_allocated_bytes=int(torch.cuda.max_memory_allocated())))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(HERE / 'pixel_costs.csv', index=False)
    pred[['instance_id', 'obs_id', 'fold', 'seed', 'valid_vv', 'valid_vh']].to_csv(
        HERE / 'pixel_cost_cases.csv', index=False)
    info = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, checks=errors,
        test_labels_used=False, head='fold0 additive seed17', both_polarizations=True,
        scope='Auxiliary native R50 C4 plus class readout, B supplied from cache. Not entire oldAD+T inference.',
        mean_reference='Same auxiliary encoder/footprint and mean projection, no readout head; lower-bound comparison, not complete mean_only pipeline.',
        io='Warm filesystem: PNG load/crop/masks counted only in warm_io; model and metadata loading excluded.',
        sharing='No assumption that oldAD can share this native R50 forward.',
        batch_note='First 32 sorted fold0 seed17 test observations with both valid polarizations; no accuracy-based selection.')
    (HERE / 'pixel_costs_summary.json').write_text(json.dumps(info, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
