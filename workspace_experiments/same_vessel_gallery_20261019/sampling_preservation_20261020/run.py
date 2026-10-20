"""Matched early/late merging of native five-point sampling; frozen old reader."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50

ROOT = Path(__file__).resolve().parent
AXIS = ROOT.parent / 'native_axis_probe_20261020'
sys.path.insert(0, str(AXIS))
import probe as old
LOCAL, prior = old.LOCAL, old.prior
ARMS = ('early_both', 'keep_refs', 'keep_roles', 'late_both', 'late_image', 'late_ship')
NSTAT = 5


def sampled(features, masks, angles, radii):
    """Numerators [B,R,C,H,W,T,K], masses [B,R,H,W,T,K]."""
    b, c, h, w = features.shape
    along = torch.stack((angles.cos(), angles.sin()), -1) * radii[:, :1, None]
    across = torch.stack((-angles.sin(), angles.cos()), -1) * radii[:, 1:, None]
    offsets = torch.stack((torch.zeros_like(along), along, -along, across, -across), -2)
    y, x = torch.meshgrid(torch.arange(h, device=features.device), torch.arange(w, device=features.device), indexing='ij')
    centers = torch.stack((x, y), -1).to(features.dtype)
    grid = centers[None, :, :, None, None] + offsets[:, None, None]
    grid = 2 * (grid + .5) / features.new_tensor((w, h)) - 1
    t = angles.shape[1]
    grid = grid[:, None].expand(-1, 3, -1, -1, -1, -1, -1).reshape(b * 3, h, w * t * 5, 2)
    source = torch.cat((features[:, None] * masks[:, :, None], masks[:, :, None]), 2)
    values = F.grid_sample(source.reshape(b * 3, c + 1, h, w), grid,
                           mode='bilinear', padding_mode='zeros', align_corners=False)
    values = values.reshape(b, 3, c + 1, h, w, t, 5)
    return values[:, :, :c], values[:, :, c]


def describe(numerator, mass, merge_refs=False, merge_roles=False):
    # Merge before the absolute differences. O/R merge does not merge the target.
    if merge_roles:
        nm = numerator[:, 1:].sum(1, keepdim=True)
        mm = mass[:, 1:].sum(1, keepdim=True)
        numerator = torch.cat((numerator[:, :1], nm, nm), 1)
        mass = torch.cat((mass[:, :1], mm, mm), 1)
    if merge_refs:
        # Each axis is unsigned independently. Average both relative endpoint
        # pairings AFTER nonlinear readout, rather than imposing a shared sign.
        direct = describe(numerator.sum(-2, keepdim=True).expand_as(numerator),
                          mass.sum(-2, keepdim=True).expand_as(mass))
        order = torch.tensor([0, 2, 1, 4, 3], device=numerator.device)
        swapped_n = torch.cat((numerator[..., :1, :], numerator[..., 1:, :].index_select(-1, order)), -2)
        swapped_m = torch.cat((mass[..., :1, :], mass[..., 1:, :].index_select(-1, order)), -2)
        reverse = describe(swapped_n.sum(-2, keepdim=True).expand_as(numerator),
                           swapped_m.sum(-2, keepdim=True).expand_as(mass))
        return (direct + reverse) / 2
    v = numerator / mass[:, :, None].clamp_min(1e-8)
    center = mass[..., 0]
    wa = center * torch.minimum(mass[..., 1], mass[..., 2])
    wc = center * torch.minimum(mass[..., 3], mass[..., 4])
    weights = (center, wa, wc, wa, wc)
    stats = (v[..., 0], (v[..., 1] + v[..., 2]) / 2,
             (v[..., 3] + v[..., 4]) / 2, (v[..., 1] - v[..., 2]).abs(),
             (v[..., 3] - v[..., 4]).abs())
    values = []
    for stat, weight in zip(stats, weights):
        denom = weight.sum((2, 3))
        pooled = (stat * weight[:, :, None]).sum((3, 4)) / denom[:, :, None].clamp_min(1e-8)
        values.append(pooled.permute(0, 1, 3, 2))
    return torch.stack(values, -2)  # B,R,T,stat,C


def geometry(view):
    p = view['priors']; axis = p.get('ship_axis')
    ship = float(axis['angle']) if axis else np.pi / 4
    radar_ok = p.get('radar_kind') == 'orbit_los' and p.get('radar_axis') is not None
    radar_ok = radar_ok and np.isfinite(p['radar_axis'])
    radar = float(p['radar_axis']) if radar_ok else 0.
    radii = [np.clip(axis['length_px'] / 32, .75, 3), np.clip(axis['width_px'] / 32, .5, 2)] if axis else [1., 1.]
    return [ship, radar, float(bool(axis)), float(radar_ok)], radii


def readout(stem, x, masks, geom, radii, arms=ARMS):
    means = x.new_tensor([.485, .456, .406])[None, :, None, None]
    stds = x.new_tensor([.229, .224, .225])[None, :, None, None]
    outputs = {a: [] for a in arms}
    for k in range(4):
        image = torch.rot90(x, k, (-2, -1)).expand(-1, 3, -1, -1)
        m = F.avg_pool2d(torch.rot90(masks, k, (-2, -1)), 4)
        features = F.avg_pool2d(stem((image - means) / stds), 2)
        first = torch.full_like(geom[:, 0], np.pi / 4)
        second = torch.zeros_like(first)
        ship = torch.where(geom[:, 2] > 0, geom[:, 0] - k * np.pi / 2, first)
        radar = torch.where(geom[:, 3] > 0, geom[:, 1] - k * np.pi / 2, second)
        physical_only = all(a not in ('late_image', 'late_ship') for a in arms)
        angles = torch.stack((ship, radar), -1) if physical_only else torch.stack((first, second, ship, radar), -1)
        num, mass = sampled(features, m, angles, radii)
        for arm in arms:
            idx = [0, 1] if physical_only or arm == 'late_image' else [2, 1] if arm == 'late_ship' else [2, 3]
            outputs[arm].append(describe(num[..., idx, :], mass[..., idx, :],
                merge_refs=arm in ('early_both', 'keep_roles'),
                merge_roles=arm in ('early_both', 'keep_refs')))
    return {a: torch.stack(v).mean(0) for a, v in outputs.items()}


def extract(repair_early=False):
    start = time.perf_counter(); torch.set_num_threads(2)
    f = pd.read_csv(prior.ROOT / 'cohort.csv', dtype={'mmsi': str})
    views = json.loads((LOCAL / 'view_snapshot.json').read_text(encoding='utf-8'))
    assert [v['obs_id'] for v in views] == f.obs_id.tolist()
    conflict = pd.read_csv(LOCAL / 'crossview_alignment.csv').set_index('obs_id').loc[f.obs_id, 'disable_new_vh'].to_numpy(bool)
    arms = ('early_both', 'keep_roles') if repair_early else ARMS
    caches = {a: np.lib.format.open_memmap(ROOT / (a + '.npy'), mode='w+', dtype=np.float32,
              shape=(len(f), 2, 3, 2, NSTAT, 64)) for a in arms}
    model = resnet50(weights=None)
    model.load_state_dict(torch.load(old.WEIGHTS, map_location='cpu', weights_only=True))
    stem = nn.Sequential(model.conv1, model.bn1, model.relu).cuda().eval(); del model
    rows = []; checks = []
    with torch.inference_mode():
        for first in range(0, len(f), 8):
            images = []; masks = []; geoms = []; radii = []; addresses = []
            for i in range(first, min(first + 8, len(f))):
                for p, pol in enumerate(('vv', 'vh')):
                    if f.iloc[i].obs_id == 31237 or (p == 1 and conflict[i]): continue
                    view = views[i]['images'][pol]
                    raw = np.asarray(Image.open(f.iloc[i]['image_' + pol]).convert('L'))
                    assert raw.shape == (view['h'], view['w'])
                    pix, mask, place = old.crop_pad(raw, old.masks_for(view), 160)
                    assert place['native_target_pixels'] == place['retained_target_pixels']
                    geom, radius = geometry(view)
                    images.append(torch.from_numpy(pix.copy())[None].float() / 255)
                    masks.append(torch.from_numpy(mask)); geoms.append(geom); radii.append(radius); addresses.append((i, p))
                    rows.append(dict(obs_id=int(f.iloc[i].obs_id), pol=pol, ship_valid=geom[2], radar_valid=geom[3],
                                     along_radius=radius[0], across_radius=radius[1], **place))
            if not images: continue
            x = torch.stack(images).cuda(); m = torch.stack(masks).cuda()
            g = torch.tensor(geoms, dtype=torch.float32, device='cuda')
            r = torch.tensor(radii, dtype=torch.float32, device='cuda')
            result = readout(stem, x, m, g, r, arms)
            for arm in arms:
                values = result[arm].cpu().numpy()
                for j, (i, p) in enumerate(addresses): caches[arm][i, p] = values[j]
            if first == 0:
                rotated_g = g[:2].clone(); rotated_g[:, :2] -= np.pi / 2
                alt = readout(stem, torch.rot90(x[:2], 1, (-2, -1)), torch.rot90(m[:2], 1, (-2, -1)), rotated_g, r[:2], arms)
                for arm in arms:
                    error = float((result[arm][:2] - alt[arm]).abs().max())
                    checks.append(dict(test='joint_C4', arm=arm, max_abs=error))
                    assert torch.allclose(result[arm][:2], alt[arm], atol=5e-5, rtol=5e-5), (arm, error)
            if first % 256 == 0: print('extract', first, len(f), round(time.perf_counter() - start, 1), flush=True)
    for v in caches.values(): v.flush()
    prefix = 'repair_' if repair_early else ''
    pd.DataFrame(rows).to_csv(ROOT / (prefix + 'coverage.csv'), index=False)
    pd.DataFrame(checks).to_csv(ROOT / (prefix + 'invariance.csv'), index=False)
    (ROOT / (prefix + 'EXTRACTION.json')).write_text(json.dumps(dict(n=len(f), seconds=time.perf_counter() - start,
        shape=[2, 3, 2, 5, 64], canvas=160, resize=False, references='image, ship, ship+LOS',
        input='native PNG; not calibrated radiometry'), indent=2), encoding='utf-8')


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.content = nn.Linear(64, 8)
        self.output = nn.Linear(60 * 8, 8)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return self.output(F.gelu(self.content(x)).flatten(1))


def baseline(raw, fold):
    h = np.load(AXIS / f'head_f{fold}_axis.npz')
    return ((raw - h['mean']) / h['std'] / np.sqrt(raw.shape[1])) @ h['coef'].T + h['intercept']


def predict(model, x, base):
    with torch.inference_mode():
        return np.concatenate([(model(x[i:i + 512]) + base[i:i + 512]).cpu().numpy() for i in range(0, len(x), 512)])


def train(objective='ce'):
    start = time.perf_counter(); torch.set_num_threads(2)
    output = ROOT if objective == 'ce' else ROOT / 'mse_aligned'
    output.mkdir(exist_ok=True)
    f = pd.read_csv(prior.ROOT / 'cohort.csv', dtype={'mmsi': str}); y = f.class_id.to_numpy()
    d = np.load(AXIS / 'descriptors.npz'); raw_base = np.concatenate((d['base'], d['axis']), 1)
    excluded = pd.read_csv(LOCAL / 'EXCLUSIONS.csv').obs_id.to_numpy()
    caches = {a: np.load(ROOT / (a + '.npy'), mmap_mode='r') for a in ARMS}
    metrics = []; selections = []; predictions = []; interventions = []; partition_rows = []
    for fold in range(3):
        train_idx, val, test = prior.indices(f, fold)
        train_idx, val, test = [ii[~f.iloc[ii].obs_id.isin(excluded).to_numpy()] for ii in (train_idx, val, test)]
        base_np = baseline(raw_base, fold).astype(np.float32)
        # Verify native old predictions; the residual experiment cannot change them.
        oldp = pd.read_csv(AXIS / 'predictions.csv')
        oldp = oldp[oldp.fold.eq(fold) & oldp.arm.eq('axis')].set_index('obs_id').loc[f.iloc[test].obs_id]
        assert np.array_equal(oldp.prediction.to_numpy(), base_np[test].argmax(1))
        base = torch.tensor(base_np, device='cuda'); targets = torch.tensor(y, device='cuda')
        weight = torch.tensor(prior.weights(f, train_idx), dtype=torch.float32, device='cuda')
        for name, ii in (('train', train_idx), ('val', val), ('test', test)):
            partition_rows.append(dict(fold=fold, partition=name, n=len(ii), ships=f.iloc[ii].mmsi.nunique(), products=f.iloc[ii]['product'].nunique()))
            metrics.append(dict(fold=fold, arm='baseline', seed=-1, step=0, partition=name,
                                **prior.metrics(y[ii], base_np[ii])))
        for arm in ARMS:
            raw = np.asarray(caches[arm]).reshape(len(f), 60, 64)
            mean = raw[train_idx].mean(0); std = np.maximum(raw[train_idx].std(0), 1e-6)
            z = (raw - mean) / std
            x = torch.from_numpy(z).cuda(); del z
            for seed in range(3):
                torch.manual_seed(20261020 + seed); torch.cuda.manual_seed_all(20261020 + seed)
                rng = np.random.default_rng(20261020 + seed + 31 * fold)
                model = Residual().cuda()
                opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
                best_key = None; best_state = None; best_step = None
                for step in range(151):
                    if step in (0, 50, 100, 150):
                        model.eval(); scores = predict(model, x[val], base[val])
                        met = prior.metrics(y[val], scores)
                        selections.append(dict(fold=fold, arm=arm, seed=seed, step=step, **met))
                        key = (met['ba'], -met['brier'])
                        if best_key is None or key > best_key:
                            best_key = key; best_step = step
                            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        if step == 0: assert np.array_equal(scores, base_np[val])
                    if step == 150: break
                    batch = rng.integers(len(train_idx), size=128); ii = train_idx[batch]
                    model.train(); opt.zero_grad(set_to_none=True)
                    scores = base[ii] + model(x[ii])
                    if objective == 'ce':
                        loss = F.cross_entropy(scores, targets[ii], reduction='none')
                    else:
                        wanted = 2 * F.one_hot(targets[ii], 8).float() - 1
                        loss = (scores - wanted).square().mean(-1)
                    loss = (loss * weight[batch]).mean()
                    if not torch.isfinite(loss): raise RuntimeError(f'Non-finite loss: {fold}/{arm}/{seed}/{step}')
                    loss.backward(); opt.step()
                model.eval()
                for name, ii in (('terminal_train', train_idx), ('terminal_val', val)):
                    terminal = predict(model, x[ii], base[ii])
                    metrics.append(dict(fold=fold, arm=arm, seed=seed, step=150, partition=name,
                                        **prior.metrics(y[ii], terminal)))
                model.load_state_dict(best_state); model.eval()
                torch.save(dict(state=best_state, mean=mean, std=std, step=best_step), output / f'head_f{fold}_{arm}_s{seed}.pt')
                for name, ii in (('train', train_idx), ('val', val), ('test', test)):
                    scores = predict(model, x[ii], base[ii])
                    metrics.append(dict(fold=fold, arm=arm, seed=seed, step=best_step, partition=name, **prior.metrics(y[ii], scores)))
                    if name == 'test':
                        np.savez_compressed(output / f'scores_f{fold}_{arm}_s{seed}.npz', rows=ii, scores=scores, base=base_np[ii])
                        other = scores.copy(); other[np.arange(len(ii)), y[ii]] = -np.inf
                        margin = scores[np.arange(len(ii)), y[ii]] - other.max(1)
                        for j, i in enumerate(ii):
                            predictions.append(dict(fold=fold, arm=arm, seed=seed, step=best_step, obs_id=int(f.iloc[i].obs_id), mmsi=f.iloc[i].mmsi,
                                port=f.iloc[i].port, product=f.iloc[i]['product'], class_id=int(y[i]), prediction=int(scores[j].argmax()),
                                base_prediction=int(base_np[i].argmax()), correct=bool(scores[j].argmax() == y[i]), margin=float(margin[j])))
                if arm == 'late_both':
                    original = predict(model, x[test], base[test])
                    for replace in ('late_image', 'late_ship'):
                        alt = np.asarray(caches[replace][test]).reshape(len(test), 60, 64)
                        alt = torch.tensor((alt - mean) / std, device='cuda')
                        ss = predict(model, alt, base[test])
                        interventions.append(dict(fold=fold, seed=seed, step=best_step, replacement=replace,
                            flips=int(np.sum(ss.argmax(1) != original.argmax(1))), score_l1=float(np.mean(np.abs(ss - original))), **prior.metrics(y[test], ss)))
                        del alt
                print('fit', fold, arm, seed, best_step, metrics[-1], flush=True)
                del model, opt
            del x
            pd.DataFrame(metrics).to_csv(output / 'metrics.csv', index=False)
            pd.DataFrame(selections).to_csv(output / 'validation_selection.csv', index=False)
    pd.DataFrame(predictions).to_csv(output / 'predictions.csv', index=False)
    pd.DataFrame(interventions).to_csv(output / 'reference_interventions.csv', index=False)
    pd.DataFrame(partition_rows).to_csv(output / 'partitions.csv', index=False)
    (output / 'RECEIPT.json').write_text(json.dumps(dict(status='complete', seconds=time.perf_counter() - start,
        fits=54, steps=150, seeds=3, identity_folds=3, frozen_old_head=True,
        objective=objective,
        trained='Residual readout, not spatial convolution; no support adaptation'), indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('extract', 'repair_early', 'train', 'train_mse'))
    args = parser.parse_args()
    if args.stage in ('train', 'train_mse'): train('mse' if args.stage == 'train_mse' else 'ce')
    else: extract(repair_early=args.stage == 'repair_early')
