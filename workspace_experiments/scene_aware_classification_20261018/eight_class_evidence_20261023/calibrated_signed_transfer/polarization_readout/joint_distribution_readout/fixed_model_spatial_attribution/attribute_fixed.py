"""Integrated gradients at the centered response node of four frozen cases."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')
import sys
sys.dont_write_bytecode = True
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.models import resnet50
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_joint as run
parent = run.parent
sys.path.insert(0, str(parent.HERE))
import extract_moments as extractor
CASES = [('pair_0004919', 'rescued'), ('pair_0007310', 'harmed'),
         ('pair_0003494', 'CC'), ('pair_0003504', 'WW')]
SCORE = [f's{c}' for c in range(8)]
BASE = [f'b{c}' for c in range(8)]


def fixed_predictions():
    meta = pd.read_csv(HERE.parent / 'predictions.csv',
                       usecols=['instance_id', 'arm', 'seed', 'role'])
    keep = set(np.flatnonzero(meta.instance_id.isin([x[0] for x in CASES]) &
        meta.arm.eq('additive') & meta.seed.eq(17) & meta.role.eq('test')) + 1)
    p = pd.read_csv(HERE.parent / 'predictions.csv',
        skiprows=lambda i: i != 0 and i not in keep, float_precision='round_trip')
    assert len(p) == 4 and p.instance_id.is_unique
    return p.set_index('instance_id')


def receptive_field(model):
    field, jump = 1, 1
    layers = [model.conv1, model.maxpool]
    for stage in (model.layer1, model.layer2, model.layer3):
        for block in stage:
            layers.extend([block.conv1, block.conv2, block.conv3])
    for layer in layers:
        k = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
        s = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
        d = layer.dilation[0] if isinstance(layer.dilation, tuple) else layer.dilation
        field += (k-1)*d*jump
        jump *= s
    return field, jump


def extract_case(backbone, row, snapshot, valid, W, device):
    images, support, pols, raw, placements = [], [], [], [], []
    for p, pol in enumerate(['vv', 'vh']):
        with Image.open(row[f'image_{pol}']) as im:
            image = np.asarray(im.convert('L'))
        view = snapshot['images'][pol]
        assert image.shape == (view['h'], view['w'])
        pixels, masks, place = extractor.crop_pad(image, extractor.masks_for(view), 160)
        raw.append(image); placements.append(place)
        if valid[p]:
            images.append(pixels[None].astype(np.float32)/255)
            support.append(masks.sum(0)[None].astype(np.float32))
            pols.append(p)
    x = torch.tensor(np.stack(images), device=device)
    mask = torch.tensor(np.stack(support), device=device)
    wp = torch.tensor(W.reshape(8, 2, 1024).transpose(1, 0, 2)[pols],
                      device=device, dtype=torch.float64)
    im_mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
    im_sd = x.new_tensor([.229, .224, .225])[None, :, None, None]
    rs, weights, mus, moments = [], [], [], []
    with torch.no_grad():
        for rotation in range(4):
            xx = x.rot90(rotation, (-2, -1)).expand(-1, 3, -1, -1)
            mm = mask.rot90(rotation, (-2, -1))
            z = backbone.relu(backbone.bn1(backbone.conv1((xx-im_mean)/im_sd)))
            z = backbone.layer3(backbone.layer2(backbone.layer1(backbone.maxpool(z))))
            h, w = z.shape[-2:]
            weight = F.interpolate(mm, size=(h, w), mode='area').flatten(2).double()
            response = torch.einsum('bck,bkp->bcp', wp, z.flatten(2).double())
            stat, mu, r = extractor.weighted_moments(response, weight)
            norm = weight/weight.sum(-1, keepdim=True)
            rr = np.zeros((2, 8, h, w), np.float64)
            ww = np.zeros((2, 1, h, w), np.float64)
            uu = np.zeros((2, 8), np.float64)
            ss = np.zeros((2, 8, 2), np.float64)
            rr[pols] = r.cpu().numpy().reshape(-1, 8, h, w)
            ww[pols] = norm.cpu().numpy().reshape(-1, 1, h, w)
            uu[pols], ss[pols] = mu.cpu().numpy(), stat.cpu().numpy()
            rs.append(rr); weights.append(ww); mus.append(uu); moments.append(ss)
    return dict(r=np.stack(rs), weight=np.stack(weights),
        mean=np.stack(mus), moments=np.stack(moments), raw=raw, placements=placements)


def moments_at_r(r, weight):
    variance = (r.square()*weight).sum((-2, -1)).clamp_min(0)
    # No sqrt(0) is evaluated in the inactive branch, including missing pols.
    safe = torch.where(variance > 0, variance, torch.ones_like(variance))
    sigma = torch.where(variance > 0, safe.sqrt(), torch.zeros_like(variance))
    kappa = (r*r.abs()*weight).sum((-2, -1))/sigma.clamp_min(1e-12)
    return torch.stack([sigma, kappa], dim=-1).mean(0)


class FixedReadout:
    def __init__(self, ck, head, base, means, valid, y, rival):
        self.head = head
        self.base = torch.tensor(base, dtype=torch.float64)
        self.means = torch.tensor(means, dtype=torch.float64)
        self.valid = torch.tensor(valid, dtype=torch.float64)
        self.mu = torch.tensor(ck['mu'], dtype=torch.float64)
        self.scale = torch.tensor(ck['scale'], dtype=torch.float64)
        self.y, self.rival = y, rival

    def inputs(self, moments):
        cols = [self.base-self.base.mean()]
        for p in range(2):
            for v in (self.means[p], moments[p, :, 0], moments[p, :, 1]):
                cols.extend([v, v-v.mean()])
        cols.extend([self.valid[p].expand(8) for p in range(2)])
        raw = torch.stack(cols, -1)[None]
        return ((raw-self.mu)/self.scale).float()

    def score(self, moments):
        return self.base + self.head(self.inputs(moments))[0].double()

    def margin(self, r, weight):
        s = self.score(moments_at_r(r, weight))
        return s[self.y]-s[self.rival]

    def branches(self, moments):
        x = self.inputs(moments)
        mean = self.head.mean(x[..., run.MEAN_COLUMNS]).flatten().double()
        dist = self.head.local(x[..., run.LOCAL_COLUMNS]).flatten().double()
        return mean-mean.mean(), dist-dist.mean()


def integrate(readout, r, weight, steps):
    grad = torch.zeros_like(r)
    for k in range(steps):
        point = (r*((k+.5)/steps)).detach().requires_grad_(True)
        value = readout.margin(point, weight)
        g, = torch.autograd.grad(value, point)
        assert torch.isfinite(g).all()
        grad += g/steps
    return r*grad


def aligned_map(ig, weight):
    aligned = np.stack([np.rot90(ig[t], -t, axes=(-2, -1)) for t in range(4)])
    support = np.stack([np.rot90(weight[t], -t, axes=(-2, -1)) for t in range(4)])
    # The derivative already contains the C4 mean. Sum, never average again.
    mass = aligned.sum((0, 2))
    covered = support.sum((0, 2)) > 0
    assert np.max(np.abs(mass[~covered]), initial=0) == 0
    assert abs(float(mass.sum()-ig.sum())) < 1e-10
    return mass, covered, aligned


def panel(row, q, data, mass, covered, scores, decomposition, outcome, info):
    fig, axes = plt.subplots(2, 4, figsize=(19, 10), layout='constrained')
    limit = max(float(np.abs(mass).max()), 1e-10)
    cmap = plt.get_cmap('RdBu_r').copy(); cmap.set_bad('#c8c8c8')
    for p, pol in enumerate(['VV', 'VH']):
        image, place = data['raw'][p], data['placements'][p]
        h, w = image.shape
        for ax in axes[p, :2]:
            ax.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            ax.set_xlabel('native pixel x'); ax.set_ylabel('native pixel y')
        axes[p, 0].set_title(f'{pol} original {w}x{h}; valid={info[f"valid_{pol.lower()}"]}')
        gh, gw = mass[p].shape
        ex = np.linspace(0, 160, gw+1)-.5 + place['crop_x']-place['pad_left']
        ey = np.linspace(0, 160, gh+1)-.5 + place['crop_y']-place['pad_top']
        grid = np.ma.masked_where(~covered[p], mass[p])
        overlay = axes[p, 1].pcolormesh(ex, ey, grid, cmap=cmap, vmin=-limit, vmax=limit,
            shading='flat', alpha=.65)
        axes[p, 1].set_xlim(-.5, w-.5); axes[p, 1].set_ylim(h-.5, -.5)
        axes[p, 1].set_title(f'{pol} IG at r; native coordinates\ncoarse area-pool bins, not pixel causes')
        coarse = axes[p, 2].imshow(grid, cmap=cmap, vmin=-limit, vmax=limit, interpolation='nearest')
        axes[p, 2].set_title(f'{pol} inverse-C4 sum {gh}x{gw}\nsum={mass[p].sum():+.6f}; gray=zero pool weight')
        axes[p, 2].set_xlabel('canvas feature column'); axes[p, 2].set_ylabel('canvas feature row')
        fig.colorbar(coarse, ax=axes[p, 2], shrink=.65, label='fixed-rival margin IG')
    colors = {'B': '#666666', 'B+mean': '#ad7618', 'B+mean+D0': '#8a55a4', 'full': '#187c58'}
    for name, s in scores.items():
        axes[0, 3].plot(range(8), s, marker='o', color=colors[name], label=name, lw=1)
    axes[0, 3].set_xticks(range(8)); axes[0, 3].legend(fontsize=8)
    axes[0, 3].set_title('Actual frozen class scores\nD0: raw sigma/kappa=0, not scaled-slot zero')
    labels = ['B', 'mean', 'D0', 'Dreal-D0', 'full']
    vals = [decomposition[k] for k in ['B_margin', 'mean_margin', 'D0_margin', 'distribution_change', 'full_margin']]
    axes[1, 3].bar(labels, vals, color=['#666666', '#ad7618', '#8a55a4', '#b44747', '#187c58'])
    axes[1, 3].axhline(0, color='black', lw=.6)
    axes[1, 3].tick_params(axis='x', labelrotation=25)
    axes[1, 3].set_title(f'Fixed rival={int(q.final_rival)} margin decomposition\nIG sum={info["ig_sum"]:+.6f}; target={info["target_change"]:+.6f}')
    fig.suptitle(f'Frozen additive266 / test seed17 / {row.instance_id} / {outcome} / fold{int(q.fold)} / epoch{int(q.epoch)}\n'
        f'{row.class_name} / {row.port}; true={int(row.class_id)}, fixed full rival={int(q.final_rival)}; '
        f'winner B/full={int(q.base_prediction)}/{int(q.prediction)}\n'
        f'Only r -> moments varies; B / mu / valid fixed; {info["steps"]} midpoint IG; '
        f'closure error={info["closure_error"]:+.2e}; theoretical RF={info["receptive_field"]} px', fontsize=11)
    path = HERE / f'{outcome}.png'
    fig.savefig(path, dpi=110); plt.close(fig)
    return str(path)


def main():
    start = time.perf_counter()
    assert (HERE / 'PROTOCOL_ZH.md').exists()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = resnet50(weights=None)
    backbone.load_state_dict(torch.load(extractor.WEIGHTS, map_location='cpu', weights_only=True))
    backbone = backbone.to(device).float().eval().requires_grad_(False)
    rf, stride = receptive_field(backbone)
    table = parent.load_table()
    lookup = table.reset_index().set_index('instance_id', drop=False)
    pred = fixed_predictions()
    snapshots = {s['obs_id']: s for s in json.loads((run.OLD / 'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    with np.load(parent.HERE / 'signed_features.npz') as z:
        names = list(z['reader_names'])
        moments, means, valid = z['moments_pol'], z['mean_pol'], z['valid'].astype(bool)
        np.testing.assert_array_equal(z['instance_ids'], table.instance_id)
    closure, details, convergence, case_rows, class_rows = [], [], [], [], []
    (HERE / 'arrays').mkdir(exist_ok=True)
    for identifier, outcome in CASES:
        row, q = lookup.loc[identifier], pred.loc[identifier]
        assert q.outcome == outcome
        idx, fold, y, rival = int(row['index']), int(q.fold), int(q.class_id), int(q.final_rival)
        with np.load(parent.PARENT / 'baseline/models' / f'fold{fold}_outer.npz') as z:
            W, intercept = z['W'], z['intercept']
        ck, head = run.restore(HERE.parent / 'models' / f'fold{fold}_additive_seed17.pt')
        assert ck['epoch'] == q.epoch and not any(p.requires_grad for p in head.parameters())
        data = extract_case(backbone, row, snapshots[int(row.obs_id)], valid[idx], W, device)
        r = torch.tensor(data['r'], dtype=torch.float64)
        weight = torch.tensor(data['weight'], dtype=torch.float64)
        fresh_mean, fresh_m = data['mean'].mean(0), data['moments'].mean(0)
        reader = names.index(f'fold{fold}_outer')
        expected_m, fixed_mean = moments[reader, idx], means[reader, idx]
        mean_error = float(np.abs(fresh_mean-fixed_mean).max())
        moment_error = float(np.abs(fresh_m-expected_m).max())
        assert mean_error < 1e-5 and moment_error < 1e-5
        base = q[BASE].to_numpy(float)
        readout = FixedReadout(ck, head, base, fixed_mean, valid[idx], y, rival)
        m1 = moments_at_r(r, weight)
        np.testing.assert_allclose(m1.numpy(), fresh_m, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(moments_at_r(r*.5, weight).numpy(), fresh_m*.5, atol=1e-12, rtol=1e-12)
        cache_m = torch.tensor(expected_m, dtype=torch.float64)
        xn, _, _ = run.inputs(base[None], fixed_mean[None], expected_m[None], valid[idx][None], ck['mu'], ck['scale'])
        input_error = float((readout.inputs(cache_m)-xn).abs().max())
        assert input_error < 1e-6
        with torch.no_grad():
            s1 = readout.score(m1).numpy()
            s0 = readout.score(torch.zeros_like(m1)).numpy()
            scache = readout.score(cache_m).numpy()
            dm, dreal = [v.numpy() for v in readout.branches(m1)]
            dm0, d0 = [v.numpy() for v in readout.branches(torch.zeros_like(m1))]
        assert np.array_equal(dm, dm0)
        saved = q[SCORE].to_numpy(float)
        cache_score_error = float(np.abs(scache-saved).max())
        fresh_score_error = float(np.abs(s1-saved).max())
        assert cache_score_error < 1e-6 and fresh_score_error < 2e-5
        assert s1.argmax() == q.prediction and np.argmax(np.where(np.arange(8) == y, -np.inf, s1)) == rival
        zero = torch.zeros_like(r, requires_grad=True)
        gz, = torch.autograd.grad(readout.margin(zero, weight), zero)
        assert torch.isfinite(gz).all()
        target = float((s1[y]-s1[rival])-(s0[y]-s0[rival]))
        tolerance = 1e-5+1e-4*abs(target)
        for steps in [64, 128]:
            ig = integrate(readout, r, weight, steps)
            total = float(ig.sum())
            error = total-target
            convergence.append(dict(instance_id=identifier, steps=steps, target_change=target,
                ig_sum=total, closure_error=error, tolerance=tolerance, passed=abs(error)<=tolerance))
            if abs(error) <= tolerance:
                break
        passed = abs(error) <= tolerance
        ig_array = ig.numpy()
        zero_weight_max = float(np.abs(ig_array*np.broadcast_to(data['weight'] == 0, ig_array.shape)).max())
        assert zero_weight_max == 0
        mass, covered, aligned = aligned_map(ig_array, data['weight'])
        if int(q.epoch) == 0:
            assert total == 0 and not ig_array.any()
        decomp = dict(instance_id=identifier, B_margin=float(base[y]-base[rival]),
            mean_margin=float(dm[y]-dm[rival]), D0_margin=float(d0[y]-d0[rival]),
            Dreal_margin=float(dreal[y]-dreal[rival]),
            distribution_change=float((dreal-d0)[y]-(dreal-d0)[rival]),
            zero_moment_margin=float(s0[y]-s0[rival]), full_margin=float(s1[y]-s1[rival]))
        decomp['branch_closure_error'] = (decomp['B_margin']+decomp['mean_margin']+
            decomp['Dreal_margin']-decomp['full_margin'])
        info = dict(instance_id=identifier, outcome=outcome, fold=fold, seed=17, epoch=int(q.epoch),
            true_class=y, fixed_rival=rival, steps=steps, target_change=target, ig_sum=total,
            closure_error=error, tolerance=tolerance, passed=passed,
            VV_IG=float(mass[0].sum()), VH_IG=float(mass[1].sum()),
            cache_mean_error=mean_error, cache_moment_error=moment_error,
            cache_score_error=cache_score_error, fresh_score_error=fresh_score_error,
            descriptor_error=input_error, zero_weight_max=zero_weight_max,
            weighted_center_error=float((r*weight).sum((-2, -1)).abs().max()),
            class_center_error=float(r.mean(2).abs().max()),
            grid_h=r.shape[-2], grid_w=r.shape[-1], receptive_field=rf, output_stride=stride,
            valid_vv=bool(valid[idx, 0]), valid_vh=bool(valid[idx, 1]),
            VV_union_cells=int(covered[0].sum()), VH_union_cells=int(covered[1].sum()))
        for t in range(4):
            for p, pol in enumerate(['VV', 'VH']):
                w = data['weight'][t, p, 0]
                n_eff = 1/float(np.square(w).sum()) if w.sum() else 0
                details.append(dict(instance_id=identifier, rotation=t, pol=pol,
                    positive_weight_cells=int((w>0).sum()), n_eff=n_eff,
                    ig_sum=float(ig_array[t, p].sum())))
                for c in range(8):
                    class_rows.append(dict(instance_id=identifier, rotation=t, pol=pol,
                        response_class=c, ig_sum=float(ig_array[t, p, c].sum())))
        scores = {'B': base, 'B+mean': base+dm, 'B+mean+D0': s0, 'full': s1}
        path = panel(row, q, data, mass, covered, scores, decomp, outcome, info)
        case_rows.append(dict(**info, panel=path, image_vv=row.image_vv, image_vh=row.image_vh,
            mean_source='unchanged cached outer mean', baseline='same head with raw moments=0'))
        closure.append(dict(**info, **{k:v for k,v in decomp.items() if k!='instance_id'}))
        np.savez_compressed(HERE / 'arrays' / f'{outcome}.npz', r=data['r'], weight=data['weight'],
            mean_per_view=data['mean'], moments_per_view=data['moments'], ig=ig_array,
            aligned_ig=aligned, aligned_mass=mass, covered=covered, fixed_mean=fixed_mean,
            base=base, zero_moment_score=s0, full_score=s1, cached_full_score=saved,
            mean_delta=dm, distribution0_delta=d0, distributionreal_delta=dreal)
        print(identifier, 'epoch', int(q.epoch), 'target', target, 'IG', total,
              'error', error, 'steps', steps, 'replay', fresh_score_error, flush=True)
    for name, rows in [('closure', closure), ('view_support', details), ('ig_convergence', convergence),
                       ('review_cases', case_rows), ('class_attribution', class_rows)]:
        pd.DataFrame(rows).to_csv(HERE / f'{name}.csv', index=False)
    summary = dict(cases=4, new_fits=0, images=sum(int(q['valid_vv'])+int(q['valid_vh']) for q in closure),
        device=str(device), seconds=time.perf_counter()-start, max_steps=max(q['steps'] for q in closure),
        all_completeness_passed=all(q['passed'] for q in closure),
        max_closure_error=max(abs(q['closure_error']) for q in closure),
        max_fresh_score_error=max(q['fresh_score_error'] for q in closure),
        max_cache_moment_error=max(q['cache_moment_error'] for q in closure),
        grid=[closure[0]['grid_h'], closure[0]['grid_w']], theoretical_receptive_field=rf,
        native_pixel_attribution=False, physical_causality=False)
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
