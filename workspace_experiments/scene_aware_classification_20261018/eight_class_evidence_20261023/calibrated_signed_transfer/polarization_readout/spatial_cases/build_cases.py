"""Four fixed-head spatial kappa decompositions; original extraction batches only."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
POL = HERE.parent
sys.path.insert(0, str(POL))
sys.path.insert(0, str(POL.parent))
import run_polarization as head
import extract_moments as extract

CASES = [('pol_preserved', 'pair_0003651', 'rescued'),
         ('pol_preserved_capped', 'pair_0007746', 'harmed'),
         ('pol_preserved', 'pair_0003494', 'CC'),
         ('pol_preserved', 'pair_0003504', 'WW')]


def theoretical_rf(model):
    receptive, jump = 1, 1
    layers = [model.conv1, model.maxpool]
    for stage in [model.layer1, model.layer2, model.layer3]:
        for block in stage:
            layers.extend([block.conv1, block.conv2, block.conv3])
    for layer in layers:
        k, s = layer.kernel_size, layer.stride
        k = k[0] if isinstance(k, tuple) else k
        s = s[0] if isinstance(s, tuple) else s
        receptive += (k-1)*jump
        jump *= s
    return receptive, jump


def native_input(row, pol, snapshots, coverage, valid):
    view = snapshots[int(row.obs_id)]['images'][pol]
    with Image.open(row[f'image_{pol}']) as im:
        raw = np.array(im.convert('L'))
    assert raw.shape == (view['h'], view['w'])
    pixels, masks, place = extract.crop_pad(raw, extract.masks_for(view), 160)
    old = coverage.loc[(row.instance_id, pol)]
    assert bool(masks[0].any() and not old.disabled_alignment) == bool(valid)
    assert all(value == old[key] for key, value in place.items())
    corners = np.asarray(view['priors']['detections'][view['priors']['target_index']]['corners'])
    canvas_corners = corners-[place['crop_x'], place['crop_y']]+[place['pad_left'], place['pad_top']]
    return dict(raw=raw, pixels=pixels, support=masks.sum(0), place=place,
                corners=corners, canvas_corners=canvas_corners, path=str(row[f'image_{pol}']))


def draw(case, observations, margin_maps, support, factors, parts, old, new, base, score):
    truth = int(case.class_id)
    limit = max(float(np.abs(margin_maps).max()), 1e-12)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.5), layout='constrained')
    for pol, name in enumerate(['VV', 'VH']):
        obs = observations[pol]
        axes[pol, 0].imshow(obs['raw'], cmap='gray', vmin=0, vmax=255, interpolation='nearest',
                            extent=(0, obs['raw'].shape[1], obs['raw'].shape[0], 0))
        axes[pol, 0].set_title(f'{name} native {obs["raw"].shape[1]}x{obs["raw"].shape[0]}')
        axes[pol, 1].imshow(obs['pixels'], cmap='gray', vmin=0, vmax=255, interpolation='nearest', extent=(0, 160, 160, 0))
        axes[pol, 1].set_title(f'{name} actual 160 crop/pad')
        values = np.ma.masked_where(support[pol] <= 0, margin_maps[pol])
        cmap = plt.get_cmap('RdBu_r').copy()
        cmap.set_bad('#dddddd')
        mesh = axes[pol, 2].pcolormesh(np.arange(11)*16, np.arange(11)*16, values,
            cmap=cmap, vmin=-limit, vmax=limit, shading='flat', edgecolors='#999999', linewidth=.45)
        axes[pol, 2].set(xlim=(0, 160), ylim=(160, 0), aspect='equal')
        integral = margin_maps[pol].sum()
        axes[pol, 2].set_title(f'{name}: gated margin, class {truth} minus {new}\n'
            f'10x10 cells; sum={integral:+.6f}; positive favors truth')
        fig.colorbar(mesh, ax=axes[pol, 2], shrink=.75, label='Logit-margin increment per cell')
        for j, corners in [(0, obs['corners']), (1, obs['canvas_corners']), (2, obs['canvas_corners'])]:
            pts = np.r_[corners, corners[:1]]
            axes[pol, j].plot(pts[:, 0], pts[:, 1], color='#10b949', lw=1.3)
            axes[pol, j].set_xlabel('Pixel x (native)' if j == 0 else 'Pixel x (160 canvas)')
            axes[pol, j].set_ylabel('Pixel y')
        axes[pol, 1].text(.01, -.19,
            f'gate truth/rival={factors["gate"][truth, pol]:+.4f}/{factors["gate"][new, pol]:+.4f}; '
            f'cap={factors["cap"][truth]:.4f}/{factors["cap"][new]:.4f}\n'
            f'saved parts truth/rival={parts[truth, pol]:+.6f}/{parts[new, pol]:+.6f}',
            transform=axes[pol, 1].transAxes, fontsize=9)
    old_a, old_b = base[truth]-base[old], score[truth]-score[old]
    new_a, new_b = base[truth]-base[new], score[truth]-score[new]
    fig.suptitle(f'{case.instance_id} | {case.arm} | seed17 | {case.outcome} | fold{int(case.fold)} epoch{int(case.epoch)}\n'
        f'truth={truth}; B prediction={int(case.base_prediction)} -> final={int(case.prediction)}; '
        f'old rival={old}, new rival={new}\n'
        f'old-rival margin {old_a:+.6f} -> {old_b:+.6f}; new-rival margin {new_a:+.6f} -> {new_b:+.6f}\n'
        'C4 inverse-mapped cell masses; fixed gates and sigma. Green=OBB; gray=no footprint. Not pixel-causal attribution.',
        fontsize=12)
    path = HERE/f'{case.arm}_{case.instance_id}.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    table = head.parent.load_table()
    cache = dict(np.load(extract.HERE/'signed_features.npz'))
    oldh = np.load(extract.PARENT/'features.npz', allow_pickle=True)['layer3']
    snapshots = {s['obs_id']: s for s in json.loads((extract.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf8'))}
    coverage = pd.read_csv(extract.PARENT/'coverage.csv').set_index(['instance_id', 'pol'])
    predictions = pd.read_csv(POL/'predictions.csv')
    predictions = predictions[predictions.role.eq('test') & predictions.seed.eq(17)]
    lookup = {ident: i for i, ident in enumerate(table.instance_id)}
    records, gated = {}, {}
    for arm, ident, outcome in CASES:
        case = predictions[predictions.arm.eq(arm) & predictions.instance_id.eq(ident)].iloc[0]
        assert case.outcome == outcome
        records[ident] = case
        group_key = (int(case.fold), arm)
        if group_key in gated:
            continue
        # Restore the full original test batch for the CPU head, not a single-case GEMM.
        with np.load(extract.BASELINE/f'fold{int(case.fold)}_inputs.npz') as z:
            ix = z['test']
        saved_rows = predictions[predictions.arm.eq(arm) & predictions.fold.eq(case.fold)].set_index('instance_id').loc[table.iloc[ix].instance_id]
        base = saved_rows[[f'b{c}' for c in range(8)]].to_numpy()
        ri = list(cache['reader_names']).index(f'fold{int(case.fold)}_outer')
        ck, model = head.restore(POL/'models'/f'fold{int(case.fold)}_{arm}_seed17.pt')
        x, carrier, _, _ = head.inputs(base, cache['moments_pol'][ri, ix], cache['valid'][ix], arm, ck['mu'], ck['scale'])
        with torch.inference_mode():
            gate = model.net(x).tanh()
            cap = carrier.sum(-1).abs()/carrier.abs().sum(-1).clamp_min(1e-12) if arm == 'pol_preserved_capped' else torch.ones_like(carrier[..., 0])
            parts = model.contributions(x, carrier)
        expected = np.stack([saved_rows[[f'{pol}_delta{c}' for c in range(8)]].to_numpy() for pol in ['vv', 'vh']], -1)
        replay_error = float(abs(parts.numpy()-expected).max())
        assert replay_error < 2e-6
        gated[group_key] = dict(rows={int(i): j for j, i in enumerate(ix)}, gate=gate.numpy(), cap=cap.numpy(),
            parts=parts.numpy(), replay_error=replay_error)
    # Keep all 15 reader axes and the original valid-pol ordering for identical extraction numerics.
    names, weights = list(cache['reader_names']), []
    for name in names:
        with np.load(extract.BASELINE/'models'/f'{name}.npz') as z:
            weights.append(z['W'].reshape(8, 2, 1024).transpose(1, 0, 2))
    w = torch.tensor(np.stack(weights), device='cuda', dtype=torch.float64)
    model = extract.resnet50(weights=None)
    model.load_state_dict(torch.load(extract.WEIGHTS, map_location='cpu', weights_only=True))
    model = model.cuda().float().eval().requires_grad_(False)
    rf, stride = theoretical_rf(model)
    mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
    batches = sorted({lookup[ident]//16*16 for ident in records})
    data, batch_rows = {}, []
    for ident, case in records.items():
        data[ident] = dict(local=np.zeros((2, 4, 8, 10, 10)), r=np.zeros((2, 4, 8, 10, 10)),
            weight=np.zeros((2, 4, 10, 10)), moments=np.zeros((2, 4, 8, 2)), observations={}, gap_error=0.)
    with torch.inference_mode():
        for first in batches:
            images, footprints, addresses = [], [], []
            for i in range(first, min(first+16, len(table))):
                row = table.iloc[i]
                for pol, name in enumerate(['vv', 'vh']):
                    obs = native_input(row, name, snapshots, coverage, cache['valid'][i, pol])
                    if row.instance_id in data:
                        data[row.instance_id]['observations'][pol] = obs
                    if not cache['valid'][i, pol]:
                        continue
                    images.append(obs['pixels'][None].astype(np.float32)/255)
                    footprints.append(obs['support'][None].astype(np.float32))
                    addresses.append((i, pol))
            x = torch.tensor(np.stack(images), device='cuda')
            support = torch.tensor(np.stack(footprints), device='cuda')
            wp = w[:, torch.tensor([p for _, p in addresses], device='cuda')]
            gaps = []
            for rot in range(4):
                xx = x.rot90(rot, (-2, -1)).expand(-1, 3, -1, -1)
                mask = support.rot90(rot, (-2, -1))
                z = model.relu(model.bn1(model.conv1((xx-mean)/std)))
                z = model.layer3(model.layer2(model.layer1(model.maxpool(z))))
                gaps.append(extract.original_pool(z, mask, maximum=False))
                weight = F.interpolate(mask, size=z.shape[-2:], mode='area').flatten(2).double()[None]
                response = torch.einsum('fbck,bkp->fbcp', wp, z.flatten(2).double())
                moments, _, r = extract.weighted_moments(response, weight)
                norm = weight/weight.sum(-1, keepdim=True)
                local = norm*r*r.abs()/moments[..., 0:1].clamp_min(1e-12)
                for b, (i, pol) in enumerate(addresses):
                    ident = table.iloc[i].instance_id
                    if ident not in data:
                        continue
                    ri = names.index(f'fold{int(records[ident].fold)}_outer')
                    q = data[ident]
                    q['local'][pol, rot] = local[ri, b].cpu().numpy().reshape(8, 10, 10)
                    q['r'][pol, rot] = r[ri, b].cpu().numpy().reshape(8, 10, 10)
                    q['weight'][pol, rot] = norm[0, b, 0].cpu().numpy().reshape(10, 10)
                    q['moments'][pol, rot] = moments[ri, b].cpu().numpy()
            gap = torch.stack(gaps).mean(0).cpu().numpy()
            for b, (i, pol) in enumerate(addresses):
                ident = table.iloc[i].instance_id
                if ident in data:
                    error = float(abs(gap[b]-oldh[i, pol*1024:(pol+1)*1024]).max())
                    data[ident]['gap_error'] = max(data[ident]['gap_error'], error)
            batch_rows.append(dict(first_row=first, objects=min(16, len(table)-first), valid_pol_images=len(addresses), rotations=4))
            print('original batch', first, 'valid images', len(addresses), flush=True)
    checks, margins, cells, cases, panels, archives = [], [], [], [], [], {}
    for ident, case in records.items():
        i, truth = lookup[ident], int(case.class_id)
        ri = names.index(f'fold{int(case.fold)}_outer')
        q = data[ident]
        cached_moments = cache['moments_pol'][ri, i]
        fresh = q['moments'].mean(1)
        moment_error = float(abs(fresh-cached_moments).max())
        assert moment_error < 1e-8 and q['gap_error'] < 1e-4, (ident, moment_error, q['gap_error'])
        pack = gated[(int(case.fold), case.arm)]
        j = pack['rows'][i]
        gate, cap = pack['gate'][j].astype(float), pack['cap'][j].astype(float)
        factors = gate*cap[:, None]
        gated_rotations = q['local']*factors.T[:, None, :, None, None]
        mapped = np.stack([np.rot90(gated_rotations[:, k], -k, axes=(-2, -1)) for k in range(4)], 1).mean(1)
        footprint = np.stack([np.rot90(q['weight'][:, k], -k, axes=(-2, -1)) for k in range(4)], 1).mean(1)
        saved_parts = np.stack([case[[f'{pol}_delta{c}' for c in range(8)]].to_numpy(float) for pol in ['vv', 'vh']], -1)
        part_integrals = mapped.sum((-2, -1)).T
        rounding = saved_parts-part_integrals
        part_error = float(abs(rounding).max())
        assert part_error < 1e-6, (ident, part_error)
        base, score = [case[[f'{prefix}{c}' for c in range(8)]].to_numpy(float) for prefix in ['b', 's']]
        old = max((c for c in range(8) if c != truth), key=lambda c: base[c])
        new = max((c for c in range(8) if c != truth), key=lambda c: score[c])
        for c in range(8):
            for pol, name in enumerate(['vv', 'vh']):
                checks.append(dict(instance_id=ident, arm=case.arm, pol=name, class_id=c,
                    cached_sigma=cached_moments[pol, c, 0], fresh_sigma=fresh[pol, c, 0],
                    cached_kappa=cached_moments[pol, c, 1], local_kappa_integral=q['local'][pol, :, c].sum((-2, -1)).mean(),
                    gate=gate[c, pol], cap=cap[c], saved_part=saved_parts[c, pol], map_integral=part_integrals[c, pol],
                    fp32_rounding_residual=rounding[c, pol]))
        for rival, label in [(old, 'old_rival'), (new, 'new_rival')]:
            mm = mapped[:, truth]-mapped[:, rival]
            expected = saved_parts[truth]-saved_parts[rival]
            actual_change = (score[truth]-score[rival])-(base[truth]-base[rival])
            assert abs(mm.sum()-actual_change) < 2e-6
            margins.append(dict(instance_id=ident, arm=case.arm, comparison=label, truth=truth, rival=rival,
                vv_integral=mm[0].sum(), vh_integral=mm[1].sum(), saved_vv=expected[0], saved_vh=expected[1],
                total_integral=mm.sum(), before_margin=base[truth]-base[rival], after_margin=score[truth]-score[rival],
                saved_margin_change=actual_change, integral_error=abs(mm.sum()-actual_change)))
            for pol, name in enumerate(['vv', 'vh']):
                for yy in range(10):
                    for xx in range(10):
                        cells.append(dict(instance_id=ident, comparison=label, rival=rival, pol=name,
                            row=yy, col=xx, x0=16*xx, y0=16*yy, x1=16*(xx+1), y1=16*(yy+1),
                            footprint_mass=footprint[pol, yy, xx], contribution=mm[pol, yy, xx]))
        margin_maps = mapped[:, truth]-mapped[:, new]
        panels.append(draw(case, q['observations'], margin_maps, footprint, dict(gate=gate, cap=cap), saved_parts, old, new, base, score))
        cases.append(dict(instance_id=ident, arm=case.arm, seed=17, fold=int(case.fold), epoch=int(case.epoch),
            truth=truth, old_rival=old, new_rival=new, rival_changed=old != new, outcome=case.outcome,
            B_prediction=int(case.base_prediction), final_prediction=int(case.prediction), row_index=i, original_batch_first=i//16*16,
            moment_max_error=moment_error, gap_max_error=q['gap_error'], parts_max_error=part_error,
            head_full_test_replay_error=pack['replay_error'], vv_native=q['observations'][0]['path'], vh_native=q['observations'][1]['path'],
            vv_shape=str(q['observations'][0]['raw'].shape), vh_shape=str(q['observations'][1]['raw'].shape),
            vv_active_cells=int((footprint[0] > 0).sum()), vh_active_cells=int((footprint[1] > 0).sum())))
        for key, value in dict(local_kappa_rot=q['local'], r_rot=q['r'], normalized_footprint_rot=q['weight'],
            moments_rot=q['moments'], gate=gate, cap=cap, gated_inverse_c4_mean=mapped,
            footprint_inverse_c4_mean=footprint, saved_parts=saved_parts, fp32_rounding_residual=rounding).items():
            archives[f'{ident}_{key}'] = value
    np.savez_compressed(HERE/'case_maps.npz', **archives)
    for name, rows in [('cases', cases), ('class_pol_closure', checks), ('margin_closure', margins),
                       ('coarse_cells', cells), ('extraction_batches', batch_rows)]:
        pd.DataFrame(rows).to_csv(HERE/f'{name}.csv', index=False)
    receipt = dict(status='complete', cases=4, panels=panels, training=False, original_batches=batch_rows,
        theory_receptive_field=rf, stride=stride, coarse_shape=[10, 10], pixels=160,
        max_moment_error=max(c['moment_max_error'] for c in cases), max_gap_error=max(c['gap_max_error'] for c in cases),
        max_parts_error=max(c['parts_max_error'] for c in cases), max_margin_error=max(m['integral_error'] for m in margins),
        max_head_replay_error=max(c['head_full_test_replay_error'] for c in cases),
        seconds=time.perf_counter()-started, gpu_peak_bytes=torch.cuda.max_memory_allocated(),
        numerical_policy='No spatial rescaling to force closure; explicit saved-parts minus map integral rounding residual retained.',
        causal_scope='Fixed observed gates, cap, sigma and weighted class/space centering; not pixel intervention.')
    (HERE/'summary.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    print(pd.DataFrame(margins).to_string(index=False), flush=True)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
