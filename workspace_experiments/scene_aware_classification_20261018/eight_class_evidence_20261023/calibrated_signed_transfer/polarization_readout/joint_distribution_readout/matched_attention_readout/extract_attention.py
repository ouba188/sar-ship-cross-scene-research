"""Two bounded extraction stages; no full feature-map cache and no reader fits."""
from common import *
import argparse
import time
from PIL import Image
from torchvision.models import resnet50
import torch.nn.functional as F


def attention(r, weight, tau):
    norm = weight / weight.sum(-1, keepdim=True)
    logw = torch.where(norm > 0, norm.log(), -torch.inf)
    logits = r / tau
    ap = (logw + logits).softmax(-1)
    am = (logw - logits).softmax(-1)
    qp, qm = (ap*r).sum(-1), (am*r).sum(-1)
    return torch.stack(((qp-qm)/2, (qp+qm)/2), -1), ap, am


def structural():
    r = torch.tensor([[[-2., 0., 1., 900.]]], dtype=torch.float64)
    w = torch.tensor([[[1., 1., 2., 0.]]], dtype=torch.float64)
    a, p, m = attention(r, w, 1.)
    v = r.numpy()[0, 0, :3]
    weights = np.array([.25, .25, .5])
    positive = weights*np.exp(v)
    negative = weights*np.exp(-v)
    qp = (positive*v).sum()/positive.sum()
    qm = (negative*v).sum()/negative.sum()
    expected = np.array([(qp-qm)/2, (qp+qm)/2])
    err = float(abs(a.numpy()[0, 0]-expected).max())
    assert err < 1e-12 and p[..., -1].item() == 0 and m[..., -1].item() == 0
    rev, _, _ = attention(-r, w, 1.)
    assert torch.allclose(rev, a*torch.tensor([1., -1.]), atol=1e-12, rtol=1e-12)
    perm = [2, 0, 3, 1]
    permuted, _, _ = attention(r[..., perm], w[..., perm], 1.)
    assert torch.allclose(a, permuted, atol=1e-12, rtol=1e-12)
    moment, mu, rr = extract.weighted_moments(torch.ones_like(r)*8, w)
    zero, _, _ = attention(rr, w, 1.)
    assert not zero.any()
    return dict(direct_formula_error=err, reversed=True, joint_permutation=True,
        zero_weight_excluded=True, constant_response_zero=True)


def main(stage):
    assert (HERE/'PROTOCOL_ZH.md').exists()
    if stage == 'test':
        assert (HERE/'source_report_frozen.json').exists()
    outpath = HERE/f'attention_{stage}.npz'
    assert not outpath.exists(), 'Do not overwrite completed extraction.'
    tick = time.perf_counter()
    event('extraction_start', stage=stage)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    checks = structural()
    table, cache = parent.load_table(), cache_load()
    np.testing.assert_array_equal(cache['instance_ids'], table.instance_id)
    names, valid = list(cache['reader_names']), cache['valid'].astype(bool)
    n, nr = len(table), len(names)
    required = np.zeros((nr, n), bool)
    tau = np.zeros((3, 2))
    scopes, origins = [], []
    for name in names:
        with np.load(parent.PARENT/'baseline/models'/f'{name}.npz') as z:
            origins.append({k: z[k] for k in ('W', 'intercept', 'fit') + (('held',) if 'inner' in name else ())})
    for fold in range(3):
        with np.load(parent.PARENT/'baseline'/f'fold{fold}_inputs.npz') as z:
            roles = ('train', 'validation') if stage == 'source' else ('test',)
            d = {k: z[k] for k in roles}
        for role, ix in d.items():
            scopes.append(dict(stage=stage, fold=fold, role=role, n=len(ix)))
        if stage == 'source':
            sig = np.zeros((n, 2, 8))
            count = np.zeros(n, int)
            for j in range(4):
                ri = names.index(f'fold{fold}_inner{j}')
                held, fit = origins[ri]['held'], origins[ri]['fit']
                assert set(held) <= set(d['train']) and set(fit) <= set(d['train'])
                assert not set(table.iloc[held].mmsi) & set(table.iloc[fit].mmsi)
                assert not set(table.iloc[held]['product']) & set(table.iloc[fit]['product'])
                required[ri, held] = True
                count[held] += 1
                sig[held] = cache['moments_pol'][ri, held, :, :, 0]
            assert (count[d['train']] == 1).all() and count.sum() == len(d['train'])
            for pol in range(2):
                tau[fold, pol] = np.median(sig[d['train']][valid[d['train'], pol], pol])
            required[names.index(f'fold{fold}_outer'), d['validation']] = True
        else:
            required[names.index(f'fold{fold}_outer'), d['test']] = True
    if stage == 'source':
        assert np.isfinite(tau).all() and (tau > 0).all()
        write_json('scales.json', dict(created_utc=now(), tau=tau.tolist(),
            definition='median C4-averaged sigma over source OOF train valid pol samples and 8 classes'))
    else:
        tau = np.array(json.loads((HERE/'scales.json').read_text())['tau'])
    chosen = np.flatnonzero(required.any(0))
    assert len(chosen) == (8750 if stage == 'source' else 4744)
    pd.DataFrame(scopes).to_csv(HERE/f'{stage}_extraction_scope.csv', index=False)
    snapshots = {s['obs_id']: s for s in json.loads((run.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    coverage = pd.read_csv(parent.PARENT/'coverage.csv').set_index(['instance_id', 'pol'])
    with np.load(parent.PARENT/'features.npz', allow_pickle=True) as z:
        oldh = z['layer3']
        np.testing.assert_array_equal(z['valid'], valid)
    att = np.zeros((nr, n, 2, 8, 2), float)
    means = np.zeros((nr, n, 2, 8), float)
    moments = np.zeros_like(att)
    seen = np.zeros((n, 2), int)
    model = resnet50(weights=None)
    model.load_state_dict(torch.load(extract.WEIGHTS, map_location='cpu', weights_only=True))
    model = model.cuda().float().eval().requires_grad_(False)
    w = torch.tensor(np.stack([o['W'].reshape(8, 2, 1024).transpose(1, 0, 2) for o in origins]), device='cuda', dtype=torch.float64)
    norm_mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
    norm_std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
    tau_tensor = torch.tensor(tau, device='cuda', dtype=torch.float64)
    diag, gapchecks = [], []

    @torch.inference_mode()
    def batch_forward(pixels, footprint, addresses, diagnostics=True):
        x = torch.tensor(pixels, device='cuda')
        support = torch.tensor(footprint, device='cuda')
        a = np.zeros((nr, len(addresses), 8, 2))
        mu = np.zeros((nr, len(addresses), 8))
        mom = np.zeros_like(a)
        gaps = []
        ids = np.array([i for i, p in addresses])
        pols = np.array([p for i, p in addresses])
        for rotation in range(4):
            xx = x.rot90(rotation, (-2, -1)).expand(-1, 3, -1, -1)
            mask = support.rot90(rotation, (-2, -1))
            z = model.relu(model.bn1(model.conv1((xx-norm_mean)/norm_std)))
            z = model.layer3(model.layer2(model.layer1(model.maxpool(z))))
            gaps.append(extract.original_pool(z, mask, maximum=False))
            weight = F.interpolate(mask, size=z.shape[-2:], mode='area').flatten(2).double()
            for ri, name in enumerate(names):
                take = np.flatnonzero(required[ri, ids])
                if not len(take):
                    continue
                tp = torch.tensor(pols[take], device='cuda')
                response = torch.einsum('bck,bku->bcu', w[ri, tp], z[take].flatten(2).double())
                ww = weight[take]
                mm, uu, r = extract.weighted_moments(response, ww)
                scale = tau_tensor[int(name[4]), tp, None, None]
                aa, ap, am = attention(r, ww, scale)
                assert torch.isfinite(aa).all()
                assert (aa[..., 0] >= -1e-10).all() and (aa[..., 1].abs() <= aa[..., 0]+1e-10).all()
                assert (ap.sum(-1)-1).abs().max() < 1e-12
                assert (am.sum(-1)-1).abs().max() < 1e-12
                assert not ap.masked_select(ww.expand_as(ap) == 0).any()
                assert not am.masked_select(ww.expand_as(am) == 0).any()
                a[ri, take] += aa.cpu().numpy()/4
                mu[ri, take] += uu.cpu().numpy()/4
                mom[ri, take] += mm.cpu().numpy()/4
                if diagnostics:
                    norm = ww/ww.sum(-1, keepdim=True)
                    lognorm = norm.clamp_min(1e-300).log()
                    for pol in np.unique(pols[take]):
                        choose = pols[take] == pol
                        for label, prob in [('plus', ap), ('minus', am)]:
                            kl = (prob*(prob.clamp_min(1e-300).log()-lognorm)).sum(-1)[choose]
                            diag.append(dict(reader=name, pol=int(pol), rotation=rotation, direction=label,
                                n=int(choose.sum()), ratio_absmax=float((r/scale)[choose].abs().max()),
                                kl_mean=float(kl.mean()), max_mass_mean=float(prob[choose].max(-1).values.mean())))
        return a, mu, mom, torch.stack(gaps).mean(0).cpu().numpy()

    last = time.perf_counter()
    forward_tick = time.perf_counter()
    replay_input = None
    batches = 0
    for first in range(0, n, 16):
        ids = [i for i in range(first, min(first+16, n)) if required[:, i].any()]
        if not ids:
            continue
        images, footprints, addresses = [], [], []
        for i in ids:
            row = table.iloc[i]
            for pol, label in enumerate(['vv', 'vh']):
                view = snapshots[int(row.obs_id)]['images'][label]
                with Image.open(row[f'image_{label}']) as im:
                    raw = np.array(im.convert('L'))
                assert raw.shape == (view['h'], view['w'])
                pix, masks, place = extract.crop_pad(raw, extract.masks_for(view), 160)
                old = coverage.loc[(row.instance_id, label)]
                assert bool(masks[0].any() and not old.disabled_alignment) == bool(valid[i, pol])
                assert all(value == old[key] for key, value in place.items())
                if not valid[i, pol]:
                    continue
                images.append(pix[None].astype(np.float32)/255)
                footprints.append(masks.sum(0)[None].astype(np.float32))
                addresses.append((i, pol))
        pixels, footprint = np.stack(images), np.stack(footprints)
        aa, mm, ss, gg = batch_forward(pixels, footprint, addresses)
        if replay_input is None:
            replay_input = (pixels[:1], footprint[:1], addresses[:1], aa[:, :1].copy())
        for b, (i, pol) in enumerate(addresses):
            att[:, i, pol], means[:, i, pol], moments[:, i, pol] = aa[:, b], mm[:, b], ss[:, b]
            err = float(abs(gg[b]-oldh[i, pol*1024:(pol+1)*1024]).max())
            assert err < 1e-4
            gapchecks.append(dict(instance_id=table.iloc[i].instance_id, pol=pol, max_error=err))
            seen[i, pol] += 1
        batches += 1
        if time.perf_counter()-last > 25:
            print(stage, 'objects', int((seen[:, 0]>0).sum()), '/', len(chosen), 'seconds', round(time.perf_counter()-tick, 1), flush=True)
            last = time.perf_counter()
    torch.cuda.synchronize()
    pixel_seconds = time.perf_counter()-forward_tick
    assert np.array_equal(seen[chosen], valid[chosen].astype(int))
    checks_rows = []
    for ri, name in enumerate(names):
        ix = np.flatnonzero(required[ri])
        if not len(ix):
            continue
        em = float(abs(means[ri, ix]-cache['mean_pol'][ri, ix]).max())
        es = float(abs(moments[ri, ix]-cache['moments_pol'][ri, ix]).max())
        assert em < 1e-4 and es < 1e-4
        before = cache['mean_pol'][ri, ix].sum(1)+origins[ri]['intercept']
        after = means[ri, ix].sum(1)+origins[ri]['intercept']
        flips = int((before.argmax(1) != after.argmax(1)).sum())
        assert flips == 0
        checks_rows.append(dict(reader=name, n=len(ix), mean_error=em, moment_error=es, reader_top1_changes=flips))
    px, fp, ad, expected = replay_input
    replay = batch_forward(px, fp, ad, diagnostics=False)[0]
    replay_error = float(abs(replay-expected).max())
    assert replay_error < 1e-4
    assert not att[:, ~valid].any()
    np.savez_compressed(outpath, reader_names=cache['reader_names'], instance_ids=cache['instance_ids'],
        valid=valid, attention_pol=att, computed=required, tau=tau)
    pd.DataFrame(checks_rows).to_csv(HERE/f'{stage}_reader_replay.csv', index=False)
    pd.DataFrame(gapchecks).to_csv(HERE/f'{stage}_gap_replay.csv', index=False)
    pd.DataFrame(diag).to_csv(HERE/f'{stage}_attention_diagnostics.csv', index=False)
    summary = dict(stage=stage, completed_utc=now(), objects=len(chosen), pol_images=int(seen.sum()),
        reader_roles=int(required.sum()), batches=batches, total_seconds=time.perf_counter()-tick,
        pixel_loop_seconds=pixel_seconds, gpu_peak_bytes=torch.cuda.max_memory_allocated(),
        single_vs_batch_attention_error=replay_error, gap_max_error=max(x['max_error'] for x in gapchecks),
        mean_max_error=max(x['mean_error'] for x in checks_rows), moment_max_error=max(x['moment_error'] for x in checks_rows),
        reader_top1_changes=0, structural=checks, gpu=torch.cuda.get_device_name(0), reader_fits=0)
    write_json(f'{stage}_extraction.json', summary)
    event('extraction_complete', **summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['source', 'test'])
    main(parser.parse_args().stage)
