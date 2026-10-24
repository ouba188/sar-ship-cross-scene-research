"""Native-image scale interventions on a frozen supplementary readout."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision.models import resnet50

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from extract_tokens import geo, slots, cell_mass
from match_readout import unit, sinkhorn, pair_score, REGIONS
from train_readout import Head, UP

BANDS = ('fine', 'middle', 'coarse')
SEEDS = (17, 29, 43)


def place(field, placement):
    out = np.zeros((160, 160), np.float32)
    x, y = placement['crop_x'], placement['crop_y']
    w, h = placement['crop_width'], placement['crop_height']
    dx, dy = placement['pad_left'], placement['pad_top']
    out[dy:dy+h, dx:dx+w] = field[y:y+h, x:x+w]
    return out


def load_view(table, snapshots, i, p):
    pix, fp, geometry, placement = geo.native(table, snapshots, i, p)
    with Image.open(table.iloc[i]['image_'+('vv', 'vh')[p]]) as im:
        raw = np.array(im.convert('L'), dtype=np.float32)/255
    g1, g4, g12 = [gaussian_filter(raw, sigma=s, mode='reflect') for s in (1, 4, 12)]
    bands = (raw-g1, g1-g4, g4-g12)
    region = geo.soft_regions(geometry)*fp
    return pix.astype(np.float32)/255, fp, geometry, placement, bands, (region[0], region[1]+region[2])


def window(field, mask):
    return mask*(field-(field*mask).sum()/max(mask.sum(), 1e-10))


def perturbations(pix, placement, field, mask, obs, pol, band):
    delta = -.25*window(place(field, placement), mask)
    real = np.clip(pix+delta, 0, 1)
    target = np.linalg.norm(real-pix)
    items = [('attenuate', real, delta)]
    for seed in (101, 307):
        rng = np.random.default_rng(np.random.SeedSequence([seed, obs, pol, band]))
        white = np.fft.rfft2(rng.normal(size=field.shape))
        phase = white/np.maximum(abs(white), 1e-20)
        scrambled = np.fft.irfft2(abs(np.fft.rfft2(field))*phase, s=field.shape)
        d = window(place(scrambled, placement), mask)
        d = d/max(np.linalg.norm(d), 1e-12)
        lo, hi = 0., max(float(target)*2, 1e-9)
        while np.linalg.norm(np.clip(pix+hi*d, 0, 1)-pix) < target:
            hi *= 2
            if hi > 1e6:
                raise RuntimeError('Cannot match clipped control energy')
        for _ in range(30):
            mid = (lo+hi)/2
            if np.linalg.norm(np.clip(pix+mid*d, 0, 1)-pix) < target:
                lo = mid
            else:
                hi = mid
        dd = ((lo+hi)/2)*d
        items.append((f'phase{seed}', np.clip(pix+dd, 0, 1), dd))
    output = []
    for kind, image, desired in items:
        actual = image-pix
        energy = float(np.linalg.norm(actual))
        if kind != 'attenuate':
            assert abs(energy-target) < 2e-5, (energy, target)
        assert np.max(abs(actual[mask == 0]), initial=0) == 0
        output.append((kind, image.astype(np.float32), dict(l2=energy,
            region_rms=energy/np.sqrt(max((mask > 0).sum(), 1)),
            mean_change=float(actual.sum()/max(mask.sum(), 1)),
            clipped_fraction=float(((pix+desired < 0) | (pix+desired > 1))[mask > 0].mean()))))
    return output


class Extractor:
    def __init__(self, projection):
        model = resnet50(weights=None)
        model.load_state_dict(torch.load(geo.WEIGHTS, map_location='cpu', weights_only=True))
        self.model = model.cuda().eval().requires_grad_(False)
        self.mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
        self.projection = projection

    @torch.inference_mode()
    def __call__(self, images, masks):
        output = []
        for first in range(0, len(images), 24):
            ims = images[first:first+24]
            ms = masks[first:first+24]
            xx = torch.tensor(np.stack(ims)[:, None], device='cuda')
            raw = np.zeros((len(ims), 20, 256), np.float32)
            for k in range(4):
                v = xx.rot90(k, (-2, -1)).expand(-1, 3, -1, -1)
                z = self.model.relu(self.model.bn1(self.model.conv1((v-self.mean)/self.std)))
                z = self.model.layer1(self.model.maxpool(z))
                w = np.stack([cell_mass(np.rot90(m, k, (-2, -1)), 4) for m in ms]).reshape(len(ms), 20, -1)
                w = w/np.maximum(w.sum(-1, keepdims=True), 1e-20)
                desc = torch.einsum('btu,bdu->btd', torch.tensor(w, device='cuda', dtype=torch.float32), z.flatten(2))
                raw += desc.cpu().numpy()/4
            x = unit((raw-self.projection['mean'])@self.projection['components'].T).astype(np.float32)
            x[np.stack(ms).sum((-1, -2)) <= 0] = 0
            output.append(x)
        return np.concatenate(output)


@torch.inference_mode()
def independent(x, mass, reference_x, reference_mass):
    output = []
    for first in range(0, len(x), 64):
        sl = slice(first, first+64)
        q, a, s, b = [torch.tensor(v, device='cuda') for v in (x[sl], mass[sl], reference_x[sl], reference_mass[sl])]
        groups = []
        for ids in REGIONS:
            sim = torch.einsum('bpid,bckpjd->bckpij', q[:, :, ids], s[..., ids, :])
            qa, sa = a[:, None, None, :, ids], b[..., ids]
            t, _, _, _ = sinkhorn(sim, qa, sa)
            groups.append(pair_score(t, sim, qa, sa))
        output.append(torch.stack(groups, -2).mean(2).cpu().numpy())
    return np.concatenate(output)


def head_scores(local, base, checkpoint):
    z = base-base.mean(-1, keepdims=True)
    rival = np.stack([np.delete(z, c, axis=1).max(1) for c in range(8)], 1)
    raw = np.concatenate([local.reshape(len(local), 8, 12), z[..., None], rival[..., None], (z-rival)[..., None]], -1)
    v = np.where(checkpoint['active'], (raw-checkpoint['mu'])/checkpoint['scale'], 0).astype(np.float32)
    model = Head(); model.load_state_dict(checkpoint['state']); model.eval()
    with torch.inference_mode():
        return base+model(torch.tensor(v)).numpy()


def main():
    tick = time.perf_counter(); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    table = pd.read_csv(UP/'table.csv', dtype={'mmsi': str})
    z = np.load(ROOT/'tokens.npz'); pr = np.load(ROOT/'projection_layer1.npz')
    np.testing.assert_array_equal(z['instance_ids'].astype(str), table.instance_id)
    x = unit((z['layer1']-pr['mean'])@pr['components'].T).astype(np.float32)
    mass = z['mass']; x[mass <= 0] = 0
    eligible = table[table.role.eq('selection') & z['valid'].all(-1) & table.obs_id.ne(31237)]
    rng = np.random.default_rng(1024); chosen = []
    for c in range(8):
        pool = eligible[eligible.class_id.eq(c)]
        pool = pool.iloc[rng.permutation(len(pool))].drop_duplicates('mmsi')
        chosen.extend(pool.index[:12])
    diagnostic = table.index[table.obs_id.isin([135, 161, 258, 445, 921]) & table.role.eq('selection')].to_numpy()
    ids = np.unique(np.r_[chosen, diagnostic]).astype(int)
    cohort = table.iloc[ids].copy(); cohort['table_index'] = ids; cohort['main_sample'] = np.isin(ids, chosen)
    cohort.to_csv(HERE/'cohort.csv', index=False)
    snapshots = {int(s['obs_id']): s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    cached = np.load(ROOT/'scores_layer1.npz'); refs = cached['refs'][ids]
    np.savez_compressed(HERE/'reference_ids.npz', ids=ids, refs=refs)
    extractor = Extractor(pr); rows = []; changed = []; artifacts = []
    raw_replay = []
    for j, i in enumerate(ids):
        variants = [('raw', -1, 'none', 'none', 'none', {}, x[i])]
        for p in range(2):
            if not z['valid'][i, p]:
                continue
            pix, fp, g, placement, bands, regions = load_view(table, snapshots, i, p)
            masks = slots(g, fp)
            images = [pix]; meta = [None]
            for r, name in enumerate(('body', 'neighbor')):
                for b, band in enumerate(BANDS):
                    for kind, im, info in perturbations(pix, placement, bands[b], regions[r], int(table.iloc[i].obs_id), p, b):
                        images.append(im); meta.append((name, band, kind, info))
            desc = extractor(images, [masks]*len(images))
            err = float(abs(desc[0]-x[i, p]).max()); raw_replay.append(err)
            assert err < 2e-4, ('token replay', int(table.iloc[i].obs_id), p, err)
            for k, m in enumerate(meta[1:], 1):
                name, band, kind, info = m; item = x[i].copy(); item[p] = desc[k]
                variants.append((f'{p}_{name}_{band}_{kind}', p, name, band, kind, info, item))
            artifacts.append(dict(obs_id=int(table.iloc[i].obs_id), pol=p, path=table.iloc[i]['image_'+('vv', 'vh')[p]], placement=placement))
        for tag, p, region, band, kind, info, item in variants:
            rows.append(dict(local_index=j, table_index=int(i), obs_id=int(table.iloc[i].obs_id), class_id=int(table.iloc[i].class_id),
                mmsi=str(table.iloc[i].mmsi), pass_group=str(table.iloc[i].pass_group), main_sample=bool(i in chosen),
                variant=tag, pol=p, region=region, band=band, kind=kind, **info))
            changed.append(item)
        if j%8 == 0:
            print('EXTRACT', j+1, '/', len(ids), round(time.perf_counter()-tick, 1), flush=True)
    f = pd.DataFrame(rows); owners = f.local_index.to_numpy(int); changed = np.stack(changed)
    local = independent(changed, mass[ids[owners]], x[refs[owners]], mass[refs[owners]])
    rawloc = np.flatnonzero(f.variant.eq('raw')); error = float(abs(local[rawloc]-cached['independent'][ids]).max())
    assert error < 2e-5, ('local replay', error)
    score = []; head_error = []
    for seed in SEEDS:
        base = np.load(UP/f'strategies_seed{seed}.npz')['logits'][:, 2].astype(np.float32)
        ck = torch.load(ROOT/'models'/f'layer1_independent_{seed}.pt', map_location='cpu', weights_only=False)
        sc = head_scores(local, base[ids[owners]], ck); score.append(sc)
        old = np.load(ROOT/'results'/f'layer1_independent_{seed}.npz')['score'][ids]
        head_error.append(float(abs(sc[rawloc]-old).max()))
        assert head_error[-1] < 2e-5, ('head replay', seed, head_error[-1])
    f.to_csv(HERE/'interventions.csv', index=False)
    np.savez_compressed(HERE/'results.npz', scores=np.stack(score), local=local, ids=ids, owners=owners, rawloc=rawloc)
    (HERE/'image_provenance.json').write_text(json.dumps(artifacts, indent=2), encoding='utf-8')
    receipt = dict(n=len(ids), main_sample=len(chosen), ships=cohort[cohort.main_sample].mmsi.nunique(),
        passes=cohort[cohort.main_sample].pass_group.nunique(), interventions=len(rows)-len(ids), token_replay=max(raw_replay),
        local_replay=error, head_replay=head_error, seconds=time.perf_counter()-tick, new_fits=0,
        target_evaluation=False, base_path_unchanged=True, refs_frozen=True)
    (HERE/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(receipt, flush=True)


if __name__ == '__main__':
    main()
