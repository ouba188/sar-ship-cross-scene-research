"""Source-guided spatial-frequency residuals with an unchanged frozen main path."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.fft import dctn, idctn
from channel_operator import BASE, CACHE as FEATURES, OUT as PRIOR, SEED, build_blocks, metrics, save_json
from scene_training import sample_weights
from target_response_adapter import phi_prior, residual_head
from target_loss_repair import OLD, support_loss, supervised_source_loss

OUT = BASE/'spectral_residual_20261010'
ARMS = ('full', 'middle', 'source_guided')
GROUPS = ('SP_near', 'SP_far', 'DP_near', 'DP_far')
BANDS = ('low', 'middle', 'high')
STEPS = 24


def masks(size=8):
    u = np.arange(size, dtype=np.float64)/(size-1)
    r2 = (u[:, None]**2+u[None, :]**2)/2
    low = np.exp(-r2/(2*.2**2))
    upper = np.exp(-r2/(2*.7**2))
    return np.stack([low, upper-low, 1-upper]).astype(np.float32)


def transform(h):
    return dctn(h, type=2, axes=(-2, -1), norm='ortho', workers=1)


def reconstruct(f):
    return idctn(f, type=2, axes=(-2, -1), norm='ortho', workers=1)


def branch(h, mask, scale):
    return reconstruct(transform(h)*mask)*scale[None, :, None, None]


class SpectralResidual(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(channels))

    def forward(self, h, residual):
        return h+self.gate[None, :, None, None]*residual

    def bound(self):
        with torch.no_grad():
            self.gate.clamp_(-.25, .25)


def guidance(obs, train, prefix):
    meta = obs.iloc[train].reset_index(drop=True)
    h = np.array(prefix[meta.manifest_row.to_numpy()], dtype=np.float32)
    rms = np.sqrt(np.mean(h.astype(np.float64)**2, axis=(0, 2, 3)))
    f = transform(h)/np.maximum(rms[None, :, None, None], 1e-6)
    sw = sample_weights(meta)
    centers = np.stack([np.average(f[meta.class_id.eq(c)], axis=0, weights=sw[meta.class_id.eq(c)])
                        for c in range(8)])
    centered = centers-centers.mean(0)
    band_masks = masks()
    between = np.array([np.mean((centered*m)**2) for m in band_masks])
    pairs = pd.read_csv(BASE/'pairs.csv')
    source_ids = set(meta.mmsi)
    target_ports = ['Santos', 'Port Klang', 'Los Angeles']
    pairs = pairs[pairs.mmsi.isin(source_ids) & pairs.group.isin(GROUPS)
                  & ~pairs.port_a.isin(target_ports) & ~pairs.port_b.isin(target_ports)].copy()
    # Type labels come only from the labeled source identities, not target queries.
    label_map = meta.drop_duplicates('mmsi').set_index('mmsi').class_id
    pairs['class_id'] = pairs.mmsi.map(label_map)
    for band in BANDS:
        pairs[band+'_drift'] = 0.
    for lo in range(0, len(pairs), 16):
        part = pairs.iloc[lo:lo+16]
        a = obs.iloc[part.a.to_numpy()]
        b = obs.iloc[part.b.to_numpy()]
        assert np.array_equal(a.mmsi.to_numpy(), b.mmsi.to_numpy())
        delta = transform(np.array(prefix[a.manifest_row], dtype=np.float32)
                          -np.array(prefix[b.manifest_row], dtype=np.float32))
        delta /= np.maximum(rms[None, :, None, None], 1e-6)
        for band, m in zip(BANDS, band_masks):
            pairs.loc[part.index, band+'_drift'] = np.mean((delta*m)**2, axis=(1, 2, 3))
    pairs.to_csv(OUT/'source_pair_drift.csv', index=False)
    drift_cols = [b+'_drift' for b in BANDS]
    vessels = pairs.groupby(['group', 'class_id', 'mmsi'])[drift_cols].mean().reset_index()
    cells = vessels.groupby(['group', 'class_id'])[drift_cols].mean().reset_index()
    means = cells.groupby('group')[drift_cols].mean()
    counts = pairs.groupby('group').agg(pairs=('mmsi', 'size'), vessels=('mmsi', 'nunique'), classes=('class_id', 'nunique'))
    means.join(counts).reset_index().to_csv(OUT/'source_drift_summary.csv', index=False)
    far_drift = means.loc[['SP_far', 'DP_far']].mean().to_numpy()
    ratios = between/np.maximum(far_drift, 1e-8)
    weights = ratios/ratios.sum()
    pd.DataFrame(dict(band=BANDS, class_centroid_signal=between, same_vessel_far_drift=far_drift,
                      signal_drift_ratio=ratios, weight=weights)).to_csv(OUT/'source_band_weights.csv', index=False)
    selected = {'full': np.ones((8, 8), np.float32), 'middle': band_masks[1],
                'source_guided': np.sum(weights[:, None, None]*band_masks, axis=0).astype(np.float32)}
    scales, records = {}, []
    fh = transform(h)
    for arm, mask in selected.items():
        raw = reconstruct(fh*mask)
        raw_rms = np.sqrt(np.mean(raw.astype(np.float64)**2, axis=(0, 2, 3)))
        factor = np.divide(rms, raw_rms, out=np.zeros_like(rms), where=raw_rms > 1e-6).astype(np.float32)
        if arm == 'full':
            factor[:] = 1
        scales[arm] = factor
        for c in range(256):
            records.append(dict(arm=arm, channel=c, original_rms=rms[c], raw_branch_rms=raw_rms[c],
                                multiplier=factor[c], matched_branch_rms=raw_rms[c]*factor[c]))
    pd.DataFrame(records).to_csv(OUT/'source_branch_scale.csv', index=False)
    return selected, scales


def encode(blocks, model, maps, residuals, rows):
    out = np.empty((len(rows), 512), np.float32)
    with torch.inference_mode():
        for lo in range(0, len(rows), 16):
            idx = rows[lo:lo+16]
            h = torch.from_numpy(np.array(maps[idx]))
            r = torch.from_numpy(np.array(residuals[idx]))
            out[lo:lo+len(idx)] = blocks(model(h, r)).mean((2, 3)).numpy()
    return out


def phi(features, state):
    return phi_prior(features, *[torch.from_numpy(state[k]) for k in
        ('source_center', 'source_scale', 'source_coef', 'source_bias')])


def fit(blocks, maps, residuals, support, labels, source_labels, source_scores, state, tag, arm):
    model = SpectralResidual()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    hs, rs = [torch.from_numpy(np.array(a[support])) for a in (maps, residuals)]
    y = torch.from_numpy(labels).long()
    counts = torch.bincount(y, minlength=8)
    present = counts > 0
    rw = len(y)/(present.sum()*counts[y].float())
    history = []
    for step in range(STEPS):
        rng = np.random.default_rng(SEED+step)
        replay = np.concatenate([rng.choice(np.flatnonzero(source_labels == c), 4, replace=False) for c in range(8)])
        hr, rr = [torch.from_numpy(np.array(a[replay])) for a in (maps, residuals)]
        p, pri = phi(blocks(model(hs, rs)).mean((2, 3)), state)
        _, loo, target = residual_head(p, pri, y, balanced=True)
        target_loss = support_loss(loo, target, rw, present, True)
        _, score = phi(blocks(model(hr, rr)).mean((2, 3)), state)
        keep = (score-source_scores[replay]).square().mean()
        source_loss = supervised_source_loss(score, torch.from_numpy(source_labels[replay]).long())
        regularizer = model.gate.square().mean()
        loss = target_loss+.25*keep+.25*source_loss+.1*regularizer
        optimizer.zero_grad()
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.))
        optimizer.step()
        model.bound()
        if step in (0, 7, 15, STEPS-1):
            history.append(dict(task=tag, arm=arm, step=step+1, target_loss=float(target_loss.detach()),
                source_preservation=float(keep.detach()), source_supervised=float(source_loss.detach()),
                gate_rms=float(regularizer.detach().sqrt()), gradient_norm=grad))
    return model.eval(), history


def run():
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    OUT.mkdir(exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Completed results must be preserved')
    start = time.perf_counter()
    obs = pd.read_csv(BASE/'observations.csv')
    with np.load(PRIOR/'split.npz') as z:
        train, test = z['train_obs_ids'], z['test_obs_ids']
    ids = np.r_[train, test]
    source_labels = obs.iloc[train].class_id.to_numpy()
    prefix = np.load(FEATURES/'source_layer3.npy', mmap_mode='r')
    save_json(OUT/'PROTOCOL.json', dict(
        purpose='Does frequency-selective residual correction reduce harm relative to full-band correction?',
        labels='Same939 source labels and exact9x64 target supports; no new ship-type annotations',
        extra_information='Guided mask uses repeated source observations and MMSI links of already-labeled source ships; this is extra source identity information, not equal-information proof against unguided arms',
        frequency='2D orthonormal DCT-II on256x8x8 frozen layer3 maps; Gaussian low=.2, upper=.7, normalized radial corner radius1; low+middle+high=1',
        masks='Full spectrum; fixed middle; source-only convex mix weighted by between-class centroid energy / same-vessel far-condition drift',
        guidance_limits='Centroid separation can contain port shortcuts; same-vessel change is not all noise; fixed weights not dynamic attention or causal identification',
        residual='h + gate[channel]*normalized_band(h); original path retained;256 parameters all arms; zero initialization; gate bounded[-.25,.25]',
        scale_control='Match branch per-channel source RMS to original source RMS; full-band multiplier1; no target fitting of scales',
        training='Same corrected masked-support LOO ridge8 +.25 source-score keep +.25 eight-class source supervision +.1 gate penalty;24 Adam steps lr.01; same32 source replay rows per step',
        scope='Earlier insertion than old affine operator and fewer parameters; frequency claim uses within-this-round matched arms, not only comparison with historical operator',
        selection='All3 arms fixed before target scoring; no cutoff sweep, target-best port policy, or extra iterations',
        boundaries='Small R18 development pilot, not full dual or Causal-Tune reproduction; support/query may share scenes; inherited pretraining exposure',
        resources='CPU1, approximately1.3GB new memory-mapped caches, no parent/GPU changes'))
    selected, scales = guidance(obs, train, prefix)
    np.savez_compressed(OUT/'frequency_masks.npz', **selected, **{k+'_scale': v for k, v in scales.items()})
    maps = np.lib.format.open_memmap(OUT/'prefix_maps.npy', mode='w+', dtype=np.float32, shape=(len(ids), 256, 8, 8))
    residuals = {k: np.lib.format.open_memmap(OUT/(k+'_branch.npy'), mode='w+', dtype=np.float32,
                                           shape=maps.shape) for k in ARMS if k != 'full'}
    rows = obs.iloc[ids].manifest_row.to_numpy()
    for lo in range(0, len(ids), 32):
        h = np.array(prefix[rows[lo:lo+32]], dtype=np.float32)
        maps[lo:lo+len(h)] = h
        for arm, array in residuals.items():
            array[lo:lo+len(h)] = branch(h, selected[arm], scales[arm])
    maps.flush()
    for a in residuals.values():
        a.flush()
    residuals['full'] = maps
    del prefix
    blocks = build_blocks()
    with np.load(sorted(OLD.glob('*_models.npz'))[0]) as z:
        state = {k: z[k] for k in z.files}
    original_features = encode(blocks, SpectralResidual(), maps, maps, np.arange(len(ids)))
    with torch.no_grad():
        all_phi, all_prior = phi(torch.from_numpy(original_features), state)
    source_scores = all_prior[:len(train)].clone()
    with np.load(PRIOR/'baseline.npz') as z:
        errors = int((all_prior[len(train):].argmax(1).numpy() != z['predictions']).sum())
    if errors:
        raise ValueError(f'Frozen prefix and full tail did not replay baseline: {errors}')
    print('PREPARATION_DONE', 'original_prediction_errors', errors, flush=True)
    records, classes, traces, replays, updates = [], [], [], [], []
    for path in sorted(OLD.glob('*_split.npz')):
        tag = path.stem.removesuffix('_split')
        with np.load(path) as z:
            support, query, labels, query_ids = [z[k] for k in
                ('support_map_rows', 'query_map_rows', 'support_labels', 'query_obs_ids')]
        with np.load(OLD/f'{tag}_models.npz') as z:
            state = {k: z[k] for k in z.files}
        predictions, saved = {}, {}
        with torch.no_grad():
            beta, _, _ = residual_head(all_phi[support], all_prior[support], torch.from_numpy(labels), balanced=True)
            head = all_prior[query]+all_phi[query]@beta
        if not np.array_equal(head.argmax(1).numpy(), state['head_only_pred']):
            raise ValueError('Head-only predictions failed replay')
        replays.append(dict(task=tag, prediction_disagreements=0))
        predictions['head_only'] = head.argmax(1).numpy()
        saved['head_only_scores'] = head.numpy()
        for arm in ARMS:
            model, history = fit(blocks, maps, residuals[arm], support, labels, source_labels,
                                 source_scores, state, tag, arm)
            traces.extend(history)
            features = encode(blocks, model, maps, residuals[arm], np.r_[support, query])
            with torch.no_grad():
                ph, pri = phi(torch.from_numpy(features), state)
                beta, _, _ = residual_head(ph[:64], pri[:64], torch.from_numpy(labels), balanced=True)
                score = pri[64:]+ph[64:]@beta
            predictions[arm] = score.argmax(1).numpy()
            saved[arm+'_scores'], saved[arm+'_prior'] = score.numpy(), pri[64:].numpy()
            saved[arm+'_gate'] = model.gate.detach().numpy()
            h = np.array(maps[support])
            delta = np.array(residuals[arm][support])*model.gate.detach().numpy()[None, :, None, None]
            updates.append(dict(task=tag, arm=arm, gate_rms=float(np.sqrt(np.mean(saved[arm+'_gate']**2))),
                relative_update_rms=float(np.linalg.norm(delta)/np.linalg.norm(h)),
                negative_activation_fraction=float(((h+delta) < 0).mean())))
            print('FIT_DONE', tag, arm, flush=True)
        truth = obs.iloc[query_ids].class_id.to_numpy()
        port, draw = obs.iloc[query_ids].port.iloc[0], int(tag.rsplit('draw', 1)[1])
        for arm, pred in predictions.items():
            records.append(dict(port=port, draw=draw, arm=arm, **metrics(pred, truth)))
            saved[arm+'_pred'] = pred
            for c in np.unique(truth):
                count = int((labels == c).sum())
                classes.append(dict(port=port, draw=draw, arm=arm, class_id=int(c), support_count=count,
                    coverage='absent' if count == 0 else 'singleton' if count == 1 else 'multiple',
                    recall=float(100*(pred[truth == c] == c).mean())))
        np.savez_compressed(OUT/f'{tag}_models.npz', **saved, query_obs_ids=query_ids)
        for name, data in [('results', records), ('by_class', classes), ('training_history', traces), ('update_diagnosis', updates)]:
            pd.DataFrame(data).to_csv(OUT/f'{name}.csv', index=False)
        save_json(OUT/'REPLAYS.json', replays)
    save_json(OUT/'COMPLETE.json', dict(tasks=9, adapter_fits=27, seconds=time.perf_counter()-start,
        baseline_prediction_disagreements=errors, query_labels_used_for_training=False))
    print(pd.DataFrame(records).groupby('arm')[['OA', 'BA']].mean().to_string())


if __name__ == '__main__':
    run()
