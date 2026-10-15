"""Frozen dynamic-versus-source-average gain diagnostic; no new training."""

from condition_kernel_residual import (
    BASE, CACHE, OLD, FACTORS, OUT, ARMS, FactorNet, KernelResidual, design,
    score, metrics, save_json, source_holdout,
)
import numpy as np
import pandas as pd
import torch
from torch import nn

DEST = OUT/'dynamic_diagnosis'


class FixedGain(nn.Module):
    def __init__(self, gain):
        super().__init__()
        self.register_buffer('gain', gain)

    def forward(self, phi, context):
        return torch.cat([phi[:, :512]*(1+self.gain), phi[:, 512:]], 1)


def run():
    torch.set_num_threads(1)
    DEST.mkdir(exist_ok=True)
    if (DEST/'COMPLETE.json').exists():
        raise ValueError('Preserve diagnostics')
    obs = pd.read_csv(BASE/'observations.csv')
    with np.load(FACTORS/'source_split.npz') as z:
        train_ids, val_ids, center, scale = [z[k] for k in ('train_obs_ids', 'diagnostic_obs_ids', 'center', 'scale')]
    raw = np.array(np.load(CACHE/'frozen_replay/source_extra.npy', mmap_mode='r')[obs.manifest_row], dtype=np.float32)
    x = (raw-center)/scale
    contexts, models, gains = {}, {}, []
    for arm, checkpoint in zip(ARMS, ('task_only', 'relations')):
        encoder = FactorNet()
        encoder.load_state_dict(torch.load(FACTORS/(checkpoint+'.pt'), map_location='cpu', weights_only=True))
        encoder.eval().requires_grad_(False)
        with np.load(OUT/(arm+'_context_scale.npz')) as z:
            dc, ds = torch.from_numpy(z['center']), torch.from_numpy(z['scale'])
        with torch.no_grad():
            d = torch.cat([encoder.condition(torch.from_numpy(x[lo:lo+256])) for lo in range(0, len(x), 256)])
            d = (d-dc)/ds
        model = KernelResidual()
        model.load_state_dict(torch.load(OUT/(arm+'.pt'), map_location='cpu', weights_only=True))
        model.eval().requires_grad_(False)
        with torch.no_grad():
            source_gain = model.gains(d[train_ids])
            mean = source_gain.mean(0)
        for mode, m in [('dynamic', model), ('source_average', FixedGain(mean))]:
            key = arm+'__'+mode
            contexts[key], models[key] = d, m
        gains.append(dict(arm=arm, source_dynamic_gain_rms=float(source_gain.square().mean().sqrt()),
            source_average_gain_rms=float(mean.square().mean().sqrt()),
            sample_specific_gain_rms=float((source_gain-mean).square().mean().sqrt())))
    records, replays = [], []
    for path in sorted(OLD.glob('*_split.npz')):
        tag = path.stem.removesuffix('_split')
        with np.load(path) as z:
            support, query, labels = [z[k] for k in ('support_obs_ids', 'query_obs_ids', 'support_labels')]
        with np.load(OLD/f'{tag}_models.npz') as z:
            state = [torch.from_numpy(z[k]) for k in ('source_center', 'source_scale', 'source_coef', 'source_bias')]
        ids = np.r_[support, query]
        truth = obs.iloc[query].class_id.to_numpy()
        port, draw = obs.iloc[query].port.iloc[0], int(tag.rsplit('draw', 1)[1])
        with torch.no_grad():
            ph, prior = design(torch.from_numpy(raw[ids]), state)
            for key, model in models.items():
                arm, mode = key.split('__')
                scores = score(model(ph, contexts[key][ids]), prior, labels)
                pred = scores.argmax(1).numpy()
                records.append(dict(arm=arm, mode=mode, port=port, draw=draw, **metrics(pred, truth)))
                if mode == 'dynamic':
                    with np.load(OUT/(tag+'.npz')) as z:
                        previous = z[arm+'_scores']
                    errors = int((pred != previous.argmax(1)).sum())
                    replays.append(dict(task=tag, arm=arm, disagreement=errors))
                    if errors:
                        raise ValueError('Dynamic prediction replay failed')
    rows = pd.DataFrame(records)
    rows.to_csv(DEST/'target_results.csv', index=False)
    summary = rows.groupby(['arm', 'mode'])[['OA', 'BA']].mean().reset_index()
    summary.to_csv(DEST/'summary.csv', index=False)
    held = pd.DataFrame(source_holdout(obs, raw, train_ids, val_ids, contexts, models))
    held.to_csv(DEST/'source_holdout.csv', index=False)
    held_summary = held.groupby('arm')[['OA', 'BA']].mean().reset_index()
    held_summary.to_csv(DEST/'source_holdout_summary.csv', index=False)
    pd.DataFrame(gains).to_csv(DEST/'source_gain_structure.csv', index=False)
    save_json(DEST/'COMPLETE.json', dict(new_training=0, tasks=9, replays=replays,
        interpretation='Fixed learned weights; source mean gains replace per-image gains and reduce RMS. Not a separately trained static control or isolated physical causal test; no target-best selection.'))
    print(summary.to_string(index=False))
    print(held_summary.to_string(index=False))
    print(pd.DataFrame(gains).to_string(index=False))


if __name__ == '__main__':
    run()
