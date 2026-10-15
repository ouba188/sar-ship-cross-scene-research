"""Source-only frozen training-task replay and local cross-port gradient agreement."""

from condition_coverage import (
    BASE, CACHE, FACTORS, PREVIOUS, OUT, ARMS, NARROW, FactorNet, make_episodes,
)
from condition_kernel_residual import KernelResidual, residual_head, sample_weights, metrics, save_json
import numpy as np
import pandas as pd
import torch

DEST = OUT/'source_diagnosis'


def objective(model, context, meta, episode):
    support, query = episode['support'], episode['query']
    ids, n = np.r_[support, query], len(support)
    ph = model(episode['phi'][ids], context[ids])
    prior = episode['prior'][ids]
    labels = torch.from_numpy(meta.class_id.to_numpy(copy=True)).long()
    beta, _, _ = residual_head(ph[:n], prior[:n], labels[support], balanced=True)
    scores = prior[n:]+ph[n:]@beta
    target = 2*torch.eye(8)[labels[query]]-1
    weights = torch.from_numpy(sample_weights(meta.iloc[query].reset_index(drop=True)).astype(np.float32))
    mse = ((scores-target).square().mean(1)*weights).mean()
    return mse, scores


def run():
    torch.set_num_threads(1)
    DEST.mkdir(exist_ok=True)
    if (DEST/'COMPLETE.json').exists():
        raise ValueError('Preserve source diagnosis')
    obs = pd.read_csv(BASE/'observations.csv')
    with np.load(FACTORS/'source_split.npz') as z:
        ids, center, scale = [z[k] for k in ('train_obs_ids', 'center', 'scale')]
    meta = obs.iloc[ids].reset_index(drop=True)
    raw = np.array(np.load(CACHE/'frozen_replay/source_extra.npy', mmap_mode='r')[meta.manifest_row], dtype=np.float32)
    x = (raw-center)/scale
    episodes, _ = make_episodes(meta, raw)
    records, alignment = [], []
    for context_name, checkpoint in zip(ARMS, ('task_only', 'relations')):
        encoder = FactorNet()
        encoder.load_state_dict(torch.load(FACTORS/(checkpoint+'.pt'), map_location='cpu', weights_only=True))
        encoder.eval().requires_grad_(False)
        with np.load(PREVIOUS/(context_name+'_context_scale.npz')) as z:
            dc, ds = torch.from_numpy(z['center']), torch.from_numpy(z['scale'])
        with torch.no_grad():
            context = (encoder.condition(torch.from_numpy(x))-dc)/ds
        models = {'identity': KernelResidual()}
        for curriculum in ('narrow32', 'broad32'):
            model = KernelResidual()
            model.load_state_dict(torch.load(OUT/(curriculum+'_'+context_name+'.pt'), map_location='cpu', weights_only=True))
            models[curriculum] = model.eval()
        gradients = {}
        for e in episodes:
            for curriculum, model in models.items():
                with torch.no_grad():
                    loss, scores = objective(model, context, meta, e)
                records.append(dict(context=context_name, curriculum=curriculum, port=e['port'], draw=e['draw'],
                    group='old2' if e['port'] in NARROW else 'added7', mse=float(loss),
                    **metrics(scores.argmax(1).numpy(), meta.iloc[e['query']].class_id.to_numpy())))
            model = models['identity']
            model.zero_grad()
            loss, _ = objective(model, context, meta, e)
            loss.backward()
            g = torch.cat([p.grad.flatten() for p in model.parameters()]).detach().numpy()
            gradients.setdefault(e['port'], []).append(g)
        gradients = {p: np.mean(v, axis=0) for p, v in gradients.items()}
        names = sorted(gradients)
        for a in range(len(names)):
            for b in range(a+1, len(names)):
                ga, gb = gradients[names[a]], gradients[names[b]]
                alignment.append(dict(context=context_name, port_a=names[a], port_b=names[b],
                    cosine=float(ga@gb/max(np.linalg.norm(ga)*np.linalg.norm(gb), 1e-12))))
    rows = pd.DataFrame(records)
    rows.to_csv(DEST/'source_task_results.csv', index=False)
    summary = rows.groupby(['context', 'curriculum', 'group'])[['mse', 'OA', 'BA']].mean().reset_index()
    summary.to_csv(DEST/'summary.csv', index=False)
    rows.groupby(['context', 'curriculum', 'port'])[['mse', 'OA', 'BA']].mean().reset_index().to_csv(DEST/'port_summary.csv', index=False)
    g = pd.DataFrame(alignment)
    g.to_csv(DEST/'gradient_pairs.csv', index=False)
    gs = g.groupby('context').agg(mean_cosine=('cosine', 'mean'), min_cosine=('cosine', 'min'),
        negative_pairs=('cosine', lambda v: int((v < 0).sum())), total_pairs=('cosine', 'size')).reset_index()
    gs.to_csv(DEST/'gradient_summary.csv', index=False)
    save_json(DEST/'COMPLETE.json', dict(new_training=0, source_tasks=len(episodes),
        query_labels='Only permitted source episode labels; no target data loaded',
        boundary='Training-task replay is not independent validation. Gradients measured at identity, mean across3draws/port; local conflict is not unique proof of final target harm. Fixed total steps do not equalize per-port exposure.'))
    print(summary.to_string(index=False))
    print(gs.to_string(index=False))


if __name__ == '__main__':
    run()
