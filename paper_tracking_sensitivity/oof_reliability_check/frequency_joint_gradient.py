"""Source-first-batch parameter-gradient attribution, without optimizer updates."""

import json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from frequency_joint_run import OUT, EXP, load_model
from prepare_learned_frequency import load_npz
from learned_frequency_model import support_scores
from frequency_joint_core import visual_scores, comparison_loss


def stats(a, b):
    na = torch.stack([g.square().sum() for g in a]).sum().sqrt()
    nb = torch.stack([g.square().sum() for g in b]).sum().sqrt()
    dot = torch.stack([(x*y).sum() for x, y in zip(a, b)]).sum()
    return dict(ce_norm=float(na), comparison_norm=float(nb),
        weighted_ratio=float(.1*nb/na.clamp_min(1e-12)),
        gradient_cosine=float(dot/(na*nb).clamp_min(1e-12)))


def main():
    maps = np.load(EXP/'maps/self24ports/maps.npy', mmap_mode='r')
    records = []
    for marker in sorted((EXP/'tasks').rglob('EVALUATED.json')):
        folder = marker.parent
        info = json.loads((folder/'PREPARED.json').read_text())
        episode_paths = sorted((folder/'episodes').glob('*.npz'))
        rng = np.random.default_rng(20261009)
        ei = int(rng.permutation(len(episode_paths))[0])
        ep = load_npz(episode_paths[ei])
        q = rng.choice(ep['query'], min(128, len(ep['query'])), replace=False)
        ix = np.r_[ep['support'], q]
        rows = ep['rows'][ix]
        x = torch.from_numpy(np.asarray(maps[rows], np.float32)).cuda()
        args = [torch.as_tensor(ep[k][ix], dtype=torch.float32, device='cuda')
            for k in ('phi', 'prior', 'mean', 'conditions')]
        labels = torch.as_tensor(ep['labels'][ix], dtype=torch.long, device='cuda')
        sy, yq = labels[:64], labels[64:]
        model = load_model(folder/'selective.pt')
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            z, _ = model(x, args[-1])
        score = support_scores(*args, z, sy, info['eta'])
        count = torch.bincount(yq, minlength=8).clamp_min(1)
        weights = 1./count[yq].float()
        ce = (F.cross_entropy(score[64:], yq, reduction='none')*weights).sum()/weights.sum()
        visual, counts = visual_scores(z, args[-1], sy)
        pair, _, _ = comparison_loss(visual, counts, score[64:], yq)
        params = list(model.named_parameters())
        g0 = torch.autograd.grad(ce, [p for _, p in params], retain_graph=True)
        g1 = torch.autograd.grad(pair, [p for _, p in params])
        for group in ('all', 'blocks', 'gate'):
            chosen = [i for i, (name, _) in enumerate(params) if group == 'all' or name.startswith(group+'.')]
            records.append(dict(dataset=info['dataset'], task=info['task'], episode=ei, group=group,
                **stats([g0[i] for i in chosen], [g1[i] for i in chosen])))
        del model, g0, g1
    result = pd.DataFrame(records)
    result.to_csv(OUT/'parameter_gradient_probe.csv', index=False)
    print(result.groupby(['dataset', 'group'])[['weighted_ratio', 'gradient_cosine']].mean().to_string())
    print('ALL PARAMETER RANGE\n'+result.loc[result.group.eq('all'), ['weighted_ratio', 'gradient_cosine']].describe().to_string())


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main()
