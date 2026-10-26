"""Two predeclared source-selected optimizer/gate-temperature candidates."""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from run import HERE, SEEDS, PREVIOUS, WEIGHT, load, batch, predict, write, op, RegionResNet50

OUT = HERE/'tuning'
CONFIGS = dict(slow=dict(lr=.0003, temperature=1., epochs=12),
               soft=dict(lr=.0003, temperature=4., epochs=12))


def fit(config, mode, seed, table, arrays):
    cfg = CONFIGS[config]
    out = OUT/config/f'{mode}_{seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists():
        return json.loads((out/'complete.json').read_text())
    tr, va = [np.flatnonzero(table.role.eq(role)) for role in ('fit_train', 'fit_validation')]
    f = table.iloc[tr]
    wt = np.zeros(len(table), np.float32)
    wt[tr] = len(f)/(8*f.groupby('class_id').mmsi.transform('nunique').to_numpy()
                    *f.groupby('mmsi').mmsi.transform('size').to_numpy())
    y = table.class_id.to_numpy(int)
    torch.manual_seed(seed)
    head = torch.nn.Linear(4098, 8).state_dict()
    torch.manual_seed(seed)
    model = RegionResNet50(WEIGHT, mode=mode, temperature=cfg['temperature']).cuda()
    model.head.load_state_dict(head)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg['lr'], weight_decay=.01)
    rng = np.random.default_rng(seed)
    best, curve, tick = None, [], time.perf_counter()
    for epoch in range(1, cfg['epochs']+1):
        model.train()
        ids = rng.permutation(tr)
        losses = []
        for first in range(0, len(ids), 24):
            take = ids[first:first+24]
            opt.zero_grad(set_to_none=True)
            score = model.forward_from(*batch(arrays, take))
            loss = (F.cross_entropy(score, torch.tensor(y[take], device='cuda'), reduction='none')
                    *torch.tensor(wt[take], device='cuda')).mean()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        score = predict(model, arrays, va)
        j = op.exact_j(y[va], score)
        m = op.metrics(y[va], score)
        curve.append(dict(epoch=epoch, loss=float(np.mean(losses)), **m))
        if best is None or j > best:
            best, selected = j, epoch
            state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('backbone.')}
            torch.save(dict(state=state, epoch=epoch, config=cfg), out/'selected.pt')
        pd.DataFrame(curve).to_csv(out/'curve.csv', index=False)
        print('TUNE', config, mode, seed, epoch, 'J', float(j), 'sec', round(time.perf_counter()-tick,1), flush=True)
    receipt = dict(config=config, mode=mode, seed=seed, selected_epoch=selected,
                   validation_j=float(best), seconds=time.perf_counter()-tick,
                   trainable=sum(p.numel() for p in model.parameters() if p.requires_grad), **cfg)
    write(out/'complete.json', receipt)
    del model, opt
    torch.cuda.empty_cache()
    return receipt


def train():
    OUT.mkdir(exist_ok=True)
    table, arrays = load()
    original = json.loads((HERE/'frozen_choices.json').read_text())
    records = [dict(r, config='original') for r in original if r['mode'] == 'regional']
    for config in CONFIGS:
        for seed in SEEDS:
            records.append(fit(config, 'regional', seed, table, arrays))
    pd.DataFrame(records).to_csv(OUT/'source_grid.csv', index=False)
    ranking = pd.DataFrame(records).groupby('config', sort=False).validation_j.mean()
    chosen = ranking.idxmax()
    write(OUT/'selected_config.json', dict(config=chosen, source_means=ranking.to_dict(),
         rule='Mean source J across three fixed seeds, original wins exact tie.'))
    if chosen == 'original':
        write(OUT/'frozen_choices.json', [])
        print('ORIGINAL_RETAINED_NO_NEW_TARGET_GRID', flush=True)
        return
    finals = [r for r in records if r['config'] == chosen]
    for mode in ('none', 'merged'):
        for seed in SEEDS:
            finals.append(fit(chosen, mode, seed, table, arrays))
    write(OUT/'frozen_choices.json', finals)
    print('FROZEN', chosen, flush=True)


def evaluate():
    records = json.loads((OUT/'frozen_choices.json').read_text())
    if not records:
        return
    assert len(records) == 9
    table, arrays = load()
    ids = np.flatnonzero(table.role.eq('evaluation'))
    truth = pd.read_csv(PREVIOUS/'source_truth.csv').set_index('obs_id')
    y = truth.loc[table.iloc[ids].obs_id, 'class_id'].to_numpy(int)
    rows = []
    for r in records:
        out = OUT/r['config']/f"{r['mode']}_{r['seed']}"
        model = RegionResNet50(WEIGHT, mode=r['mode'], temperature=r['temperature']).cuda().eval()
        ck = torch.load(out/'selected.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(ck['state'], strict=False)
        score = predict(model, arrays, ids)
        rows.append(dict(**r, **op.metrics(y, score)))
        np.savez_compressed(out/'evaluation.npz', ids=ids, obs_id=table.iloc[ids].obs_id.to_numpy(), y=y, score=score)
        print('TUNED_EVAL', r['mode'], r['seed'], op.metrics(y, score), flush=True)
        del model
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('train', 'evaluate'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    (train if args.stage == 'train' else evaluate)()
