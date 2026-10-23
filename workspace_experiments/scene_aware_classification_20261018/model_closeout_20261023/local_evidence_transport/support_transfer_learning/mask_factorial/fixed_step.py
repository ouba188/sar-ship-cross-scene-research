"""Isolate training-edge masks at step100 and a common frozen residual gain."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
P = HERE.parent
sys.path.insert(0, str(P))
import run


def main():
    torch.set_num_threads(2)
    table = pd.read_csv(run.UP / 'table.csv')
    data = np.load(run.ROOT / 'tokens.npz')
    cache = np.load(P / 'kernel.npz')
    idx, bank = cache['ids'][64:], cache['bank']
    y = torch.tensor(table.iloc[bank.ravel()].class_id.to_numpy())
    sameclass = y.numpy()[:, None] == y.numpy()[None]
    group = table.iloc[bank.ravel()].pass_group.to_numpy()
    samepass = group[:, None] == group[None]
    diag = np.eye(64, dtype=bool)
    masks = {'image': ~diag, 'pass': ~samepass,
             'crosspos': ~(diag | (sameclass & samepass)),
             'crossneg': ~(diag | (~sameclass & samepass))}
    reader = run.Readout(cache['kernel'][64:], data['mass'], bank, idx, np.ones((len(idx), 8, 8), bool))
    scale = json.loads((P / 'receipt.json').read_text())['scale']
    selected = pd.read_csv(P / 'source_metrics.csv')
    te = np.flatnonzero(table.iloc[idx].role.eq('evaluation'))
    labels = table.iloc[idx[te]].class_id.to_numpy()
    rows, values, checks = [], {}, {}
    for name, mask in masks.items():
        tr = run.Readout(cache['kernel'][:64], data['mass'], bank, bank.ravel(), mask.reshape(64, 8, 8))
        v = torch.nn.Parameter(torch.zeros(8, 8, 2, 20))
        opt = torch.optim.Adam([v], lr=.03)
        for _ in range(100):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(tr(v, 'token'), y) + .01 * v.square().mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            value = reader(v, 'token').numpy()
        values[name] = value
        rows.append(dict(mask=name, level='local', seed=0, gain=1., **run.metrics(labels, value[te])))
        if name in ('image', 'crossneg'):
            original = np.load((P if name == 'image' else HERE) / 'weights.npz')['token_' + name]
            checks[name] = float(np.max(abs(original - v.detach().numpy())))
            assert checks[name] < 1e-5
        for seed in (17, 29, 43):
            base = np.load(P / f'predictions_{seed}.npz')['full'][idx]
            gain = float(selected.query('seed == @seed and arm == "uniform" and role == "evaluation"').gain.iloc[0])
            score = base + gain * value / scale
            rows.append(dict(mask=name, level='full', seed=seed, gain=gain, **run.metrics(labels, score[te])))
    pd.DataFrame(rows).to_csv(HERE / 'fixed_step_metrics.csv', index=False)
    np.savez_compressed(HERE / 'fixed_step_scores.npz', idx=idx, **values)
    (HERE / 'fixed_step_receipt.json').write_text(json.dumps(dict(step=100, no_new_selection=True,
        same_uniform_RMS_and_gain=True, selected_checkpoint_replay_errors=checks), indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).groupby(['level','mask'])[['OA','BA','NLL','Brier']].mean().to_string())


if __name__ == '__main__':
    main()
