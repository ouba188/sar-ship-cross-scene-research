"""Check real-source full-loss gradient equivalence for computational chunking."""
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tune import HERE, PREV, load, write, ta, op, PairMetric, FrozenDecision


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = load(PREV/'bundle/features.npz')
    table = pd.read_csv(PREV/'bundle/table.csv', dtype={'mmsi': str, 'pass_group': str})
    groups = ta.validate_roles(table, data)
    prior = load(PREV/'training/seed17/predictions.npz')
    init = PREV/'initial_assets/warm/seed17'
    matcher = op.Matcher({k: data[k] for k in ta.FEATURES}, data['refs'],
                         op.load_checkpoint(init/'metric_scaler.pt'), 'cuda')
    decision = FrozenDecision(PREV/'training/seed17/warm', prior, prior['bare'], 'cuda')
    y, ids = table.class_id.to_numpy(int), groups['utility_train']
    yt = torch.tensor(y[ids], device='cuda')
    wt = op.balanced_weights(y, ids, 'cuda', classes=8)
    gradients, states, records = [], [], []
    for batch in (24, 128):
        model = PairMetric().cuda()
        model.load_state_dict(op.load_checkpoint(init/'both_metric.pt')['state'])
        opt = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
        opt.zero_grad(set_to_none=True)
        start, total = time.perf_counter(), 0.
        for k in range(0, len(ids), batch):
            ix = ids[k:k+batch]
            s = decision(matcher.local(ix, model), ix)
            loss = (F.cross_entropy(s, yt[k:k+batch], reduction='none')*wt[k:k+batch]).sum()
            loss.backward()
            total += float(loss.detach())
        gradients.append(torch.cat([p.grad.ravel() for p in model.parameters()]).cpu().numpy())
        opt.step()
        states.append(torch.cat([p.detach().ravel() for p in model.parameters()]).cpu().numpy())
        records.append(dict(batch=batch, seconds=time.perf_counter()-start, loss=total))
    np.testing.assert_allclose(gradients[0], gradients[1], atol=2e-7, rtol=2e-4)
    np.testing.assert_allclose(states[0], states[1], atol=2e-6, rtol=1e-5)
    report = dict(records=records, gradient_max_error=float(abs(gradients[0]-gradients[1]).max()),
        one_step_state_max_error=float(abs(states[0]-states[1]).max()),
        source_queries=len(ids), evaluation_labels_used=False, passed=True)
    write(HERE/'batching_check.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
