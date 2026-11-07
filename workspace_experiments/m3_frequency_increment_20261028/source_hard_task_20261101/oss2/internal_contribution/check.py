"""Detect changed historical output or inadvertently coupled ablation factors."""
import json
import numpy as np
import torch
from ablate import HERE, AblationModel
from model_task import records, arrays, predict, tensor, legal, current

current.setup('cuda')
rows, train, hold, rid = records()
a = arrays()
model = AblationModel('aggregate', 17, 'full')
state = torch.load(HERE.parent/'aggregate_17/last.pt', map_location='cpu', weights_only=True)['trainable_state']
model.restore(state)
expected = np.load(HERE.parent/'last_epoch/aggregate.npz')
actual = predict(model, a, rows, hold, rid)['score']
error = float(np.abs(actual-expected['score']).max())
assert error < 1e-6, error
ids = train[:8]
inputs = [tensor(a[k], ix) for k, ix in [('desc',ids),('desc',rid),('mass',ids),('mass',rid),('write',ids)]]
pair = legal(rows,ids,rid)[...,None] & tensor(a['available'],ids)[:,None].bool() & tensor(a['available'],rid)[None].bool()
labels = torch.tensor([rows[i]['label'] for i in rid],device='cuda')
traces = {}
with torch.no_grad():
    for variant in ('full', 'shared_metric', 'no_competition'):
        model.matcher.variant = variant
        delta, _, _, t = model.matcher(*inputs,pair,labels,arm='aggregate',details=True)
        assert torch.isfinite(delta).all()
        traces[variant] = t
metric_change = 0.
for p in range(2):
    for full, shared, no in zip(*(traces[k][p]['regions'] for k in traces)):
        assert torch.equal(full['metric'], no['metric'])
        assert not torch.count_nonzero(no['competition'])
        assert torch.allclose(shared['metric'], shared['metric'][:,:1].expand_as(shared['metric']))
        metric_change = max(metric_change, float((full['metric']-shared['metric']).abs().max()))
assert metric_change > 0
report = dict(historical_replay_max_error=error,metric_change=metric_change,
              isolated_factors=True,scope='same trained weights; 228 replay, 8 source trace diagnostic')
(HERE/'checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(report)
