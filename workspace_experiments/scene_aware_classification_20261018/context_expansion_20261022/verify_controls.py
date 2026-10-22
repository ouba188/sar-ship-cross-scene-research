"""Replay saved constant controls and join them to already reviewed native cases."""
import json
import numpy as np
import pandas as pd
import torch
from train_expanded import HERE, Relation, inputs_for, SEEDS

torch.set_num_threads(2)
t = pd.read_csv(HERE/'readout/targets.csv')
p = pd.concat([pd.read_csv(HERE/'content_control_predictions.csv'),
               pd.read_csv(HERE/'information_source_predictions.csv')], ignore_index=True)
rows = []
for fold in range(6):
    d = dict(np.load(HERE/f'fold{fold}_inputs.npz'))
    train, test = d['train'], d['test']
    source = train[d['m'][train].any(1)]
    for seed in SEEDS:
        for arm, key in [('fixed_split_mean', 'split_unit'), ('trained_context_constant', 'context_constant'),
                         ('image_constant_geometry_real', 'image_constant_geometry_real'),
                         ('image_real_geometry_constant', 'image_real_geometry_constant')]:
            x = inputs_for('split_unit', d, 'cuda')
            if arm != 'image_real_geometry_constant':
                x[1][:] = torch.as_tensor(d['split'][train][d['m'][train]].mean(0), device='cuda')
            if arm != 'image_constant_geometry_real':
                x[3][:] = torch.as_tensor(d['stats'][source].mean(0), device='cuda')
            model = Relation('split_unit').cuda().eval()
            model.load_state_dict(torch.load(HERE/'models'/f'fold{fold}_{key}_{seed}.pt', weights_only=True, map_location='cpu'))
            with torch.no_grad():
                score = d['base'][test]+model(*[a[test] for a in x]).cpu().numpy()
            q = p[p.fold.eq(fold)&p.seed.eq(seed)&p.arm.eq(arm)].set_index('instance_id').loc[t.iloc[test].instance_id]
            err = float(np.abs(score-q[['s0','s1','s2']].to_numpy()).max())
            assert err < 1e-4 and np.array_equal(score.argmax(1), q.prediction)
            rows.append(dict(fold=fold, seed=seed, arm=arm, error=err))
pd.DataFrame(rows).to_csv(HERE/'content_control_replay.csv', index=False)
cases = pd.read_csv(HERE/'review_cases.csv').instance_id.unique()
p[p.instance_id.isin(cases)].to_csv(HERE/'review_controls.csv', index=False)
(HERE/'content_control_verification.json').write_text(json.dumps(dict(head_replays=len(rows),
    max_error=max(x['error'] for x in rows), top1_identical=True,
    case_note='Same eight original VV/VH case panels as analyze_expanded; controls change readers, not images.'), indent=2), encoding='utf-8')
print('verified', len(rows), 'max error', max(x['error'] for x in rows))
