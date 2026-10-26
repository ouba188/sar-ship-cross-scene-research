"""Locate weak feedback and distinguish direct feedback from refitted heads."""
import json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from run import HERE, SEEDS, load, batch, write
from analyze import restored


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    table, arrays = load()
    records = []
    for seed in SEEDS:
        model = restored('regional', seed)
        for role in ('fit_train', 'fit_validation', 'evaluation'):
            ids = np.flatnonzero(table.role.eq(role))
            gates, ratios, masses = [[], []], [[], []], [[], []]
            for first in range(0, len(ids), 32):
                x, masks, valid, avail = batch(arrays, ids[first:first+32])
                for p in range(2):
                    _, d = model.blocks[f'{p}_2'](x[:, p], masks[:, p], valid[:, p], return_details=True)
                    occupancy = masks[:, p]/masks[:, p].sum(1, keepdim=True).clamp_min(1)
                    active = occupancy > .01
                    gate = d['weights'][active]/occupancy[active]
                    gates[p].append(gate.cpu().numpy())
                    region = masks[:, p].sum(1) > .01
                    ratios[p].append((d['update'].norm(dim=1)[region]/x[:, p].norm(dim=1)[region].clamp_min(1e-6)).cpu().numpy())
                    masses[p].append(d['mass'].cpu().numpy())
            for p in range(2):
                g, r, m = np.concatenate(gates[p]), np.concatenate(ratios[p]), np.concatenate(masses[p])
                records.append(dict(seed=seed, role=role, pol=('VV','VH')[p],
                    gate_mean=float(g.mean()), gate_below_001=float((g < .01).mean()),
                    gate_above_099=float((g > .99).mean()), relative_update_mean=float(r.mean()),
                    relative_update_median=float(np.median(r)),
                    body_mass_median=float(np.median(m[m[:, 0] > 0, 0]))))
        del model
    pd.DataFrame(records).to_csv(HERE/'gate_diagnostics.csv', index=False)
    cases = json.loads((HERE/'analysis/cases.json').read_text(encoding='utf-8'))
    interventions = []
    for case in cases:
        model = restored('regional', case['seed'])
        idx = int(np.flatnonzero(table.obs_id.eq(case['obs_id']))[0])
        x, masks, valid, avail = batch(arrays, [idx])
        for arm in ('full', 'no_feedback', 'no_body', 'no_end', 'no_side', 'no_vv_feedback', 'no_vh_feedback'):
            modified = masks.clone()
            if arm == 'no_feedback':
                modified.zero_()
            if arm in ('no_body', 'no_end', 'no_side'):
                modified[:, :, ('no_body', 'no_end', 'no_side').index(arm)] = 0
            if arm in ('no_vv_feedback', 'no_vh_feedback'):
                modified[:, ('no_vv_feedback', 'no_vh_feedback').index(arm)] = 0
            score = model.forward_from(x, modified, valid, avail)[0].cpu().numpy()
            y = case['true']
            interventions.append(dict(obs_id=case['obs_id'], kind=case['kind'], seed=case['seed'], arm=arm,
                y=y, pred=int(score.argmax()), margin=float(score[y]-np.delete(score, y).max()),
                scores=json.dumps(score.tolist())))
        del model
    pd.DataFrame(interventions).to_csv(HERE/'case_interventions.csv', index=False)
    print(pd.DataFrame(records).to_string(index=False), flush=True)
    print(pd.DataFrame(interventions).drop(columns='scores').to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
