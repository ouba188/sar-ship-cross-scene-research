"""Frozen inference replacements; no checkpoint updates or target fitting."""
from pathlib import Path
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'm3_explained_20261029'))
import trace_relations as tr

exp, joint = tr.exp, tr.j
ARMS = ('soft', 'pool_keep_beta_mass', 'uniform_refs', 'pool_uniform_mass',
        'simple_region_mean', 'm3_off')


def downstream(model, arrays, ids, updated):
    valid, slots, energy = [exp.tensor(arrays[k], ids) for k in ('valid', 'slots', 'energy')]
    available = exp.tensor(arrays['available'], ids)
    deep = model.backbone.model.layer4(updated.flatten(0, 1))
    glob = exp.regional_mean(deep, valid.flatten(0, 1), valid.flatten(0, 1))[0][:, 0].reshape(len(ids), 2, -1)
    low = exp.tensor(arrays['low'], ids)
    pooled = []
    for p in range(2):
        reg, en = model.region_integrals(slots[:, p], energy[:, p])
        extra = model.deep_pool[p](updated[:, p], reg, valid[:, p], en)
        pooled.append((low[:, p] + model.deep_output[p](extra['pooled'])).flatten(1))
    values = torch.cat((glob, torch.stack(pooled, 1)), -1) * available[..., None]
    return model.head(torch.cat((values.flatten(1), available), -1))


def reference_values(model, arrays, rid, p):
    source = exp.tensor(arrays['x'], rid)[:, p]
    masks = exp.tensor(arrays['slots'], rid)[:, p]
    valid = exp.tensor(arrays['valid'], rid)[:, p]
    mass = exp.tensor(arrays['mass'], rid)[:, p]
    masks = torch.minimum(masks.clamp(0, 1), joint.grid_occupancy(valid, source.shape[-2:]).clamp(0, 1))
    z = F.layer_norm(source.permute(0, 2, 3, 1), (source.shape[1],)).permute(0, 3, 1, 2)
    values = torch.einsum('kshw,kdhw->ksd', masks, model.m3[0][p].project(z)) / mass.clamp_min(1e-8)[..., None]
    return values, mass


def replace_update(op, detail, values, rm, qm, slots, valid, legal, arm):
    parts = []
    for r, indexes in enumerate(exp.tokens.REGIONS):
        t = detail['transport'][r]
        beta = detail['reference_weights'][:, :, indexes]
        offered = qm[:, None, indexes]
        offered = offered / offered.sum(-1, keepdim=True).clamp_min(1e-20)
        refmass = rm[None, :, indexes]
        refmass = refmass / refmass.sum(-1, keepdim=True).clamp_min(1e-20)
        if arm in ('pool_keep_beta_mass', 'pool_uniform_mass'):
            t = t.sum(-1, keepdim=True) * refmass[..., None, :]
        if arm == 'simple_region_mean':
            t = offered[..., :, None] * refmass[..., None, :] * legal[..., None, None]
        if arm in ('uniform_refs', 'pool_uniform_mass', 'simple_region_mean'):
            live = t.sum(-1) > 0
            beta = live.to(t) / live.sum(1, keepdim=True).clamp_min(1)
        moved = torch.einsum('bkij,kjd->bkid', t, values[:, indexes])
        moved = moved / offered[..., None].clamp_min(1e-20)
        parts.append((beta[..., None] * moved).sum(1))
    masks = torch.minimum(slots.clamp(0, 1), joint.grid_occupancy(valid, slots.shape[-2:]).clamp(0, 1))
    write = masks / masks.sum(1, keepdim=True).clamp_min(1)
    return op.gain * op.output(torch.einsum('bshw,bsd->bdhw', write, torch.cat(parts, 1)))


@torch.no_grad()
def main():
    exp.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    exp.current.setup(exp.DEVICE)
    torch.set_num_threads(2)
    rows, nf, refs, rid, arrays = exp.load()
    cache = joint.pair_cache()
    ids = np.arange(nf, len(rows))
    y = np.array([rows[i]['label'] for i in ids])
    records, metrics, diagnostic = [], [], []
    checks = {'logit_parity_max': 0., 'update_parity_max': 0., 'pool_rowmass_max': 0.}
    tick = time.perf_counter()
    for seed in (17, 29, 43):
        model = tr.review.restore('N_level_pattern', seed)
        model.requires_grad_(False)
        for op in model.m3[0]:
            op.frequency = False
        rv = [reference_values(model, arrays, rid, p) for p in range(2)]
        outputs = {arm: [] for arm in ARMS}
        for start in range(0, len(ids), 8):
            batch = ids[start:start+8]
            original_score, details = joint.forward(model, 'N_level_pattern', cache, arrays, batch, rid, rows, details=True)
            x, qm, slots, valid = [exp.tensor(arrays[k], batch) for k in ('x', 'mass', 'slots', 'valid')]
            avail = exp.tensor(arrays['available'], batch).bool()
            ravail = exp.tensor(arrays['available'], rid).bool()
            legal = exp.data.pair_mask([rows[i] for i in batch], [rows[i] for i in rid], exp.DEVICE)
            legal &= model.m3[0][0].cross_orbit
            for arm in ARMS:
                updates = []
                for p, op in enumerate(model.m3[0]):
                    if arm == 'm3_off':
                        update = torch.zeros_like(x[:, p])
                    else:
                        update = replace_update(op, details[p], *rv[p], qm[:, p], slots[:, p], valid[:, p],
                                                legal & avail[:, p, None] & ravail[None, :, p], arm)
                    if arm == 'soft':
                        checks['update_parity_max'] = max(checks['update_parity_max'], float((update-details[p]['update']).abs().max()))
                    updates.append(update)
                update = torch.stack(updates, 1)
                score = downstream(model, arrays, batch, x + update)
                if arm == 'soft':
                    checks['logit_parity_max'] = max(checks['logit_parity_max'], float((score-original_score).abs().max()))
                outputs[arm].append(tr.arr(score))
                ratios = tr.arr(update.flatten(2).norm(dim=-1) / x.flatten(2).norm(dim=-1).clamp_min(1e-8))
                for j, idx in enumerate(batch):
                    diagnostic.append(dict(seed=seed, id=str(rows[idx]['id']), arm=arm, vv_update_ratio=float(ratios[j, 0]), vh_update_ratio=float(ratios[j, 1])))
            for p in range(2):
                for r, indexes in enumerate(exp.tokens.REGIONS):
                    t = details[p]['transport'][r]
                    rm = rv[p][1][None, :, indexes]
                    simple = t.sum(-1, keepdim=True) * (rm/rm.sum(-1, keepdim=True).clamp_min(1e-20))[..., None, :]
                    checks['pool_rowmass_max'] = max(checks['pool_rowmass_max'], float((simple.sum(-1)-t.sum(-1)).abs().max()))
            print('BATCH', seed, start+len(batch), len(ids), round(time.perf_counter()-tick, 1), flush=True)
        for arm in ARMS:
            scores = np.concatenate(outputs[arm])
            np.save(ROOT/f'{seed}_{arm}_logits.npy', scores)
            metrics.append(dict(seed=seed, arm=arm, **exp.metrics(y, scores)))
            margin = tr.review.margin(scores, y)
            for j, idx in enumerate(ids):
                records.append(dict(seed=seed, arm=arm, index=int(idx), id=str(rows[idx]['id']),
                    label=int(y[j]), prediction=int(scores[j].argmax()), margin=float(margin[j]),
                    port=rows[idx].get('port'), pass_group=rows[idx].get('pass_group'), mmsi=rows[idx].get('mmsi')))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assert checks['update_parity_max'] < 3e-5 and checks['logit_parity_max'] < 1e-4, checks
    assert checks['pool_rowmass_max'] < 1e-6, checks
    pd.DataFrame(records).to_csv(ROOT/'predictions.csv', index=False)
    pd.DataFrame(metrics).to_csv(ROOT/'metrics.csv', index=False)
    pd.DataFrame(diagnostic).to_csv(ROOT/'update_norms.csv', index=False)
    pd.DataFrame([rows[i] for i in rid]).to_csv(ROOT/'references.csv', index=False)
    receipt = dict(candidate='N_level_pattern', seeds=[17, 29, 43], m3_frequency=False,
        queries=len(ids), unique_queries=len(set(str(rows[i]['id']) for i in ids)),
        references=[str(rows[i]['id']) for i in rid], checks=checks,
        trained=False, device=exp.DEVICE, elapsed_seconds=time.perf_counter()-tick,
        protocol='Historical development cohort; frozen checkpoint inference replacements, not matched retraining.')
    (ROOT/'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
