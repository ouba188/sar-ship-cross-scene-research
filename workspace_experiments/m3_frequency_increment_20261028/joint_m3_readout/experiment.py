"""Joint M3-only adaptation from fixed upstream caches; no main model edits."""
from pathlib import Path
import sys
import time
import argparse
import json

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent/'readout_feasibility'))
import run as exp
import probe
import analyze as old_analysis
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sar_m123.m3 import sinkhorn
from sar_m123.pool_replacement import grid_occupancy

ARMS = ('S_spatial', 'W_weighted', 'C_cosine', 'L_level', 'N_level_pattern')
STREAMS = {'S_spatial': [0, 1, 2, 3, 4], 'W_weighted': [0, 1, 2, 3, 4],
           'C_cosine': [0, 1, 5, 6, 7], 'L_level': [0, 1, 8, 9, 10],
           'N_level_pattern': [0, 1, 11, 12, 13]}
SEEDS = (17, 29, 43)
EPOCHS = 4
BATCH = 24


def prepare(rows, nf, rid, arrays):
    dest = HERE/'pair_cache'
    dest.mkdir(parents=True, exist_ok=True)
    train, hold = probe.split(rows, nf, rid)
    exp.write(HERE/'split.json', dict(train=train.tolist(), source_holdout=hold.tolist(),
        development=list(range(nf, len(rows))), references=rid.tolist()))
    if (dest/'complete.json').exists():
        return train, hold
    raw = np.asarray(arrays['layout'][:, :, 0])
    occupied = np.asarray(arrays['occupied'])
    scales = probe.fit_scales(raw, occupied, train)
    exp.write(HERE/'scales.json', {k: v.tolist() for k, v in scales.items()})
    tick = time.perf_counter()
    for region, slots in enumerate(exp.tokens.REGIONS):
        ns = len(slots)
        cache = np.lib.format.open_memmap(dest/f'evidence{region}.npy', mode='w+', dtype='float32',
            shape=(len(rows), 2, len(rid), ns, ns, 14))
        usable = np.lib.format.open_memmap(dest/f'usable{region}.npy', mode='w+', dtype='bool',
            shape=(len(rows), 2, len(rid), ns, ns))
        for first in range(0, len(rows), BATCH):
            ids = np.arange(first, min(first+BATCH, len(rows)))
            for p in range(2):
                deepq, deepr = np.asarray(arrays['deep'][ids])[:, p, slots], np.asarray(arrays['deep'][rid])[:, p, slots]
                fineq, finer = np.asarray(arrays['fine'][ids])[:, p, slots], np.asarray(arrays['fine'][rid])[:, p, slots]
                values = [np.einsum('qid,kjd->qkij', deepq, deepr)]
                values += [np.einsum('qid,kjd->qkij', fineq[:, :, b], finer[:, :, b]) for b in range(4)]
                other = {'cos': [], 'level': [], 'separate': []}
                for b in range(3):
                    comparison, good = probe.compare(raw[ids][:, p, slots, b], raw[rid][:, p, slots, b],
                        occupied[ids][:, p, slots], occupied[rid][:, p, slots], scales['meanbin'][p, b],
                        scales['level'][p, region, b], scales['shape'][p, region, b])
                    for key in other:
                        # Match the original cosine range; absent layouts cannot enter the selector.
                        other[key].append(np.where(good, 2*comparison[key]-1, 0))
                values += other['cos']+other['level']+other['separate']
                cache[ids, p] = np.stack(values, -1)
                usable[ids, p] = good
        cache.flush()
        usable.flush()
        print('PAIR_CACHE', region, round(time.perf_counter()-tick, 1), flush=True)
    exp.write(dest/'complete.json', dict(n=len(rows), train=len(train), holdout=len(hold)))
    return train, hold


def pair_cache():
    return [(np.load(HERE/'pair_cache'/f'evidence{r}.npy', mmap_mode='r'),
             np.load(HERE/'pair_cache'/f'usable{r}.npy', mmap_mode='r')) for r in range(3)]


class MatrixMatch(exp.CachedMatch):
    """Original M3 transport/writeback, supplied with frozen comparison matrices."""
    def forward(self, x, anchor, local, slots, local_slots, valid, energy, local_energy,
                reference, pair_valid, return_details=False):
        qm, rm = local, reference['local']
        pair_valid = pair_valid & self.cross_orbit
        source = reference['features']
        read_masks = torch.minimum(reference['slots'].clamp(0, 1),
            grid_occupancy(reference['valid'].to(source), source.shape[-2:]).clamp(0, 1))
        z = F.layer_norm(source.permute(0, 2, 3, 1), (source.shape[1],)).permute(0, 3, 1, 2)
        value = torch.einsum('kshw,kdhw->ksd', read_masks, self.project(z))/rm.clamp_min(1e-8)[..., None]
        parts, weights_all, ref_scores, ref_live = [], [], [], []
        traces = {k: [] for k in ('transport', 'similarity', 'branch_similarity', 'branch_weights')}
        for region, indexes in enumerate(exp.tokens.REGIONS):
            ix = indexes.tolist()
            evidence = self.comparisons[region]
            if not self.frequency:
                evidence = evidence.clone()
                evidence[..., 2:] = 0
            logits = self.branch_bias.expand_as(evidence)+self.selector(evidence)
            active = torch.ones_like(evidence, dtype=torch.bool)
            active[..., 2:] = self.layout_valid[region][..., None] if self.layout else True
            if not self.frequency:
                active[..., 2:] = False
            bw = logits.masked_fill(~active, -torch.inf).softmax(-1)
            sim = (bw*evidence).sum(-1)
            qa, ra = qm[:, None, ix], rm[None, :, ix]
            allowed = (qa > 0)[..., :, None] & (ra > 0)[..., None, :] & pair_valid[..., None, None]
            transport = sinkhorn(sim, qa, ra)*allowed.to(sim)
            row_mass = transport.sum(-1)
            score = (transport*sim).sum(-1)/row_mass.clamp_min(1e-20)
            available = row_mass > 0
            weights = (score/self.temperature).masked_fill(~available, -1e4).softmax(1)*available
            weights = weights/weights.sum(1, keepdim=True).clamp_min(1e-20)
            moved = torch.einsum('bkij,kjd->bkid', transport, value[:, ix])
            offered = qa/qa.sum(-1, keepdim=True).clamp_min(1e-20)
            moved = moved/offered[..., None].clamp_min(1e-20)
            parts.append((weights[..., None]*moved).sum(1))
            weights_all.append(weights)
            mass = row_mass.sum(-1)
            ref_scores.append((score*row_mass).sum(-1)/mass.clamp_min(1e-20))
            ref_live.append(mass > 0)
            if return_details:
                for key, value_trace in zip(traces, (transport, sim, evidence, bw)):
                    traces[key].append(value_trace)
        live = torch.stack(ref_live, -1)
        relation = (torch.stack(ref_scores, -1)*live).sum(-1)/live.sum(-1).clamp_min(1)
        relation = (relation/self.temperature).masked_fill(~live.any(-1), -1e4)
        write_masks = torch.minimum(slots.clamp(0, 1), grid_occupancy(valid.to(x), x.shape[-2:]).clamp(0, 1))
        write = write_masks/write_masks.sum(1, keepdim=True).clamp_min(1)
        mixed = torch.einsum('bshw,bsd->bdhw', write, torch.cat(parts, 1))
        update = self.gain*self.output(mixed)
        detail = dict(update=update, reference_logits=relation, reference_valid=live.any(-1),
            reference_weights=torch.cat(weights_all, -1), **{k: tuple(v) for k, v in traces.items()})
        return x+update, detail


def make(arm, seed):
    model = exp.current.load_model(exp.PACKAGE/'weights/selected_s17.pt', exp.DEVICE)
    model.requires_grad_(False)
    for p in range(2):
        op = MatrixMatch()
        op.load_state_dict(model.m3[0][p].state_dict())
        op.requires_grad_(False)
        if seed is not None:
            torch.manual_seed(seed+100*p)
            op.selector = nn.Sequential(nn.Linear(5, 16), nn.GELU(), nn.Linear(16, 5))
            nn.init.zeros_(op.selector[-1].weight)
            nn.init.zeros_(op.selector[-1].bias)
            op.branch_bias = nn.Parameter(torch.tensor([.5, .25, 1/12, 1/12, 1/12]).log())
            op.selector.requires_grad_(True)
            op.branch_bias.requires_grad_(True)
            op.project.requires_grad_(True)
            op.output.requires_grad_(True)
        op.frequency = arm != 'S_spatial'
        op.layout = arm in ('C_cosine', 'L_level', 'N_level_pattern')
        model.m3[0][p] = op.to(exp.DEVICE)
    model.eval()
    return model


def forward(model, arm, cache, arrays, ids, rid, rows, details=False, exclude_orbit=True):
    legal = torch.tensor([[probe.orbit(rows[i]) != probe.orbit(rows[j]) for j in rid] for i in ids],
                         device=exp.DEVICE)
    if not exclude_orbit:
        legal[:] = True
    for p, op in enumerate(model.m3[0]):
        op.cross_orbit = legal
        op.comparisons = [exp.tensor(a, ids)[:, p][..., STREAMS[arm]] for a, _ in cache]
        op.layout_valid = [exp.tensor(b, ids)[:, p].bool() for _, b in cache]
    return exp.forward(model, arrays, ids, rid, rows, details=details)


@torch.no_grad()
def predict(model, arm, cache, arrays, ids, rid, rows):
    scores, relations, validity, norms = [], [], [], []
    for first in range(0, len(ids), BATCH):
        batch = ids[first:first+BATCH]
        score, ds = forward(model, arm, cache, arrays, batch, rid, rows)
        scores.append(score.cpu().numpy())
        live = torch.stack([d['reference_valid'] for d in ds])
        rel = (torch.stack([d['reference_logits'] for d in ds])*live).sum(0)/live.sum(0).clamp_min(1)
        relations.append(rel.cpu().numpy())
        validity.append(live.any(0).cpu().numpy())
        updates = torch.stack([d['update'] for d in ds], 1)
        original = exp.tensor(arrays['x'], batch)
        norms.append((updates.flatten(2).norm(dim=-1)/original.flatten(2).norm(dim=-1).clamp_min(1e-8)).cpu().numpy())
    return dict(ids=ids, score=np.concatenate(scores), relation=np.concatenate(relations),
                valid=np.concatenate(validity), update_ratio=np.concatenate(norms))


def check(rows, nf, rid, arrays, cache, train):
    # Detect unintended changes to the transport or full classification path.
    original, norm = exp.make('B_current', None, arrays, nf)
    rebuilt = make('W_weighted', None)
    ids = train[:16]
    with torch.no_grad():
        a, _ = exp.forward(original, arrays, ids, rid, rows, norm)
        b, _ = forward(rebuilt, 'W_weighted', cache, arrays, ids, rid, rows, exclude_orbit=False)
    error = float((a-b).abs().max())
    assert error < 1e-4, error
    del original, rebuilt
    gradients = []
    y = torch.tensor([rows[i]['label'] for i in ids], device=exp.DEVICE)
    ry = torch.tensor([rows[i]['label'] for i in rid], device=exp.DEVICE)
    for arm in ARMS:
        model = make(arm, 17)
        score, info = forward(model, arm, cache, arrays, ids, rid, rows)
        classification, relation = F.cross_entropy(score, y), exp.relation_loss(info, y, ry).mean()
        # Classification must reach both the comparator and the value path.
        classification.backward(retain_graph=True)
        ce = {k: float(v.grad.norm()) for k, v in model.named_parameters() if v.grad is not None}
        assert ce and all(k.startswith('m3.') for k in ce)
        for part in ('.selector.', '.project.', '.output.'):
            assert any(part in k and v > 0 for k, v in ce.items()), (arm, part)
        model.zero_grad(set_to_none=True)
        relation.backward()
        rel = {k: float(v.grad.norm()) for k, v in model.named_parameters() if v.grad is not None}
        gradients.append(dict(arm=arm, trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                              classification=ce, relation=rel))
        del model, score, info, classification, relation
    exp.write(HERE/'checks.json', dict(original_forward_max_error=error, gradients=gradients))
    print('CHECK_PASS', error, flush=True)


def train_all(rows, nf, rid, arrays, cache, train, hold):
    y = np.array([r['label'] for r in rows])
    cohorts = {'source_holdout': hold, 'development': np.arange(nf, len(rows))}
    frame = pd.DataFrame([rows[i] for i in train])
    weights = np.zeros(len(rows), np.float32)
    weights[train] = len(train)/(8*frame.groupby('label').mmsi.transform('nunique').to_numpy()
                               *frame.groupby('mmsi').mmsi.transform('size').to_numpy())
    ry = torch.tensor(y[rid], device=exp.DEVICE)
    for seed in SEEDS:
        for arm in ARMS:
            dest = HERE/'training'/f'{arm}_{seed}'
            dest.mkdir(parents=True, exist_ok=True)
            if (dest/'complete.json').exists():
                continue
            tick = time.perf_counter()
            model = make(arm, seed)
            selector, values = [], []
            for name, p in model.named_parameters():
                if p.requires_grad:
                    (values if '.project.' in name or '.output.' in name else selector).append(p)
            optimizer = torch.optim.AdamW([dict(params=selector, lr=.001), dict(params=values, lr=.0001)], weight_decay=.01)
            for cohort, ids in cohorts.items():
                np.savez_compressed(dest/f'initial_{cohort}.npz', **predict(model, arm, cache, arrays, ids, rid, rows))
            curve = []
            rng = np.random.default_rng(seed)
            for epoch in range(1, EPOCHS+1):
                order, loss_values = rng.permutation(train), []
                for first in range(0, len(train), BATCH):
                    ids = order[first:first+BATCH]
                    optimizer.zero_grad(set_to_none=True)
                    score, info = forward(model, arm, cache, arrays, ids, rid, rows)
                    target = torch.tensor(y[ids], device=exp.DEVICE)
                    per_sample = F.cross_entropy(score, target, reduction='none')+.1*exp.relation_loss(info, target, ry)
                    loss = (per_sample*torch.tensor(weights[ids], device=exp.DEVICE)).mean()
                    assert torch.isfinite(loss)
                    loss.backward()
                    optimizer.step()
                    loss_values.append(float(loss.detach()))
                outcome = predict(model, arm, cache, arrays, hold, rid, rows)
                curve.append(dict(epoch=epoch, loss=float(np.mean(loss_values)), **exp.metrics(y[hold], outcome['score'])))
                pd.DataFrame(curve).to_csv(dest/'curve.csv', index=False)
                print('EPOCH', arm, seed, epoch, curve[-1], round(time.perf_counter()-tick, 1), flush=True)
            for cohort, ids in cohorts.items():
                np.savez_compressed(dest/f'final_{cohort}.npz', **predict(model, arm, cache, arrays, ids, rid, rows))
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items() if k.startswith('m3.')}, dest/'m3.pt')
            exp.write(dest/'complete.json', dict(arm=arm, seed=seed, epochs=EPOCHS,
                seconds=time.perf_counter()-tick, trainable=sum(p.numel() for p in selector+values)))
            print('DONE', arm, seed, round(time.perf_counter()-tick, 1), flush=True)
            del model, optimizer, selector, values, score, info, loss, per_sample
            torch.cuda.empty_cache()


def tables(rows, rid):
    ry = np.array([rows[i]['label'] for i in rid])
    summary, paired, classes = [], [], []
    for cohort in ('source_holdout', 'development'):
        for seed in SEEDS:
            results = {}
            for arm in ARMS:
                path = HERE/'training'/f'{arm}_{seed}'
                for when in ('initial', 'final'):
                    z = dict(np.load(path/f'{when}_{cohort}.npz'))
                    y = np.array([rows[i]['label'] for i in z['ids']])
                    summary.append(dict(cohort=cohort, arm=arm, seed=seed, when=when,
                        **exp.metrics(y, z['score']), **old_analysis.relation_metrics(y, ry, z['relation'], z['valid']),
                        mean_update_ratio=float(z['update_ratio'].mean())))
                results[arm] = z
                for c in np.unique(y):
                    classes.append(dict(cohort=cohort, arm=arm, seed=seed, label=int(c), n=int((y==c).sum()),
                        recall=float((z['score'].argmax(1)[y==c]==c).mean()*100)))
            for base in ARMS[:-1]:
                a, b = results['N_level_pattern']['score'], results[base]['score']
                ca, cb = a.argmax(1)==y, b.argmax(1)==y
                paired.append(dict(cohort=cohort, seed=seed, base=base, rescue=int((ca & ~cb).sum()),
                    harm=int((~ca & cb).sum()), changed=int((a.argmax(1)!=b.argmax(1)).sum()),
                    **{k: exp.metrics(y, a)[k]-exp.metrics(y, b)[k] for k in ('oa','ba','nll','brier')}))
    pd.DataFrame(summary).to_csv(HERE/'metrics.csv', index=False)
    pd.DataFrame(paired).to_csv(HERE/'paired.csv', index=False)
    pd.DataFrame(classes).to_csv(HERE/'classes.csv', index=False)
    print(pd.DataFrame(summary).query("when=='final'").groupby(['cohort', 'arm'])[['oa','ba','nll','map_at_r','mean_update_ratio']].agg(['mean','std']).to_string(), flush=True)
    print(pd.DataFrame(paired).to_string(index=False), flush=True)


if __name__ == '__main__':
    exp.current.setup(exp.DEVICE)
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare','check','train','tables','all'))
    action = parser.parse_args().action
    rows, nf, refs, rid, arrays = exp.load()
    train, hold = prepare(rows, nf, rid, arrays)
    cache = pair_cache()
    if action in ('check','all'): check(rows, nf, rid, arrays, cache, train)
    if action in ('train','all'): train_all(rows, nf, rid, arrays, cache, train, hold)
    if action in ('tables','all'): tables(rows, rid)
