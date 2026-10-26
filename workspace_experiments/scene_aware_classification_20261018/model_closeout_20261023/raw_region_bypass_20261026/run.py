"""Minimal raw-region bypass at M2; fixed source selection and three controls."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'readout_bottleneck_20261026'))
import probe_compression as prior
base = prior.base
REPAIR = ROOT/'public_m123_cross_dataset_20261026/normalization_repair_20261026'


def train(x, ext, aux, eaux, anchors, table, fit, val, seed, arm, out, hidden=128):
    torch.manual_seed(seed)
    raw_head = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.GELU(), nn.Linear(hidden, 4))
    if arm == 'raw_only':
        model = raw_head
        xx, ee = x, ext
    else:
        model = nn.Sequential(nn.Linear(x.shape[1]+aux.shape[1], hidden), nn.GELU(), nn.Linear(hidden, 4))
        with torch.no_grad():
            model[0].weight.zero_()
            model[0].weight[:, :x.shape[1]].copy_(raw_head[0].weight)
            model[0].bias.copy_(raw_head[0].bias)
            model[2].load_state_dict(raw_head[2].state_dict())
        xx, ee = np.c_[x, aux], np.c_[ext, eaux]
    initial = {k:v.detach().clone() for k,v in model.state_dict().items()}
    is_anchor = arm == 'anchored_raw_summary'
    sb, eb = anchors if is_anchor else (np.zeros((len(xx), 4), np.float32), np.zeros((len(ee), 4), np.float32))
    xt = torch.tensor(xx)
    bt = torch.tensor(sb)
    y = table.class_id.to_numpy(int)
    labels = torch.tensor(y[fit])
    weights = torch.tensor(base.sample_weights(table, fit), dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    best = base.ranking(y[val], sb[val]) if is_anchor else None
    chosen, selected_step = None, 0
    curve = []
    if is_anchor:
        curve.append(dict(step=0, loss=None, **base.classification_metrics(y[val], sb[val])))
    start = time.perf_counter()
    for step in range(1, 201):
        optimizer.zero_grad(set_to_none=True)
        score = bt[fit]+model(xt[fit])
        loss = (nn.functional.cross_entropy(score, labels, reduction='none')*weights).mean()
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        if step % 5 == 0:
            with torch.inference_mode():
                v = sb[val]+model(xt[val]).numpy()
            j = base.ranking(y[val], v)
            curve.append(dict(step=step, loss=float(loss.detach()), **base.classification_metrics(y[val], v)))
            if best is None or j > best:
                best, selected_step = j, step
                chosen = {k:z.detach().clone() for k,z in model.state_dict().items()}
    if chosen is None:
        source, external = sb.copy(), eb.copy()
    else:
        model.load_state_dict(chosen)
        with torch.inference_mode():
            source = sb+model(xt).numpy()
            external = eb+model(torch.tensor(ee)).numpy()
    torch.save(dict(state=chosen, initial=initial, step=selected_step, raw_dim=x.shape[1], aux_dim=0 if arm=='raw_only' else aux.shape[1]), out/f'{arm}_seed{seed}.pt')
    pd.DataFrame(curve).to_csv(out/f'{arm}_seed{seed}_curve.csv', index=False)
    return source, external, dict(arm=arm, seed=seed, step=selected_step,
        seconds=time.perf_counter()-start, parameters=sum(p.numel() for p in model.parameters()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, choices=('oss1', 'oss2'))
    source = parser.parse_args().source
    torch.set_num_threads(2)
    root, cross, table, target, raw, eraw, local, elocal = base.dataset(source)
    fit = np.flatnonzero(table.role.eq('fit_train'))
    val = np.flatnonzero(table.role.eq('fit_validation'))
    for field in ('mmsi', 'pass_group'):
        assert not set(table.iloc[fit][field]) & set(table.iloc[val][field])
        assert not set(table.loc[table.role.ne('evaluation'), field]) & set(target[field])
    out = HERE/source
    out.mkdir(parents=True, exist_ok=True)
    x = base.representations(raw, local)['regional']
    ext = base.representations(eraw, elocal)['regional']
    aux, eaux = prior.summary_inputs(root, cross, table, target)
    sc, sa = base.standardize(x, fit), base.standardize(aux, fit)
    np.savez_compressed(out/'raw_scaler.npz', **sc)
    np.savez_compressed(out/'summary_scaler.npz', **sa)
    x, ext = base.transform(x, sc), base.transform(ext, sc)
    aux, eaux = base.transform(aux, sa), base.transform(eaux, sa)
    src, dst, receipts, checks = {}, {}, [], []
    for seed in base.SEEDS:
        s = np.load(REPAIR/f'source_{source}/seed{seed}/predictions.npz')
        e = np.load(REPAIR/cross.name/f'seed{seed}/predictions.npz')
        np.testing.assert_array_equal(e['obs_id'], target.obs_id)
        for z, t in ((s, table), (e, target)):
            if 'obs_id' in z.files:
                np.testing.assert_array_equal(z['obs_id'], t.obs_id)
        anchors = s['s12'].astype(np.float32), e['s12'].astype(np.float32)
        for arm in ('raw_only', 'raw_summary', 'anchored_raw_summary'):
            key = f'{arm}_seed{seed}'
            src[key], dst[key], receipt = train(x, ext, aux, eaux, anchors, table, fit, val, seed, arm, out)
            receipts.append(receipt)
        # Simple control must be the same model, not a weaker convenient baseline.
        for side, scores in (('source', src), ('cross', dst)):
            old = np.load(prior.PRIOR/f'classical_fit_only/{source}/{side}_predictions.npz')
            expected = old[f'regional_mlp_seed{seed}']
            actual = scores[f'raw_only_seed{seed}']
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-5)
            np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
            checks.append(dict(side=side, seed=seed, raw_replay_error=float(abs(actual-expected).max())))
        print('FROZEN', source, seed, flush=True)
    np.savez_compressed(out/'source_predictions.npz', obs_id=table.obs_id.to_numpy(), **src)
    np.savez_compressed(out/'cross_predictions.npz', obs_id=target.obs_id.to_numpy(), **dst)
    pd.DataFrame(receipts).to_csv(out/'training.csv', index=False)
    base.write_json(out/'checks.json', checks)
    truth = base.tr.read_table(cross/'target_table.csv', labels=True)
    np.testing.assert_array_equal(target.obs_id, truth.obs_id)
    rows = []
    for key, score in src.items():
        for role, ids in table.groupby('role').indices.items():
            rows.append(dict(source=source, cohort=role, method=key, **base.classification_metrics(table.iloc[ids].class_id.to_numpy(int), score[ids])))
        rows.append(dict(source=source, cohort='cross', method=key, **base.classification_metrics(truth.class_id.to_numpy(int), dst[key])))
    pd.DataFrame(rows).to_csv(out/'metrics.csv', index=False)
    base.write_json(out/'complete.json', dict(source=source, fits=9, seeds=base.SEEDS, target_selection=False, raw_review_pending=True))
    print(pd.DataFrame(rows).query("cohort=='cross'").to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
