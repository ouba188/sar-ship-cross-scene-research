"""Same source bank/local features: simple matching and matched s12 additions."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from run_classical import HERE, ROOT, TRANSFER, SEEDS, tr, write_json, load_selected
from local_baselines import dn4_scores, regional_prototype_scores


class SharedHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 12), nn.GELU(), nn.Linear(12, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        d = self.net(x).squeeze(-1)
        return d-d.mean(1, keepdim=True)


def head_input(score, available, base):
    z = base-base.mean(1, keepdims=True)
    rival = np.stack([np.delete(z, c, 1).max(1) for c in range(4)], 1)
    return np.stack((score, available.astype(float), z, rival, z-rival), -1).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', choices=('oss1', 'oss2'), required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    name = args.source
    other = 'oss2' if name == 'oss1' else 'oss1'
    root = ROOT/f'{name}_m123_replication_20261026'
    cross = TRANSFER/f'{name}-to-{other}'
    repair = TRANSFER/'normalization_repair_20261026'
    mroot = root/('m3_results_grid01' if name == 'oss1' else 'm3_results')
    table = pd.read_csv(root/'stack_table.csv', dtype={'mmsi': str})
    target = tr.read_table(cross/'target_table.csv')
    protocol = tr.read_json(cross/'protocol.json')
    local = tr.load_npz(mroot/'local.npz')
    bank_rows = tr.load_npz(mroot/'references.npz')['bank'].ravel()
    projection = tr.load_npz(mroot/'projection.npz')
    eraw = load_selected(__import__('pathlib').Path(protocol['target_cache'])/'m3_raw.npz',
                        target, ('layer1', 'mass'))
    eq = ((eraw['layer1']-projection['mean'])@projection['components'].T).astype(np.float32)
    eq /= np.maximum(np.linalg.norm(eq, axis=-1, keepdims=True), 1e-8)
    eq[eraw['mass'] <= 0] = 0
    queries = np.concatenate((local['x'], eq))
    valid = np.concatenate((local['mass'] > 0, eraw['mass'] > 0))
    ids = np.r_[table.mmsi.to_numpy(), target.mmsi.to_numpy()]
    bank_ids = table.iloc[bank_rows].mmsi.to_numpy()
    bank_y = table.iloc[bank_rows].class_id.to_numpy(int)
    assert table.iloc[bank_rows].role.eq('fit_train').all()
    train = np.flatnonzero(table.role.eq('utility_train'))
    val = np.flatnonzero(table.role.eq('selection'))
    test = np.flatnonzero(table.role.eq('evaluation'))
    y = table.class_id.to_numpy(int)
    out = HERE/'local'/name
    out.mkdir(parents=True, exist_ok=True)
    table.iloc[bank_rows].to_csv(out/'bank.csv', index=False)
    common = dict(query=queries, query_valid=valid, bank=local['x'][bank_rows],
                  bank_valid=local['mass'][bank_rows] > 0, bank_y=bank_y,
                  query_ids=ids, bank_ids=bank_ids, num_classes=4, return_available=True)
    operators = {'dn4': lambda: dn4_scores(**common),
                 'region_prototype': lambda: regional_prototype_scores(**common),
                 'merged_prototype': lambda: regional_prototype_scores(**common, merge_regions=True)}
    frozen, receipts = {}, []
    for method, op in operators.items():
        tick = time.perf_counter()
        score, available = op()
        assert np.isfinite(score).all()
        # No-evidence classes cannot win by a zero sentinel over negative cosines.
        direct = np.where(available, score, -1e4)
        direct[~available.any(1)] = 0
        inverse_temperature = min((1.,3.,10.,30.),
            key=lambda t: tr.m3.metrics(y[val], direct[val]*t)['NLL'])
        frozen[f'{method}_direct'] = direct*inverse_temperature
        np.savez_compressed(out/f'{method}_evidence.npz', score=score, available=available)
        receipts.append(dict(method=method, seconds=time.perf_counter()-tick,
                             inverse_temperature=inverse_temperature, no_evidence=int((~available.any(1)).sum())))
        operators[method] = (score, available)
    operators['context_only'] = (np.zeros((len(queries),4),np.float32),
                                 np.ones((len(queries),4),bool))
    for seed in SEEDS:
        source_base = tr.load_npz(repair/f'source_{name}/seed{seed}/predictions.npz')['s12']
        cross_base = tr.load_npz(repair/f'{name}-to-{other}/seed{seed}/predictions.npz')['s12']
        base = np.r_[source_base,cross_base].astype(np.float32)
        for method, (score,available) in operators.items():
            dest = out/f'{method}_seed{seed}'
            dest.mkdir(exist_ok=True)
            raw = head_input(score, available, base)
            scaler = tr.m3.scale_fit(raw, train)
            x = tr.m3.transform(raw, scaler)
            torch.manual_seed(seed)
            head = SharedHead()
            delta, step = tr.m3.fit_head('head', head, x, base, y, train, val, dest)
            frozen[f'{method}_residual_seed{seed}'] = base+delta
            np.savez_compressed(dest/'scaler.npz', **scaler)
            receipts.append(dict(method=f'{method}_residual', seed=seed, selected_step=step,
                                 parameters=sum(z.numel() for z in head.parameters())))
        frozen[f's12_seed{seed}'] = base
    np.savez_compressed(out/'source_predictions.npz', obs_id=table.obs_id.to_numpy(),
                        **{k:v[:len(table)] for k,v in frozen.items()})
    np.savez_compressed(out/'cross_predictions.npz', obs_id=target.obs_id.to_numpy(),
                        **{k:v[len(table):] for k,v in frozen.items()})
    external = tr.read_table(cross/'target_table.csv', labels=True)
    rows = []
    for k,z in frozen.items():
        rows.append(dict(cohort='within', method=k, **tr.m3.metrics(y[test],z[test])))
        rows.append(dict(cohort='cross', method=k,
                         **tr.m3.metrics(external.class_id.to_numpy(int),z[len(table):])))
    pd.DataFrame(rows).to_csv(out/'metrics.csv', index=False)
    write_json(out/'receipts.json', receipts)
    write_json(out/'complete.json',dict(source=name, references=32, train=len(train),selection=len(val),
                                       within=len(test), cross=len(target), target_fit=False))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
