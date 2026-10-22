"""Entire-stack source cross-fitting for a matched residual-strength gate."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import gc
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'geometry_evidence_gate_20261021'
spec = importlib.util.spec_from_file_location('previous_gate', PREVIOUS / 'run.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
old = previous.old
ARMS = ('in_constant', 'oof_constant', 'in_score', 'oof_score')


class Gate(nn.Module):
    def __init__(self, constant=False):
        super().__init__()
        self.constant = constant
        if constant:
            self.bias = nn.Parameter(torch.zeros(()))
        else:
            self.first = nn.Linear(16, 8)
            self.last = nn.Linear(8, 1)
            nn.init.zeros_(self.last.weight)
            nn.init.zeros_(self.last.bias)

    def forward(self, a, d):
        if self.constant:
            z = self.bias.expand(len(a))
        else:
            z = self.last(F.gelu(self.first(torch.cat((a, d), 1)))).squeeze(1)
        return 2 * torch.sigmoid(z)


def predict(head, x, base=None):
    values = []
    with torch.no_grad():
        for first in range(0, len(x), 512):
            end = first + 512
            out = head(x[first:end])
            if base is not None:
                out = out + base[first:end]
            values.append(out.cpu().numpy())
    return np.concatenate(values)


def fit_residual(head, x, base, f, tr, val, seed, fold, stage, inner, logs):
    y = f.class_id.to_numpy()
    yy = torch.tensor(y, device='cuda')
    weights = torch.tensor(old.prior.weights(f, tr), device='cuda', dtype=torch.float32)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.001)
    rng = np.random.default_rng((20261020 if stage == 'keep_refs' else 20261021) + seed + 31 * fold)
    best, state, chosen = None, None, 0
    for step in range(151):
        if step in (0, 50, 100, 150):
            with torch.no_grad():
                scores = (base[val] + head(x[val])).cpu().numpy()
            met = old.control.metrics(y[val], scores)
            logs.append(dict(fold=fold, inner=inner, seed=seed, stage=stage, step=step, **met))
            rank = (met['ba'], -met['brier'])
            if best is None or rank > best:
                best, chosen = rank, step
                state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            if step == 150:
                break
        pick = rng.integers(len(tr), size=128)
        ii = tr[pick]
        target = 2 * F.one_hot(yy[ii], 8).float() - 1
        loss = ((base[ii] + head(x[ii]) - target).square().mean(1) * weights[pick]).mean()
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    head.load_state_dict(state)
    head.eval()
    torch.save(dict(state=state, step=chosen), ROOT / f'inner_f{fold}_k{inner}_{stage}_s{seed}.pt')
    return predict(head, x), chosen


def inner_partition(f, tr, fold):
    ships = f.iloc[tr][['mmsi', 'class_id']].drop_duplicates()
    assert ships.mmsi.nunique() == len(ships)
    assignment = {}
    rng = np.random.default_rng(20261022 + fold)
    for _, part in ships.groupby('class_id'):
        ids = rng.permutation(np.sort(part.mmsi.to_numpy()))
        assert len(ids) >= 3
        assignment.update({v: j % 3 for j, v in enumerate(ids)})
    return np.array([assignment[v] for v in f.iloc[tr].mmsi])


def record_predictions(records, f, test, fold, seed, arm, scores, w, step):
    y = f.class_id.to_numpy()[test]
    wrong = scores.copy()
    wrong[np.arange(len(test)), y] = -np.inf
    for j, i in enumerate(test):
        row = f.iloc[i]
        records.append(dict(fold=fold, seed=seed, arm=arm, obs_id=int(row.obs_id),
            mmsi=row.mmsi, port=row.port, product=row['product'], class_id=int(y[j]),
            prediction=int(scores[j].argmax()), correct=bool(scores[j].argmax() == y[j]),
            margin=float(scores[j, y[j]] - wrong[j].max()), weight=float(w[j]), step=step))


def fit():
    torch.set_num_threads(2)
    start = time.perf_counter()
    f, splits = old.relation.load_data()
    y = f.class_id.to_numpy()
    raw = np.load(old.five.AXIS / 'descriptors.npz')
    raw_base = np.concatenate((raw['base'], raw['axis']), 1)
    raw_five = np.load(old.strip.PREV / 'keep_refs.npy', mmap_mode='r').reshape(len(f), 60, 64)
    cache = np.load(old.ROOT / 'features.npz')
    logs, coverage, partitions, exposure, checks = [], [], [], [], []
    gate_logs, metrics, predictions, weights_out = [], [], [], []

    for fold, (tr, val, test) in enumerate(splits):
        assignment = inner_partition(f, tr, fold)
        partitions.extend(dict(fold=fold, row=int(i), obs_id=int(f.iloc[i].obs_id),
            mmsi=f.iloc[i].mmsi, inner=int(k)) for i, k in zip(tr, assignment))
        oof_a = np.full((3, len(f), 8), np.nan, np.float32)
        oof_d = np.full_like(oof_a, np.nan)
        for inner in range(3):
            fit_rows = tr[assignment != inner]
            held = tr[assignment == inner]
            held_ids = set(f.iloc[held].mmsi)
            assert not held_ids & set(f.iloc[fit_rows].mmsi)
            assert not held_ids & set(f.iloc[val].mmsi)
            assert not held_ids & set(f.iloc[test].mmsi)
            assert set(y[fit_rows]) == set(range(8))
            checks.append(dict(fold=fold, inner=inner, check='held_identity_exclusion_all_fit_stages', error=0))
            for part, ii in (('fit', fit_rows), ('held', held), ('validation', val)):
                coverage.append(dict(fold=fold, inner=inner, partition=part, n=len(ii),
                    ships=f.iloc[ii].mmsi.nunique(), classes=len(np.unique(y[ii])),
                    products=f.iloc[ii]['product'].nunique()))
            # All normalization statistics and supervised fits use exactly fit_rows.
            mean = raw_base[fit_rows].mean(0)
            std = np.maximum(raw_base[fit_rows].std(0), 1e-6)
            x = (raw_base - mean) / std / np.sqrt(raw_base.shape[1])
            w = old.prior.weights(f, fit_rows)
            candidates = []
            for alpha in (.1, 1., 10.):
                ridge = RidgeClassifier(alpha=alpha, solver='cholesky').fit(x[fit_rows], y[fit_rows], sample_weight=w)
                assert np.array_equal(ridge.classes_, np.arange(8))
                met = old.control.metrics(y[val], ridge.decision_function(x[val]))
                logs.append(dict(fold=fold, inner=inner, seed=-1, stage='ridge', step=alpha, **met))
                candidates.append((met['ba'], -met['brier'], alpha, ridge))
            _, _, alpha, ridge = max(candidates, key=lambda t: t[:2])
            ridge_scores = ridge.decision_function(x).astype(np.float32)
            five_mean = raw_five[fit_rows].mean(0)
            five_std = np.maximum(raw_five[fit_rows].std(0), 1e-6)
            xf = torch.empty(raw_five.shape, device='cuda', dtype=torch.float32)
            for first in range(0, len(f), 512):
                xf[first:first+512] = torch.tensor((raw_five[first:first+512]-five_mean)/five_std, device='cuda')
            matrices, stats = old.standardize(cache, fit_rows)
            xd = torch.tensor(matrices['separated'], device='cuda')
            np.savez_compressed(ROOT / f'inner_f{fold}_k{inner}_normalization.npz',
                ridge_mean=mean, ridge_std=std, ridge_coef=ridge.coef_, ridge_intercept=ridge.intercept_,
                ridge_alpha=alpha, five_mean=five_mean, five_std=five_std, **stats)
            rb = torch.tensor(ridge_scores, device='cuda')
            for seed in range(3):
                torch.manual_seed(20261020 + seed)
                head = old.five.Residual().cuda()
                supplement, keep_step = fit_residual(head, xf, rb, f, fit_rows, val,
                    seed, fold, 'keep_refs', inner, logs)
                a = ridge_scores + supplement
                at = torch.tensor(a, device='cuda')
                torch.manual_seed(20261021 + seed)
                residual = old.Residual().cuda()
                d, residual_step = fit_residual(residual, xd, at, f, fit_rows, val,
                    seed, fold, 'neighborhood', inner, logs)
                oof_a[seed, held] = a[held]
                oof_d[seed, held] = d[held]
                for part, ii in (('inner_fit', fit_rows), ('inner_held', held), ('source_val', val)):
                    for stage, scores in (('A', a), ('A_plus_D', a+d)):
                        exposure.append(dict(fold=fold, inner=inner, seed=seed, partition=part,
                            stage=stage, keep_step=keep_step, residual_step=residual_step,
                            residual_abs=float(np.abs(d[ii]).mean()), **old.control.metrics(y[ii], scores[ii])))
                print('inner', fold, inner, seed, 'steps', keep_step, residual_step, flush=True)
                del head, residual, at
            del xf, xd, rb, matrices, x, candidates, ridge
            gc.collect()

        assert np.isfinite(oof_a[:, tr]).all() and np.isfinite(oof_d[:, tr]).all()
        np.savez_compressed(ROOT / f'oof_f{fold}.npz', rows=tr, inner=assignment,
                            base=oof_a[:, tr], residual=oof_d[:, tr])
        for seed in range(3):
            frozen = np.load(PREVIOUS / f'inputs_f{fold}_s{seed}.npz')
            a, d = frozen['base'], frozen['residual']
            reference = a+d
            saved = np.load(old.ROOT / f'scores_f{fold}_separated_s{seed}.npz')
            np.testing.assert_array_equal(test, saved['rows'])
            error = float(np.abs(reference[test]-saved['scores']).max())
            assert error < 2e-5
            checks.append(dict(fold=fold, seed=seed, check='unchanged_full_reference', error=error))
            for label, aa, dd in (('full_in_sample', a[tr], d[tr]),
                                  ('oof_training', oof_a[seed,tr], oof_d[seed,tr])):
                for stage, ss in (('A', aa), ('A_plus_D', aa+dd)):
                    exposure.append(dict(fold=fold, inner=-1, seed=seed, partition=label, stage=stage,
                        residual_abs=float(np.abs(dd).mean()), **old.control.metrics(y[tr], ss)))
            eval_a, eval_d = torch.tensor(a, device='cuda'), torch.tensor(d, device='cuda')
            source_weight = torch.tensor(old.prior.weights(f, tr), device='cuda', dtype=torch.float32)
            yy = torch.tensor(y[tr], device='cuda')
            for arm in ARMS:
                aa = oof_a[seed, tr] if arm.startswith('oof') else a[tr]
                dd = oof_d[seed, tr] if arm.startswith('oof') else d[tr]
                train_a = torch.tensor(aa, device='cuda')
                train_d = torch.tensor(dd, device='cuda')
                torch.manual_seed(20261022+seed)
                gate = Gate(arm.endswith('constant')).cuda()
                optimizer = torch.optim.AdamW(gate.parameters(), lr=.001, weight_decay=.001)
                rng = np.random.default_rng(20261022+seed+31*fold)
                best, state, chosen = None, None, 0
                for step in range(151):
                    if step in (0,50,100,150):
                        with torch.no_grad():
                            ss = (eval_a[val]+gate(eval_a[val],eval_d[val])[:,None]*eval_d[val]).cpu().numpy()
                        met = old.control.metrics(y[val],ss)
                        gate_logs.append(dict(fold=fold,seed=seed,arm=arm,step=step,**met))
                        rank = (met['ba'],-met['brier'])
                        if best is None or rank > best:
                            best, chosen = rank,step
                            state = {k:v.detach().cpu().clone() for k,v in gate.state_dict().items()}
                        if step==0:
                            np.testing.assert_array_equal(ss,reference[val])
                        if step==150:
                            break
                    pick = rng.integers(len(tr),size=128)
                    ss = train_a[pick]+gate(train_a[pick],train_d[pick])[:,None]*train_d[pick]
                    target = 2*F.one_hot(yy[pick],8).float()-1
                    loss = ((ss-target).square().mean(1)*source_weight[pick]).mean()
                    assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                gate.load_state_dict(state)
                torch.save(dict(state=state,step=chosen,arm=arm),ROOT/f'head_f{fold}_{arm}_s{seed}.pt')
                with torch.no_grad():
                    ww = gate(eval_a,eval_d).cpu().numpy()
                    train_w = gate(train_a,train_d).cpu().numpy()
                replay = Gate(arm.endswith('constant')).cuda()
                replay.load_state_dict(state)
                with torch.no_grad():
                    np.testing.assert_array_equal(ww,replay(eval_a,eval_d).cpu().numpy())
                checks.append(dict(fold=fold,seed=seed,arm=arm,check='selected_gate_replay',error=0))
                score = a+ww[:,None]*d
                for part, ii in (('full_train',tr),('val',val),('test',test)):
                    metrics.append(dict(fold=fold,seed=seed,arm=arm,partition=part,step=chosen,
                                        **old.control.metrics(y[ii],score[ii])))
                    weights_out.append(dict(fold=fold,seed=seed,arm=arm,partition=part,step=chosen,
                        mean=float(ww[ii].mean()),std=float(ww[ii].std()),minimum=float(ww[ii].min()),
                        maximum=float(ww[ii].max()),residual_abs=float(np.abs(d[ii]).mean())))
                metrics.append(dict(fold=fold,seed=seed,arm=arm,partition='gate_training',step=chosen,
                    **old.control.metrics(y[tr],aa+train_w[:,None]*dd)))
                np.savez_compressed(ROOT/f'scores_f{fold}_{arm}_s{seed}.npz',rows=test,
                                    scores=score[test],weight=ww[test],reference=reference[test])
                record_predictions(predictions,f,test,fold,seed,arm,score[test],ww[test],chosen)
                if arm.endswith('score'):
                    wm = np.full(len(test),ww[tr].mean(),np.float32)
                    record_predictions(predictions,f,test,fold,seed,arm+'_source_mean',
                                       a[test]+wm[:,None]*d[test],wm,chosen)
                print('gate',fold,seed,arm,chosen,metrics[-2]['oa'],flush=True)
                del gate,replay,optimizer,train_a,train_d
            record_predictions(predictions,f,test,fold,seed,'reference',reference[test],np.ones(len(test)),-1)
            record_predictions(predictions,f,test,fold,seed,'baseline',a[test],np.zeros(len(test)),-1)
        for name,data in (('stack_selection',logs),('coverage',coverage),('inner_partitions',partitions),
                          ('exposure',exposure),('checks',checks),('validation_selection',gate_logs),
                          ('metrics',metrics),('predictions',predictions),('weights',weights_out)):
            pd.DataFrame(data).to_csv(ROOT/f'{name}.csv',index=False,lineterminator='\n')
        gc.collect()
    (ROOT/'TRAINING.json').write_text(json.dumps(dict(status='complete',seconds=time.perf_counter()-start,
        inner_ridge_fits=27,inner_neural_fits=54,gate_fits=36,steps=150,score_gate_parameters=145,
        test_rows_per_seed=4744,source_only=True,scope='development; no full target-support adaptation'),indent=2),encoding='utf-8')


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        fit()
