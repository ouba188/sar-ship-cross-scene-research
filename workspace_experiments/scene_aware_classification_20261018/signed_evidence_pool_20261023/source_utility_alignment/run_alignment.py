"""Predeclared loss x source-selection experiment on frozen signed-only inputs."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import sys
sys.dont_write_bytecode = True
import json
import time
from fractions import Fraction
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
import train_signed as parent

LOSSES = ['balancedCE', 'jointCE']
SELECTORS = ['balancedNLL', 'J']
SEEDS = [17, 29, 43]


def assessment(y, score):
    pred = score.argmax(1)
    correct = pred == y
    counts = np.bincount(y, minlength=3)
    hits = np.bincount(y[correct], minlength=3)
    nll = logsumexp(score, axis=1)-score[np.arange(len(y)), y]
    oa, ba = correct.mean(), np.mean(hits/counts)
    prob = np.exp(score-logsumexp(score, axis=1, keepdims=True))
    return dict(OA=oa*100, BA=ba*100, J=.5*(oa+ba), NLL=float(nll.mean()),
        balancedNLL=float(np.mean([nll[y == c].mean() for c in range(3)])),
        Brier=float(((prob-np.eye(3)[y])**2).sum(1).mean()))


def exact_j(y, pred):
    correct = pred == y
    return Fraction(int(correct.sum()), 2*len(y))+sum(
        (Fraction(int(correct[y == c].sum()), 6*int((y == c).sum())) for c in range(3)), Fraction(0))


def frozen_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def inputs(fold, cache, names):
    d = dict(np.load(parent.NEW/f'fold{fold}_inputs.npz'))
    outer = cache['moments'][names.index(f'fold{fold}_outer')]
    oof = np.zeros_like(outer)
    for j, name in enumerate(names):
        if name.startswith(f'fold{fold}_inner'):
            held = np.load(parent.NEW/'origins'/f'{name}.npz')['held']
            oof[held] = cache['moments'][j, held]
    raw = parent.descriptors(d['oof'], oof)
    mu = raw[d['train']].mean((0, 1))
    scale = raw[d['train']].std((0, 1)).clip(1e-8)
    return d, outer, oof, raw, mu, scale


def main():
    assert (HERE/'PROTOCOL_ZH.md').exists()
    started = time.perf_counter()
    torch.set_num_threads(2)
    (HERE/'models').mkdir(exist_ok=True)
    table = pd.read_csv(parent.EXP/'readout/targets.csv', dtype={'mmsi': str})
    y = table.class_id.to_numpy()
    yy = torch.tensor(y, dtype=torch.long)
    cache = np.load(parent.CACHE/'signed_features.npz')
    names = list(cache['reader_names'])
    original_sel = pd.read_csv(PARENT/'signed_only_retrain/selections.csv')
    records, curves, selection_rows, checks, fold_sources = [], [], [], [], []
    for fold in range(6):
        d, outer, oof, raw, mu, scale = inputs(fold, cache, names)
        tr, va = d['train'], d['validation']
        xvraw = parent.descriptors(d['base'], outer)
        x = torch.tensor((raw-mu)/scale, dtype=torch.float32)
        xv = torch.tensor((xvraw-mu)/scale, dtype=torch.float32)
        m, mv = torch.tensor(oof, dtype=torch.float32), torch.tensor(outer, dtype=torch.float32)
        b, bv = torch.tensor(d['oof'], dtype=torch.float32), torch.tensor(d['base'], dtype=torch.float32)
        counts = np.bincount(y[va], minlength=3)
        fold_sources.append(dict(fold=fold, train_n=len(tr), validation_n=len(va),
            validation_passes=table.iloc[va].pass_group.nunique(),
            **{f'val_class_{c}':int(counts[c]) for c in range(3)},
            **{f'one_correct_flip_J_pp_class_{c}':100*(.5/len(va)+1/(6*counts[c])) for c in range(3)}))
        for loss_name in LOSSES:
            for seed in SEEDS:
                tick = time.perf_counter()
                torch.manual_seed(seed)
                model = parent.Correction('signed_only')
                assert sum(p.numel() for p in model.parameters()) == 113
                with torch.no_grad():
                    assert model(xv[va], mv[va]).abs().max() == 0
                initial = frozen_state(model)
                initial_risk = float(parent.balanced_loss(bv[va], yy[va]))
                initial_j = exact_j(y[va], bv[va].numpy().argmax(1))
                best = {'balancedNLL': dict(epoch=0, state=initial, value=initial_risk),
                        'J': dict(epoch=0, state=initial, value=initial_j)}
                meta = dict(fold=fold, loss=loss_name, seed=seed)
                curve = [dict(**meta, epoch=0, train_loss=np.nan,
                    **assessment(y[va], bv[va].numpy().astype(float)),
                    selection_balancedNLL=initial_risk)]
                opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
                for epoch in range(1, 201):
                    model.train()
                    opt.zero_grad()
                    score = b[tr]+model(x[tr], m[tr])
                    balanced = parent.balanced_loss(score, yy[tr])
                    loss = balanced if loss_name == 'balancedCE' else .5*balanced+.5*torch.nn.functional.cross_entropy(score, yy[tr])
                    loss.backward()
                    opt.step()
                    if epoch % 5:
                        continue
                    model.eval()
                    with torch.no_grad():
                        val_score = bv[va]+model(xv[va], mv[va])
                        risk = float(parent.balanced_loss(val_score, yy[va]))
                    arr = val_score.numpy().astype(float)
                    jvalue = exact_j(y[va], arr.argmax(1))
                    curve.append(dict(**meta, epoch=epoch, train_loss=float(loss.detach()),
                        **assessment(y[va], arr), selection_balancedNLL=risk))
                    if risk < best['balancedNLL']['value']-1e-8:
                        best['balancedNLL'] = dict(epoch=epoch, state=frozen_state(model), value=risk)
                    if jvalue > best['J']['value']:
                        best['J'] = dict(epoch=epoch, state=frozen_state(model), value=jvalue)
                elapsed = time.perf_counter()-tick
                curves.extend(curve)
                records.append(dict(**meta, steps=200, seconds=elapsed))
                for selector, chosen in best.items():
                    cell = f'{loss_name}__{selector}'
                    ck = dict(state=chosen['state'], arm='signed_only', mu=mu, scale=scale,
                        epoch=chosen['epoch'], fold=fold, loss=loss_name, selector=selector, seed=seed)
                    path = HERE/'models'/f'fold{fold}_{cell}_seed{seed}.pt'
                    torch.save(ck, path)
                    row = next(row for row in curve if row['epoch'] == chosen['epoch'])
                    jvals = sorted(set(row['J'] for row in curve), reverse=True)
                    selection_rows.append(dict(**meta, selector=selector, cell=cell,
                        epoch=chosen['epoch'], checkpoint=str(path), parameters=113,
                        validation_OA=row['OA'], validation_BA=row['BA'], validation_J=row['J'],
                        validation_NLL=row['NLL'], validation_balancedNLL=row['selection_balancedNLL'],
                        J_gain=row['J']-float(initial_j), balancedNLL_gain=row['selection_balancedNLL']-initial_risk,
                        J_best_minus_next_distinct=jvals[0]-jvals[1] if len(jvals)>1 else 0.,
                        J_best_equal_checkpoints=sum(r['J']==jvals[0] for r in curve),
                        J_within_0_1pp_of_best=sum(r['J']>=jvals[0]-.001 for r in curve)))
                    if loss_name == 'balancedCE' and selector == 'balancedNLL':
                        old = torch.load(PARENT/'signed_only_retrain/models'/f'fold{fold}_signed_only_seed{seed}.pt', map_location='cpu', weights_only=False)
                        old_sel = original_sel[(original_sel.fold==fold)&(original_sel.arm=='signed_only')&(original_sel.seed==seed)].iloc[0]
                        state_error = max(float((ck['state'][k]-old['state'][k]).abs().max()) for k in ck['state'])
                        assert ck['epoch']==old['epoch'] and state_error < 1e-7
                        assert abs(row['selection_balancedNLL']-old_sel.validation_loss)<2e-6
                        checks.append(dict(**meta, check='original_selection_state', error=state_error, epoch=chosen['epoch']))
                print(f'fold={fold} loss={loss_name} seed={seed} selected NLL={best["balancedNLL"]["epoch"]} J={best["J"]["epoch"]}', flush=True)
    pd.DataFrame(curves).to_csv(HERE/'selection_curves.csv', index=False)
    pd.DataFrame(selection_rows).to_csv(HERE/'selections.csv', index=False)
    pd.DataFrame(records).to_csv(HERE/'trajectory_costs.csv', index=False)
    pd.DataFrame(fold_sources).to_csv(HERE/'source_counts.csv', index=False)
    (HERE/'selection_frozen.json').write_text(json.dumps(dict(trajectories=len(records), selections=len(selection_rows),
        test_evaluated=False, seconds=time.perf_counter()-started), indent=2), encoding='utf8')

    # Test labels and scores are used only after every source-selected state is saved.
    evaluation_start = time.perf_counter()
    original_predictions = pd.read_csv(PARENT/'signed_only_retrain/predictions.csv')
    predictions, validation = [], []
    for fold in range(6):
        d = dict(np.load(parent.NEW/f'fold{fold}_inputs.npz'))
        outer = cache['moments'][names.index(f'fold{fold}_outer')]
        raw = parent.descriptors(d['base'], outer)
        for chosen in (r for r in selection_rows if r['fold']==fold):
            ck = torch.load(chosen['checkpoint'], map_location='cpu', weights_only=False)
            model = parent.Correction('signed_only')
            model.load_state_dict(ck['state'])
            model.eval()
            for role in ['validation','test']:
                ix = d[role]
                with torch.inference_mode():
                    delta = model(torch.tensor((raw[ix]-ck['mu'])/ck['scale'], dtype=torch.float32),
                                  torch.tensor(outer[ix], dtype=torch.float32)).numpy().astype(float)
                base = d['base'][ix].astype(float)
                score = base+delta
                frame = table.iloc[ix][['instance_id','class_id','mmsi','port','pass_group']].copy()
                for key in ['fold','loss','selector','seed','epoch','cell']:
                    frame[key] = chosen[key]
                frame['prediction'], frame['base_prediction'] = score.argmax(1), base.argmax(1)
                frame['net'] = (frame.prediction==frame.class_id).astype(int)-(frame.base_prediction==frame.class_id).astype(int)
                frame['outcome'] = np.select([frame.net.eq(1),frame.net.eq(-1),frame.prediction.eq(frame.class_id)], ['rescued','harmed','CC'], default='WW')
                for c in range(3):
                    frame[f's{c}'], frame[f'b{c}'], frame[f'delta{c}'] = score[:,c], base[:,c], delta[:,c]
                (validation if role=='validation' else predictions).append(frame)
                if role=='test' and ck['loss']=='balancedCE' and ck['selector']=='balancedNLL':
                    old = original_predictions[(original_predictions.fold==fold)&(original_predictions.arm=='signed_only')&(original_predictions.seed==ck['seed'])]
                    old = old.set_index('instance_id').loc[frame.instance_id]
                    error = float(abs(old[['s0','s1','s2']].to_numpy()-score).max())
                    assert error < 2e-6 and np.array_equal(old.prediction, frame.prediction)
                    checks.append(dict(fold=fold, loss=ck['loss'], seed=ck['seed'], check='original_test_replay', error=error, epoch=ck['epoch']))
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(HERE/'predictions.csv', index=False)
    pd.concat(validation, ignore_index=True).to_csv(HERE/'validation_predictions.csv', index=False)
    pooled, classes, passes = [], [], []
    for (cell, seed), g in pred.groupby(['cell','seed']):
        assert len(g)==len(table) and g.instance_id.is_unique
        meta = dict(cell=cell, seed=seed)
        pooled.append(dict(**meta, rescued=int(g.net.eq(1).sum()), harmed=int(g.net.eq(-1).sum()), net=int(g.net.sum()),
            **assessment(g.class_id.to_numpy(), g[['s0','s1','s2']].to_numpy())))
        for c, group in g.groupby('class_id'):
            classes.append(dict(**meta, class_id=c, n=len(group), net=int(group.net.sum()),
                rescued=int(group.net.eq(1).sum()), harmed=int(group.net.eq(-1).sum()),
                base_recall=100*group.base_prediction.eq(c).mean(), recall=100*group.prediction.eq(c).mean()))
        for unit in ['fold','pass_group','port']:
            for name, group in g.groupby(unit):
                passes.append(dict(**meta, unit=unit, name=name, n=len(group),
                    net=int(group.net.sum()), rescued=int(group.net.eq(1).sum()), harmed=int(group.net.eq(-1).sum())))
    pd.DataFrame(pooled).to_csv(HERE/'pooled.csv', index=False)
    pd.DataFrame(classes).to_csv(HERE/'class_effects.csv', index=False)
    pd.DataFrame(passes).to_csv(HERE/'acquisition_effects.csv', index=False)
    pd.DataFrame(checks).to_csv(HERE/'replay_checks.csv', index=False)
    receipt = dict(trajectories=len(records), selected_heads=len(selection_rows), parameters=113,
        training_seconds=sum(row['seconds'] for row in records), evaluation_seconds=time.perf_counter()-evaluation_start,
        total_seconds=time.perf_counter()-started, original_reproduced_heads=18, test_used_for_selection=False,
        cache=str(parent.CACHE/'signed_features.npz'), new_backbone_fits=0)
    (HERE/'summary.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(pd.DataFrame(pooled).to_string(index=False), flush=True)
    print(json.dumps(receipt,indent=2), flush=True)


if __name__ == '__main__':
    main()
