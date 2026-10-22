"""Read-only replay of selected heads; no optimization or checkpoint selection."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import sys
sys.dont_write_bytecode = True
import json
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
import train_signed as source


def sample_values(y, base, score):
    ix = np.arange(len(y))
    wrong = base.copy()
    wrong[ix, y] = -np.inf
    competitor = wrong.argmax(1)
    margin = base[ix, y] - wrong.max(1)
    bp, pred = base.argmax(1), score.argmax(1)
    bnll = logsumexp(base, axis=1) - base[ix, y]
    nll = logsumexp(score, axis=1) - score[ix, y]
    return pd.DataFrame(dict(class_id=y, base_correct=bp == y, correct=pred == y,
        base_prediction=bp, prediction=pred, base_nll=bnll, nll=nll,
        base_margin=margin, margin=score[ix, y]-score[ix, competitor],
        boundary=np.select([abs(margin) <= .25, abs(margin) <= 1.],
                           ['near_le_0.25', 'middle_0.25_1'], default='far_gt_1')))


def metrics(g):
    by_class = g.groupby('class_id')
    base_oa, oa = g.base_correct.mean()*100, g.correct.mean()*100
    base_ba = by_class.base_correct.mean().mean()*100
    ba = by_class.correct.mean().mean()*100
    base_bnll = by_class.base_nll.mean().mean()
    bnll = by_class.nll.mean().mean()
    rescued = int((~g.base_correct & g.correct).sum())
    harmed = int((g.base_correct & ~g.correct).sum())
    return dict(n=len(g), unique_instances=g.instance_id.nunique(),
        base_OA=base_oa, selected_OA=oa, delta_OA_pp=oa-base_oa,
        base_BA=base_ba, selected_BA=ba, delta_BA_pp=ba-base_ba,
        base_NLL=g.base_nll.mean(), selected_NLL=g.nll.mean(),
        delta_NLL=(g.nll-g.base_nll).mean(), base_balNLL=base_bnll,
        selected_balNLL=bnll, delta_balNLL=bnll-base_bnll,
        rescued=rescued, harmed=harmed, net=rescued-harmed,
        delta_nll_sum=float((g.nll-g.base_nll).sum()),
        mean_margin_delta=float((g.margin-g.base_margin).mean()))


def main():
    start = time.perf_counter()
    torch.set_num_threads(2)
    table = pd.read_csv(source.EXP/'readout/targets.csv', dtype={'mmsi': str})
    cache_path = source.CACHE/'signed_features.npz'
    cache = np.load(cache_path)
    names, moments = list(cache['reader_names']), cache['moments']
    np.testing.assert_array_equal(cache['instance_ids'], table.instance_id)
    paths = sorted(p for p in PARENT.rglob('models/*.pt') if HERE not in p.parents)
    inventory, head_rows, checks, frames = [], [], [], []
    disk_predictions, selections = {}, {}
    review_ids = set(pd.read_csv(PARENT/'review_cases.csv').instance_id)
    case_rows = []
    for fold in range(6):
        d = dict(np.load(source.NEW/f'fold{fold}_inputs.npz'))
        outer = moments[names.index(f'fold{fold}_outer')]
        oof = np.zeros_like(outer)
        for j, name in enumerate(names):
            if name.startswith(f'fold{fold}_inner'):
                held = np.load(source.NEW/'origins'/f'{name}.npz')['held']
                oof[held] = moments[j, held]
        raw_x = {'train_oof': source.descriptors(d['oof'], oof),
                 'validation': source.descriptors(d['base'], outer)}
        for path in paths:
            match = re.fullmatch(r'fold(\d+)_(.+)_seed(-?\d+)\.pt', path.name)
            if not match or int(match[1]) != fold:
                continue
            seed = int(match[3])
            run = path.parent.parent
            run_name = run.relative_to(PARENT).as_posix()
            ck = torch.load(path, map_location='cpu', weights_only=False)
            arm = ck['arm']
            model = source.Correction(arm)
            model.load_state_dict(ck['state'])
            model.eval()
            if run not in disk_predictions:
                disk_predictions[run] = pd.read_csv(run/'predictions.csv',
                    usecols=['fold','arm','seed','instance_id','s0','s1','s2'])
                selections[run] = pd.read_csv(run/'selections.csv')
            sel = selections[run]
            sel = sel[(sel.fold == fold) & (sel.arm == arm) & (sel.seed == seed)].iloc[0]
            meta = dict(run=run_name, fold=fold, arm=arm, seed=seed, epoch=ck['epoch'])
            inventory.append(dict(**meta, checkpoint=str(path), cache=str(cache_path),
                selected_validation_loss=float(sel.validation_loss)))
            for role, key in [('train_oof','train'),('validation','validation'),('test','test')]:
                ix = d[key]
                is_train = role == 'train_oof'
                raw = raw_x['train_oof' if is_train else 'validation'][ix]
                x = torch.tensor((raw-ck['mu'])/ck['scale'], dtype=torch.float32)
                mom = torch.tensor((oof if is_train else outer)[ix], dtype=torch.float32)
                base = d['oof' if is_train else 'base'][ix].astype(float)
                y = table.iloc[ix].class_id.to_numpy()
                with torch.inference_mode():
                    correction = model(x, mom)
                    delta = correction.numpy().astype(float)
                score = base+delta
                g = sample_values(y, base, score)
                for col in ['instance_id','mmsi','pass_group','port']:
                    g[col] = table.iloc[ix][col].to_numpy()
                for col, value in meta.items():
                    g[col] = value
                g['role'] = role
                head_rows.append(dict(**meta, role=role, **metrics(g)))
                frames.append(g)
                if role == 'validation':
                    with torch.inference_mode():
                        risk = float(source.balanced_loss(
                            torch.tensor(base, dtype=torch.float32)+correction,
                            torch.tensor(y, dtype=torch.long)))
                    error = abs(risk-float(sel.validation_loss))
                    assert error < 2e-6, (path, error)
                    checks.append(dict(**meta, check='selected_validation_risk', max_error=error))
                if role == 'test':
                    disk = disk_predictions[run]
                    disk = disk[(disk.fold == fold) & (disk.arm == arm) & (disk.seed == seed)]
                    disk = disk.set_index('instance_id').loc[g.instance_id]
                    error = float(abs(score-disk[['s0','s1','s2']].to_numpy()).max())
                    assert error < 2e-6, (path, error)
                    checks.append(dict(**meta, check='test_prediction_replay', max_error=error))
                    if seed in [-1,17]:
                        case_rows.append(g[g.instance_id.isin(review_ids)].copy())
    all_rows = pd.concat(frames, ignore_index=True)
    heads = pd.DataFrame(head_rows)
    pd.DataFrame(inventory).to_csv(HERE/'inventory.csv', index=False)
    pd.DataFrame(checks).to_csv(HERE/'replay_checks.csv', index=False)
    heads.to_csv(HERE/'head_role_metrics.csv', index=False)
    pooled, classes, boundaries = [], [], []
    keys = ['run','arm','seed','role']
    for values, g in all_rows.groupby(keys):
        meta = dict(zip(keys, values))
        pooled.append(dict(**meta, **metrics(g)))
        for c, group in g.groupby('class_id'):
            classes.append(dict(**meta, class_id=c, **metrics(group)))
        for (correct, boundary), group in g.groupby(['base_correct','boundary']):
            class_counts = g.class_id.value_counts()
            balanced_contribution = ((group.nll-group.base_nll)/
                group.class_id.map(class_counts)).sum()/3
            boundaries.append(dict(**meta, base_correct=correct, boundary=boundary,
                **metrics(group), nll_improved=int((group.nll < group.base_nll).sum()),
                balanced_NLL_delta_contribution=float(balanced_contribution)))
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(HERE/'role_pooled.csv', index=False)
    pd.DataFrame(classes).to_csv(HERE/'class_role_metrics.csv', index=False)
    pd.DataFrame(boundaries).to_csv(HERE/'boundary_role_metrics.csv', index=False)
    cases = pd.concat(case_rows, ignore_index=True)
    cases['existing_panel'] = cases.instance_id.map(lambda i: str(PARENT/f'case_{i}.png'))
    cases.to_csv(HERE/'existing_case_replay.csv', index=False)
    paired = []
    for values, g in heads.groupby(['run','arm','seed','fold']):
        out = dict(zip(['run','arm','seed','fold'], values))
        for role, row in g.set_index('role').iterrows():
            for col in ['delta_OA_pp','delta_BA_pp','delta_NLL','delta_balNLL','net']:
                out[f'{role}_{col}'] = row[col]
        out['epoch'] = int(g.epoch.iloc[0])
        out['val_risk_improves_OA_worsens'] = out['validation_delta_balNLL'] < -1e-7 and out['validation_delta_OA_pp'] < -1e-7
        out['val_risk_improves_BA_worsens'] = out['validation_delta_balNLL'] < -1e-7 and out['validation_delta_BA_pp'] < -1e-7
        out['val_OA_nonnegative_test_negative'] = out['validation_delta_OA_pp'] >= -1e-7 and out['test_delta_OA_pp'] < -1e-7
        out['val_BA_nonnegative_test_negative'] = out['validation_delta_BA_pp'] >= -1e-7 and out['test_delta_BA_pp'] < -1e-7
        paired.append(out)
    paired = pd.DataFrame(paired)
    paired.to_csv(HERE/'selection_vs_transfer.csv', index=False)
    flags = ['val_risk_improves_OA_worsens','val_risk_improves_BA_worsens',
             'val_OA_nonnegative_test_negative','val_BA_nonnegative_test_negative']
    summary = paired.groupby(['run','arm']).agg(heads=('fold','size'), epoch0=('epoch',lambda a:int((a==0).sum())),
        **{flag:(flag,'sum') for flag in flags}).reset_index()
    summary.to_csv(HERE/'selection_summary.csv', index=False)
    # Macro over saved seed runs, not an ensemble and not independent replicates.
    numeric = ['base_OA','selected_OA','delta_OA_pp','base_BA','selected_BA','delta_BA_pp',
               'base_NLL','selected_NLL','delta_NLL','base_balNLL','selected_balNLL','delta_balNLL','net']
    pooled.groupby(['run','arm','role'])[numeric].mean().reset_index().to_csv(HERE/'arm_role_summary.csv',index=False)
    receipt = dict(heads=len(inventory), head_role_rows=len(heads),
        cache=str(cache_path), fits=0, seconds=time.perf_counter()-start,
        thresholds=[.25,1.], source_roles='fold-sample events, repeated IDs across folds',
        train_scope='base OOF; correction head in-sample',
        max_replay_error=max(row['max_error'] for row in checks))
    (HERE/'summary.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    print(json.dumps(receipt, indent=2), flush=True)
    print(summary.to_string(index=False), flush=True)
    print(pooled[['arm','seed','role','delta_OA_pp','delta_BA_pp','delta_NLL','delta_balNLL','net']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
