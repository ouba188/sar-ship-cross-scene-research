"""Four matched arms: legacy, common access, final margin, and their combination."""
from pathlib import Path
import json
import time
import sys
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from model import (prior, EvidenceReader, read, predict, loss_parts,
                   decision_margin_loss, all_readouts, COMMON_SHARE, MARGIN_WEIGHT)

HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent/'support_task_alignment_20261017'
sys.path.insert(0, str(LEGACY))
from sampling import training_schedule

ARMS = {'baseline': (False, False), 'access': (True, False),
        'margin': (False, True), 'joint': (True, True)}
PKEYS = ['fold', 'seed', 'origin', 'draw', 'global_row']


@torch.no_grad()
def evaluate(model, features, tasks, labels, use_common, use_margin, *, axis_weight=1-COMMON_SHARE):
    records = []
    for task in tasks:
        loss, brier, aux, margin, p, _ = loss_parts(model, features, task, labels, use_common, use_margin,
                                                  axis_weight=axis_weight)
        y = labels[task['q']]
        scores = prior.old.scores(y.numpy(), p.numpy())
        records.append(dict(objective=float(loss), brier=float(brier), retrieval_ce=float(aux),
            margin_loss=float(margin), oa=scores['oa'], ba=scores['ba'],
            risk_gain=float(prior.old.risk(task['old'], y)-brier)))
    return pd.DataFrame(records).mean().to_dict()


def fit(arm, seed, features, fit_tasks, val_tasks, labels, folder, *, axis_weight=1-COMMON_SHARE):
    use_common, use_margin = ARMS[arm]
    torch.manual_seed(20261017+seed)
    model = EvidenceReader('static')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.0001)
    schedule, _ = training_schedule(seed, len(fit_tasks))
    np.savez_compressed(folder/'schedule.npz', episode=schedule)
    history = []
    for step in range(201):
        if step:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.stack([loss_parts(model, features, fit_tasks[index], labels,
                use_common, use_margin, axis_weight=axis_weight)[0] for index in schedule[step-1]]).mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
        if step in (0, 50, 100, 200):
            for name, tasks in [('fit', fit_tasks), ('selection', val_tasks)]:
                record = evaluate(model, features, tasks, labels, use_common, use_margin,
                                  axis_weight=axis_weight)
                history.append(dict(step=step, evaluation=name, **record,
                    training_batch_loss=np.nan if not step else float(loss.detach()),
                    gradient_norm=np.nan if not step else float(norm)))
    torch.save(model.state_dict(), folder/'model.pt')
    pd.DataFrame(history).to_csv(folder/'learning.csv', index=False)
    return model


def source_access(data, features, tasks, kind):
    rows = []
    for index, task in enumerate(tasks):
        axis = features['valid'].numpy()
        common = features['common_valid'].numpy()
        sy, qy = data['labels'][task['s']], data['labels'][task['q']]
        old_classes = np.unique(sy[axis[task['s']]])
        new_classes = np.unique(sy[(axis | common)[task['s']]])
        for j, q in enumerate(task['q']):
            old = bool(axis[q] and qy[j] in old_classes)
            new = bool((axis[q] or common[q]) and qy[j] in new_classes)
            rows.append(dict(fold=data['fold'], kind=kind, episode=index, origin=task['origin'],
                global_row=int(data['rows'][q]), truth=int(qy[j]), axis_valid=bool(axis[q]),
                common_valid=bool(common[q]), old_aux_eligible=old, new_aux_eligible=new,
                newly_aux_eligible=new and not old, old_support_classes=len(old_classes),
                new_support_classes=len(new_classes)))
    return rows


def actual_gradient_check(data, features, tasks, labels):
    torch.manual_seed(20261017)
    model = EvidenceReader('static')
    for task in tasks:
        ids, ns = task['ids'], task['ns']
        invalid = ~features['valid'][task['q']] & features['common_valid'][task['q']]
        if not invalid.any():
            continue
        axis = read(model, features, ids)[0]
        common = read(model, features, ids, True)[0]
        p, _, _ = predict(axis, features['valid'][ids], task, common, features['common_valid'][ids])
        brier = prior.old.risk(p, labels[task['q']])
        gradient = torch.autograd.grad(brier, common)[0][ns:][invalid].abs().sum(1)
        assert bool(torch.isfinite(gradient).all()) and float(gradient.max()) > 0
        return dict(fold=data['fold'], recovered_source_queries=int(invalid.sum()),
            maximum_query_gradient=float(gradient.max()), nonzero_queries=int((gradient > 0).sum()),
            global_rows=data['rows'][task['q']][invalid.numpy()].tolist())
    raise AssertionError('Expected real source queries without accepted axes')


@torch.no_grad()
def score(model, data, features, tasks, arm, seed, folder, use_common, *, axis_weight=1-COMMON_SHARE):
    output = all_readouts(model, features, use_common)
    axis_raw, axis_weights, _ = output[0]
    common_raw = output[1][0] if use_common else None
    arrays = dict(global_rows=data['rows'], axis_raw=axis_raw.numpy(), axis_valid=features['valid'].numpy())
    if use_common:
        arrays.update(common_raw=common_raw.numpy(), common_valid=features['common_valid'].numpy())
    np.savez_compressed(folder/'readouts.npz', **arrays)
    metrics, predictions = [], []
    for task in tasks:
        ids, q, ns = task['ids'], task['q'], task['ns']
        cv = features['common_valid'][ids] if use_common else None
        cr = common_raw[ids] if use_common else None
        tic = time.perf_counter()
        p, h, valid = predict(axis_raw[ids], features['valid'][ids], task, cr, cv,
                              axis_weight=axis_weight)
        elapsed = time.perf_counter()-tic
        before, probability, y = task['old'].numpy(), p.numpy(), data['labels'][q]
        sy = data['labels'][task['s']]
        axis_active = features['valid'][q] & (features['valid'][task['s']].sum() >= 2)
        common_active = (features['common_valid'][q] & (features['common_valid'][task['s']].sum() >= 2)
                         if use_common else torch.zeros(len(q), dtype=torch.bool))
        active = (axis_active | common_active).numpy()
        k0 = task['qs'].numpy()
        k1 = k0+(h[ns:]@h[:ns].T).numpy()
        k1[~active] = k0[~active]
        ap0, ap1 = prior.probe.map_at_r(k0, y, sy), prior.probe.map_at_r(k1, y, sy)
        result, original = prior.old.scores(y, probability), prior.old.scores(y, before)
        macro, macro0 = float(prior.old.risk(p, torch.tensor(y))), float(prior.old.risk(task['old'], torch.tensor(y)))
        result.update(fold=data['fold'], arm=arm, seed=seed, origin=task['origin'], draw=task['draw'],
            query=len(q), valid_query=int(active.sum()), baseline_oa=original['oa'], baseline_ba=original['ba'],
            delta_oa=result['oa']-original['oa'], delta_ba=result['ba']-original['ba'],
            delta_brier=result['brier']-original['brier'], delta_nll=result['nll']-original['nll'],
            macro_brier=macro, baseline_macro_brier=macro0, kernel_risk_gain=macro0-macro,
            margin_loss=float(decision_margin_loss(p, torch.tensor(y))),
            support_map_r=float(np.nanmean(ap1)), baseline_support_map_r=float(np.nanmean(ap0)),
            descriptor_energy=float(h[ns:].square().sum(1).mean()), cached_solve_seconds=elapsed,
            rescued=int(((before.argmax(1) != y) & (probability.argmax(1) == y)).sum()),
            harmed=int(((before.argmax(1) == y) & (probability.argmax(1) != y)).sum()))
        metrics.append(result)
        for j, row in enumerate(q):
            predictions.append(dict(fold=data['fold'], arm=arm, seed=seed, origin=task['origin'], draw=task['draw'],
                global_row=int(data['rows'][row]), truth=int(y[j]), baseline=int(before[j].argmax()),
                prediction=int(probability[j].argmax()), axis_valid=bool(features['valid'][row]), active=bool(active[j]),
                baseline_true_probability=float(before[j,y[j]]), true_probability=float(probability[j,y[j]]),
                baseline_ap=ap0[j], ap=ap1[j], same_class_support=int((sy==y[j]).sum()),
                **{f'p{k}':float(probability[j,k]) for k in range(8)}))
    return metrics, predictions


def main():
    torch.set_num_threads(1)
    quality = dict(np.load(prior.HERE/'quality.npz'))
    manifest = pd.read_csv(prior.old.MANIFEST, dtype={'mmsi':str})
    previous = pd.read_csv(LEGACY/'predictions.csv')
    previous = previous[previous.arm.eq('fixed_aux')].set_index(PKEYS)
    metrics, predictions, timings, replay, access, gradients = [], [], [], [], [], []
    with threadpool_limits(limits=1):
        for args in prior.old.folds():
            data, features, fit_tasks, val_tasks, tasks, _, _ = prior.prepare_fold(args, manifest, quality)
            folder = HERE/data['fold']
            folder.mkdir(exist_ok=True)
            cached = np.load(folder/'universal_features.npz')
            np.testing.assert_array_equal(cached['sample_ids'].astype(str), manifest.sample_id.to_numpy(str))
            features['quality'] = torch.zeros_like(features['quality'])
            for key in ('body', 'tokens', 'positions', 'valid'):
                features['common_'+key] = torch.tensor(cached[key][data['rows']])
            labels = torch.tensor(data['labels'], dtype=torch.long)
            for kind, source_tasks in [('fit',fit_tasks), ('selection',val_tasks)]:
                access.extend(source_access(data, features, source_tasks, kind))
            gradients.append(actual_gradient_check(data, features, fit_tasks, labels))
            (folder/'data_protocol.json').write_text(json.dumps(dict(train=data['rows'][data['train']].tolist(),
                selection=data['rows'][data['val']].tolist(), query=data['rows'][data['query']].tolist(),
                fit_tasks=len(fit_tasks), selection_tasks=len(val_tasks), steps=200,
                common_share=COMMON_SHARE, margin_weight=MARGIN_WEIGHT), indent=2), encoding='utf-8')
            print('FOLD',data['fold'],'SOURCE',len(fit_tasks),len(val_tasks),'QUERY',len(data['query']),flush=True)
            for seed in range(3):
                for name, use_common in [('initial',False), ('initial_access',True)]:
                    torch.manual_seed(20261017+seed)
                    model = EvidenceReader('static')
                    out = folder/f'{name}_{seed}'
                    out.mkdir(exist_ok=True)
                    m,p = score(model,data,features,tasks,name,seed,out,use_common)
                    metrics.extend(m); predictions.extend(p)
                for arm, (use_common, use_margin) in ARMS.items():
                    out = folder/f'{arm}_{seed}'
                    out.mkdir(exist_ok=True)
                    tic = time.perf_counter()
                    if arm == 'baseline':
                        model = EvidenceReader('static')
                        model.load_state_dict(torch.load(LEGACY/data['fold']/f'fixed_aux_{seed}'/'model.pt', weights_only=True))
                        history = [dict(step=200,evaluation=name,**evaluate(model,features,source_tasks,labels,False,False))
                                   for name,source_tasks in [('fit',fit_tasks),('selection',val_tasks)]]
                        pd.DataFrame(history).to_csv(out/'learning.csv',index=False)
                    else:
                        model = fit(arm,seed,features,fit_tasks,val_tasks,labels,out)
                    seconds = time.perf_counter()-tic
                    m,p = score(model,data,features,tasks,arm,seed,out,use_common)
                    if arm == 'baseline':
                        current = pd.DataFrame(p).set_index(PKEYS)
                        columns = [f'p{k}' for k in range(8)]
                        error = float(np.max(abs(current[columns].to_numpy()-previous.loc[current.index,columns].to_numpy())))
                        if error > 1e-10:
                            raise AssertionError(f'Baseline probability drift {error}')
                        replay.append(dict(fold=data['fold'],seed=seed,maximum_probability_error=error))
                    metrics.extend(m); predictions.extend(p)
                    timings.append(dict(fold=data['fold'],seed=seed,arm=arm,seconds=seconds,
                        reused_model=arm=='baseline',trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
                    print('FIT',data['fold'],seed,arm,round(seconds,2),flush=True)
                pd.DataFrame(metrics).to_csv(HERE/'metrics.csv',index=False)
                pd.DataFrame(predictions).to_csv(HERE/'predictions.csv',index=False)
                pd.DataFrame(timings).to_csv(HERE/'timings.csv',index=False)
                pd.DataFrame(replay).to_csv(HERE/'baseline_replay.csv',index=False)
            pd.DataFrame(access).to_csv(HERE/'source_access.csv',index=False)
            (HERE/'actual_gradient_checks.json').write_text(json.dumps(gradients,indent=2),encoding='utf-8')
    print(pd.DataFrame(metrics).groupby(['origin','arm'])[['oa','ba','delta_oa','delta_ba','kernel_risk_gain']].mean().to_string(),flush=True)


if __name__ == '__main__':
    main()
