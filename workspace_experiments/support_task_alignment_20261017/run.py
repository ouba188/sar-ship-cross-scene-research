"""Matched 2x2 intervention: support diversity and auxiliary retrieval loss."""
from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent/'local_evidence_reliability_20261017'
sys.path.insert(0, str(PREVIOUS))
import train_reliability as prior
from local_evidence import EvidenceReader, retrieval_loss
from sampling import support_variants, training_schedule

ARMS = {'fixed_aux': (False, .05), 'fixed_task': (False, 0.),
        'diverse_aux': (True, .05), 'diverse_task': (True, 0.)}


def loss_parts(model, features, task, labels):
    raw, _, _ = prior.forward(model, features, task['ids'])
    valid = features['valid'][task['ids']]
    p, h = prior.old.predict(raw, valid, task, 1.)
    risk = prior.old.risk(p, labels[task['q']])
    auxiliary = retrieval_loss(h, labels[task['ids']], task['ns'], valid)
    return risk, auxiliary, p


@torch.no_grad()
def evaluate(model, features, tasks, labels):
    records = []
    for task in tasks:
        risk, aux, p = loss_parts(model, features, task, labels)
        metrics = prior.old.scores(labels[task['q']].numpy(), p.numpy())
        before = prior.old.scores(labels[task['q']].numpy(), task['old'].numpy())
        records.append(dict(risk=float(risk), retrieval_ce=float(aux), oa=metrics['oa'], ba=metrics['ba'],
            delta_oa=metrics['oa']-before['oa'], delta_ba=metrics['ba']-before['ba'],
            risk_gain=float(prior.old.risk(task['old'], labels[task['q']])-risk)))
    return pd.DataFrame(records).mean().to_dict()


def assemble_banks(data, fit_tasks, selection_tasks, fold_index):
    source = np.r_[data['train'], data['val']]
    g, d, c = [prior.public.pca(raw, source) for raw in (data['raw']['G50'], data['raw']['D18'], data['c4'])]
    base = prior.public.design_block(c, np.arange(len(c)))
    z = prior.public.design_block((g,d), np.arange(len(g)))
    cache, result, receipts = {}, {}, []
    for offset, (kind, original_tasks) in enumerate([('fit', fit_tasks), ('selection', selection_tasks)]):
        episodes = prior.old.make_episodes(data, kind)
        variants, rows = support_variants(data, episodes, kind, 20263017+fold_index*10+offset)
        tasks = []
        for i, episode in enumerate(episodes):
            bank = []
            for draw, support in enumerate(variants[i]):
                updated = {**episode, 's': support}
                task = prior.old.make_task(data, base, z, updated, cache)
                if draw == 0:
                    for key in ('old','ss','qs','prior_s','prior_q','base'):
                        np.testing.assert_allclose(task[key].numpy(), original_tasks[i][key].numpy(), atol=1e-12, rtol=0)
                    task = original_tasks[i]
                bank.append(task)
            tasks.append(bank)
        result[kind] = tasks
        for row in rows:
            row['fold'] = data['fold']
            receipts.append(row)
    return result, receipts


def fit(arm, seed, features, banks, labels, folder):
    diversified, auxiliary_weight = ARMS[arm]
    torch.manual_seed(20261017+seed)
    model = EvidenceReader('static')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.0001)
    families, draws = training_schedule(seed, len(banks['fit']))
    np.savez_compressed(folder/'schedule.npz', episode=families, draw=draws if diversified else np.zeros_like(draws))
    evaluation = {'fit_fixed': [b[0] for b in banks['fit']],
                  'selection_fixed': [b[0] for b in banks['selection']],
                  'selection_diverse': [t for b in banks['selection'] for t in b]}
    history = []
    for step in range(201):
        if step:
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for family, draw in zip(families[step-1], draws[step-1]):
                task = banks['fit'][family][draw if diversified else 0]
                risk, auxiliary, _ = loss_parts(model, features, task, labels)
                losses.append(risk+auxiliary_weight*auxiliary)
            loss = torch.stack(losses).mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
        if step in (0,50,100,200):
            for name, tasks in evaluation.items():
                row = evaluate(model, features, tasks, labels)
                history.append(dict(step=step, evaluation=name, **row,
                    training_batch_loss=np.nan if not step else float(loss.detach()),
                    gradient_norm=np.nan if not step else float(norm)))
    torch.save(model.state_dict(), folder/'model.pt')
    pd.DataFrame(history).to_csv(folder/'learning.csv', index=False)
    return model


def main():
    torch.set_num_threads(1)
    quality = dict(np.load(PREVIOUS/'quality.npz'))
    manifest = pd.read_csv(prior.old.MANIFEST, dtype={'mmsi':str})
    metrics, predictions, timings, replay, receipts = [], [], [], [], []
    with threadpool_limits(limits=1):
        for fold_index, args in enumerate(prior.old.folds()):
            data, features, fit_tasks, selection_tasks, target_tasks, _, _ = prior.prepare_fold(args, manifest, quality)
            features = {**features, 'quality':torch.zeros_like(features['quality'])}
            labels = torch.tensor(data['labels'], dtype=torch.long)
            tic = time.perf_counter()
            banks, rows = assemble_banks(data, fit_tasks, selection_tasks, fold_index)
            receipts.extend(rows)
            folder = HERE/data['fold']
            folder.mkdir(exist_ok=True)
            (folder/'support_bank.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
            print('BANK',data['fold'],len(fit_tasks),len(selection_tasks),round(time.perf_counter()-tic,2),flush=True)
            for seed in range(3):
                torch.manual_seed(20261017+seed)
                initial = EvidenceReader('static')
                output = folder/f'initial_{seed}'
                output.mkdir(exist_ok=True)
                m,p = prior.score_model(initial, data, features, target_tasks, 'initial', seed, output)
                metrics.extend(m)
                predictions.extend(p)
                for arm in ARMS:
                    output = folder/f'{arm}_{seed}'
                    output.mkdir(exist_ok=True)
                    tic = time.perf_counter()
                    model = fit(arm, seed, features, banks, labels, output)
                    elapsed = time.perf_counter()-tic
                    if arm == 'fixed_aux':
                        old_state = torch.load(PREVIOUS/data['fold']/f'static_no_quality_{seed}'/'model.pt', weights_only=True)
                        error = max(float((value-old_state[key]).abs().max()) for key,value in model.state_dict().items())
                        replay.append(dict(fold=data['fold'], seed=seed, model_error=error))
                        if error > 1e-6:
                            raise AssertionError(f'Legacy fixed_aux fit drift: {error}')
                    m,p = prior.score_model(model, data, features, target_tasks, arm, seed, output)
                    metrics.extend(m)
                    predictions.extend(p)
                    timings.append(dict(fold=data['fold'], seed=seed, arm=arm, seconds=elapsed))
                    print('FIT',data['fold'],seed,arm,round(elapsed,2),flush=True)
            pd.DataFrame(metrics).to_csv(HERE/'metrics.csv', index=False)
            pd.DataFrame(predictions).to_csv(HERE/'predictions.csv', index=False)
            pd.DataFrame(timings).to_csv(HERE/'timings.csv', index=False)
            pd.DataFrame(replay).to_csv(HERE/'legacy_replay.csv', index=False)
    (HERE/'support_banks.json').write_text(json.dumps(receipts, indent=2), encoding='utf-8')
    pd.DataFrame(receipts).drop(columns=['support_global_rows','query_global_rows','prior_global_rows','class_counts']).to_csv(HERE/'support_diversity.csv',index=False)
    print(pd.DataFrame(metrics).groupby(['origin','arm'])[['oa','ba','delta_oa','delta_ba','kernel_risk_gain']].mean().to_string(),flush=True)


if __name__ == '__main__':
    main()
