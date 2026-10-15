"""Fixed-budget factorial: missing-support-class loss and supervised source replay."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import time
import numpy as np
import pandas as pd
import torch
from channel_operator import BASE, OUT as CACHE, SEED, build_blocks, metrics, save_json
from target_response_adapter import ResponseAdapter, residual_head, phi_prior, encode, STEPS

OLD = BASE/'target_response_adapter_balanced_20261010'
OUT = BASE/'target_loss_repair_20261010'
ARMS = [('mask_only', True, False), ('source_supervision', False, True),
        ('mask_and_source', True, True)]


def support_loss(scores, target, row_weight, present, mask_absent):
    error = (scores-target).square()
    if mask_absent:
        error = error*present[None, :]
    # Keep division by eight: retained class gradients do not get rescaled.
    return (error.mean(1)*row_weight).mean()


def supervised_source_loss(scores, labels):
    target = 2*torch.eye(8, dtype=scores.dtype)[labels]-1
    return (scores-target).square().mean()


def fit(block, maps, support, labels, source_labels, source_scores, state,
        mask_absent, source_supervised, tag, initial=None):
    adapter = ResponseAdapter(state['channel_scale'])
    if initial is not None:
        with torch.no_grad():
            adapter.log_gain.copy_(torch.as_tensor(initial[0]))
            adapter.shift.copy_(torch.as_tensor(initial[1]))
    optimizer = torch.optim.Adam(adapter.parameters(), lr=.01)
    center, scale, weight, bias = [torch.from_numpy(state[k]) for k in
        ('source_center', 'source_scale', 'source_coef', 'source_bias')]
    hs = torch.from_numpy(np.array(maps[support]))
    y = torch.from_numpy(labels).long()
    counts = torch.bincount(y, minlength=8)
    present = counts > 0
    rw = len(y)/(present.sum()*counts[y].float())
    history = []
    for step in range(STEPS):
        rng = np.random.default_rng(SEED+step)
        replay = np.concatenate([rng.choice(np.flatnonzero(source_labels == c), 4, replace=False)
                                 for c in range(8)])
        hr = torch.from_numpy(np.array(maps[replay]))
        phi, prior = phi_prior(block(adapter(hs)).mean((2, 3)), center, scale, weight, bias)
        _, loo, target = residual_head(phi, prior, y, balanced=True)
        target_loss = support_loss(loo, target, rw, present, mask_absent)
        _, scores = phi_prior(block(adapter(hr)).mean((2, 3)), center, scale, weight, bias)
        preservation = (scores-source_scores[replay]).square().mean()
        supervised = supervised_source_loss(scores, torch.from_numpy(source_labels[replay]).long())
        regularizer = (adapter.log_gain.square()+adapter.shift.square()).mean()
        loss = target_loss+.25*preservation+.1*regularizer
        if source_supervised:
            loss = loss+.25*supervised
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.)
        optimizer.step()
        adapter.bound()
        if step in (0, 7, 15, STEPS-1):
            history.append(dict(task=tag, step=step+1, target_loss=float(target_loss.detach()),
                source_preservation=float(preservation.detach()), source_supervised=float(supervised.detach()),
                regularizer=float(regularizer.detach())))
    return adapter.eval(), history


def load_adapter(state, prefix='ordinary_adapter'):
    model = ResponseAdapter(state['channel_scale'])
    with torch.no_grad():
        model.log_gain.copy_(torch.from_numpy(state[prefix+'_log_gain']))
        model.shift.copy_(torch.from_numpy(state[prefix+'_shift']))
    return model.eval()


def evaluate(block, maps, support, query, labels, adapter, state):
    features = encode(block, adapter, maps, np.concatenate([support, query]))
    with torch.no_grad():
        phi, prior = phi_prior(torch.from_numpy(features), *[torch.from_numpy(state[k]) for k in
            ('source_center', 'source_scale', 'source_coef', 'source_bias')])
        beta, _, _ = residual_head(phi[:64], prior[:64], torch.from_numpy(labels).long(), balanced=True)
        score = prior[64:]+phi[64:]@beta
    return score.numpy(), prior[64:].numpy(), beta.numpy()


def run():
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    OUT.mkdir(exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Do not overwrite completed experiments')
    start = time.perf_counter()
    save_json(OUT/'PROTOCOL.json', dict(
        comparison='ordinary baseline vs mask only vs added source supervision vs both',
        labels='Same939 source labels and exact previous9x64 target supports; no additional labels',
        split='Load previous support/query indices; source excludes all3 target ports and identities; query excludes support identities',
        architecture='Frozen R18;1024 affine parameters before final residual block; source geometry and source classifier frozen',
        target_loss='Balanced support analytic leave-one-out MSE; missing columns masked only in enabled arms, divisor remains8',
        residual_head='Same balanced ridge8 and zero residual columns for support-absent classes in all arms',
        source_loss='Original0.25 score-preservation always retained; enabled arms add0.25 eight-class supervised MSE on same balanced32 replay samples',
        optimization='Adam lr0.01,24 steps, same replay draws, same0.1 identity regularizer and parameter bounds',
        guidance='Ordinary adaptation only; previous scene guidance had negligible extra gain; no target-selected new channel rules',
        selection='Coefficients and all4 cells fixed before new query results; no grid or port-specific choice',
        reuse='Previous baseline reused; mask inactive with all8 support classes, so exact objective-equivalent cells reused and labeled',
        limitations='Development pilot, not complete dual; supports/queries can share scenes;3 support draws not3 independent cohorts; inherited pretraining exposure',
        resources='CPU1; no parent modifications'))
    obs = pd.read_csv(BASE/'observations.csv')
    with np.load(CACHE/'split.npz') as z:
        source_obs = obs.iloc[z['train_obs_ids']]
    source_labels = source_obs.class_id.to_numpy()
    maps = np.load(CACHE/'boundary_maps.npy', mmap_mode='r')
    block = build_blocks()[1]
    results, classes, histories, replays, reuse_records = [], [], [], [], []
    fits = 0
    source_scores = None
    for split_path in sorted(OLD.glob('*_split.npz')):
        tag = split_path.stem.removesuffix('_split')
        with np.load(split_path) as z:
            support, query, labels, query_ids = [z[k] for k in
                ('support_map_rows', 'query_map_rows', 'support_labels', 'query_obs_ids')]
        with np.load(OLD/f'{tag}_models.npz') as z:
            state = {k: z[k] for k in z.files}
        port = obs.iloc[query_ids].port.iloc[0]
        draw = int(tag.rsplit('draw', 1)[1])
        if source_scores is None:
            identity = ResponseAdapter(state['channel_scale']).eval()
            f = encode(block, identity, maps, np.arange(len(source_labels)))
            with torch.no_grad():
                _, source_scores = phi_prior(torch.from_numpy(f), *[torch.from_numpy(state[k]) for k in
                    ('source_center', 'source_scale', 'source_coef', 'source_bias')])
        adapters = {'ordinary_adapter': load_adapter(state)}
        # Reproduce one old fit to detect unintended changes outside the two switches.
        if not replays:
            reproduced, _ = fit(block, maps, support, labels, source_labels, source_scores,
                                state, False, False, tag)
            err = max(float((a-b).abs().max()) for a, b in
                      zip(reproduced.parameters(), adapters['ordinary_adapter'].parameters()))
            if err > 1e-6:
                raise ValueError(f'Unmodified training failed replay: {err}')
            replays.append(dict(task=tag, parameter_max_abs=err, type='baseline_training'))
        absent = np.bincount(labels, minlength=8) == 0
        for arm, mask, supervised in ARMS:
            if mask and not absent.any():
                reused = 'source_supervision' if supervised else 'ordinary_adapter'
                adapters[arm] = adapters[reused]
                reuse_records.append(dict(task=tag, arm=arm, reused_from=reused,
                    reason='All8 support classes present, mask is identity'))
            else:
                adapters[arm], trace = fit(block, maps, support, labels, source_labels,
                                           source_scores, state, mask, supervised, tag)
                histories.extend(dict(arm=arm, **r) for r in trace)
                fits += 1
            print('FIT', tag, arm, 'absent_classes', int(absent.sum()), flush=True)
        # All cells finished adaptation before query labels enter scoring.
        truth = obs.iloc[query_ids].class_id.to_numpy()
        arrays = {}
        for arm, adapter in adapters.items():
            scores, prior, beta = evaluate(block, maps, support, query, labels, adapter, state)
            pred = scores.argmax(1)
            if arm == 'ordinary_adapter':
                errors = int((pred != state['ordinary_adapter_pred']).sum())
                if errors:
                    raise ValueError('Baseline query predictions did not replay')
                replays.append(dict(task=tag, prediction_disagreements=errors, type='baseline_inference'))
            arrays[arm+'_scores'], arrays[arm+'_prior'] = scores, prior
            arrays[arm+'_pred'], arrays[arm+'_residual_coef'] = pred, beta
            arrays[arm+'_log_gain'] = adapter.log_gain.detach().numpy()
            arrays[arm+'_shift'] = adapter.shift.detach().numpy()
            results.append(dict(port=port, draw=draw, arm=arm, **metrics(pred, truth)))
            for c in np.unique(truth):
                m = truth == c
                count = int((labels == c).sum())
                classes.append(dict(port=port, draw=draw, arm=arm, class_id=int(c), n=int(m.sum()),
                    support_count=count, coverage='absent' if count == 0 else 'singleton' if count == 1 else 'multiple',
                    recall=float(100*(pred[m] == c).mean()), mean_true_prior=float(prior[m, c].mean())))
        for arm in ('source_only', 'head_only'):
            pred = state[arm+'_pred']
            results.append(dict(port=port, draw=draw, arm=arm, **metrics(pred, truth)))
        np.savez_compressed(OUT/f'{tag}_models.npz', **arrays, query_obs_ids=query_ids, support_labels=labels)
        for name, rows in [('results', results), ('by_class', classes), ('training_history', histories)]:
            pd.DataFrame(rows).to_csv(OUT/f'{name}.csv', index=False)
        save_json(OUT/'REPLAYS.json', replays)
        save_json(OUT/'OBJECTIVE_EQUIVALENT_REUSE.json', reuse_records)
        print('TASK_DONE', tag, flush=True)
    save_json(OUT/'COMPLETE.json', dict(tasks=9, new_adapter_fits=fits, baseline_reproduction_fits=1,
        seconds=time.perf_counter()-start, resources='CPU1', query_labels_used_for_fitting=False))
    print(pd.DataFrame(results).groupby('arm')[['OA', 'BA']].mean().to_string(), flush=True)


if __name__ == '__main__':
    run()
