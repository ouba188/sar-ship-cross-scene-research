"""One bounded source-episode initialization attempt before target adaptation."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import time
import numpy as np
import pandas as pd
import torch
from channel_operator import BASE, OUT as CACHE, SEED, build_blocks, fit_classifier, save_json, metrics
from target_response_adapter import ResponseAdapter, support_design, phi_prior, residual_head, encode
from target_loss_repair import OLD, OUT as REPAIR, fit, evaluate, supervised_source_loss

OUT = BASE/'source_episode_closeout_20261010'
STEPS = 24


def parameters(adapter):
    return (adapter.log_gain.detach().numpy().copy(), adapter.shift.detach().numpy().copy())


def phi(features, state):
    return phi_prior(features, *[torch.from_numpy(state[k]) for k in
                     ('source_center', 'source_scale', 'source_coef', 'source_bias')])


def prepare_episodes(meta, features, state):
    episodes, designs = [], []
    for port, group in meta.groupby('port', sort=True):
        if group.mmsi.nunique() < 80:
            continue
        forbidden = set(group.mmsi)
        reference = np.flatnonzero(~meta.port.eq(port) & ~meta.mmsi.isin(forbidden))
        if meta.iloc[reference].class_id.nunique() != 8:
            continue
        center = features[reference].mean(0)
        scale = np.maximum(features[reference].std(0), 1e-6)
        weight, bias = fit_classifier(features[reference], meta.iloc[reference].reset_index(drop=True), center, scale)
        prior_state = dict(source_center=center, source_scale=scale, source_coef=weight, source_bias=bias,
                           channel_scale=state['channel_scale'])
        group_rows = group.index.to_numpy()
        local = group.reset_index(drop=True)
        for draw in range(3):
            support, query = support_design(local, draw)
            support, query = group_rows[support], group_rows[query]
            assert not set(meta.iloc[support].mmsi) & set(meta.iloc[query].mmsi)
            assert not set(meta.iloc[reference].mmsi) & set(meta.iloc[np.r_[support, query]].mmsi)
            episodes.append(dict(tag=port.replace(' ', '_')+f'_draw{draw}', support=support, query=query,
                                 reference=reference, state=prior_state))
            designs.append(dict(port=port, draw=draw, support=len(support), query=len(query),
                support_classes=meta.iloc[support].class_id.nunique(), query_classes=meta.iloc[query].class_id.nunique(),
                reference=len(reference), shared_scenes=len(set(meta.iloc[support]['product']) & set(meta.iloc[query]['product']))))
    return episodes, pd.DataFrame(designs)


def weighted_mse(scores, labels):
    counts = torch.bincount(labels, minlength=8)
    weights = len(labels)/((counts > 0).sum()*counts[labels].float())
    targets = 2*torch.eye(8, dtype=scores.dtype)[labels]-1
    return ((scores-targets).square().mean(1)*weights).mean()


def warmup(block, maps, meta, original_scores, state, episodes, episodic):
    adapter = ResponseAdapter(state['channel_scale'])
    optimizer = torch.optim.Adam(adapter.parameters(), lr=.01)
    y = meta.class_id.to_numpy()
    histories = []
    for step in range(STEPS):
        ep = episodes[step % len(episodes)]
        rng = np.random.default_rng(SEED+step)
        reference = ep['reference']
        replay = np.concatenate([rng.choice(reference[y[reference] == c], 4, replace=False) for c in range(8)])
        h = torch.from_numpy(np.array(maps[replay]))
        _, scores = phi(block(adapter(h)).mean((2, 3)), state)
        preservation = (scores-original_scores[replay]).square().mean()
        supervised = supervised_source_loss(scores, torch.from_numpy(y[replay]).long())
        if episodic:
            idx = np.r_[ep['support'], ep['query']]
            f = block(adapter(torch.from_numpy(np.array(maps[idx])))).mean((2, 3))
            ph, pri = phi(f, ep['state'])
            beta, _, _ = residual_head(ph[:64], pri[:64], torch.from_numpy(y[ep['support']]).long(), balanced=True)
            query_scores = pri[64:]+ph[64:]@beta
            main_loss = weighted_mse(query_scores, torch.from_numpy(y[ep['query']]).long())
        else:
            main_loss = supervised
        reg = (adapter.log_gain.square()+adapter.shift.square()).mean()
        loss = main_loss+.25*preservation+.1*reg
        if episodic:
            loss = loss+.25*supervised
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.)
        optimizer.step()
        adapter.bound()
        histories.append(dict(step=step+1, episode=ep['tag'], main_loss=float(main_loss.detach()),
            source_supervised=float(supervised.detach()), source_preservation=float(preservation.detach())))
    return adapter.eval(), histories


def run():
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    OUT.mkdir(exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Completed runs must be preserved')
    start = time.perf_counter()
    obs = pd.read_csv(BASE/'observations.csv')
    with np.load(CACHE/'split.npz') as z:
        source = obs.iloc[z['train_obs_ids']].reset_index(drop=True)
    source_labels = source.class_id.to_numpy()
    maps = np.load(CACHE/'boundary_maps.npy', mmap_mode='r')
    first = sorted(OLD.glob('*_models.npz'))[0]
    with np.load(first) as z:
        state = {k: z[k] for k in z.files}
    block = build_blocks()[1]
    identity = ResponseAdapter(state['channel_scale']).eval()
    features = encode(block, identity, maps, np.arange(len(source)))
    with torch.no_grad():
        _, source_scores = phi(torch.from_numpy(features), state)
    episodes, designs = prepare_episodes(source, features, state)
    if not episodes:
        raise ValueError('No64-support episodes with independent source queries')
    designs.to_csv(OUT/'source_episode_design.csv', index=False)
    for ep in episodes:
        np.savez_compressed(OUT/(ep['tag']+'_source_split.npz'), support_rows=ep['support'],
                            query_rows=ep['query'], reference_rows=ep['reference'])
    save_json(OUT/'PROTOCOL.json', dict(
        purpose='Final bounded attempt on channel-affine adaptation: source query-supervised few-shot initialization',
        source_labels=len(source), target_supports='Exact previous9x64 support/query splits; no extra target labels',
        source_episodes='Ports with >=80 distinct source vessels and8-class remaining reference;64 support vessels, remaining vessels query;3 draws each',
        reference='Per-episode standardization and source head fitted excluding entire pseudo-target port and all its identities',
        source_initialization='Shared1024 affine parameters,24 Adam steps lr0.01; episode query gradient through64-support differentiable ridge8',
        comparison='Ordinary source-supervised warmup also24 steps same1024 parameters; equal steps, not equal forward FLOPs',
        target_adaptation='24-step corrected mask+source objective from previous run; initialize from warmup, regularizer still anchored to identity',
        arms=['previous_repaired', 'supervised_init_head', 'supervised_init_adapt', 'episode_init_head', 'episode_init_adapt'],
        no_target_selection='Final24-step weights, no checkpoint/grid/port-specific selection',
        operational_stop='Keep as positive pilot only if episode_init_adapt mean BA exceeds previous repair and ordinary warmup+adapt, and missing-class recall is not worse than previous repair; otherwise stop iterating this channel route',
        limitations='Only2 eligible source ports expected; repeated source-query episodes are training, not heldout evidence; source and target support/query can share scenes; development pilot and inherited pretraining exposure',
        paper_relation='Not a Causal-Tune reproduction: no DCT filter or causal tokens; closes current channel route before spatial-frequency alternative',
        resources='CPU1, no GPU or parent edits'))
    initializations, traces = {}, []
    for name, episodic in [('supervised', False), ('episode', True)]:
        adapter, trace = warmup(block, maps, source, source_scores, state, episodes, episodic)
        initializations[name] = parameters(adapter)
        np.savez_compressed(OUT/f'{name}_initialization.npz', log_gain=initializations[name][0], shift=initializations[name][1])
        traces.extend(dict(arm=name, **r) for r in trace)
        print('WARMUP_DONE', name, flush=True)
    pd.DataFrame(traces).to_csv(OUT/'source_training_history.csv', index=False)
    records, classes, target_histories, replays = [], [], [], []
    for path in sorted(OLD.glob('*_split.npz')):
        tag = path.stem.removesuffix('_split')
        with np.load(path) as z:
            support, query, labels, ids = [z[k] for k in
                ('support_map_rows', 'query_map_rows', 'support_labels', 'query_obs_ids')]
        with np.load(OLD/f'{tag}_models.npz') as z:
            state = {k: z[k] for k in z.files}
        if not replays:
            replay, _ = fit(block, maps, support, labels, source_labels, source_scores, state, True, True, tag,
                            initial=(np.zeros(512, np.float32), np.zeros(512, np.float32)))
            with np.load(REPAIR/f'{tag}_models.npz') as z:
                error = max(float(np.max(abs(a-z['mask_and_source_'+k]))) for a, k in
                            zip(parameters(replay), ['log_gain', 'shift']))
            if error > 1e-6:
                raise ValueError('Identity initialization changed the previous corrected fit')
            replays.append(dict(task=tag, max_parameter_error=error))
        scores_by_arm, arrays = {}, {}
        for name, initial in initializations.items():
            warm = ResponseAdapter(state['channel_scale'])
            with torch.no_grad():
                warm.log_gain.copy_(torch.from_numpy(initial[0]))
                warm.shift.copy_(torch.from_numpy(initial[1]))
            scores_by_arm[name+'_init_head'], _, _ = evaluate(block, maps, support, query, labels, warm.eval(), state)
            adapted, trace = fit(block, maps, support, labels, source_labels, source_scores, state,
                                 True, True, tag, initial=initial)
            target_histories.extend(dict(arm=name, **r) for r in trace)
            scores_by_arm[name+'_init_adapt'], _, _ = evaluate(block, maps, support, query, labels, adapted, state)
            arrays[name+'_log_gain'], arrays[name+'_shift'] = parameters(adapted)
            print('TARGET_DONE', tag, name, flush=True)
        with np.load(REPAIR/f'{tag}_models.npz') as z:
            scores_by_arm['previous_repaired'] = z['mask_and_source_scores']
        truth = obs.iloc[ids].class_id.to_numpy()
        port, draw = obs.iloc[ids].port.iloc[0], int(tag.rsplit('draw', 1)[1])
        for arm, score in scores_by_arm.items():
            pred = score.argmax(1)
            records.append(dict(port=port, draw=draw, arm=arm, **metrics(pred, truth)))
            arrays[arm+'_scores'], arrays[arm+'_pred'] = score, pred
            for c in np.unique(truth):
                count = int((labels == c).sum())
                classes.append(dict(port=port, draw=draw, arm=arm, class_id=int(c), support_count=count,
                    coverage='absent' if count == 0 else 'singleton' if count == 1 else 'multiple',
                    recall=float(100*(pred[truth == c] == c).mean())))
        np.savez_compressed(OUT/f'{tag}_models.npz', **arrays, query_obs_ids=ids)
        pd.DataFrame(records).to_csv(OUT/'results.csv', index=False)
        pd.DataFrame(classes).to_csv(OUT/'by_class.csv', index=False)
        pd.DataFrame(target_histories).to_csv(OUT/'target_training_history.csv', index=False)
        save_json(OUT/'REPLAYS.json', replays)
    save_json(OUT/'COMPLETE.json', dict(tasks=9, source_warmup_fits=2, target_adapter_fits=18,
        baseline_reproduction_fits=1, source_episode_ports=int(designs.port.nunique()),
        source_episodes=len(episodes), seconds=time.perf_counter()-start))
    print(pd.DataFrame(records).groupby('arm')[['OA', 'BA']].mean().to_string())


if __name__ == '__main__':
    run()
