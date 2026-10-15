"""Bounded source-episodic pooling pilot; no auxiliary source-score head."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'

import copy
import gc
import pickle
import time

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from episodic_spatial_pooling import EpisodicPool, adapted_scores, balanced_ce, balanced_subset
from evidence_aware_handoff import fit_gate, predict_gate, route_scores
from projection_residual_expert import discarded_coordinates, fit_kernel_correction
from run_stable_guided_spatial_pooling import (
    ROOT, OUT as PREVIOUS, PARENT, EVIDENCE, SOURCE, PORTS, archive, read_json,
    inner_held_views, gate_features, collect_record)
from sar_joint_completion import write_json
from stable_guided_spatial_pooling import predicted_pair_indices


OUT = ROOT / 'episodic_spatial_pooling_pilot_20261009'
ARMS = ('original', 'matched_refit', 'fixed_ordinary', 'fixed_stable',
        'learned_ordinary', 'learned_stable')
KINDS = ('ordinary', 'stable')
EPOCHS = 8
SEED = 20261009


def token_inputs(folder, g, rows, selector):
    local = np.searchsorted(g['rows'], rows)
    np.testing.assert_array_equal(g['rows'][local], rows)
    tokens = np.asarray(np.load(folder / 'projected_tokens.npy', mmap_mode='r')[local])
    selected = predicted_pair_indices(selector, g['pairs'])
    logits = {}
    for kind in KINDS:
        arm = kind + '_pool'
        evidence = np.load(folder / (arm + '_evidence.npy'), mmap_mode='r')
        logits[kind] = (np.abs(evidence[local[:, None], np.arange(4)[None, :], selected[:, None]])
                       / g[arm + '_evidence_scale'][selected, None]).astype(np.float32)
    return tokens, logits


def prepare_source(port, g):
    folder = OUT / 'ports' / port
    folder.mkdir(parents=True, exist_ok=True)
    episodes = read_json(PARENT / port / 'EPISODES_COMPLETE.json')['episodes']
    receipts = {e['port']: e for e in read_json(EVIDENCE / 'source_episodes' / port / 'COMPLETE.json')['episodes']}
    source = g['source_rows']
    np.testing.assert_array_equal(source, np.load(EVIDENCE / 'source_features' / port / 'rows.npy'))
    extra = np.load(EVIDENCE / 'source_features' / port / 'features.npy', mmap_mode='r')
    result, maximum_error = [], 0.
    for index, description in enumerate(episodes):
        path = folder / f'episode_{index:02d}.npz'
        result.append(dict(path=path, port=description['port'], split='validation' if index % 3 == 0 else 'train'))
        if path.exists():
            continue
        ep = archive(description['path'])
        receipt = receipts[description['port']]
        old = archive(receipt['path'])
        np.testing.assert_array_equal(ep['rows'], old['rows'])
        np.testing.assert_array_equal(ep['support'], old['support'])
        local = np.searchsorted(source, ep['rows'])
        np.testing.assert_array_equal(source[local], ep['rows'])
        with np.load(EVIDENCE / 'source_pair_cache' / (receipt['pair_key'] + '.npz')) as saved:
            selector = ep['xs'] @ saved['base_coef'].T + saved['base_intercept']
        prior, kernel = inner_held_views(ep, extra[local])
        labels, support, query = ep['labels'], ep['support'], ep['query']
        assert not np.intersect1d(support, query).size
        replay = prior + kernel @ fit_kernel_correction(kernel[support], labels[support], prior[support])
        error = float(np.max(np.abs(replay - old['expert_scores'])))
        maximum_error = max(maximum_error, error)
        np.testing.assert_allclose(replay, old['expert_scores'], atol=2e-6, rtol=2e-6)
        tokens, logits = token_inputs(PREVIOUS / 'ports' / port, g, ep['rows'], selector)
        np.savez(path, tokens=tokens, ordinary_logits=logits['ordinary'], stable_logits=logits['stable'],
            prior=prior, kernel=kernel, labels=labels, support=support, query=query, rows=ep['rows'],
            base=old['base_scores'], expert=old['expert_scores'])
    write_json(folder / 'SOURCE_PREPARED.json', dict(episodes=[dict(port=e['port'], split=e['split']) for e in result],
        max_original_replay_error=maximum_error, target_query_labels_used=0))
    return result


def batch(episode, query):
    support = episode['support']
    rows = np.r_[support, query]
    return dict(tokens=torch.from_numpy(episode['tokens'][rows]),
        prior=torch.from_numpy(episode['prior'][rows]), kernel=torch.from_numpy(episode['kernel'][rows]),
        support=torch.arange(len(support)), labels=torch.from_numpy(episode['labels'][rows]).long(),
        **{k + '_logits': torch.from_numpy(episode[k + '_logits'][rows]) for k in KINDS})


def forward(model, data, kind, token_scale, residual_scale):
    z, alpha = model(data['tokens'], data[kind + '_logits'], token_scale, residual_scale)
    scores = adapted_scores(data['prior'], data['kernel'], z, data['support'], data['labels'][data['support']])
    return scores, alpha


def validation(model, descriptions, kind, token_scale, residual_scale):
    losses, bas = [], []
    with torch.no_grad():
        for description in descriptions:
            ep = archive(description['path'])
            query = balanced_subset(ep['labels'], ep['query'], 64, np.random.default_rng(SEED))
            data = batch(ep, query)
            scores, _ = forward(model, data, kind, token_scale, residual_scale)
            y, scores = data['labels'][64:], scores[64:]
            losses.append(float(balanced_ce(scores, y)))
            correct = scores.argmax(1) == y
            bas.append(float(torch.stack([correct[y == c].double().mean() for c in y.unique()]).mean()))
    return float(np.mean(losses)), float(np.mean(bas))


def train_models(port, g, descriptions):
    folder = OUT / 'ports' / port
    saved_path = folder / 'pooling_models.pt'
    if saved_path.exists():
        return torch.load(saved_path, weights_only=False)
    projected = np.load(PREVIOUS / 'ports' / port / 'projected_tokens.npy', mmap_mode='r')
    local = np.searchsorted(g['rows'], g['source_rows'])
    squares = np.zeros(128)
    for lo in range(0, len(local), 512):
        t = projected[local[lo:lo + 512]].astype(float)
        squares += np.square(t).sum((0, 1))
    token_scale = torch.tensor(np.maximum(np.sqrt(squares / (4 * len(local))), 1e-6), dtype=torch.float32)
    train = [d for d in descriptions if d['split'] == 'train']
    valid = [d for d in descriptions if d['split'] == 'validation']
    results, curves, diagnostics = dict(token_scale=token_scale), [], []
    for kind in KINDS:
        torch.manual_seed(SEED)
        model = EpisodicPool()
        residual_scale = torch.tensor(g[kind + '_pool_residual_scale'], dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.005, weight_decay=.001)
        initial_loss, initial_ba = validation(model, valid, kind, token_scale, residual_scale)
        best_ba, best_loss, best_epoch = initial_ba, initial_loss, 0
        best_state = copy.deepcopy(model.state_dict())
        curves.append(dict(port=port, kind=kind, epoch=0, val_ce=initial_loss, val_ba=initial_ba))
        max_gradient = 0.
        rng = np.random.default_rng(SEED)
        for epoch in range(1, EPOCHS + 1):
            train_loss = []
            for index in rng.permutation(len(train)):
                ep = archive(train[index]['path'])
                query = balanced_subset(ep['labels'], ep['query'], 32, rng)
                data = batch(ep, query)
                optimizer.zero_grad()
                scores, _ = forward(model, data, kind, token_scale, residual_scale)
                loss = balanced_ce(scores[64:], data['labels'][64:])
                loss.backward()
                gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.))
                if not np.isfinite(gradient):
                    raise ValueError('Nonfinite gradient through support adaptation')
                max_gradient = max(max_gradient, gradient)
                optimizer.step()
                train_loss.append(float(loss.detach()))
            val_loss, val_ba = validation(model, valid, kind, token_scale, residual_scale)
            if val_ba > best_ba + 1e-12 or (abs(val_ba - best_ba) <= 1e-12 and val_loss < best_loss):
                best_ba, best_loss, best_epoch = val_ba, val_loss, epoch
                best_state = copy.deepcopy(model.state_dict())
            curves.append(dict(port=port, kind=kind, epoch=epoch, val_ce=val_loss,
                val_ba=val_ba, train_ce=float(np.mean(train_loss))))
            print(f'TRAIN {port} {kind} epoch {epoch}/{EPOCHS}: source-val BA {100*val_ba:.3f}, best epoch {best_epoch}', flush=True)
        results[kind] = dict(state=best_state, residual_scale=residual_scale, selected_epoch=best_epoch)
        diagnostics.append(dict(port=port, kind=kind, selected_epoch=best_epoch,
            initial_val_ba=initial_ba, selected_val_ba=best_ba, initial_val_ce=initial_loss,
            selected_val_ce=best_loss, max_gradient=max_gradient,
            selected_output_weight_norm=float(best_state['output.weight'].norm())))
    torch.save(results, saved_path)
    pd.DataFrame(curves).to_csv(folder / 'learning_curves.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(folder / 'learning_diagnostics.csv', index=False)
    return results


def feature_predictions(tokens, logits, fitted):
    features, diagnostics = {}, {}
    for kind in KINDS:
        for prefix in ('fixed', 'learned'):
            model = EpisodicPool()
            if prefix == 'learned':
                model.load_state_dict(fitted[kind]['state'])
            chunks, max_weights = [], []
            with torch.no_grad():
                for lo in range(0, len(tokens), 512):
                    z, alpha = model(torch.from_numpy(tokens[lo:lo + 512]),
                        torch.from_numpy(logits[kind][lo:lo + 512]), fitted['token_scale'], fitted[kind]['residual_scale'])
                    chunks.append(z.numpy())
                    max_weights.append(alpha.max(1).values.numpy())
            arm = prefix + '_' + kind
            features[arm] = np.concatenate(chunks)
            diagnostics[arm] = float(np.concatenate(max_weights).mean())
    return features, diagnostics


def scores_for_features(prior, kernel, features, support, labels):
    result = {}
    for arm, z in features.items():
        z = z.astype(float)
        augmented = kernel + .25 * z @ z[support].T / z.shape[1]
        result[arm] = prior + augmented @ fit_kernel_correction(augmented[support], labels, prior[support])
    return result


def train_gates(port, descriptions, fitted):
    folder = OUT / 'ports' / port
    if (folder / 'gates.pkl').exists():
        with (folder / 'gates.pkl').open('rb') as stream:
            return pickle.load(stream)
    groups = {a: [] for a in ARMS[1:]}
    for description in descriptions:
        ep = archive(description['path'])
        y, support, query, base = ep['labels'], ep['support'], ep['query'], ep['base']
        features, _ = feature_predictions(ep['tokens'], {k: ep[k + '_logits'] for k in KINDS}, fitted)
        experts = scores_for_features(ep['prior'], ep['kernel'], features, support, y[support])
        experts['matched_refit'] = ep['expert']
        _, inverse, counts = np.unique(y[query], return_inverse=True, return_counts=True)
        weight = 1 / (len(descriptions) * len(counts) * counts[inverse])
        for arm, expert in experts.items():
            eligible = (base.argmax(1) != expert.argmax(1)) & np.isin(expert.argmax(1), y[support])
            groups[arm].append(dict(base=base[query], expert=expert[query], labels=y[query],
                features=gate_features(base, expert)[query], eligible=eligible[query], weights=weight,
                rows=ep['rows'][query], support_labels=y[support], held_source=description['port'], split=description['split']))
    gates, records = {}, []
    for arm, episodes in groups.items():
        joined = {k: np.concatenate([ep[k] for ep in episodes])
                  for k in ('features', 'base', 'expert', 'labels', 'eligible', 'weights')}
        gates[arm] = fit_gate(**joined)
        for ep in episodes:
            score = predict_gate(gates[arm], ep['features'])
            routed = route_scores(ep['base'], ep['expert'], score, ep['support_labels'],
                np.ones(len(score), dtype=bool), ep['rows'])
            record = collect_record(port, ep['held_source'], arm, ep['labels'], ep['base'],
                ep['expert'], routed['predictions'], routed['selected'])
            record['pool_split'] = ep['split']
            records.append(record)
    with (folder / 'gates.pkl').open('wb') as stream:
        pickle.dump(gates, stream)
    pd.DataFrame(records).to_csv(folder / 'source_metrics.csv', index=False)
    return gates


def freeze_target(port, g, fitted, gates, raw, support_truth):
    folder = OUT / 'ports' / port
    if (folder / 'TARGET_FROZEN.json').exists():
        return
    old = archive(PARENT / port / 'matched_joint/joint_policies.npz')
    rows = old['query_rows']
    with (PARENT / port / 'source_model.pkl').open('rb') as stream:
        model = pickle.load(stream)
    extra = np.load(PARENT / port / 'matched_joint/joint_extra.npy', mmap_mode='r')
    with np.load(ROOT / 'pred_cache2' / f'{port}.npz') as saved:
        xs = saved['Xs'][rows]
    selector = model['base'].decision_function(xs)
    tokens, logits = token_inputs(PREVIOUS / 'ports' / port, g, rows, selector)
    features, diag = feature_predictions(tokens, logits, fitted)
    z = model['pca_ssl'].transform(extra).astype(np.float32)
    xe = (np.column_stack((xs, z)) - model['expert_center']) / model['expert_scale']
    prior = model['expert'].decision_function(xe).astype(np.float32).astype(float)
    retained = (xe / np.sqrt(192)).astype(float)
    stats = archive(PARENT / port / 'deployment_complement.npz')
    discarded = np.column_stack((discarded_coordinates(raw[rows], model['pca']), discarded_coordinates(extra, model['pca_ssl'])))
    complement = (discarded - stats['discarded_center']) / stats['discarded_scale'] / np.sqrt(2560)
    outputs = {k: [] for k in ('expert_scores', 'predictions', 'selected', 'gate_scores')}
    max_error = 0.
    for draw, ev in enumerate(old['eval_mask']):
        support = np.searchsorted(rows, old['support_rows'][draw])
        np.testing.assert_array_equal(rows[support], old['support_rows'][draw])
        assert len(support) == 64 and not ev[support].any()
        y, base = support_truth[draw], old['base_scores'][draw]
        kernel = .5 * retained @ retained[support].T + .5 * complement @ complement[support].T + 1
        replay = prior + kernel @ fit_kernel_correction(kernel[support], y, prior[support])
        max_error = max(max_error, float(np.max(np.abs(replay - old['expert_scores'][draw]))))
        np.testing.assert_allclose(replay, old['expert_scores'][draw], atol=2e-6, rtol=2e-6)
        experts = scores_for_features(prior, kernel, features, support, y)
        experts['matched_refit'] = old['expert_scores'][draw]
        values = {k: [old[k][draw]] for k in outputs}
        for arm in ARMS[1:]:
            score = predict_gate(gates[arm], gate_features(base, experts[arm]))
            routed = route_scores(base, experts[arm], score, y, ev, rows)
            assert routed['cap'] == old['cap'][draw]
            for key, value in dict(expert_scores=experts[arm], predictions=routed['predictions'],
                                   selected=routed['selected'], gate_scores=score).items():
                values[key].append(value)
        for key in outputs:
            outputs[key].append(np.stack(values[key]))
    np.savez_compressed(folder / 'target_policies.npz', arms=np.array(ARMS), query_rows=rows,
        support_rows=old['support_rows'], eval_mask=old['eval_mask'], base_scores=old['base_scores'],
        cap=old['cap'], **{k: np.stack(v) for k, v in outputs.items()})
    write_json(folder / 'TARGET_FROZEN.json', dict(port=port, draws=5, query_labels_used=0,
        original_prior_unchanged=True, max_original_replay_error=max_error, mean_max_attention=diag))
    print('TARGET FROZEN ' + port, flush=True)


def evaluate(frame):
    assert read_json(OUT / 'ALL_FROZEN.json')['ports'] == list(PORTS)
    records = []
    for port in PORTS:
        saved = archive(OUT / 'ports' / port / 'target_policies.npz')
        labels = frame.class_id.to_numpy()[saved['query_rows']]
        for draw, ev in enumerate(saved['eval_mask']):
            for index, arm in enumerate(ARMS):
                r = collect_record(port, '', arm, labels[ev], saved['base_scores'][draw, ev],
                    saved['expert_scores'][draw, index, ev], saved['predictions'][draw, index, ev],
                    saved['selected'][draw, index, ev])
                r['draw'] = draw
                records.append(r)
    df = pd.DataFrame(records)
    df.to_csv(OUT / 'target_draw_metrics.csv', index=False)
    values = ['oa', 'ba', 'base_ba', 'expert_ba', 'rescues', 'harms', 'admissions']
    ports = df.groupby(['port', 'arm'], as_index=False)[values].mean()
    overall = ports.groupby('arm', as_index=False)[values].mean()
    ports.to_csv(OUT / 'target_port_means.csv', index=False)
    overall.to_csv(OUT / 'target_equal_port_means.csv', index=False)
    pd.concat([pd.read_csv(OUT / 'ports' / p / 'source_metrics.csv') for p in PORTS]).to_csv(OUT / 'source_metrics.csv', index=False)
    pd.concat([pd.read_csv(OUT / 'ports' / p / 'learning_diagnostics.csv') for p in PORTS]).to_csv(OUT / 'learning_diagnostics.csv', index=False)
    write_json(OUT / 'COMPLETE.json', dict(status='complete', ports=list(PORTS), draws=5,
        metrics=overall.to_dict('records'), main_model_changed=False))
    print(overall.to_string(index=False), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'COMPLETE.json').exists():
        print('Completed output preserved: ' + str(OUT))
        return
    write_json(OUT / 'PROTOCOL.json', dict(ports=list(PORTS), arms=list(ARMS), seed=SEED,
        support=64, draws=5, handoff=.05, epochs=EPOCHS, train_query_per_class=32, validation_query_per_class=64,
        learning_rate=.005, optimizer='AdamW, weight decay .001, gradient clip5', hidden=16,
        selection='Maximum equal-port validation BA, tie minimum balanced CE; epoch0 eligible, no target selection',
        split='Existing source episode order: index modulo3==0 validation, others meta-train',
        objective='Equal-source-episode balanced query CE through exact balanced ridge8 support64 solve',
        pool='Original fixed top2 direction logits PLUS trainable128->16 tanh->1 token scorer',
        fixed_controls='Same kernel-only formulation, zero scorer; original source residual prior NOT added',
        kernel='.5 retained + .5 complement +1 +.25 spatial/128', prior='Exact original expert prior unchanged',
        boundaries='Encoder, geometry and source scaling use all outer source, including meta-validation ports. Not nested OOF. Gate fit on all source episodes, including meta-training query labels; source gate metrics descriptive.',
        labels='No target query labels until all6target policies frozen; only same64 target labels used to adapt',
        main_model_changed=False, development_only=True, no_physical_semantics=True))
    torch.set_num_threads(2)
    frame = pd.read_csv(SOURCE / 'manifest.csv', usecols=['port', 'class_id'])
    raw = np.load(SOURCE / 'pooled_features.npy', mmap_mode='r')
    with threadpool_limits(limits=2):
        for port in PORTS:
            start = time.monotonic()
            g = archive(PREVIOUS / 'ports' / port / 'geometry.npz')
            assert not (frame.port.to_numpy()[g['source_rows']] == port).any()
            descriptions = prepare_source(port, g)
            fitted = train_models(port, g, descriptions)
            gates = train_gates(port, descriptions, fitted)
            with np.load(PARENT / port / 'matched_joint/joint_policies.npz') as old:
                support_truth = frame.class_id.to_numpy()[old['support_rows']]
            freeze_target(port, g, fitted, gates, raw, support_truth)
            write_json(OUT / 'STATUS.json', dict(last_frozen=port, elapsed_port_seconds=time.monotonic() - start))
            gc.collect()
        write_json(OUT / 'ALL_FROZEN.json', dict(ports=list(PORTS), arms=list(ARMS), query_labels_read=False))
        evaluate(frame)


if __name__ == '__main__':
    main()
