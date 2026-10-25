"""Post-training paired outcomes, source utility diagnostics and native case panels."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE / '.mplcache')
import json
import numpy as np
import pandas as pd
import torch
import train as run

# These modules import this experiment's train module; only reuse their public helpers.
native = run.load('intervention_native_analysis', HERE.parent / 'anchored_m3_20261025/analyze.py')
stats = native.previous
native.HERE = stats.HERE = HERE
stats.PAIRS = [(a, 'old_full') for a in run.ARMS] + [
    ('effect', 'ce_only'), ('effect', 'shuffled')]
os.environ['MPLCONFIGDIR'] = str(HERE / '.mplcache')
SIGN_EPS = 1e-6
REGIONS = ('body', 'sides', 'ends')


def type_fields(t):
    return dict(type=int(t), polarization=('VV', 'VH')[t // 9],
                region=REGIONS[(t % 9) // 3], wavelength_px=(4, 8, 16)[t % 3])


def active_operations(reader, ids):
    v = reader.valid[ids]
    return v[..., 0] & (v[..., 1] | v[..., 2])


def weighted_type_mean(target, active, query_weights):
    entry_weight = query_weights[:, None] * active / active.sum(1).clip(1)[:, None]
    total = entry_weight.sum(0)
    return np.divide((target * entry_weight).sum(0), total,
                     out=np.zeros(target.shape[1]), where=total > 0)


def utility_metrics(prediction, target, raw, active, labels, query_weights, normalizer):
    error = (prediction - target) ** 2
    n = active.sum(1)
    query_mse = (error * active).sum(1) / n.clip(1)
    residual = np.abs(prediction - target)
    huber = np.where(residual <= 1, .5 * error, residual - .5)
    classes = np.unique(labels[n > 0])
    signed = active & (np.abs(raw) > SIGN_EPS)
    positive, negative = signed & (raw > 0), signed & (raw < 0)

    def fraction(values, mask):
        return float(values[mask].mean()) if mask.any() else np.nan

    return dict(n_queries=len(target), active_entries=int(active.sum()),
                target_mse=fraction(error, active),
                query_equal_mse=float(query_mse[n > 0].mean()) if (n > 0).any() else np.nan,
                macroclass_query_mse=float(np.mean([
                    query_mse[(labels == c) & (n > 0)].mean() for c in classes]))
                if len(classes) else np.nan,
                macroclass_count=len(classes),
                matched_weighted_mse=float(np.mean(
                    query_weights * (error * active).sum(1) / normalizer)),
                matched_weighted_huber=float(np.mean(
                    query_weights * (huber * active).sum(1) / normalizer)),
                mean_training_query_weight=float(query_weights.mean()),
                sign_entries=int(signed.sum()), positive_entries=int(positive.sum()),
                negative_entries=int(negative.sum()),
                sign_accuracy=fraction(np.sign(prediction) == np.sign(raw), signed),
                positive_recall=fraction(prediction > 0, positive),
                negative_recall=fraction(prediction < 0, negative),
                zero_prediction_fraction=fraction(prediction == 0, signed))


@torch.inference_mode()
def source_utility(data, reader, models, receipts, source):
    """Selected and fixed-final predictions on source only; fit means/RMS on 644."""
    seed = reader.seed
    ids, raw = source['ids'], source['effect']
    np.testing.assert_array_equal(ids, np.r_[data.tr, data.va])
    ntrain = len(data.tr)
    scale = max(float(np.sqrt(np.mean(raw[:ntrain] ** 2))), 1e-6)
    target = np.tanh(raw / scale)
    active = active_operations(reader, ids).cpu().numpy().reshape(len(ids), 18)
    labels = data.table.iloc[ids].class_id.to_numpy(int)
    class_counts = np.bincount(labels[:ntrain], minlength=8)
    query_weights = .5 + .5 * ntrain / (8 * class_counts[labels])
    normalizer = active.sum(1).clip(1)
    counts = active[:ntrain].sum(0)
    mean = np.divide((target[:ntrain] * active[:ntrain]).sum(0), counts,
                     out=np.zeros(18), where=counts > 0)
    baseline = np.broadcast_to(mean, target.shape) * active
    weighted_mean = weighted_type_mean(target[:ntrain], active[:ntrain], query_weights[:ntrain])
    weighted_baseline = np.broadcast_to(weighted_mean, target.shape) * active
    arrays = dict(ids=ids, up_index=data.up[ids], raw_effect=raw, target=target,
                  active=active, source_rms=np.array(scale), source_type_mean=mean,
                  source_type_counts=counts, source_weighted_type_mean=weighted_mean,
                  source_class_counts=class_counts, labels=labels, training_query_weights=query_weights)
    rows = []

    def record(prediction, arm, state, step):
        for role, take in (('utility_train', np.arange(ntrain)),
                           ('selection', np.arange(ntrain, len(ids)))):
            for t in (-1, *range(18)):
                columns = np.arange(18) if t == -1 else np.array([t])
                ix = np.ix_(take, columns)
                args = (target[ix], raw[ix], active[ix], labels[take], query_weights[take], normalizer[take])
                m = utility_metrics(prediction[ix], *args)
                base = utility_metrics(baseline[ix], *args)
                weighted_base = utility_metrics(weighted_baseline[ix], *args)
                rows.append(dict(seed=seed, arm=arm, state=state, checkpoint_step=step,
                                 role=role, source_rms=scale, type=t,
                                 **({} if t == -1 else {k: v for k, v in type_fields(t).items() if k != 'type'}),
                                 **m, mse_gain_vs_type_mean=base['target_mse'] - m['target_mse'],
                                 matched_mse_gain_vs_weighted_type_mean=
                                 weighted_base['matched_weighted_mse'] - m['matched_weighted_mse'],
                                 matched_huber_gain_vs_weighted_type_mean=
                                 weighted_base['matched_weighted_huber'] - m['matched_weighted_huber']))

    record(baseline, 'source_type_mean', 'source_fit', 0)
    record(weighted_baseline, 'source_weighted_type_mean', 'source_fit', 0)
    record(np.zeros_like(target), 'zero', 'constant', 0)
    for arm in run.ARMS:
        receipt = receipts[arm]
        assert np.isclose(scale, receipt['source_effect_rms'], rtol=1e-5)
        for state in ('selected', 'final'):
            if state == 'selected':
                model, step = models[arm], receipt['selected_step']
            else:
                checkpoint = torch.load(HERE / 'models' / f'final_{arm}_{seed}.pt',
                                        map_location='cpu', weights_only=False)
                assert checkpoint['diagnostic_only']
                model = run.make_model(seed)
                model.load_state_dict(checkpoint['state'])
                model.eval()
                step = checkpoint['receipt']['steps']
            predictions = []
            for first in range(0, len(ids), 64):
                take = ids[first:first + 64]
                _, utility = model(reader.x[take], reader.valid[take], reader.context[take])
                predictions.append(utility.cpu().numpy().reshape(len(take), 18))
            prediction = np.concatenate(predictions)
            arrays[f'{state}_{arm}'] = prediction
            record(prediction, arm, state, step)
            if state == 'final':
                del model
    np.savez_compressed(HERE / 'analysis' / f'source_utility_{seed}.npz', **arrays)
    return rows


@torch.inference_mode()
def predict_override(reader, ids, model, gates):
    scores = []
    for first in range(0, len(ids), 32):
        take = ids[first:first + 32]
        override = torch.as_tensor(gates[first:first + 32], device=reader.x.device,
                                   dtype=reader.x.dtype)
        scores.append(reader.forward(take, model, override=override).cpu().numpy())
    return np.concatenate(scores)


def single_gate(gates):
    """Keep the maximum deployed amplitude, without rescaling it to a full endpoint."""
    flat = gates.reshape(len(gates), 18)
    out = np.zeros_like(flat)
    index = flat.argmax(1)
    out[np.arange(len(flat)), index] = flat[np.arange(len(flat)), index]
    return out.reshape(gates.shape)


@torch.inference_mode()
def fixed_diagnostics(data, reader, models, receipts, scores, gates):
    seed = reader.seed
    rows, cache = [], {}
    y = data.table.iloc[data.te].class_id.to_numpy(int)
    old = scores[seed, 'old_full'][data.te]
    for arm in run.ARMS:
        full = scores[seed, arm][data.te]
        for mode in ('both', 'initial', 'score', 'transport'):
            value = reader.predict(data.te, models[arm], mode=mode, batch=32)
            if mode in ('both', 'initial'):
                expected = full if mode == 'both' else old
                assert float(np.abs(value - expected).max()) < 3e-5, (seed, arm, mode)
                np.testing.assert_array_equal(value.argmax(1), expected.argmax(1))
            cache[f'{arm}_{mode}'] = value
            rows.append(dict(seed=seed, arm=arm, mode=mode,
                             selected_step=receipts[arm]['selected_step'],
                             **run.metrics(y, value),
                             **{f'vs_old_{k}': v for k, v in stats.effect(value, old, y).items()},
                             **{f'both_vs_mode_{k}': v for k, v in stats.effect(full, value, y).items()},
                             max_score_difference_from_both=float(np.abs(value - full).max())))

    dynamic = gates[seed, 'effect']
    mean = dynamic[data.tr].mean(0)
    ids = np.arange(data.n)
    active = active_operations(reader, ids).cpu().numpy()
    constant = np.broadcast_to(mean, dynamic.shape).copy() * active
    sparse = single_gate(dynamic)
    extra = []
    for name, override in (('source_mean_gate', constant), ('single_largest_gate', sparse)):
        value = predict_override(reader, ids, models['effect'], override)
        cache[name] = value
        for role, take in (('utility_train', data.tr), ('selection', data.va), ('evaluation', data.te)):
            yy = data.table.iloc[take].class_id.to_numpy(int)
            base, actual = scores[seed, 'old_full'][take], scores[seed, 'effect'][take]
            extra.append(dict(seed=seed, diagnostic=name, role=role,
                              selected_step=receipts['effect']['selected_step'],
                              **run.metrics(yy, value[take]),
                              **{f'vs_old_{k}': v for k, v in stats.effect(value[take], base, yy).items()},
                              **{f'dynamic_vs_diagnostic_{k}': v for k, v in stats.effect(actual, value[take], yy).items()},
                              mean_gate=float(override[take].mean()),
                              max_score_difference_from_dynamic=float(np.abs(actual - value[take]).max())))
    np.savez_compressed(HERE / 'analysis' / f'fixed_diagnostics_{seed}.npz',
                        evaluation_ids=data.te, all_ids=ids, source_mean_gate=mean,
                        **{('score_' + k): v for k, v in cache.items()})
    return rows, extra


def choose_cases(data, scores, receipts, sources):
    """Selected-effect cases, signed source extremes, and two requested source examples."""
    chosen, used = [], set()
    ids = data.te
    y = data.table.iloc[ids].class_id.to_numpy(int)
    candidates = {name: [] for name in ('rescue', 'harm', 'CC', 'WW')}
    for seed in run.SEEDS:
        new = scores[seed, 'effect'][ids].argmax(1) == y
        old = scores[seed, 'old_full'][ids].argmax(1) == y
        step = int(receipts.loc[(receipts.seed == seed) & (receipts.arm == 'effect'), 'selected_step'].iloc[0])
        for name, mask in (('rescue', new & ~old), ('harm', ~new & old), ('CC', new & old), ('WW', ~new & ~old)):
            for k in np.flatnonzero(mask):
                j = int(ids[k]); row = data.table.iloc[j]
                candidates[name].append((step == 0, seed, int(row.obs_id), j))
    for name in ('rescue', 'harm', 'CC', 'WW', 'CC', 'WW', 'CC', 'WW'):
        if len(chosen) >= 4:
            break
        for _, seed, obs, j in sorted(candidates[name]):
            vessel = str(data.table.iloc[j].mmsi)
            if vessel not in used:
                chosen.append(dict(arm='effect', seed=seed, obs_id=obs, index=j, case=name,
                                   kind='selected_gate', sampling='active checkpoint first, then seed/obs; distinct vessels'))
                used.add(vessel)
                break
    teacher_used = set()
    for sign, name in ((1, 'teacher_positive'), (-1, 'teacher_negative')):
        options = []
        for seed in run.SEEDS:
            source = sources[seed]
            mask = np.isin(source['ids'], data.va)
            ids, effects = source['ids'][mask], source['effect'][mask]
            for k, t in np.argwhere(sign * effects > SIGN_EPS):
                j = int(ids[k]); value = float(effects[k, t])
                options.append((-sign * value, seed, j, int(t), value))
        for _, seed, j, t, value in sorted(options):
            obs = int(data.table.iloc[j].obs_id)
            if obs not in teacher_used:
                chosen.append(dict(arm='teacher_endpoint', seed=seed, obs_id=obs, index=j,
                                   case=name, kind='teacher_endpoint', operation=t, raw_effect=value,
                                   sampling='strongest signed source-selection endpoint, descriptive selection only'))
                teacher_used.add(obs)
                break
    for obs, name in ((1761, 'teacher_rescue'), (1822, 'teacher_third_class_harm')):
        matches = data.va[data.table.iloc[data.va].obs_id.to_numpy() == obs]
        assert len(matches) == 1, (obs, matches)
        j = int(matches[0]); source = sources[29]
        k = int(np.flatnonzero(source['ids'] == j)[0])
        chosen = [c for c in chosen if not (c['kind'] == 'teacher_endpoint'
                  and c['seed'] == 29 and c['obs_id'] == obs and c['operation'] == 0)]
        chosen.append(dict(arm='teacher_endpoint', seed=29, obs_id=obs, index=j,
                           case=name, kind='teacher_endpoint', operation=0,
                           raw_effect=float(source['effect'][k, 0]),
                           sampling='prespecified source-table failure example; local mechanism only'))
    return chosen


def actual_reference_map(data, reader, up):
    # The panels use this selected reader's real reference IDs, not a fresh retrieval.
    refs = reader.refs[:data.n].cpu().numpy()
    mapped = np.full((len(up), *refs.shape[1:]), -1, dtype=np.int64)
    mapped[data.up[:data.n]] = data.up[refs]
    return mapped


def rival_summary(before, after, true):
    wrong = np.arange(len(before)) != true
    old_rival = int(np.where(wrong, before, -np.inf).argmax())
    new_rival = int(np.where(wrong, after, -np.inf).argmax())
    m0, m1 = float(before[true] - before[old_rival]), float(after[true] - after[new_rival])
    fixed = float(after[true] - after[old_rival])
    return dict(old_strongest_rival=old_rival, new_strongest_rival=new_rival,
                old_fixed_margin_before=m0, old_fixed_margin_after=fixed,
                new_rival_margin_before=float(before[true] - before[new_rival]),
                new_rival_margin_after=m1, fixed_margin_change=fixed - m0,
                dynamic_margin_before=m0, dynamic_margin_after=m1, dynamic_margin_change=m1 - m0)


@torch.inference_mode()
def explain_case(data, reader, model, case, before, after, rival, expected_gate):
    j, seed = case['index'], case['seed']
    true = int(data.table.iloc[j].class_id)
    ids = [j]
    teacher_rival = int(np.where(np.arange(8) == true, -np.inf, before).argmax())
    raw, endpoints = reader.teaching(ids, return_scores=True)
    if case['kind'] == 'teacher_endpoint':
        gate = torch.zeros((1, 2, 3, 3), device=reader.x.device)
        gate.reshape(1, 18)[0, case['operation']] = 1
        score, local, gate = reader.forward(ids, model, override=gate, return_local=True)
        rows_to_disable = [case['operation']]
    else:
        score, local, gate = reader.forward(ids, model, return_local=True)
        np.testing.assert_allclose(gate[0].cpu().numpy(), expected_gate, atol=1e-6, rtol=1e-5)
        rows_to_disable = range(18)
    replay = float(np.abs(score[0].cpu().numpy() - after).max())
    assert replay < 3e-5, (seed, j, replay)
    full_margin = float(score[0, true] - score[0, rival])
    rows = []
    for t in rows_to_disable:
        disabled = gate.clone()
        disabled.reshape(1, 18)[0, t] = 0
        # Keep all other deployed gates fixed, including their existing normalization.
        alt = reader.forward(ids, model, override=disabled)[0].cpu().numpy()
        ep = endpoints[0, t]
        rows.append(dict(**type_fields(t), gate=float(gate.reshape(1, 18)[0, t]),
                         gate_off_fixed_margin=float(alt[true] - alt[rival]),
                         retained_gate_fixed_margin_effect=full_margin - float(alt[true] - alt[rival]),
                         gate_off_prediction=int(alt.argmax()), gate_off_scores=alt,
                         teacher_rival=teacher_rival, teacher_raw_effect=float(raw[0, t]),
                         teacher_endpoint_prediction=int(ep.argmax()),
                         teacher_endpoint_dynamic_margin=float(stats.margin(ep[None], np.array([true]))[0]),
                         **{f'endpoint_{k}': v for k, v in rival_summary(before, ep, true).items()}))
    top = torch.as_tensor(before, device=reader.x.device).topk(2).indices.cpu().numpy()
    actual = reader.refs[j].cpu().numpy()
    context_refs = [dict(predicted_rank=rank + 1, predicted_class=int(c),
                         indices=actual[c], up_indices=data.up[actual[c]],
                         obs_ids=data.table.iloc[actual[c]].obs_id.to_numpy(),
                         paths=data.table.iloc[actual[c]][['image_vv', 'image_vh']].to_dict('records'))
                    for rank, c in enumerate(top)]
    return dict(replay_max_abs=replay, fixed_rival=rival, teacher_rival=teacher_rival,
                competition=rival_summary(before, after, true),
                gate=gate[0].cpu().numpy(), local=local[0].cpu().numpy(),
                validity=reader.valid[j].cpu().numpy(), uncertainty_context=reader.context[j].cpu().numpy(),
                old_top2_reference_context=context_refs, operations=rows,
                interpretation='Same-model gate-off conditional effects; other gates fixed, not additive contributions. '
                'Teacher endpoints use the old strongest rival and are separate from learned gates.')


def main():
    assert (HERE / 'complete.json').is_file(), 'Wait for all selected checkpoints and evaluation outputs.'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    (HERE / 'analysis').mkdir(exist_ok=True)
    data = run.Data()
    scores, gates, sources = {}, {}, {}
    for seed in run.SEEDS:
        for arm in ('old_full', *run.ARMS):
            with np.load(HERE / 'results' / f'{arm}_{seed}.npz') as z:
                np.testing.assert_array_equal(z['up_index'], data.up[:data.n])
                scores[seed, arm] = z['score'].copy()
                if arm != 'old_full':
                    gates[seed, arm] = z['gate'].copy()
        with np.load(HERE / 'results' / f'source_interventions_{seed}.npz') as z:
            sources[seed] = {key: z[key].copy() for key in ('ids', 'effect', 'score')}
    receipts = pd.read_csv(HERE / 'training_receipts.csv')
    receipt = stats.statistics(data, scores)
    native.source_and_development(data, scores)
    chosen = choose_cases(data, scores, receipts, sources)
    base = HERE.parents[1]
    geo = run.load('intervention_panel_geo', base.parent / 'eight_class_evidence_20261023/obb_dynamic_readout/common.py')
    up = pd.read_csv(base / 'regional_utility_pilot/upstream/table.csv')
    with np.load(base / 'regional_utility_pilot/upstream/summaries.npz') as z:
        valid = z['valid'].copy()
    snapshots = {int(s['obs_id']): s for s in json.loads(
        (geo.OLD / 'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    utility_rows, switch_rows, preference_rows, cases, gate_rows = [], [], [], [], []
    for seed in run.SEEDS:
        reader = run.Reader(data, seed)
        models, selected = {}, {}
        for arm in run.ARMS:
            models[arm], selected[arm] = run.restore(reader, arm, seed)
        utility_rows.extend(source_utility(data, reader, models, selected, sources[seed]))
        switches, preference = fixed_diagnostics(data, reader, models, selected, scores, gates)
        switch_rows.extend(switches)
        preference_rows.extend(preference)
        refs = actual_reference_map(data, reader, up)
        for case in (c for c in chosen if c['seed'] == seed):
            j = case['index']; before = scores[seed, 'old_full'][j]
            if case['kind'] == 'teacher_endpoint':
                k = int(np.flatnonzero(sources[seed]['ids'] == j)[0])
                after = sources[seed]['score'][k, case['operation']]
                expected_gate = None
            else:
                after = scores[seed, 'effect'][j]
                expected_gate = gates[seed, 'effect'][j]
            path, provenance, true, rival = native.panel(case, data, before, after, geo, up, valid, snapshots, refs)
            diagnostic = explain_case(data, reader, models['effect'], case, before, after, rival, expected_gate)
            extra_panels = []
            if case['case'] == 'teacher_third_class_harm':
                new_case = dict(case, arm='teacher_endpoint_new_rival', case='harm')
                extra_path, extra_provenance, _, new_rival = native.panel(
                    new_case, data, before, after, geo, up, valid, snapshots, refs)
                extra_panels.append(dict(figure=str(extra_path), rival=new_rival, provenance=extra_provenance,
                                         purpose='Post-intervention strongest rival; fixed-old-rival panel retained.'))
            cases.append(dict(**case, true=true, rival=rival, before=before, after=after,
                              selected_step=selected['effect']['selected_step'],
                              comparisons={a: scores[seed, a][j] for a in ('old_full', *run.ARMS)},
                              figure=str(path), provenance=provenance, extra_panels=extra_panels, diagnostics=diagnostic))
            for row in diagnostic['operations']:
                gate_rows.append(dict(seed=seed, obs_id=case['obs_id'], case=case['case'], kind=case['kind'],
                                      true=true, rival=rival, **{k: v for k, v in row.items() if k != 'gate_off_scores'}))
            print('CASE', case, flush=True)
        del reader, models
        torch.cuda.empty_cache()
        print('ANALYZED', seed, flush=True)
    pd.DataFrame(utility_rows).to_csv(HERE / 'source_utility_prediction.csv', index=False)
    pd.DataFrame(switch_rows).to_csv(HERE / 'fixed_model_switches.csv', index=False)
    pd.DataFrame(preference_rows).to_csv(HERE / 'fixed_gate_preferences.csv', index=False)
    pd.DataFrame(gate_rows).to_csv(HERE / 'case_gate_effects.csv', index=False)
    run.write_json(HERE / 'analysis/cases.json', cases)
    receipt.update(cases=len(cases), figures_rendered_not_human_reviewed=True,
                   case_selection='Four distinct-vessel selected-effect cases; fill missing flips with CC/WW. '
                   'Up to two strongest signed source-selection teacher endpoints plus requested seed29 '
                   '1761/type0 rescue and 1822/type0 third-class harm; not prevalence estimates or an explanation of all failures.',
                   source_utility='Unclipped signed utility vs tanh(effect/source644 RMS); source644 active-type mean and zero baselines. '
                   'Also report macroclass query MSE and training-weighted MSE/Huber. Weighted type means fit source644 '
                   'targets with training class weight divided by each query active-operation count. Per-type weighted '
                   'losses retain the full-query denominator and sum to the aggregate loss. The weighted mean minimizes '
                   'matched MSE, not necessarily Huber. Selected checkpoints and fixed final100 source only; '
                   'selection reused for J is not an independent test.',
                   raw_effect_sign_tolerance=SIGN_EPS,
                   diagnostics='Prespecified fixed-model K/T switches, source644 mean deployed gate with validity mask, '
                   'and single maximum positive deployed gate without amplitude rescaling. Never selected by development outcomes.',
                   uncertainty='2000 acquisition-cluster bootstrap draws after seed averaging; BA uses all-eight-class draws.',
                   scope='No final checkpoint development inference; no training or checkpoint selection. '
                   'Source action coverage and operator scale are reported by source_scope.py/operator_scale separately.',
                   small_dose='Preserve small_dose_source_pilot as a source-only amplitude failure; do not infer that the full operator has no utility.')
    run.write_json(HERE / 'analysis_receipt.json', receipt)
    print(pd.DataFrame(utility_rows).query("role=='selection' and type==-1").to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
