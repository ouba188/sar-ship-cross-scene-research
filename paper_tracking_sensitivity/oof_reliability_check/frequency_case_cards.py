"""CPU-only, post-hoc case cards for the completed frequency residual run."""

import os
import sys

sys.dont_write_bytecode = True
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'

import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from c4_condition_descriptors import MANIFESTS, load_manifest, read_pair
from conditional_direction_residual import direction_kernel, fit_predict
from cross_dataset_c4 import CLASSES
from evaluate_learned_frequency import predict


RUN = Path('E:/Hermes/paper_tracking/sensitivity/learned_frequency_residual_20261009')
OUT = RUN/'mechanism_20261009/cases'
TASKS = [('self24ports', 'Hamburg', ('rescue',)),
         ('self24ports', 'Callao', ('harm',)),
         ('f0921', 'f0921', ('rescue', 'harm')),
         ('oss2', 'oss2', ('rescue', 'harm'))]
BETA = .25
DRAW = 0
GREEN, RED, BLUE, GRAY = '#16765b', '#be3745', '#296caa', '#60656b'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def load_arrays(path, keys=None):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in (archive.files if keys is None else keys)}


def class_record(value):
    return dict(id=int(value), name=CLASSES[int(value)])


def margin(score, positive, negative):
    return float(score[positive]-score[negative])


def replay(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    data = load_arrays(folder/'target.npz')
    embedding = load_arrays(folder/'selective_embedding.npz', ['rows', 'embedding'])
    policies = load_arrays(folder/'policies.npz')
    np.testing.assert_array_equal(embedding['rows'], data['rows'])
    np.testing.assert_array_equal(policies['query_rows'], data['rows'])
    np.testing.assert_array_equal(policies['sample_ids'], data['ids'])
    np.testing.assert_array_equal(policies['support_rows'][DRAW], data['support'][DRAW])
    np.testing.assert_array_equal(policies['eval_masks'][DRAW], data['eval_masks'][DRAW])
    support, labels = data['support'][DRAW], data['support_labels'][DRAW]
    assert len(support) == len(np.unique(support)) == 64
    assert not data['eval_masks'][DRAW, support].any()
    eta = info['eta']
    phi, prior = data['phi'], data['prior']
    mean_kernel = eta*direction_kernel(data['mean'], data['conditions'], support, True)
    freq_kernel = direction_kernel(embedding['embedding'].astype(float), data['conditions'], support, True)
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    root = np.sqrt(len(labels)/(len(counts)*counts[inverse]))
    desired = 2*np.eye(prior.shape[1])[labels]-1
    ks = phi[support]@phi[support].T+(mean_kernel+BETA*freq_kernel)[support]
    alpha = root[:, None]*np.linalg.solve(
        root[:, None]*ks*root[None, :]+8*np.eye(len(support)),
        root[:, None]*(desired-prior[support]))
    # Reproduce the evaluator's summation order before checking saved float32 scores.
    exact = prior+phi@(phi[support].T@alpha)+(mean_kernel+BETA*freq_kernel)@alpha
    reference = predict(data, embedding['embedding'], support, labels, eta)
    np.testing.assert_array_equal(exact, reference)
    baseline = fit_predict(phi, prior, data['mean'], data['conditions'], support, labels,
                           eta, conditional=True)
    slots = {mode: list(policies['modes']).index(mode) for mode in ('baseline', 'selective')}
    effective, selected = {}, {}
    for mode, score in [('baseline', baseline), ('selective', exact)]:
        slot = slots[mode]
        np.testing.assert_array_equal(score.astype(np.float32), policies['scores'][DRAW, slot])
        np.testing.assert_array_equal(score.argmax(1), policies['base_predictions'][DRAW, slot])
        selected[mode] = policies['selected'][DRAW, slot]
        effective[mode] = np.where(selected[mode][:, None], data['expert'][DRAW], score)
        np.testing.assert_array_equal(effective[mode].argmax(1), policies['predictions'][DRAW, slot])
    # This is the existing-kernel component under the already reconstructed new alpha.
    existing_new = prior+phi@(phi[support].T@alpha)+mean_kernel@alpha
    direct = BETA*freq_kernel@alpha
    np.testing.assert_allclose(existing_new+direct, exact, rtol=1e-12, atol=1e-12)
    return dict(info=info, data=data, embedding=embedding['embedding'], alpha=alpha,
                baseline=baseline, exact=exact, existing_new=existing_new,
                freq_kernel=freq_kernel, mean_kernel=mean_kernel,
                effective=effective, selected=selected,
                replay=dict(unrounded_evaluator_exact=True, saved_float32_scores_exact=True,
                            base_predictions_exact=True, saved_route_predictions_exact=True,
                            all_rows_checked=len(phi), routing_masks='Existing completed draw0 masks',
                            component_max_abs_error=float(np.max(np.abs(existing_new+direct-exact)))))


def select_case(state, y, event):
    old = state['effective']['baseline'].argmax(1)
    new = state['effective']['selective'].argmax(1)
    ev = state['data']['eval_masks'][DRAW]
    mask = ev & ((old != y) & (new == y) if event == 'rescue' else (old == y) & (new != y))
    candidates = np.flatnonzero(mask)
    if not len(candidates):
        raise ValueError(f'No {event} cases in {state["info"]["task"]}')
    groups, counts = np.unique(np.c_[old[candidates], new[candidates]], axis=0, return_counts=True)
    group = groups[np.argmax(counts)]  # Sorted class IDs break equal-count ties.
    group_rows = candidates[(old[candidates] == group[0]) & (new[candidates] == group[1])]
    values = state['effective']['selective'][group_rows, group[1]]-state['effective']['selective'][group_rows, group[0]]
    median = float(np.median(values))
    rank = np.lexsort((state['data']['ids'][group_rows], np.abs(values-median)))
    q = int(group_rows[rank[0]])
    selection = dict(rule='Largest (old full prediction, new full prediction) event group; nearest to its median new full pair margin; ties by class IDs then exact sample ID.',
                     task_chosen_in_advance=True, eligible_event_count=len(candidates),
                     prediction_group=group.tolist(), group_size=len(group_rows),
                     group_median_new_full_margin=median,
                     selected_distance_from_median=float(abs(values[rank[0]]-median)),
                     filters='Evaluation rows only; no filter on images, support signs, or handoff status.',
                     all_event_groups=[dict(old_class=int(g[0]), new_class=int(g[1]), count=int(n))
                                       for g, n in zip(groups, counts)])
    return q, int(old[q]), int(new[q]), selection


def image_record(dataset, data, local, manifest, pixels):
    row = int(data['rows'][local])
    item = manifest.iloc[row]
    sample_id = str(data['ids'][local])
    assert str(item.sample_id) == sample_id, (dataset, row, sample_id, item.sample_id)
    record = dict(sample_id=sample_id, local_task_row=int(local), manifest_row=row)
    if dataset == 'f0921':
        pair = np.asarray(pixels[row])
        record.update(inputs_path=str(MANIFESTS[dataset].parent/'inputs128.npy'),
                      inputs_index=row, source_row=int(item['row']))
    else:
        fields = ['image_relpath_vv', 'image_relpath_vh'] if dataset == 'self24ports' else ['chip_path_vv', 'chip_path_vh']
        paths = [str(item[k]) for k in fields]
        pair, sizes = read_pair(paths)
        record.update(paths=dict(zip(('VV', 'VH'), paths)), native_pixels=sizes)
    assert pair.shape == (2, 128, 128) and np.isfinite(pair).all()
    record['display'] = '128x128 uint8; grayscale fixed [0,255]; no per-chip contrast stretching'
    return pair, record


def build_case(state, dataset, event, y, manifest, pixels):
    data = state['data']
    q, old, new, selection = select_case(state, y, event)
    positive, negative = (int(y[q]), old) if event == 'rescue' else (new, int(y[q]))
    alpha_delta = state['alpha'][:, positive]-state['alpha'][:, negative]
    contributions = BETA*state['freq_kernel'][q]*alpha_delta
    pos = sorted(np.flatnonzero(contributions > 0), key=lambda i: (-contributions[i], i))[:2]
    neg = sorted(np.flatnonzero(contributions < 0), key=lambda i: (contributions[i], i))[:1]
    if len(pos) != 2 or len(neg) != 1:
        raise ValueError('Selected case lacks the requested two positive and one negative supports')
    selected_supports = pos+neg
    pair, query = image_record(dataset, data, q, manifest, pixels)
    images = [pair]
    supports = data['support'][DRAW]
    support_records = []
    e = state['embedding'].astype(float)
    for s, local in enumerate(supports):
        row = int(data['rows'][local])
        assert str(manifest.iloc[row].sample_id) == str(data['ids'][local])
        assert int(y[local]) == int(data['support_labels'][DRAW, s])
        distance = np.sum((data['conditions'][q]-data['conditions'][local])**2)
        support_records.append(dict(support_slot=s, local_task_row=int(local), manifest_row=row,
            sample_id=str(data['ids'][local]), label=class_record(y[local]),
            frequency_cosine=float(e[q]@e[local]), condition_rbf=float(np.exp(-distance/(2*.5**2))),
            k_freq=float(state['freq_kernel'][q, s]),
            k_mean_unscaled=float(data['mean'][q]@data['mean'][local]*np.exp(-distance/(2*.5**2))),
            alpha_positive=float(state['alpha'][s, positive]), alpha_negative=float(state['alpha'][s, negative]),
            alpha_difference=float(alpha_delta[s]), direct_margin_contribution=float(contributions[s]),
            shown=s in selected_supports))
    for s in selected_supports:
        support_pair, image_info = image_record(dataset, data, int(supports[s]), manifest, pixels)
        support_records[s]['image'] = image_info
        images.append(support_pair)
    base_old = margin(state['baseline'][q], positive, negative)
    base_new = margin(state['exact'][q], positive, negative)
    direct = float(contributions.sum())
    refit = margin(state['existing_new'][q]-state['baseline'][q], positive, negative)
    full_old = margin(state['effective']['baseline'][q], positive, negative)
    full_new = margin(state['effective']['selective'][q], positive, negative)
    route_old, route_new = full_old-base_old, full_new-base_new
    error = abs((base_old+refit+direct)-base_new)
    full_error = abs(full_old+refit+direct+(route_new-route_old)-full_new)
    np.testing.assert_allclose(base_old+refit+direct, base_new, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(full_old+refit+direct+route_new-route_old, full_new, rtol=1e-10, atol=1e-11)
    assert full_old < 0 and full_new > 0
    scores = {key: state[key][q].tolist() for key in ('baseline', 'exact', 'existing_new')}
    scores.update(source_prior=data['prior'][q].tolist(), expert=data['expert'][DRAW, q].tolist(),
                  full_old=state['effective']['baseline'][q].tolist(), full_new=state['effective']['selective'][q].tolist())
    record = dict(dataset=dataset, task=state['info']['task'], event=event, draw=DRAW,
        query=query, truth=class_record(y[q]), old_full_prediction=class_record(old),
        new_full_prediction=class_record(new),
        old_base_prediction=class_record(state['baseline'][q].argmax()),
        new_base_prediction=class_record(state['exact'][q].argmax()),
        pair_positive=class_record(positive), pair_negative=class_record(negative),
        eta=state['info']['eta'], beta=BETA, ridge=8., selection=selection,
        routing=dict(old_selected=bool(state['selected']['baseline'][q]),
                     new_selected=bool(state['selected']['selective'][q]),
                     note='Full margin is from the actually selected score vector (base or saved expert); it is not a separately trained full head.'),
        margins=dict(base_old=base_old, base_new=base_new, full_old=full_old, full_new=full_new,
            direct_sum=direct, refit_margin=refit, route_old=route_old, route_new=route_new,
            routing_delta=route_new-route_old, full_delta=full_new-full_old,
            shown_direct_sum=float(contributions[selected_supports].sum()),
            remaining_61_direct_sum=float(direct-contributions[selected_supports].sum()),
            base_decomposition_abs_error=error, full_decomposition_abs_error=full_error),
        scores=scores, shown_support_slots=selected_supports, supports=support_records,
        verification=state['replay'],
        limitation='Outcome-selected illustration, not population proof. Exact support-score attribution, not a pixel attribution map or physical/causal frequency interpretation.')
    # Convert numpy integer slots once, keeping all IDs and floats unrounded in JSON.
    record['shown_support_slots'] = [int(s) for s in selected_supports]
    return record, images


def render(record, images, stem):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig = plt.figure(figsize=(16, 11.8), facecolor='white')
    grid = fig.add_gridspec(3, 4, left=.06, right=.97, top=.80, bottom=.14,
                           height_ratios=[1, 1, 1.1], hspace=.60, wspace=.22)
    m = record['margins']
    title = f"{record['dataset']} / {record['task']} | {record['event'].upper()} | draw 0"
    fig.text(.06, .965, title, fontsize=20, weight='bold', va='top')
    names = [record[k]['name'].replace('_', ' ') for k in ('truth', 'old_full_prediction', 'new_full_prediction')]
    fig.text(.06, .922, f'Truth: {names[0]}     Full prediction: {names[1]}  ->  {names[2]}', fontsize=12)
    p, n = record['pair_positive']['name'].replace('_', ' '), record['pair_negative']['name'].replace('_', ' ')
    fig.text(.06, .892, f'Margin = {p} minus {n}     |     eta={record["eta"]:g}, beta=0.25', fontsize=11)
    shown = [record['supports'][s] for s in record['shown_support_slots']]
    headings = ['QUERY']+[f'POSITIVE SUPPORT {i+1}' for i in range(2)]+['STRONGEST NEGATIVE']
    for col, (pair, heading) in enumerate(zip(images, headings)):
        item = record['query'] if col == 0 else shown[col-1]
        label = record['truth']['name'] if col == 0 else item['label']['name']
        detail = 'Held-out query' if col == 0 else f"Direct contribution: {item['direct_margin_contribution']:+.6f}"
        color = GRAY if col == 0 else GREEN if col < 3 else RED
        for channel in range(2):
            ax = fig.add_subplot(grid[channel, col])
            ax.imshow(pair[channel], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(1.3)
            ax.text(.025, .965, ('VV', 'VH')[channel], color='white', transform=ax.transAxes,
                    va='top', fontsize=10, bbox=dict(facecolor='black', alpha=.65, pad=2, edgecolor='none'))
            if channel == 0:
                ax.set_title(f'{heading}\n{label.replace("_", " ")}\n{detail}', fontsize=10, color=color, pad=9)
            else:
                ax.set_xlabel(textwrap.fill(item['sample_id'], 35), fontsize=8, labelpad=7)
    ax = fig.add_subplot(grid[2, :2])
    order = np.argsort([s['direct_margin_contribution'] for s in record['supports']])[::-1]
    values = np.array([record['supports'][s]['direct_margin_contribution'] for s in order])
    ax.bar(np.arange(64), values, color=np.where(values >= 0, GREEN, RED), width=.85)
    ax.axhline(0, color=GRAY, linewidth=.6)
    ax.set_title(f'All 64 direct support contributions | sum {m["direct_sum"]:+.6f}', fontsize=11, loc='left')
    ax.set_xlabel('Supports sorted by signed contribution (not image similarity)', fontsize=9)
    ax.set_ylabel('Pair-margin contribution', fontsize=9)
    for name, s in zip(('P1', 'P2', 'N1'), record['shown_support_slots']):
        rank = int(np.flatnonzero(order == s)[0])
        value = values[rank]
        ax.annotate(name, (rank, value), xytext=(6 if name == 'P2' else -2,
                    18 if name == 'P2' else 7 if value >= 0 else -12),
                    textcoords='offset points', ha='center', fontsize=8)
    ax.margins(y=.24)
    ax = fig.add_subplot(grid[2, 2:])
    terms = [m['full_old'], m['refit_margin'], m['direct_sum'], m['routing_delta'], m['full_new']]
    names = ['Old full', 'Coefficient refit', 'Direct frequency', 'Routing change', 'New full']
    starts = [0, terms[0], terms[0]+terms[1], terms[0]+terms[1]+terms[2], 0]
    for i, (value, start) in enumerate(zip(terms, starts)):
        ax.bar(i, value, bottom=start, color=BLUE if i in (0, 4) else GREEN if value >= 0 else RED, width=.66)
        endpoint = start+value
        ax.annotate(f'{value:+.5f}', (i, max(start, endpoint)), xytext=(0, 5),
                    textcoords='offset points', ha='center', fontsize=9)
    ax.axhline(0, color=GRAY, linewidth=.6)
    ax.set_xticks(range(5), [x.replace(' ', '\n') for x in names], fontsize=9)
    ax.set_title('Exact full-margin accounting', fontsize=11, loc='left')
    endpoints = np.r_[starts, np.asarray(starts)+terms]
    span = float(endpoints.max()-endpoints.min())
    ax.set_ylim(endpoints.min()-.17*span, endpoints.max()+.24*span)
    for ax in fig.axes[-2:]:
        ax.spines[['top', 'right']].set_visible(False)
    route = record['routing']
    selection = record['selection']
    fig.text(.06, .083, f"Base margin: {m['base_old']:+.6f} -> {m['base_new']:+.6f}   |   Full margin: {m['full_old']:+.6f} -> {m['full_new']:+.6f}   |   Expert handoff: {route['old_selected']} -> {route['new_selected']}", fontsize=10)
    fig.text(.06, .058, f"Selection: {selection['eligible_event_count']} {record['event']} events in this task; largest prediction group n={selection['group_size']}; closest to median new full margin.", fontsize=10)
    fig.text(.06, .032, 'Selected illustration, not population proof. Chips show inputs; contributions explain support scores, not causal pixels or physical frequency bands.', fontsize=10, color=GRAY)
    fig.savefig(OUT/f'{stem}.png', dpi=160, facecolor='white')
    fig.savefig(OUT/f'{stem}.pdf', facecolor='white')
    plt.close(fig)


def write_report(records):
    lines = ['# Learned Frequency Residual: Six Selected Cases', '',
        'These cards visualize the completed model, using draw 0 only. No neural training, parameter search, new model selection, feature extraction, or all-task control experiment was performed. The existing 64-support closed-form solve is replayed to recover its exact coefficients. Parent files and completed outputs are read-only.', '',
        'The self24ports examples use Hamburg for rescue and Callao for harm, as specified for this illustration; they are not pooled across ports. F0921 and OSS2 each contribute one rescue and one harm. Target outcomes are used only for retrospective illustration. These selected examples are not population-level evidence or independent confirmation.', '',
        '## Selection', '',
        'For each event type, use all draw-0 evaluation rows whose final prediction changes from wrong to correct (rescue) or correct to wrong (harm). Choose the largest (old full prediction, new full prediction) group, breaking group-size ties by ascending class IDs. Select the row closest to the median new full pair margin within that group, breaking ties by exact sample ID. Do not filter by image appearance, attribution sign, or routing state. IDs, counts and selection distances are recorded in each JSON.', '',
        '## Exact Decomposition', '',
        '`K0(q,s) = phi(q) phi(s)^T + eta * mean_cosine(q,s) * RBF(q,s)`', '',
        '`Kfreq(q,s) = learned_embedding(q) learned_embedding(s)^T * RBF(q,s)`', '',
        '`alpha = W^(1/2) [W^(1/2) (K0_SS + 0.25 Kfreq_SS) W^(1/2) + 8I]^-1 W^(1/2) (Y - prior_S)`', '',
        '`W_ss = 64 / (number_of_observed_support_classes * count_of_support_class_s)`; `Y` is +1 for the support class and -1 otherwise. The RBF bandwidth is the existing 0.5. `eta` is read from each PREPARED.json, including Hamburg eta=0.', '',
        'For rescue, `(a,b) = (truth, old wrong full prediction)`. For harm, `(a,b) = (new wrong full prediction, truth)`. Each direct contribution is exactly `0.25 * Kfreq(q,s) * (alpha[s,a] - alpha[s,b])`. Positive support contributions favor a; they need not come from class-a supports because coefficients are jointly solved residual corrections, not label votes.', '',
        '`new base margin = old base margin + coefficient-refit margin + sum(direct support contributions)`', '',
        'The refit term is the change in the existing-kernel correction under the new support coefficients. It excludes the frequency term; the source prior stays fixed. Computing it as the existing-kernel score under new alpha minus the exact baseline score also preserves the eta=0 evaluator path. Direct frequency is an additive accounting term with the actual joint alpha, not an independently fitted branch or causal ablation.', '',
        'The final pipeline can hand off to its saved expert. A full margin here uses the score vector actually selected by the saved route. Thus `new full - old full = refit + direct + (new route offset - old route offset)`, where each route offset is full margin minus base margin. Scores from different heads are not calibrated probabilities; the routing term is arithmetic bookkeeping of the final decision, not a learned additive feature.', '',
        'The replay matches evaluate_learned_frequency.predict before rounding, then exactly matches saved float32 scores, base predictions and final predictions using the saved routing masks for all rows of each of the four inspected tasks. Both scalar decompositions are checked against unrounded scores. This sidecar does not rerun gate admission or the main all-25-task controls.', '',
        '## Cases', '',
        '| Dataset / task | Event | Query ID | Full prediction | Direct | Refit | Routing delta | Full margin old -> new |',
        '|---|---|---|---|---:|---:|---:|---|']
    for r in records:
        m = r['margins']
        lines.append(f"| {r['dataset']} / {r['task']} | {r['event']} | {r['query']['sample_id']} | {r['old_full_prediction']['name']} -> {r['new_full_prediction']['name']} | {m['direct_sum']:+.8f} | {m['refit_margin']:+.8f} | {m['routing_delta']:+.8f} | {m['full_old']:+.8f} -> {m['full_new']:+.8f} |")
    for r in records:
        m = r['margins']
        stem = f"{r['dataset']}_{r['task']}_{r['event']}_draw0"
        if r['routing']['new_selected'] and not r['routing']['old_selected']:
            interpretation = ('The new pipeline admits the saved expert, whose prediction is correct. '
                'The new base prediction remains wrong; its negative pair margin shows that the direct frequency and refit terms do not themselves complete this rescue. '
                'This is a changed-handoff rescue, with the direct term and refit contribution shown separately. The card does not localize the gate-ranking reason for admission.')
        elif r['routing']['old_selected'] and not r['routing']['new_selected']:
            interpretation = ('The old pipeline used the correct expert prediction, while the new pipeline no longer hands off this query. '
                'Both old and new base predictions are already wrong. The full-policy harm is therefore a loss of the correct expert handoff; '
                'it is not a base prediction newly flipped by the frequency term. The card does not localize the gate-ranking reason for losing admission.')
        else:
            interpretation = ('Neither policy hands off this query. The displayed full prediction change is the base prediction change, '
                'and the sum of direct frequency and coefficient-refit terms crosses the zero pair-margin boundary. '
                'This is exact additive score accounting, not a counterfactual claim about individual pixels or support deletion.')
        lines += ['', f"### {r['dataset']} / {r['task']}: {r['event']}", '',
            f"![{stem}]({stem}.png)", '',
            interpretation, '',
            f"The direct frequency term {'favors' if m['direct_sum'] > 0 else 'opposes'} the new winning class relative to the old class. Coefficient refitting contributes {m['refit_margin']:+.8f}; the change in route offset contributes {m['routing_delta']:+.8f}. The three displayed supports sum to {m['shown_direct_sum']:+.8f}; the remaining 61 sum to {m['remaining_61_direct_sum']:+.8f}. The full 64-term list and all score vectors are in `{stem}.json`."]
    lines += ['', '## Image And Interpretation Boundaries', '',
        'Query and support chips are the actual model input pairs. self24ports and OSS2 use c4_condition_descriptors.load_manifest/read_pair; F0921 uses inputs128.npy in read-only mmap mode, with feature_rows.npy checked against manifest source rows. IDs are checked against task ordering. VV and VH use fixed grayscale [0,255], without individual stretching. The two strongest positive contributors and strongest negative contributor are displayed; all 64 are present in JSON.', '',
        'These are support-level score explanations, not saliency maps. Resized display chips and learned embedding similarities do not identify physical scattering mechanisms, spatially causal pixels, calibrated radiometry, or universally useful frequency bands. A routed harm must not be described as a direct-frequency-caused harm. Population support matching and no-image controls belong to the separate main analysis.', '',
        'Outputs: six PNG cards, six PDF cards, six per-case JSON files, cases.json, VERIFICATION.json and this README. JSON numbers retain full Python float precision; displayed numbers are rounded only for readability.', '']
    (OUT/'README.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, checks = [], {}
    manifests = {}
    for dataset, task, events in TASKS:
        folder = RUN/'tasks'/dataset/task if dataset == 'self24ports' else RUN/'tasks'/dataset
        print(f'REPLAY {dataset}/{task}: exact draw0 scores and row identities', flush=True)
        state = replay(folder)
        if dataset not in manifests:
            manifests[dataset] = load_manifest(dataset)
        manifest = manifests[dataset]
        labels = pd.read_csv(MANIFESTS[dataset], usecols=['class_id']).class_id.to_numpy()
        y = labels[state['data']['rows']]
        np.testing.assert_array_equal(manifest.sample_id.to_numpy(str)[state['data']['rows']], state['data']['ids'])
        pixels = None
        if dataset == 'f0921':
            pixels = np.load(MANIFESTS[dataset].parent/'inputs128.npy', mmap_mode='r')
            np.testing.assert_array_equal(np.load(MANIFESTS[dataset].parent/'feature_rows.npy', mmap_mode='r'), manifest['row'])
        for event in events:
            record, images = build_case(state, dataset, event, y, manifest, pixels)
            stem = f'{dataset}_{task}_{event}_draw0'
            write_json(OUT/f'{stem}.json', record)
            render(record, images, stem)
            records.append(record)
            print(f"CARD {stem} {record['query']['sample_id']} {json.dumps(record['margins'])}", flush=True)
        checks[f'{dataset}/{task}'] = state['replay']
        del state
    write_json(OUT/'cases.json', records)
    write_report(records)
    write_json(OUT/'VERIFICATION.json', dict(cases=len(records), tasks=checks, cpu_only=True,
        neural_training=False, parameter_search=False, draw=DRAW, beta=BETA,
        image_identity_checks=True, max_base_decomposition_error=max(r['margins']['base_decomposition_abs_error'] for r in records),
        max_full_decomposition_error=max(r['margins']['full_decomposition_abs_error'] for r in records),
        scope='Four selected tasks, six visual cases; all25 quantitative controls remain with main analysis',
        interpreter=sys.executable, code=str(Path(__file__).resolve()), output=str(OUT)))
    print(f'COMPLETE {len(records)} cases: {OUT}', flush=True)


if __name__ == '__main__':
    main()
