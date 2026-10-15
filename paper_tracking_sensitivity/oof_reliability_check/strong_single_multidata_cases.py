"""Real-pixel cases versus the task-trained single, with signed support accounting."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from strong_single_multidata import OUT, TASKS, SEEDS
from strong_single_r50_task import ROOT, SOURCE, support_scores, write_json
from dual_evidence_core import EvidenceContext, load_task, base_cross, MODES
from dual_evidence_cases import image_pair
from c4_condition_descriptors import load_manifest, MANIFESTS
from cross_dataset_c4 import CLASSES


def truth_for(dataset, data):
    path = SOURCE/'manifest.csv' if dataset == 'self24ports' else ROOT/'f0921_detector_recrop_20261007/target_manifest.csv'
    idcol = 'instance_id' if dataset == 'self24ports' else 'sample_id'
    frame = pd.read_csv(path, usecols=[idcol, 'class_id'])
    np.testing.assert_array_equal(frame[idcol].to_numpy(str)[data['rows']], data['ids'])
    return frame.class_id.to_numpy()[data['rows']]


def select(out, seeds):
    choices, done = [], set()
    for folder in sorted((out/'targets').glob('*/*')):
        if not all((folder/f'seed_{s}/COMPLETE.json').exists() for s in seeds):
            continue
        dataset, name = folder.parent.name, folder.name
        with np.load(folder/f'seed_{seeds[0]}/predictions.npz') as saved:
            data = {k: saved[k] for k in ('rows', 'ids', 'eval_masks', 'support')}
        predictions = []
        for seed in seeds:
            with np.load(folder/f'seed_{seed}/predictions.npz') as saved:
                for k in data:
                    np.testing.assert_array_equal(saved[k], data[k])
                predictions.append(saved['predictions'][0, list(saved['arms']).index('task_adapted_single_same_readout')])
        predictions = np.stack(predictions)
        with np.load(ROOT/'backbone_ablation_20261010/targets'/dataset/name/'predictions.npz') as saved:
            for k in data:
                np.testing.assert_array_equal(saved[k], data[k])
            dual = saved['predictions'][0, list(saved['modes']).index('dual_refit_late')]
        y = truth_for(dataset, data)
        single_right, single_wrong = (predictions == y).all(0), (predictions != y).all(0)
        events = {'dual_rescue': single_wrong & (dual == y),
                  'dual_harm': single_right & (dual != y),
                  'both_wrong': single_wrong & (dual != y)}
        for event, mask in events.items():
            if (dataset, event) in done:
                continue
            candidates = np.flatnonzero(mask & data['eval_masks'][0])
            if not len(candidates):
                continue
            q = int(candidates[np.argsort(data['ids'][candidates], kind='stable')[0]])
            choices.append(dict(dataset=dataset, task=name, folder=str(folder), event=event, draw=0,
                query_local=q, truth=int(y[q]), single_prediction=int(predictions[0, q]),
                single_predictions_all_seeds=predictions[:, q].tolist(), dual_prediction=int(dual[q]),
                seeds=seeds, sample_id=str(data['ids'][q]), candidates_in_selected_task=int(len(candidates))))
            done.add((dataset, event))
    return choices


@torch.no_grad()
def build(case, output):
    dataset, name = case['dataset'], case['task']
    folder = TASKS/'self24ports'/name if dataset == 'self24ports' else TASKS/dataset
    info = json.loads((folder/'PREPARED.json').read_text())
    data = load_task(folder)
    q, support, labels = case['query_local'], data['support'][0], data['support_labels'][0]
    run = Path(case['folder'])/f'seed_{case["seeds"][0]}'
    phi = np.load(run/'task_adapted_single_same_readout_phi.npy', mmap_mode='r')
    state = torch.load(run/'final_tail.pt', map_location='cpu', weights_only=False)
    z = torch.from_numpy(np.array(phi[np.r_[support, q]]))
    prior = F.linear(z, state['prior']['weight'])
    ys = torch.as_tensor(labels).long()
    _, inv, counts = torch.unique(ys, return_inverse=True, return_counts=True)
    root = (64/(len(counts)*counts[inv].float())).sqrt()[:, None]
    weighted = z[:64]*root
    desired = 2*F.one_hot(ys, 8).float()-1
    alpha = root*torch.linalg.solve(weighted@weighted.T+8*torch.eye(64), root*(desired-prior[:64]))
    terms_single = ((z[64:]@z[:64].T).T*alpha).numpy()
    single_score = support_scores(z[:64], ys, z[64:], support_prior=prior[:64], query_prior=prior[64:]).numpy()[0]
    reassembly_error = float(np.max(abs(prior[64].numpy()+terms_single.sum(0)-single_score)))
    np.testing.assert_allclose(prior[64].numpy()+terms_single.sum(0), single_score, atol=3e-4, rtol=1e-5)
    assert int(single_score.argmax()) == case['single_prediction']
    context = EvidenceContext(folder, data)
    ref, query = context.features(support), context.features(np.array([q]))
    kernel = context.cross(query, ref)['D'][0]
    evidence = ROOT/'dual_evidence_20261010/targets'/dataset/name
    with np.load(evidence/'coefficients.npz') as saved:
        ae, ab = saved['0_DD_expert'], saved['0_base']
    terms_expert = kernel[:, None]*ae
    terms_base = base_cross(data, np.array([q]), support, info['eta'])[0, :, None]*ab
    expert_prior, base_prior = query['priors']['dual'][0], data['prior'][q]
    expert_score, base_score = expert_prior+terms_expert.sum(0), base_prior+terms_base.sum(0)
    stored = np.load(evidence/'scores.npy', mmap_mode='r')
    np.testing.assert_allclose(expert_score, stored[0, MODES.index('DD_expert'), q], atol=3e-5, rtol=1e-6)
    np.testing.assert_allclose(base_score, stored[0, MODES.index('base'), q], atol=3e-5, rtol=1e-6)
    single_prob, dual_prob = softmax(single_score), .5*(softmax(base_score)+softmax(expert_score))
    assert int(dual_prob.argmax()) == case['dual_prediction']
    y = case['truth']
    rival = case['single_prediction'] if case['single_prediction'] != y else case['dual_prediction']
    dst = output/dataset/case['event']
    dst.mkdir(parents=True, exist_ok=True)
    rows = []
    for position, local in enumerate(support):
        for c in range(8):
            rows.append(dict(support_position=position, support_id=str(data['ids'][local]),
                support_label=int(labels[position]), output_class=c,
                trained_single_term=float(terms_single[position, c]),
                dual_base_term=float(terms_base[position, c]),
                dual_expert_term=float(terms_expert[position, c])))
    pd.DataFrame(rows).to_csv(dst/'all64_signed_terms.csv', index=False)
    branches = [('Trained single', prior[64].numpy(), terms_single, single_score),
                ('Dual base', base_prior, terms_base, base_score),
                ('Dual expert', expert_prior, terms_expert, expert_score)]
    margin = lambda a: float(a[y]-a[rival])
    branch_summary = [dict(branch=n, prior_margin=margin(p), support_margin=margin(t.sum(0)),
                           final_logit_margin=margin(s)) for n, p, t, s in branches]
    delta_terms = (terms_expert[:, y]-terms_expert[:, rival])-(terms_single[:, y]-terms_single[:, rival])
    order = np.argsort(-np.abs(delta_terms), kind='stable')[:4]
    write_json(dst/'CASE.json', dict(**case, rival=rival, class_names=list(CLASSES),
        trained_single_scores=single_score.tolist(), trained_single_probability=single_prob.tolist(),
        dual_base_scores=base_score.tolist(), dual_expert_scores=expert_score.tolist(),
        dual_probability=dual_prob.tolist(), branch_margins=branch_summary,
        all64_terms_reassembly_max_error=reassembly_error,
        displayed_support_positions=order.tolist(),
        caveats=['Support terms explain each head, not physically exclusive cues',
                 'Different logit scales; term differences are not a causal allocation of the final gap',
                 'Softmax-normalized scores are not demonstrated calibrated probabilities',
                 'Posthoc fixed-rule illustrations, not a prevalence estimate']))
    manifest = load_manifest(dataset)
    pixels = np.load(MANIFESTS[dataset].parent/'inputs128.npy', mmap_mode='r') if dataset == 'f0921' else None
    fig = plt.figure(figsize=(15, 9), layout='constrained')
    grid = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1.4])
    for column, local in enumerate([q]+[int(support[i]) for i in order]):
        pair = image_pair(dataset, manifest, pixels, data['rows'][local])
        for pol in range(2):
            ax = fig.add_subplot(grid[pol, column])
            ax.imshow(pair[pol], cmap='gray', vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if column == 0:
                ax.set_ylabel(('VV', 'VH')[pol])
            if pol == 0:
                title = f'Query true={y}\nSingle={case["single_prediction"]}, dual={case["dual_prediction"]}' if column == 0 else f'Support {order[column-1]}\nClass={labels[order[column-1]]}'
                ax.set_title(title, fontsize=10)
    ax = fig.add_subplot(grid[2, :2])
    x = np.arange(3)
    for offset, field, color in [(-.23, 'prior_margin', '#287ba5'), (0, 'support_margin', '#15936b'), (.23, 'final_logit_margin', '#9564ad')]:
        ax.bar(x+offset, [r[field] for r in branch_summary], .23, label=field.replace('_', ' '), color=color)
    ax.set_xticks(x, [r['branch'] for r in branch_summary])
    ax.axhline(0, color='black', linewidth=.6)
    ax.set_title(f'True class {y} versus fixed rival {rival}\nEach head: prior + support = final logits', fontsize=10)
    ax.legend(fontsize=8)
    ax = fig.add_subplot(grid[2, 2:4])
    ax.barh(['Trained single', 'Dual fusion'], [margin(single_prob), margin(dual_prob)], color=['#287ba5', '#15936b'])
    ax.axvline(0, color='black', linewidth=.6)
    ax.set_title('Final normalized-score margin\nNot calibrated probabilities', fontsize=10)
    ax = fig.add_subplot(grid[2, 4])
    ax.axis('off')
    ax.text(0, .98, f'Event: {case["event"]}\nTruth: {CLASSES[y]}\n\nDraw 0, seed {case["seeds"][0]}\nEvent stable in {len(case["seeds"])} seeds.\n\nAll 64 signed terms saved.\nFixed posthoc selection.\nNo physical specialization\nclaim.', va='top', fontsize=9, wrap=True)
    fig.suptitle(f'{dataset} | {name} | {case["event"]}', fontsize=14)
    fig.savefig(dst/'card.png', dpi=130)
    plt.close(fig)
    print('CASE', dataset, name, case['event'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    destination = args.destination or args.out/'cases'
    choices = select(args.out, args.seeds)
    write_json(destination/'SELECTION.json', dict(rule='Draw0; sorted task, then lexicographic ID; all supplied seeds share event',
        cases=choices, posthoc=True, no_representativeness_claim=True))
    for case in choices:
        build(case, destination)
        gc.collect()
    write_json(destination/'COMPLETE.json', dict(cases=len(choices), seeds=args.seeds,
        all_signed_terms_saved=True, prediction_replay=True, physical_causality=False))


if __name__ == '__main__':
    main()
