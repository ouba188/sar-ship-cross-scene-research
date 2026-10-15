"""Nine fixed posthoc cases with every support term and native VV/VH images."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'

import json
import gc
import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from backbone_ablation_core import folders, info_for, write
from dual_evidence_core import OUT, MODES, EvidenceContext, subset_archive
from dual_evidence_report import truth_for
from c4_condition_descriptors import load_manifest, MANIFESTS, read_pair
from cross_dataset_c4 import CLASSES

EVENTS = ('dual_rescue', 'dual_harm', 'both_wrong')


def select_cases():
    choices = []
    done = set()
    for folder in folders():
        info, suffix = info_for(folder)
        dataset = info['dataset']
        if all((dataset, event) in done for event in EVENTS):
            continue
        p = subset_archive(OUT/'targets'/suffix/'predictions.npz',
                           ('rows', 'ids', 'predictions', 'eval_masks'))
        y = truth_for(info, p['rows'], p['ids'])
        rr = p['predictions'][0, MODES.index('RR_late')]
        dd = p['predictions'][0, MODES.index('DD_late')]
        for event, mask in [('dual_rescue', (rr != y)&(dd == y)),
                            ('dual_harm', (rr == y)&(dd != y)),
                            ('both_wrong', (rr != y)&(dd != y))]:
            if (dataset, event) in done:
                continue
            indices = np.flatnonzero(mask&p['eval_masks'][0])
            if len(indices):
                q = int(indices[np.argsort(p['ids'][indices], kind='stable')[0]])
                choices.append(dict(dataset=dataset, task=info['task'], folder=str(folder),
                    suffix=suffix, event=event, draw=0, q=q, truth=int(y[q]),
                    rr_prediction=int(rr[q]), dd_prediction=int(dd[q]),
                    sample_id=str(p['ids'][q]), candidates_in_selected_task=int(len(indices))))
                done.add((dataset, event))
    if len(choices) != 9:
        raise ValueError(f'Expected9 fixed cases, found {len(choices)}')
    write(OUT/'CASE_SELECTION.json', dict(rule='Draw0, sorted task order; first lexicographic sample ID satisfying each event; labels used only posthoc',
        events=EVENTS, cases=choices, cherry_picked_best_effect=False,
        representative_prevalence_claim=False))
    return choices


def image_pair(dataset, manifest, pixels, row):
    item = manifest.iloc[int(row)]
    if dataset == 'f0921':
        return np.asarray(pixels[int(row)])
    columns = ['image_relpath_vv', 'image_relpath_vh'] if dataset == 'self24ports' else ['chip_path_vv', 'chip_path_vh']
    return read_pair([str(item[c]) for c in columns])[0]


def build_task_cases(folder, choices):
    from pathlib import Path
    folder = Path(folder)
    info, suffix = info_for(folder)
    data = subset_archive(folder/'target.npz', ('rows', 'ids', 'source_rows', 'support', 'support_labels'))
    context = EvidenceContext(folder, data)
    saved = subset_archive(OUT/'targets'/suffix/'coefficients.npz', ('0_RR_expert', '0_DD_expert'))
    support, labels = data['support'][0], data['support_labels'][0]
    reference = context.features(support)
    scores = np.load(OUT/'targets'/suffix/'scores.npy', mmap_mode='r')
    ar, ad = saved['0_RR_expert'], saved['0_DD_expert']
    dataset = info['dataset']
    manifest = load_manifest(dataset)
    pixels = np.load(MANIFESTS[dataset].parent/'inputs128.npy', mmap_mode='r') if dataset == 'f0921' else None
    for case in choices:
        q = case['q']
        query = context.features(np.array([q]))
        kernels = context.cross(query, reference)
        kr, kd = kernels['R'][0], kernels['D'][0]
        pr, pd_ = query['priors']['r50'][0], query['priors']['dual'][0]
        er, ed = pr+kr@ar, pd_+kd@ad
        np.testing.assert_allclose(er, scores[0, MODES.index('RR_expert'), q], atol=3e-5, rtol=1e-6)
        np.testing.assert_allclose(ed, scores[0, MODES.index('DD_expert'), q], atol=3e-5, rtol=1e-6)
        term_r = kr[:, None]*ar
        term_reweight = kr[:, None]*(ad-ar)
        term_ssl = (kd-kr)[:, None]*ad
        delta_prior = pd_-pr
        np.testing.assert_allclose(delta_prior+term_reweight.sum(0)+term_ssl.sum(0), ed-er, atol=1e-10)
        y = case['truth']
        rival_values = scores[0, MODES.index('RR_expert'), q].copy()
        rival_values[y] = -np.inf
        rival = int(rival_values.argmax())
        total_support_delta = term_reweight+term_ssl
        margin_terms = total_support_delta[:, y]-total_support_delta[:, rival]
        order = np.argsort(-np.abs(margin_terms), kind='stable')[:4]
        base = scores[0, MODES.index('base'), q]
        rlate, dlate = .5*(softmax(base)+softmax(er)), .5*(softmax(base)+softmax(ed))
        np.testing.assert_allclose(rlate, scores[0, MODES.index('RR_late'), q], atol=3e-6)
        np.testing.assert_allclose(dlate, scores[0, MODES.index('DD_late'), q], atol=3e-6)
        # Explicitly order nonlinear steps. This is not an order-invariant attribution.
        states = [er, er+delta_prior, er+delta_prior+term_reweight.sum(0), ed]
        prob_steps = [.5*(softmax(states[i+1])-softmax(states[i])) for i in range(3)]
        np.testing.assert_allclose(np.sum(prob_steps, axis=0), dlate-rlate, atol=1e-14)
        dst = OUT/'cases'/dataset/case['event']
        dst.mkdir(parents=True, exist_ok=True)
        records = []
        for j, si in enumerate(support):
            for c in range(8):
                records.append(dict(support_position=j, support_local_row=int(si),
                    support_manifest_row=int(data['rows'][si]), support_id=str(data['ids'][si]),
                    support_label=int(labels[j]), output_class=c,
                    kr50=float(kr[j]), kdual=float(kd[j]), kssl_delta=float(kd[j]-kr[j]),
                    alpha_r50=float(ar[j, c]), alpha_dual=float(ad[j, c]),
                    old_r50_support_term=float(term_r[j, c]),
                    old_relation_reweight_term=float(term_reweight[j, c]),
                    added_ssl_relation_term=float(term_ssl[j, c]),
                    total_support_score_delta=float(total_support_delta[j, c])))
        pd.DataFrame(records).to_csv(dst/'all64_signed_terms.csv', index=False)
        details = dict(**case, manifest_row=int(data['rows'][q]), truth_name=CLASSES[y],
            margin_rival=rival, margin_rival_name=CLASSES[rival],
            source_prior_r50=pr.tolist(), source_prior_dual=pd_.tolist(),
            expert_score_r50=er.tolist(), expert_score_dual=ed.tolist(),
            base_logits=base.tolist(), fusion_probability_r50=rlate.tolist(),
            fusion_probability_dual=dlate.tolist(),
            expert_score_delta_parts=dict(prior=delta_prior.tolist(),
                r50_relation_reweight=term_reweight.sum(0).tolist(), ssl_direct=term_ssl.sum(0).tolist()),
            ordered_fusion_probability_delta=dict(order=['prior', 'r50_relation_reweight', 'ssl_direct'],
                values=[x.tolist() for x in prob_steps], order_dependent=True),
            displayed_support_positions=order.tolist(), all_support_terms=64,
            physical_semantics_claim=False)
        write(dst/'CASE.json', details)
        image_rows = [q]+[int(support[i]) for i in order]
        fig, axes = plt.subplots(3, 5, figsize=(15, 9), gridspec_kw={'height_ratios': [1, 1, 1.35]})
        for column, local in enumerate(image_rows):
            pair = image_pair(dataset, manifest, pixels, data['rows'][local])
            for pol in range(2):
                axes[pol, column].imshow(pair[pol], cmap='gray', vmin=0, vmax=255)
                axes[pol, column].set_xticks([])
                axes[pol, column].set_yticks([])
                if column == 0:
                    axes[pol, column].set_ylabel(('VV', 'VH')[pol])
            if column == 0:
                title = f'Query: true={y}\nR50={case["rr_prediction"]}, dual={case["dd_prediction"]}'
            else:
                position = order[column-1]
                title = f'Support {position}: label={labels[position]}\nDelta margin={margin_terms[position]:+.4f}'
            axes[0, column].set_title(title, fontsize=10)
        axes[2, 0].barh(['R50 expert', 'Dual expert'], [er[y]-er[rival], ed[y]-ed[rival]], color=['#777777', '#196d89'])
        axes[2, 0].set_title(f'True({y}) vs fixed rival({rival})\nExpert LOGIT margin', fontsize=10)
        values = [delta_prior[y]-delta_prior[rival],
                  term_reweight[:, y].sum()-term_reweight[:, rival].sum(),
                  term_ssl[:, y].sum()-term_ssl[:, rival].sum()]
        axes[2, 1].barh(['Prior', 'Reweight', 'Added SSL'], values,
                       color=['#34865d' if v >= 0 else '#b74648' for v in values])
        axes[2, 1].set_title('Exact expert margin change', fontsize=10)
        axes[2, 2].barh(['R50 fusion', 'Dual fusion'], [rlate[y]-rlate[rival], dlate[y]-dlate[rival]], color=['#777777', '#196d89'])
        axes[2, 2].set_title('Final PROBABILITY margin', fontsize=10)
        for ax in axes[2, :3]:
            ax.axvline(0, color='black', linewidth=.6)
            ax.tick_params(labelsize=8)
        axes[2, 3].axis('off')
        axes[2, 4].axis('off')
        axes[2, 3].text(0, .95, 'All 64 signed support terms\nare available in the CSV.\n\nImages are real query/support\nVV/VH pixels, fixed 0..255.\n\nTerms describe computation,\nnot separate physical mechanisms.', va='top', fontsize=10)
        axes[2, 4].text(0, .95, f'Case: {case["event"]}\nTask: {case["task"]}\nDraw: 0\n\nPosthoc deterministic selection.\nNot a prevalence estimate.\n\nClass {y}:\n{CLASSES[y]}', va='top', fontsize=9)
        fig.suptitle(f'{dataset} | {case["event"]} | {case["sample_id"]}', fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .95), w_pad=2.5)
        fig.savefig(dst/'card.png', dpi=130)
        plt.close(fig)
        print('CASE', dataset, case['event'], flush=True)


def main():
    choices = select_cases()
    with threadpool_limits(limits=2):
        for folder in dict.fromkeys(c['folder'] for c in choices):
            build_task_cases(folder, [c for c in choices if c['folder'] == folder])
            gc.collect()
    write(OUT/'CASES_COMPLETE.json', dict(cases=9, draw=0, signed_supports_per_case=64,
        output_classes_per_support=8, exact_logit_decomposition=True,
        nonlinear_softmax_steps_explicitly_ordered=True, visual_cards=9))


if __name__ == '__main__':
    main()
