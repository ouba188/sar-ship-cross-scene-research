"""Post-freeze outcome, confidence, and exact mechanism summaries."""

import json
import numpy as np
import pandas as pd

from backbone_ablation_core import folders, info_for, write, SOURCE, ROOT
from dual_evidence_core import OUT, ARMS, HEADS, MODES
from prepare_learned_frequency import DATASETS

CONTRASTS = [('total', 'DD', 'RR'), ('prior_only', 'DR', 'RR'),
             ('kernel_only', 'RD', 'RR'), ('kernel_given_dual_prior', 'DD', 'DR'),
             ('prior_given_dual_kernel', 'DD', 'RD')]


def truth_for(info, rows, ids):
    manifest, idcol = (SOURCE/'manifest.csv', 'instance_id') if info['dataset'] == 'self24ports' \
        else (ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', 'sample_id')
    frame = pd.read_csv(manifest, usecols=[idcol, 'class_id'], keep_default_na=False).iloc[rows]
    np.testing.assert_array_equal(frame[idcol].to_numpy(str), ids)
    return frame.class_id.to_numpy(int)


def ba(y, p):
    return 100*float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


def report():
    task_folders = folders()
    if len(task_folders) != 25 or not all((OUT/'targets'/info_for(f)[1]/'COMPLETE.json').exists()
                                         for f in task_folders):
        raise RuntimeError('All25 prediction tasks must finish before query evaluation')
    write(OUT/'ALL_PREDICTIONS_SAVED.json', dict(tasks=25, draws=125, modes=MODES))
    metrics, effects, classes, strata, decomposition, receipts = [], [], [], [], [], []
    for folder in task_folders:
        info, suffix = info_for(folder)
        dst = OUT/'targets'/suffix
        saved = np.load(dst/'predictions.npz')
        predictions, masks = saved['predictions'], saved['eval_masks']
        truth = truth_for(info, saved['rows'], saved['ids'])
        scores = np.load(dst/'scores.npy', mmap_mode='r')
        parts = np.load(dst/'expert_delta_parts.npy', mmap_mode='r')
        receipts.append(json.loads((dst/'COMPLETE.json').read_text()))
        common = dict(dataset=info['dataset'], task=info['task'])
        for draw, mask in enumerate(masks):
            yy = truth[mask]
            pred = {mode: predictions[draw, mi, mask] for mi, mode in enumerate(MODES)}
            for mode, pp in pred.items():
                metrics.append(dict(**common, draw=draw, mode=mode, n=len(yy),
                    oa=100*np.mean(pp == yy), ba=ba(yy, pp)))
            for head in HEADS:
                for contrast, candidate, control in CONTRASTS:
                    a, b = pred[candidate+'_'+head], pred[control+'_'+head]
                    rescue, harm = (a == yy)&(b != yy), (a != yy)&(b == yy)
                    effects.append(dict(**common, draw=draw, head=head, contrast=contrast,
                        delta_oa=100*np.mean((a == yy).astype(float)-(b == yy)),
                        delta_ba=ba(yy, a)-ba(yy, b), rescued=int(rescue.sum()), harmed=int(harm.sum()),
                        changed=int(np.sum(a != b)), n=len(yy)))
                    for c in np.unique(yy):
                        ix = yy == c
                        classes.append(dict(**common, draw=draw, head=head, contrast=contrast,
                            class_id=int(c), n=int(ix.sum()), rescued=int(rescue[ix].sum()),
                            harmed=int(harm[ix].sum()),
                            delta_recall=100*np.mean((a[ix] == c).astype(float)-(b[ix] == c)),
                            ba_contribution=100*np.mean((a[ix] == c).astype(float)-(b[ix] == c))/len(np.unique(yy))))
            rr, dd = pred['RR_late'], pred['DD_late']
            pp = scores[draw, MODES.index('RR_late'), mask]
            top = np.sort(pp, axis=1)[:, -2:]
            confidence = top[:, 1]-top[:, 0]
            events = {'rescue': (rr != yy)&(dd == yy), 'harm': (rr == yy)&(dd != yy),
                      'both_correct': (rr == yy)&(dd == yy), 'both_wrong': (rr != yy)&(dd != yy)}
            for name, ix in [('uncertain_gap_lt_005', confidence < .05),
                              ('moderate_gap_005_020', (confidence >= .05)&(confidence < .2)),
                              ('strong_gap_ge_020', confidence >= .2)]:
                n = int(ix.sum())
                old_correct, old_wrong = int(np.sum(ix&(rr == yy))), int(np.sum(ix&(rr != yy)))
                strata.append(dict(**common, draw=draw, stratum=name, n=n,
                    r50_correct=old_correct, r50_wrong=old_wrong,
                    rescued=int(np.sum(ix&events['rescue'])), harmed=int(np.sum(ix&events['harm'])),
                    r50_oa=100*np.mean(rr[ix] == yy[ix]) if n else np.nan,
                    dual_oa=100*np.mean(dd[ix] == yy[ix]) if n else np.nan))
            # Always true-class versus a fixed RR expert competitor, including harmed cases.
            expert_r = scores[draw, MODES.index('RR_expert'), mask]
            competitor = expert_r.copy()
            competitor[np.arange(len(yy)), yy] = -np.inf
            rival = competitor.argmax(1)
            components = np.asarray(parts[draw, :, mask, :])
            # Advanced indexing places the selected-query axis first.
            if components.shape[0] == len(yy):
                components = components.transpose(1, 0, 2)
            margins = components[:, np.arange(len(yy)), yy] - components[:, np.arange(len(yy)), rival]
            original_margin = expert_r[np.arange(len(yy)), yy]-expert_r[np.arange(len(yy)), rival]
            ed = scores[draw, MODES.index('DD_expert'), mask]
            expected = ed[np.arange(len(yy)), yy]-ed[np.arange(len(yy)), rival]-original_margin
            np.testing.assert_allclose(margins.sum(0), expected, atol=1e-10)
            for event, ix in events.items():
                for j, component in enumerate(('prior_delta', 'r50_relation_reweight', 'ssl_kernel_direct')):
                    decomposition.append(dict(**common, draw=draw, event=event, component=component,
                        n=int(ix.sum()), margin_sum=float(margins[j, ix].sum()),
                        positive=int(np.sum(margins[j, ix] > 0)), negative=int(np.sum(margins[j, ix] < 0))))
        saved.close()
        del scores, parts
    frames = dict(metrics=pd.DataFrame(metrics), effects=pd.DataFrame(effects),
                  class_effects=pd.DataFrame(classes), confidence_strata=pd.DataFrame(strata),
                  margin_decomposition=pd.DataFrame(decomposition))
    for name, frame in frames.items():
        frame.to_csv(OUT/(name+'.csv'), index=False)
    summary = frames['metrics'].groupby(['dataset', 'mode'])[['oa', 'ba']].mean().reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    effect_summary = frames['effects'].groupby(['dataset', 'head', 'contrast']).agg(
        delta_oa=('delta_oa', 'mean'), delta_ba=('delta_ba', 'mean'),
        rescued=('rescued', 'sum'), harmed=('harmed', 'sum')).reset_index()
    effect_summary.to_csv(OUT/'effect_summary.csv', index=False)
    cc = frames['class_effects'].groupby(['dataset', 'head', 'contrast', 'class_id']).agg(
        ba_contribution=('ba_contribution', 'sum'), rescued=('rescued', 'sum'),
        harmed=('harmed', 'sum'), n=('n', 'sum')).reset_index()
    task_n = frames['metrics'].groupby('dataset').task.nunique()
    cc['ba_contribution'] /= cc.dataset.map(task_n)*5
    cc.to_csv(OUT/'class_contributions.csv', index=False)
    for row in effect_summary.itertuples():
        selected = cc[(cc.dataset == row.dataset)&(cc['head'] == row.head)&(cc.contrast == row.contrast)]
        np.testing.assert_allclose(selected.ba_contribution.sum(), row.delta_ba, atol=1e-10)
    st = frames['confidence_strata'].groupby(['dataset', 'stratum'])[
        ['n', 'r50_correct', 'r50_wrong', 'rescued', 'harmed']].sum().reset_index()
    st['rescue_pct_of_wrong'] = 100*st.rescued/st.r50_wrong.replace(0, np.nan)
    st['harm_pct_of_correct'] = 100*st.harmed/st.r50_correct.replace(0, np.nan)
    st['net_pp_in_stratum'] = 100*(st.rescued-st.harmed)/st.n.replace(0, np.nan)
    st.to_csv(OUT/'confidence_summary.csv', index=False)
    dc = frames['margin_decomposition'].groupby(['dataset', 'event', 'component'])[
        ['n', 'margin_sum', 'positive', 'negative']].sum().reset_index()
    dc['mean_true_vs_rr_rival_logit_delta'] = dc.margin_sum/dc.n.replace(0, np.nan)
    dc.to_csv(OUT/'margin_summary.csv', index=False)
    ef = frames['effects']
    ef.groupby(['dataset', 'task', 'head', 'contrast'])[['delta_oa', 'delta_ba']].mean().reset_index().to_csv(
        OUT/'port_effects.csv', index=False)
    write(OUT/'COMPLETE.json', dict(tasks=25, draws=125, metric_rows=len(metrics),
        modes=MODES, predictions_exact=True, no_training=True, no_target_tuning=True,
        maximum_expert_decomposition_error=max(r['maximum_expert_score_decomposition_error'] for r in receipts),
        maximum_probability_delta_error=max(r['maximum_final_probability_delta_error'] for r in receipts),
        maximum_diagonal_coefficient_error=max(r['maximum_diagonal_coefficient_error'] for r in receipts),
        total_prediction_seconds=sum(r['seconds'] for r in receipts),
        five_support_draws_not_independent_replications=True))
    print(effect_summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    report()
