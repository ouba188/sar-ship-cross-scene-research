"""Source-only condition x evidence discovery, frozen features and no model promotion."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from incidence_transport import matched_pairs, pair_weights
from condition_evidence_response import ARMS, fit_model, predict_model, response_target, evaluate

BASE = Path(__file__).resolve().parent
OUT = BASE/'condition_evidence_20261011'
PARENT = BASE/'incidence_transport_20261011'
CROSSFIT = BASE/'crossfit_evidence_20261011'
SEEDS = (31415, 27182)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def score_bank(meta, fit, held):
    bank = {}
    for seed in SEEDS:
        score = np.full((len(meta), 8), np.nan)
        with np.load(PARENT/f'full_{seed}_inputs.npz') as f:
            score[held] = f['prior'][held]
        assigned = []
        for fold in range(3):
            with np.load(PARENT/f'fold{fold}_{seed}_inputs.npz') as f:
                ids = f['excluded_obs_ids']
                score[ids] = f['prior'][ids]
                assigned.extend(ids.tolist())
        if len(assigned) != len(fit) or set(assigned) != set(fit):
            raise ValueError('OOF scores must cover the fitting cohort exactly once')
        bank[seed] = score
    return bank


def make_tasks(meta, fit, held, features, names, scores):
    tasks, rows = [], []
    # Three confusion pairs are defined by source OOF errors, never held labels.
    count = np.zeros((8, 8), int)
    yfit = meta.iloc[fit].class_id.to_numpy()
    for score in scores.values():
        pred = score[fit].argmax(1)
        for y, p in zip(yfit, pred):
            if y != p:
                count[min(y, p), max(y, p)] += 1
    pairs = sorted([(count[a, b], a, b) for a in range(8) for b in range(a+1, 8)],
                   key=lambda x: (-x[0], x[1], x[2]))[:3]
    for c, a, b in pairs:
        rows.append(dict(class_a=a, class_b=b, source_oof_confusions=int(c)))
    pd.DataFrame(rows).to_csv(OUT/'source_confusion_pairs.csv', index=False)
    total = np.full(len(meta), np.nan)
    available = np.isfinite(features[:, :3]).all(axis=1)
    total[available] = np.logaddexp.reduce(features[available, :3], axis=1)
    for j, name in enumerate(names):
        for mode in ('signed', 'magnitude'):
            tasks.append(dict(name=str(name), family='spatial', seed=0, mode=mode,
                              anchor=features[:, j], context=total, values=features[:, j]))
    ids = np.r_[fit, held]
    for seed, score in scores.items():
        order = np.sort(score, axis=1)
        gap, top = order[:, -1]-order[:, -2], order[:, -1]
        for _, a, b in pairs:
            values = score[:, a]-score[:, b]
            for mode in ('signed', 'magnitude'):
                tasks.append(dict(name=f'class_{a}_vs_{b}', family='decision', seed=seed,
                                  mode=mode, anchor=values, context=gap, values=values))
        margin = np.full(len(meta), np.nan)
        error = margin.copy()
        label = meta.iloc[ids].class_id.to_numpy()
        rival = score[ids].copy()
        rival[np.arange(len(ids)), label] = -np.inf
        margin[ids] = score[ids, label]-rival.max(1)
        error[ids] = (score[ids].argmax(1) != label).astype(float)
        # True labels define answers only, not anchor/context inputs.
        tasks.append(dict(name='true_margin_drop', family='risk', seed=seed,
                          mode='deterioration', anchor=gap, context=top, values=margin))
        tasks.append(dict(name='next_view_error', family='risk', seed=seed,
                          mode='error', anchor=gap, context=top, values=error))
    return tasks


def identify(task):
    return {k: task[k] for k in ('name', 'family', 'seed', 'mode')}


def run_task(meta, task, fitting, pairs, evaluation_pairs, tag, export=False):
    results, archives = [], {}
    angles = meta.incidence_deg.to_numpy(float)
    directed = np.r_[evaluation_pairs, evaluation_pairs[:, ::-1]]
    a, b = directed.T
    truth = response_target(task['values'], directed, task['mode'])
    for arm in ARMS:
        model = fit_model(meta, fitting, pairs, task['anchor'], task['context'],
                          task['values'], task['mode'], arm)
        prediction = predict_model(task['anchor'][a], task['context'][a], angles[a], angles[b], model)
        results.append(dict(**identify(task), arm=arm, split=tag,
                            **evaluate(meta, evaluation_pairs, truth, prediction, model['constant'])))
        archives[arm] = prediction
        if export:
            key = f"{task['seed']}_{task['name']}_{task['mode']}_{arm}"
            np.savez_compressed(OUT/'models'/f'{key}.npz', **model)
    if export:
        key = f"{task['seed']}_{task['name']}_{task['mode']}"
        np.savez_compressed(OUT/'predictions'/f'{key}.npz', directed=directed,
                            truth=truth, **archives)
    return results, archives, truth


def comparisons(frame, columns):
    table = frame.pivot(index=columns, columns='arm', values='mse').reset_index()
    for arm in ('anchor', 'capacity', 'sham'):
        table['gain_vs_'+arm] = 100*(1-table['condition']/table[arm].clip(lower=1e-12))
    return table


def main():
    start = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    if (OUT/'PROTOCOL.json').exists():
        raise ValueError('Do not overwrite an existing analysis')
    (OUT/'models').mkdir(exist_ok=True)
    (OUT/'predictions').mkdir(exist_ok=True)
    meta = pd.read_csv(BASE/'observations.csv')
    if not np.array_equal(meta.obs_id.to_numpy(), np.arange(len(meta))):
        raise ValueError('Metadata global row alignment changed')
    with np.load(PARENT/'source_cohorts.npz') as f:
        fit, held, train_pairs, held_pairs = [f[k] for k in
            ('fit_obs_ids', 'held_obs_ids', 'fit_pairs', 'held_pairs')]
    if set(meta.iloc[fit].mmsi) & set(meta.iloc[held].mmsi):
        raise ValueError('Held vessels overlap response training')
    with np.load(OUT/'evidence_features.npz') as f:
        names = f['feature_names']
        features = np.full((len(meta), len(names)), np.nan)
        features[f['obs_ids']] = f['features']
    if not np.isfinite(features[np.r_[fit, held]]).all():
        raise ValueError('Evidence cache does not cover all source observations')
    with np.load(CROSSFIT/'source_folds.npz') as f:
        np.testing.assert_array_equal(f['fit_obs_ids'], fit)
        assignment = f['assignment']
    scores = score_bank(meta, fit, held)
    tasks = make_tasks(meta, fit, held, features, names, scores)
    save_json(OUT/'PROTOCOL.json', dict(
        objective='Find partial incidence-dependent evidence changes or reliability changes in unseen source vessels; no factor separation or recognition deployment yet.',
        fixed='Original source-fit1296,held823; matchedpairs384/189. Known incidence/platform/pass; same ship,port,platform,pass,different product. Equal class/ship/pair weights; two directions are not independent samples.',
        evidence='Six fixed descriptors of cached CNN spatial maps, not radar frequencies or actual ship segmentation. Three confusion score differences from sourceOOF only; actual true-margin decrease and B-view error as distinct risk targets.',
        inputs='Only A evidence and A context plus A/B incidence. B image evidence is never a regressor input; true labels only define risk outcomes and weights.',
        estimators='Anchor2+intercept; capacity adds4 anchor-only nonlinear terms; condition adds4 condition terms; sham identical to condition with withinship/port/platform/pass training-angle permutation. Equal parameter count is not equal hypothesis space. Fixed ridge.05 after training-only weighted standardization; clip magnitude/drop>=0,error in[0,1]. No sweep.',
        condition_terms='signed:d,d*mid,d*u,d*v; magnitude/drop:abs(d),d*d,mid,abs(d)*u; error:a,b,b*b,d*v. a/b=(incidence-38)/5; u/v are source-pair-normalized anchor/context.',
        selection='Three source-vessel response folds; freeze one candidate per family/mode by worst gain across anchor/capacity/sham averaged across folds/seeds if all three mean gains>0. Report fold consistency. SourceOOF confusion-pair selection precedes response CV, so screening is developmental, not nested independent assessment.',
        upstream='Spatial CNN cache is historically trained. Decision scores are ship-OOF on sourcefit and full-source on held; source response CV is not nested upstream refitting. Fixed semantic score axes, but OOF/full distribution mismatch remains. No full-pipeline unseen-vessel/scene claim.',
        held='All candidates reported transparently but only source-selected candidates considered for follow-up. Original held development data reused, not independent confirmation. Report near/far and acquisition coverage. No target ports or other datasets.',
        boundary='Magnitude means evidence volatility, NOT classification unreliability. Error and true-margin-drop are separate tests. No causal identification from predictive association alone.',
        tasks=len(tasks), neural_fits=0, main_model_modified=False))
    coverage = []
    scene_fit = set(meta.iloc[fit]['product'])
    angle = meta.incidence_deg.to_numpy(float)
    for role, ids, pairs in [('fit', fit, train_pairs), ('held', held, held_pairs)]:
        a, b = pairs.T
        for subset, mask in [('all', np.ones(len(pairs), bool)),
                             ('gap2', np.abs(angle[b]-angle[a]) >= 2),
                             ('both_new_scene', (~meta.iloc[a]['product'].isin(scene_fit).to_numpy()) &
                              (~meta.iloc[b]['product'].isin(scene_fit).to_numpy()))]:
            p = pairs[mask]
            coverage.append(dict(role=role, subset=subset, pairs=len(p),
                                 ships=meta.iloc[p[:, 0]].mmsi.nunique(),
                                 classes=meta.iloc[p[:, 0]].class_id.nunique()))
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv', index=False)
    cv = []
    for fold in range(3):
        fitting, excluded = fit[assignment != fold], fit[assignment == fold]
        train, val = matched_pairs(meta, fitting), matched_pairs(meta, excluded)
        for task in tasks:
            rows, _, _ = run_task(meta, task, fitting, train, val, f'fold{fold}')
            cv.extend(rows)
        print('SOURCE_RESPONSE_FOLD', fold, 'pairs', len(train), len(val), flush=True)
    cv = pd.DataFrame(cv)
    cv.to_csv(OUT/'source_cv_results.csv', index=False)
    key = ['name', 'family', 'seed', 'mode', 'split']
    cmp = comparisons(cv, key)
    cmp.to_csv(OUT/'source_cv_comparisons.csv', index=False)
    gains = ['gain_vs_anchor', 'gain_vs_capacity', 'gain_vs_sham']
    summary = cmp.groupby(['name', 'family', 'mode'])[gains].mean().reset_index()
    summary['screen_score'] = summary[gains].min(axis=1)
    summary['selected'] = False
    for _, group in summary.groupby(['family', 'mode']):
        eligible = group[group.screen_score > 0]
        if len(eligible):
            summary.loc[eligible.screen_score.idxmax(), 'selected'] = True
    summary.to_csv(OUT/'source_selection.csv', index=False)
    save_json(OUT/'SELECTION_FROZEN.json', dict(held_metrics_read=False,
        selected=summary[summary.selected][['name', 'family', 'mode', 'screen_score']].to_dict('records')))
    print('SOURCE_SELECTION', summary[summary.selected].to_string(index=False), flush=True)
    held_rows, subrows, relations = [], [], []
    a, b = held_pairs.T
    masks = dict(all=np.ones(len(held_pairs), bool), near=np.abs(angle[b]-angle[a]) < 2,
                 gap2=np.abs(angle[b]-angle[a]) >= 2,
                 both_new_scene=(~meta.iloc[a]['product'].isin(scene_fit).to_numpy()) &
                                (~meta.iloc[b]['product'].isin(scene_fit).to_numpy()))
    for task in tasks:
        rows, predictions, truth = run_task(meta, task, fit, train_pairs, held_pairs, 'held', True)
        held_rows.extend(rows)
        for subset, mask in masks.items():
            directed_mask = np.r_[mask, mask]
            for arm in ARMS:
                constant = np.load(OUT/'models'/f"{task['seed']}_{task['name']}_{task['mode']}_{arm}.npz")['constant']
                subrows.append(dict(**identify(task), subset=subset, arm=arm,
                    **evaluate(meta, held_pairs[mask], truth[directed_mask], predictions[arm][directed_mask], constant)))
        # Descriptive condition-response means are saved for both cohorts, not used for selection.
        for role, pairs in [('fit', train_pairs), ('held', held_pairs)]:
            directed = np.r_[pairs, pairs[:, ::-1]]
            u, v = directed.T
            outcome = response_target(task['values'], directed, task['mode'])
            for label, low, high in [('near', 0., 2.), ('moderate', 2., 5.), ('large', 5., np.inf)]:
                gap = np.abs(angle[pairs[:, 1]]-angle[pairs[:, 0]])
                mask = (gap >= low) & (gap < high)
                if not mask.any():
                    continue
                w = np.tile(pair_weights(meta, pairs[mask])/2, 2)
                yy = outcome[np.r_[mask, mask]]
                # Slope respects angle sign, unlike a mean signed difference over reversals.
                da = angle[v]-angle[u]
                dd = da[np.r_[mask, mask]]
                relations.append(dict(**identify(task), role=role, gap_bin=label,
                    pairs=int(mask.sum()), ships=meta.iloc[pairs[mask, 0]].mmsi.nunique(),
                    outcome_mean=float(w@yy), outcome_rms=float(np.sqrt(w@(yy*yy))),
                    angle_slope=float(w@(dd*yy)/max(w@(dd*dd), 1e-12)) if task['mode']=='signed' else np.nan))
    held_frame = pd.DataFrame(held_rows)
    held_frame.to_csv(OUT/'held_results.csv', index=False)
    hcmp = comparisons(held_frame, key)
    hcmp.to_csv(OUT/'held_comparisons.csv', index=False)
    pd.DataFrame(subrows).to_csv(OUT/'held_subsets.csv', index=False)
    pd.DataFrame(relations).to_csv(OUT/'condition_relations.csv', index=False)
    hsummary = hcmp.groupby(['name', 'family', 'mode'])[gains].mean().reset_index()
    summary.merge(hsummary, on=['name', 'family', 'mode'], suffixes=('_cv', '_held')).to_csv(OUT/'summary.csv', index=False)
    save_json(OUT/'COMPLETE.json', dict(seconds=time.perf_counter()-start,
        response_fits=len(tasks)*4*4, tasks=len(tasks), neural_fits=0, target_evaluations=0,
        selected_candidates=int(summary.selected.sum()), main_model_modified=False))
    print('COMPLETE', hsummary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
