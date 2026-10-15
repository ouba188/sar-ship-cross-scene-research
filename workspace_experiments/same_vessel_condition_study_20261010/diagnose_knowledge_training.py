"""Post-freeze attribute readability and geometry; never select a classifier."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scene_training import sample_weights
from probe_knowledge_attributes import fit_ridge, predict_ridge, class_means, write_json
from candidate_embedding_diagnostics import embedding_diagnostics

BASE = Path(__file__).resolve().parent
ROOT = BASE/'knowledge_train_only_20261011'


def readable_attributes(zfit, zheld, attributes, fm, hm, tag):
    """Refit the SAME small linear readout for every representation, source only."""
    fit_mask, held_mask = attributes['fit_valid'], attributes['held_valid']
    x, q = zfit[fit_mask].astype(float), zheld[held_mask].astype(float)
    ft, ht = fm.loc[fit_mask].reset_index(drop=True), hm.loc[held_mask].reset_index(drop=True)
    y, truth = np.log(attributes['fit_values'][fit_mask]), np.log(attributes['held_values'][held_mask])
    labels, qlabels = ft.class_id.to_numpy(), ht.class_id.to_numpy()
    weights = sample_weights(ft); weights /= weights.sum()
    mean = weights@y
    scale = np.sqrt(weights@np.square(y-mean)).clip(1e-5)
    cm, xm = class_means(y, labels, weights), class_means(x, labels, weights)
    direct = fit_ridge(x, y, weights)
    conditional = fit_ridge(x-xm[labels], y-cm[labels], weights)
    rows, held_prediction = [], {}
    for cohort, xhere, yhere, cls, mm in [('fit', x, y, labels, ft), ('held', q, truth, qlabels, ht)]:
        predictions = dict(global_mean=np.broadcast_to(mean, yhere.shape), class_mean=cm[cls],
            sar=predict_ridge(direct, xhere),
            within_class=cm[cls]+predict_ridge(conditional, xhere-xm[cls]))
        if cohort == 'held':
            held_prediction = predictions
        for c in [-1]+list(range(8)):
            selected = np.ones(len(mm), bool) if c == -1 else cls == c
            if not selected.any():
                continue
            w = sample_weights(mm.loc[selected]); w /= w.sum()
            for d, attribute in enumerate(('length', 'beam')):
                errors = {k: float(w@np.square((v[selected, d]-yhere[selected, d])/scale[d]))
                          for k, v in predictions.items()}
                for name, mse in errors.items():
                    rows.append(dict(**tag, cohort=cohort, class_id=c, attribute=attribute, reader=name,
                        MSE=mse, skill_vs_class=100*(1-mse/errors['class_mean']),
                        ships=int(mm.loc[selected, 'mmsi'].nunique())))
    return rows, held_prediction


def auxiliary_errors(state, zfit, zheld, attributes, fm, hm, normalization, tag):
    rows = []
    for cohort, z, mm in [('fit', zfit, fm), ('held', zheld, hm)]:
        valid = attributes[cohort+'_valid']
        if not valid.any():
            continue
        with torch.no_grad():
            prediction = F.linear(torch.from_numpy(z[valid]), state['attribute_head.weight'],
                                  state['attribute_head.bias']).numpy().astype(float)
        logs = np.log(attributes[cohort+'_values'][valid])
        truth = (logs-normalization['log_mean'])/normalization['log_scale']
        selected_meta = mm.loc[valid].reset_index(drop=True)
        class_prediction = (normalization['class_log_means'][selected_meta.class_id.to_numpy()]
                            -normalization['log_mean'])/normalization['log_scale']
        w = sample_weights(selected_meta); w /= w.sum()
        for d, name in enumerate(('length', 'beam')):
            err = float(w@np.square(prediction[:, d]-truth[:, d]))
            baseline = float(w@np.square(class_prediction[:, d]-truth[:, d]))
            rows.append(dict(**tag, cohort=cohort, attribute=name, MSE=err,
                skill_vs_class=100*(1-err/baseline),
                auxiliary_was_trained=tag['arm'] in ('C', 'D')))
    return rows


def main(run_name='training'):
    from relative_content_condition import load_data
    from knowledge_train_only import ARMS, SEEDS
    torch.set_num_threads(1)
    train = ROOT/run_name
    if not (train/'COMPLETE.json').is_file():
        raise RuntimeError('Freeze all fits and finish classification before diagnostics')
    arms = json.loads((train/'PROTOCOL.json').read_text(encoding='utf-8'))['arms']
    out = ROOT/('mechanism' if run_name == 'training' else run_name+'_mechanism')
    out.mkdir(exist_ok=False)
    meta, split, roles, source, raw, norm, fixed, h = load_data()
    del norm, fixed, h
    fit, held = split['factor_fit'], split['test_source']
    fm, hm = [meta.iloc[ids].reset_index(drop=True) for ids in (fit, held)]
    with np.load(ROOT/'attributes.npz') as f:
        attributes = {k: f[k] for k in f.files}
    with np.load(train/'attribute_normalization.npz') as f:
        normalization = {k: f[k] for k in f.files}
    rows, head_rows, geometry = [], [], []
    rr, pp = readable_attributes(raw[fit], raw[held], attributes, fm, hm,
                                dict(arm='raw512', seed=0, snapshot='reference'))
    rows.extend(rr)
    np.savez_compressed(out/'raw512_attribute_predictions.npz', **pp)
    for seed in SEEDS:
        with np.load(train/f'initial_{seed}_vectors.npz') as f:
            zi = f['z']
        tag = dict(arm='initial32', seed=seed, snapshot='reference')
        rr, pp = readable_attributes(zi[fit], zi[held], attributes, fm, hm, tag)
        rows.extend(rr)
        geometry.append(dict(**tag, **embedding_diagnostics(zi[held], hm)))
        np.savez_compressed(out/f'initial32_{seed}_attribute_predictions.npz', **pp)
        for arm in arms:
            state = torch.load(train/f'{arm}_{seed}.pt', map_location='cpu', weights_only=True)
            for snapshot in ('selected', 'last'):
                tag = dict(arm=arm, seed=seed, snapshot=snapshot)
                with np.load(train/f'{arm}_{seed}_{snapshot}.npz') as f:
                    np.testing.assert_array_equal(f['obs_ids'], np.arange(len(meta)))
                    z = f['z']
                rr, pp = readable_attributes(z[fit], z[held], attributes, fm, hm, tag)
                rows.extend(rr)
                head_rows.extend(auxiliary_errors(state[snapshot], z[fit], z[held], attributes, fm, hm, normalization, tag))
                np.savez_compressed(out/f'{arm}_{seed}_{snapshot}_attribute_predictions.npz', **pp)
                if snapshot == 'selected':
                    geometry.append(dict(**tag, **embedding_diagnostics(z[held], hm)))
    pd.DataFrame(rows).to_csv(out/'readability.csv', index=False)
    pd.DataFrame(head_rows).to_csv(out/'auxiliary_head.csv', index=False)
    pd.DataFrame(geometry).to_csv(out/'geometry.csv', index=False)
    counts, events = [], []
    for path in sorted(train.glob('target_*.npz')):
        seed = int(path.stem.rsplit('_', 1)[1])
        task = path.stem[len('target_'):].rsplit('_', 1)[0]
        with np.load(path) as f:
            with np.load(ROOT/'training'/path.name) as original:
                original_scores = {k: original[k] for k in original.files}
            np.testing.assert_array_equal(f['query_obs_ids'], original_scores['query_obs_ids'])
            ids = f['query_obs_ids']
            mm = meta.iloc[ids].reset_index(drop=True)
            truth = mm.class_id.to_numpy()
            for new, old in [('C', 'B'), ('C', 'D'), ('B', 'A'), ('C', 'A')]:
                if new+'_selected' not in f:
                    continue
                ns = f[new+'_selected']
                bs = f[old+'_selected'] if old+'_selected' in f else original_scores[old+'_selected']
                npred, bpred = ns.argmax(1), bs.argmax(1)
                rescue = (npred == truth) & (bpred != truth)
                harm = (npred != truth) & (bpred == truth)
                for cls in [-1]+list(np.unique(truth)):
                    selected = np.ones(len(ids), bool) if cls == -1 else truth == cls
                    counts.append(dict(seed=seed, task=task, port=str(mm.port.iloc[0]), comparison=new+'-'+old,
                        class_id=int(cls), queries=int(selected.sum()), rescues=int((rescue & selected).sum()),
                        harms=int((harm & selected).sum()), changed=int(((npred != bpred) & selected).sum())))
                for i in np.flatnonzero(rescue | harm):
                    events.append(dict(seed=seed, task=task, comparison=new+'-'+old, obs_id=int(ids[i]),
                        mmsi=int(mm.iloc[i].mmsi), port=str(mm.iloc[i].port), class_id=int(truth[i]),
                        old_prediction=int(bpred[i]), new_prediction=int(npred[i]),
                        event='rescue' if rescue[i] else 'harm'))
    pd.DataFrame(counts).to_csv(out/'rescue_harm.csv', index=False)
    pd.DataFrame(events).to_csv(out/'changed_cases.csv', index=False)
    write_json(out/'DIAGNOSTIC_PROTOCOL.json', dict(
        selection_changed=False, classification_refits=0,
        readout='Same fixed relative ridge0.1 on clean source attributes for each frozen representation; selected and last separately.',
        oracle='Within-class reader/class_mean uses true class only for diagnosis, not classifier inference.',
        head='A/B auxiliary heads are untrained, so primary cross-arm readability uses newly fitted matched ridge, not those heads.',
        geometry='Held normalized_drift uses each representation own variance; matched same/other identity control retained.',
        counts='Rescue/harm are query exposures across repeated support draws; changed_cases.csv retains IDs for concentration analysis.',
        boundary='New-vessel readable structure is not causal identification; no pure-noise complement or guaranteed invariance.'))
    summary = pd.DataFrame(rows)
    print(summary[(summary.class_id == -1) & summary.cohort.eq('held') &
                  summary.snapshot.isin(['selected', 'reference'])].groupby(['arm', 'reader', 'attribute'])[
                      ['MSE', 'skill_vs_class']].mean().to_string())


if __name__ == '__main__':
    main()
