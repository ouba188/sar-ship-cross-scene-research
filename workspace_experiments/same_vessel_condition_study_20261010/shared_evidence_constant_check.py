"""Fixed source-mean residual control: sample-dependent evidence or class offset?"""

import numpy as np
import pandas as pd
import torch
from shared_evidence_distillation import (
    OUT, SEEDS, SUPPORTS, EvidenceStudent, centered, previous, FactorNet,
    BoundedOperator, encode_two, standardize, load_data, sample_weights,
    residual_head, metrics, save_json, vessel_targets,
)
from shared_evidence_diagnose import diagnose_scores
from decision_condition_consistency import pair_metrics


def main():
    torch.set_num_threads(1)
    out = OUT/'constant_control'
    out.mkdir(exist_ok=False)
    meta, split, roles, source, raw, norm, fixed, h = load_data()
    fit, test = split['factor_fit'], split['test_source']
    xt = torch.from_numpy(norm)
    probability = sample_weights(meta.iloc[fit].reset_index(drop=True))
    probability /= probability.sum()
    source_rows, rows, class_rows, frozen, means = [], [], [], {}, {}
    for seed in SEEDS:
        factor = FactorNet()
        factor.load_state_dict(torch.load(previous.old.OUT/f'anchor_{seed}_factor.pt', map_location='cpu', weights_only=True)['model'])
        factor.eval()
        _, condition = encode_two(factor, norm)
        context = torch.from_numpy(standardize(condition, roles['fit']).astype(np.float32))
        operator = BoundedOperator(fixed['basis'], (h[roles['fit']]@fixed['basis']).square().mean(0).sqrt(), 'residual')
        operator.load_state_dict(torch.load(previous.OUT/f'anchor_residual_{seed}_operator.pt', map_location='cpu', weights_only=True))
        with torch.no_grad():
            base = operator(h, context)@fixed['weight']+fixed['bias']
        frozen[seed] = (operator, context)
        mm = meta.iloc[test].reset_index(drop=True)
        _, teacher, valid = vessel_targets(base[test].numpy(), mm)
        for arm in ('self_teacher', 'other_acquisition_teacher'):
            model = EvidenceStudent(centered(base[fit]).square().mean(0).sqrt())
            model.load_state_dict(torch.load(OUT/f'{arm}_{seed}.pt', map_location='cpu', weights_only=True)['selected'])
            with torch.no_grad():
                delta = model(xt[fit], base[fit])-base[fit]
                mean = (torch.tensor(probability, dtype=torch.float32)[:, None]*delta).sum(0)
            means[arm, seed] = mean
            scores = (base[test]+mean).numpy()
            source_rows.append(dict(arm=arm, seed=seed, **diagnose_scores(scores, base[test].numpy(), teacher, mm, valid),
                **pair_metrics(scores, mm)))
        for arm in ('baseline', 'supervised', 'self_teacher', 'other_acquisition_teacher'):
            with np.load(OUT/f'{arm}_{seed}_source_scores.npz') as z:
                assert np.array_equal(z['obs_ids'], test)
                score = z['scores']
            y, bp, pred = mm.class_id.to_numpy(), base[test].argmax(1).numpy(), score.argmax(1)
            for c in np.unique(y):
                mask = y == c
                class_rows.append(dict(arm=arm, seed=seed, class_id=int(c), n=int(mask.sum()),
                    recall=float(100*(pred[mask] == c).mean()),
                    rescue=int((mask&(bp != y)&(pred == y)).sum()),
                    harm=int((mask&(bp == y)&(pred != y)).sum())))
    pd.DataFrame(source_rows).to_csv(out/'source_diagnostics.csv', index=False)
    pd.DataFrame(class_rows).to_csv(out/'source_classes.csv', index=False)
    torch.save({f'{a}_{s}': m for (a, s), m in means.items()}, out/'source_mean_residuals.pt')
    tasks = []
    for path in sorted(SUPPORTS.glob('*_split.npz')):
        task = path.stem.removesuffix('_split')
        tasks.append(task)
        with np.load(path) as z:
            support, query, sy = [z[k] for k in ('support_obs_ids', 'query_obs_ids', 'support_labels')]
        ids = np.r_[support, query]
        archive = dict(query_obs_ids=query)
        for seed in SEEDS:
            operator, context = frozen[seed]
            with torch.no_grad():
                changed = operator(h[ids], context[ids])
                base = changed@fixed['weight']+fixed['bias']
            phi = torch.cat([changed, torch.ones((len(ids), 1))], 1)
            for arm in ('self_teacher', 'other_acquisition_teacher'):
                prior = base+means[arm, seed]
                beta, _, _ = residual_head(phi[:64], prior[:64], torch.tensor(sy), balanced=True)
                archive[f'{arm}__{seed}'] = (prior[64:]+phi[64:]@beta).numpy()
        np.savez_compressed(out/f'{task}.npz', **archive)
    for task in tasks:
        with np.load(out/f'{task}.npz') as z:
            qm = meta.iloc[z['query_obs_ids']].reset_index(drop=True)
            y = qm.class_id.to_numpy()
            with np.load(OUT/f'{task}_predictions.npz') as old:
                for seed in SEEDS:
                    bp = old[f'baseline__{seed}'].argmax(1)
                    for arm in ('self_teacher', 'other_acquisition_teacher'):
                        score = z[f'{arm}__{seed}']
                        pred = score.argmax(1)
                        rows.append(dict(arm=arm, seed=seed, task=task, port=qm.port.iloc[0],
                            **metrics(pred, y), rescue=int(((bp != y)&(pred == y)).sum()),
                            harm=int(((bp == y)&(pred != y)).sum()), **pair_metrics(score, qm)))
    data = pd.DataFrame(rows)
    data.to_csv(out/'target_results.csv', index=False)
    data.groupby('arm').mean(numeric_only=True).to_csv(out/'target_summary.csv')
    save_json(out/'COMPLETE.json', dict(new_training=0, source_mean='Original training sampling weights over source factor-fit1296 only.',
        purpose='Posthoc fixed diagnostic; do not target-select or substitute a new primary method.',
        constant_effect='Centered score differences between observations exactly cancel the fixed added vector. Normalized angular distance may nevertheless change.',
        tasks=len(tasks), seeds=SEEDS))
    print(data.groupby('arm')[['OA', 'BA']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
