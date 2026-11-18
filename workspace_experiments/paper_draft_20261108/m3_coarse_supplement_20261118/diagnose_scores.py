"""Posthoc cached-score diagnostics; no fitting, selection, or inference."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def score(seed, arm, split, variant):
    return np.load(ROOT/'fits'/f'{arm}_s{seed}'/f'{split}_{variant}_local.npz')['score'].astype(np.float64)


def centered(x):
    return x-x.mean(-1, keepdims=True)


def main():
    effects, source = [], []
    for seed in (17, 29, 43):
        for split in ('validation', 'expansion'):
            contrasts = {}
            for arm in ('pointwise', 'coarse_plus'):
                delta = centered(score(seed, arm, split, 'on')-score(seed, arm, split, 'off'))
                contrasts[arm] = delta
                common = delta.mean(0)
                total = np.square(delta).sum()
                effects.append(dict(seed=seed, split=split, arm=arm,
                    common_class_offset_energy_fraction=float(len(delta)*np.square(common).sum()/max(total, 1e-20)),
                    mean_class_centered_offset=common.tolist(),
                    mean_query_specific_l2=float(np.linalg.norm(delta-common, axis=1).mean()),
                    mean_effect_l2=float(np.linalg.norm(delta, axis=1).mean())))
            p, c = contrasts['pointwise'], contrasts['coarse_plus']
            effects.append(dict(seed=seed, split=split, comparison='pointwise/coarse_plus',
                centered_effect_cosine=float((p*c).sum()/max(np.linalg.norm(p)*np.linalg.norm(c), 1e-20)),
                relative_effect_difference=float(np.linalg.norm(p-c)/max(np.linalg.norm(p), 1e-20))))
        base = read(ROOT/'fits'/f'continue_s{seed}/receipt.json')['best']['metrics']
        b = np.array(base['confusion'])
        for arm in ('pointwise', 'coarse_plus'):
            metrics = read(ROOT/'fits'/f'{arm}_s{seed}/receipt.json')['best']['metrics']
            a = np.array(metrics['confusion'])
            assert np.array_equal(b.sum(1), a.sum(1))
            source.append(dict(seed=seed, arm=arm, delta_oa_pp=metrics['oa']-base['oa'],
                delta_ba_pp=metrics['ba']-base['ba'], class_counts=b.sum(1).tolist(),
                delta_correct_by_class=(a.diagonal()-b.diagonal()).tolist(),
                delta_recall_pp=(100*(a.diagonal()-b.diagonal())/b.sum(1)).tolist()))
    result = dict(definition='Remove class-common logit offset within each query; decompose remaining intervention effect into sample-mean class vector plus query-specific residual. Descriptive decomposition only, not causal attribution or deployable bias subtraction.',
                  effects=effects, source_selected_comparisons=source,
                  note='Source uses separately selected checkpoints. Off/on uses identical weights. No extra inference or validation-based selection.')
    (ROOT/'aggregate/score_diagnosis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
