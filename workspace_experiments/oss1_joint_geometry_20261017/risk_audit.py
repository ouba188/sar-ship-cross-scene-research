"""Macro-risk transfer and exact support-solve accounting for two changed cases."""
import importlib.util
import numpy as np
import pandas as pd
import torch
from common import HERE, FULL, MANIFEST, folds, public
from local_reader import LocalReader

spec = importlib.util.spec_from_file_location('oss1_local_fit', HERE / 'train.py')
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)


def main():
    torch.set_num_threads(2)
    manifest = pd.read_csv(MANIFEST)
    prediction = pd.read_csv(HERE / 'CASE_DIAGNOSTICS.csv')
    risks, cases, terms = [], [], []
    for fold, cache, episode, prefix in folds():
        data = public.load_fold('oss1', fold, cache, episode, prefix, manifest,
                                np.load(FULL / 'oss1_c4.npy'), np.load(FULL / 'oss1_zero.npy'))
        source = np.r_[data['train'], data['val']]
        g = public.pca(data['raw']['G50'], source)
        d = public.pca(data['raw']['D18'], source)
        c = public.pca(data['c4'], source)
        base = public.design_block(c, np.arange(len(c)))
        z = public.design_block((g, d), np.arange(len(g)))
        prior_cache = {tuple(sorted(source)): (
            base @ public.fit_head(c, data['labels'], source),
            z @ public.fit_head((g, d), data['labels'], source),
            public.class_map(z[:, :-1], data['labels'], data['train']))}
        qrows = data['rows'][data['query']]
        truth = data['labels'][data['query']]
        with np.load(HERE / fold / 'local_features.npz') as f:
            features = {k: torch.tensor(f[key][data['rows']]) for k, key in
                        [('body', 'body'), ('valid', 'valid'), ('tokens', 'joint_tokens'),
                         ('positions', 'joint_positions')]}
        for (origin, draw), s in data['support'].items():
            task = fit.make_task(data, base, z, dict(q=data['query'], s=s, prior_rows=source,
                origin=origin, acquisition=str(data['query_frame'].acquisition.iloc[0])), prior_cache)
            old = task['old'].numpy()
            old_loss = ((old - np.eye(8)[truth]) ** 2).sum(1)
            old_macro = np.mean([old_loss[truth == k].mean() for k in np.unique(truth)])
            selected = prediction[(prediction.fold == fold) & (prediction.origin == origin) &
                                  (prediction.draw == draw)]
            for (arm, seed), p in selected.groupby(['arm', 'seed']):
                p = p.set_index('global_row').loc[qrows]
                np.testing.assert_array_equal(p.baseline, old.argmax(1))
                new = p[[f'p{k}' for k in range(8)]].to_numpy()
                loss = ((new - np.eye(8)[truth]) ** 2).sum(1)
                macro = np.mean([loss[truth == k].mean() for k in np.unique(truth)])
                risks.append(dict(fold=fold, origin=origin, draw=draw, arm=arm, seed=seed,
                    baseline_macro_brier=old_macro, macro_brier=macro, delta_macro_brier=macro-old_macro))
                if arm != 'joint' or origin != 'cross':
                    continue
                for row in (1038, 591):
                    if row not in p.index or any(x['global_row'] == row for x in cases):
                        continue
                    wanted = p.loc[row].net_base > 0 if row == 1038 else p.loc[row].net_base < 0
                    if not wanted:
                        continue
                    reader = LocalReader().eval()
                    state = torch.load(HERE / fold / f'joint_{seed}' / 'reader.pt',
                                       map_location='cpu', weights_only=False)
                    reader.load_state_dict(state['selected'])
                    with torch.no_grad():
                        raw, _ = fit.forward(reader, features, task)
                        probability, h = fit.predict(raw, features['valid'][task['ids']], task,
                                                     state['best']['alpha'])
                    baseline_replay = float(np.max(np.abs(old[np.arange(len(truth)), truth] -
                                                   p.baseline_true_probability.to_numpy())))
                    full_replay = float(np.max(np.abs(probability.numpy() - new)))
                    print('CASE_REPLAY', fold, row, 'baseline', baseline_replay, 'full', full_replay,
                          flush=True)
                    # Recomputed float32 PCA/prior already has micro-scale drift before this reader.
                    # Keep classification exact and record probability drift in the case receipt.
                    np.testing.assert_array_equal(probability.argmax(1).numpy(), new.argmax(1))
                    np.testing.assert_allclose(probability.numpy(), new, atol=1e-5, rtol=0)
                    h = h.numpy(); ns = task['ns']; alpha = state['best']['alpha']
                    ss = task['ss'].numpy(); qs = task['qs'].numpy()
                    ss_new = ss + alpha * h[:ns] @ h[:ns].T
                    qs_new = qs + alpha * h[ns:] @ h[:ns].T
                    c0 = public.coefficients(ss, task['prior_s'].numpy(), data['labels'][s], 8.)
                    c1 = public.coefficients(ss_new, task['prior_s'].numpy(), data['labels'][s], 8.)
                    j = int(np.flatnonzero(qrows == row)[0]); true = int(truth[j])
                    rival = int(p.loc[row, 'baseline' if row == 1038 else 'prediction'])
                    a0, a1 = c0[:, true] - c0[:, rival], c1[:, true] - c1[:, rival]
                    kernel = (qs_new[j] - qs[j]) * (a0 + a1) / 2
                    refit = (qs_new[j] + qs[j]) / 2 * (a1 - a0)
                    prior = task['prior_q'].numpy()[j]
                    margin0 = prior[true] - prior[rival] + qs[j] @ a0
                    margin1 = prior[true] - prior[rival] + qs_new[j] @ a1
                    np.testing.assert_allclose((kernel + refit).sum(), margin1-margin0, atol=1e-12)
                    cases.append(dict(global_row=row, fold=fold, draw=draw, seed=seed, truth=true,
                        rival=rival, alpha=alpha, old_true_probability=old[j, true],
                        new_true_probability=new[j, true], old_expert_margin=margin0,
                        new_expert_margin=margin1, kernel_term=kernel.sum(), refit_term=refit.sum(),
                        support_same_class=int((data['labels'][s] == true).sum()),
                        baseline_probability_replay_max_error=baseline_replay,
                        full_probability_replay_max_error=full_replay))
                    for k, sr in enumerate(data['rows'][s]):
                        terms.append(dict(query_row=row, support_row=int(sr),
                            support_class=int(data['labels'][s[k]]), kernel_term=kernel[k],
                            refit_term=refit[k], total=kernel[k]+refit[k]))
    risk = pd.DataFrame(risks)
    risk.to_csv(HERE / 'MACRO_RISK.csv', index=False)
    pd.DataFrame(cases).to_csv(HERE / 'MECHANISM_CASES.csv', index=False)
    pd.DataFrame(terms).to_csv(HERE / 'SUPPORT_SCORE_CONTRIBUTIONS.csv', index=False)
    prediction[prediction.arm.eq('joint')].groupby(['origin', 'fold', 'draw']).net_base.mean().mul(100).to_csv(
        HERE / 'JOINT_SUPPORT_DRAW_EFFECTS.csv')
    print('TARGET_MACRO_RISK', risk.groupby(['origin', 'arm'])[
          ['baseline_macro_brier', 'macro_brier', 'delta_macro_brier']].mean().to_string())
    print('MECHANISM_CASES', pd.DataFrame(cases).to_string(index=False))
    print('SOURCE_SELECTION', pd.read_csv(HERE / 'SOURCE_SELECTION_GAINS.csv').groupby('arm')[
          ['old_source_brier', 'chosen_source_brier', 'delta_source_brier']].mean().to_string())


if __name__ == '__main__':
    main()
