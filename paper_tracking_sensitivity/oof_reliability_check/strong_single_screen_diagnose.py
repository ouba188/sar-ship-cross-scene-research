"""Fixed prior/kernel crossover: functional attribution, not model selection."""

from strong_single_screen import ROOT, SOURCE, OUT, inputs, coefficients, ba, write
import numpy as np
import pandas as pd


def main():
    metrics = []
    for receipt in sorted((OUT/'targets').glob('*/*/PREDICTED.json')):
        dest = receipt.parent
        dataset, task = dest.parent.name, dest.name
        with np.load(dest/'predictions.npz') as p:
            data = {k: p[k] for k in p.files}
        priors, kernels = {}, {}
        for short, arm in [('S', 'sarclip'), ('R', 'r50_c4_common_readout')]:
            source, target = inputs(arm, dataset)
            with np.load(dest/f'{arm}_head.npz') as h:
                z = (np.asarray(target[data['rows']], float)-h['center'])/h['scale']
                priors[short] = z@h['coef']+h['intercept']
                # Absorb selected lambda into K: all arms then solve with lambda1.
                kernels[short] = [(z@z[s].T+1)/float(h['residual_lambda']) for s in data['support']]
            del source, target, z
        arms = ('RR', 'SR', 'RS', 'SS', 'S_prior', 'R_prior', 'S_support', 'R_support')
        predicted = np.empty((5, len(arms), len(data['rows'])), dtype=np.int8)
        for draw, support in enumerate(data['support']):
            for ai, arm in enumerate(arms):
                if arm.endswith('_prior'):
                    values = priors[arm[0]]
                else:
                    if arm.endswith('_support'):
                        prior = np.zeros_like(priors[arm[0]])
                        kernel = kernels[arm[0]][draw]
                    else:
                        prior, kernel = priors[arm[0]], kernels[arm[1]][draw]
                    coef = coefficients(kernel[support], prior[support], data['support_labels'][draw], 1.)
                    values = prior+kernel@coef
                predicted[draw, ai] = values.argmax(1)
            np.testing.assert_array_equal(predicted[draw, arms.index('RR')], data['predictions'][draw, 1])
            np.testing.assert_array_equal(predicted[draw, arms.index('SS')], data['predictions'][draw, 0])
        np.savez_compressed(dest/'diagnostic_predictions.npz', predictions=predicted, modes=np.array(arms))
        path = SOURCE/'manifest.csv' if dataset == 'self24ports' else ROOT/'opensarship2_transfer_20261007/target_manifest.csv'
        col = 'instance_id' if dataset == 'self24ports' else 'sample_id'
        frame = pd.read_csv(path, usecols=[col, 'class_id'])
        np.testing.assert_array_equal(frame[col].to_numpy(str)[data['rows']], data['ids'])
        truth = frame.class_id.to_numpy()[data['rows']]
        for draw, mask in enumerate(data['eval_masks']):
            for ai, arm in enumerate(arms):
                pred = predicted[draw, ai, mask]
                metrics.append(dict(dataset=dataset, task=task, draw=draw, arm=arm,
                    oa=100*np.mean(pred == truth[mask]), ba=ba(truth[mask], pred)))
        print(f'crossover replay exact: {dataset}/{task}', flush=True)
    table = pd.DataFrame(metrics)
    table.to_csv(OUT/'diagnostic_metrics.csv', index=False)
    summary = table.groupby(['dataset', 'arm'])[['oa', 'ba']].mean()
    summary.to_csv(OUT/'diagnostic_summary.csv')
    write(OUT/'DIAGNOSED.json', dict(tasks=table[['dataset', 'task']].drop_duplicates().to_dict('records'),
        diagonals_exact=True, target_tuning=False, meaning='Source prior vs source-selected scaled support kernel; no physical attribution'))
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    main()
