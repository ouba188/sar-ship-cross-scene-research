"""Small relative-utility probes for two frozen complete classifiers."""

import numpy as np
from scipy.special import softmax, expit
from sklearn.linear_model import LogisticRegression

NAMES = (
    'raw_max', 'initial_max', 'raw_margin', 'initial_margin',
    'probability_l1', 'js_divergence', 'base_agrees_raw', 'base_agrees_initial',
    'support_gain_raw_role', 'support_gain_initial_role',
    'support_sd_raw_role', 'support_sd_initial_role',
    'support_fraction_raw_role', 'support_fraction_initial_role',
    'support_class_coverage', 'support_macro_gain',
)


def class_weights(y):
    _, ix, counts = np.unique(y, return_inverse=True, return_counts=True)
    return 1. / (len(counts) * counts[ix])


def gain_labels(raw, initial, y):
    return ((initial.argmax(1) == y).astype(int)
            - (raw.argmax(1) == y).astype(int))


def support_profile(prior, raw_residual, initial_residual, base_probability, y):
    n = len(y)
    repeats = len(prior) // n
    raw = (.5 * (base_probability + softmax(prior + raw_residual, axis=1)))
    initial = (.5 * (base_probability + softmax(prior + initial_residual, axis=1)))
    raw = raw.reshape(repeats, n, -1).mean(0)
    initial = initial.reshape(repeats, n, -1).mean(0)
    gain = gain_labels(raw, initial, y)
    counts = np.bincount(y, minlength=8)
    rescue = np.bincount(y[gain == 1], minlength=8)
    harm = np.bincount(y[gain == -1], minlength=8)
    total = counts + 4.
    mean = (rescue - harm) / total
    variance = ((rescue + harm + 2.) / total - mean**2) / (total + 1.)
    return dict(mean=mean, sd=np.sqrt(variance), counts=counts,
                rescue=rescue, harm=harm, oof_gain=gain)


def features(raw, initial, base_probability, profile):
    r, i, b = raw.argmax(1), initial.argmax(1), base_probability.argmax(1)
    sr, si = np.sort(raw, axis=1), np.sort(initial, axis=1)
    midpoint = (raw + initial) / 2
    js = .5 * np.sum(raw * np.log(np.maximum(raw, 1e-30) / midpoint)
                    + initial * np.log(np.maximum(initial, 1e-30) / midpoint), axis=1)
    count = profile['counts']
    answer = np.column_stack((
        sr[:, -1], si[:, -1], sr[:, -1] - sr[:, -2], si[:, -1] - si[:, -2],
        abs(raw - initial).sum(1), js, b == r, b == i,
        profile['mean'][r], profile['mean'][i], profile['sd'][r], profile['sd'][i],
        count[r] / count.sum(), count[i] / count.sum(),
        np.full(len(raw), np.mean(count > 0)),
        np.full(len(raw), profile['mean'].mean()),
    ))
    assert answer.shape[1] == len(NAMES) and np.isfinite(answer).all()
    return answer


def fit_probe(x, gain, weights, width):
    mask = gain != 0
    xx, yy, w = x[mask, :width], (gain[mask] > 0).astype(int), weights[mask]
    mean = np.average(xx, axis=0, weights=w)
    scale = np.sqrt(np.average((xx - mean)**2, axis=0, weights=w))
    scale[scale < 1e-12] = 1.
    # Keep rescue/harm prevalence; do not reweight the binary outcomes equally.
    model = LogisticRegression(C=.1, solver='lbfgs', max_iter=1000, tol=1e-9)
    model.fit((xx - mean) / scale, yy, sample_weight=w / w.mean())
    return dict(mean=mean, scale=scale, coef=model.coef_[0],
                intercept=model.intercept_[0], width=width,
                iterations=int(model.n_iter_[0]), events=int(mask.sum()))


def predict_probe(model, x):
    z = (x[:, :int(model['width'])] - model['mean']) / model['scale']
    return expit(z @ model['coef'] + model['intercept'])


def selected_probabilities(raw, initial, choose):
    return np.where(choose[:, None], initial, raw)
