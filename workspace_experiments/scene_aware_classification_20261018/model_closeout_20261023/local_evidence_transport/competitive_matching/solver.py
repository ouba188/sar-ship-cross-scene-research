"""Prototype shared-query-mass matching; no learned classifier or data tuning."""
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp


def transport_values(sim, a, b, w, eps=.1, steps=300):
    """Return regularized optima and row-marginal dual derivatives.

    Both sides have a unit dustbin, so real-query rejection is variable.
    The KL reference stays fixed at a, rather than following optimized w.
    This prototype takes strictly positive valid tokens; callers remove masks.
    """
    c, n, m = sim.shape
    aa = np.r_[a, 1.]
    ww = np.r_[w, 1.]
    bb = np.c_[b, np.ones(c)]
    ref = aa[None, :, None] * bb[:, None, :] / 2
    ss = np.pad(sim, ((0, 0), (0, 1), (0, 1)))
    logk = ss / eps + np.log(ref)
    u = np.zeros((c, n + 1))
    v = np.zeros((c, m + 1))
    for _ in range(steps):
        u = np.log(ww)[None] - logsumexp(logk + v[:, None], axis=2)
        v = np.log(bb) - logsumexp(logk + u[:, :, None], axis=1)
    logt = logk + u[:, :, None] + v[:, None]
    t = np.exp(logt)
    kl = (t * (logt - np.log(ref)) - t + ref).sum((1, 2))
    value = (t * ss).sum((1, 2)) - eps * kl
    error = max(abs(t.sum(2) - ww).max(), abs(t.sum(1) - bb).max())
    return value, -eps * u[:, :-1], t, float(error)


def capped_normalize(proposal, cap):
    low, high = 0., 1. / proposal.min()
    for _ in range(80):
        mid = (low + high) / 2
        if np.minimum(cap, mid * proposal).sum() < 1:
            low = mid
        else:
            high = mid
    return np.minimum(cap, high * proposal)


def optimize(sim, a, b, candidate, penalty=.1, rounds=20):
    """Locally improve hard-rival objective, retaining the fixed-mass start."""
    w = a.copy()
    trajectory = []
    rivals = np.delete(np.arange(len(b)), candidate)

    def evaluate(weights):
        values, gradients, plan, error = transport_values(sim, a, b, weights)
        if error > 1e-7:
            raise RuntimeError(f'Inner marginal error {error}; outer update invalid')
        rival = rivals[np.argmax(values[rivals])]
        kl = np.sum(weights * np.log(weights / a))
        objective = values[candidate] - values[rival] - penalty * kl
        gradient = gradients[candidate] - gradients[rival]
        gradient -= penalty * (np.log(weights / a) + 1)
        return float(objective), gradient, int(rival), plan

    current, gradient, rival, plan = evaluate(w)
    trajectory.append(current)
    for _ in range(rounds):
        accepted = False
        rate = 1.
        for _ in range(12):
            logproposal = np.log(w) + rate * gradient
            proposal = np.exp(logproposal - logproposal.max())
            trial = capped_normalize(proposal, 2 * a)
            value, grad, other, newplan = evaluate(trial)
            if value > current + 1e-10:
                w, current, gradient, rival, plan = trial, value, grad, other, newplan
                trajectory.append(current)
                accepted = True
                break
            rate *= .5
        if not accepted:
            break
    return dict(score=current, weights=w, rival=rival, plan=plan,
                trajectory=np.asarray(trajectory))


def checks():
    # These checks catch mass, entropy, gradient and spurious-count mechanisms.
    rng = np.random.default_rng(17)
    sim = rng.uniform(-.2, .8, (3, 4, 5))
    a = np.array([.1, .2, .3, .4])
    b = rng.uniform(.1, 1., (3, 5))
    b /= b.sum(1, keepdims=True)
    values, gradient, plan, marginal = transport_values(sim, a, b, a)
    assert marginal < 1e-8, marginal
    direction = np.array([.2, -.1, .1, -.2])
    h = 1e-5
    plus = transport_values(sim, a, b, a + h * direction)[0]
    minus = transport_values(sim, a, b, a - h * direction)[0]
    gradient_error = float(abs((plus - minus) / (2 * h) - gradient @ direction).max())
    assert gradient_error < 1e-6, gradient_error

    repeated = np.repeat(sim, 2, axis=2)
    repeated_b = np.repeat(b / 2, 2, axis=1)
    duplicate_error = float(abs(transport_values(repeated, a, repeated_b, a)[0] - values).max())
    assert duplicate_error < 1e-8, duplicate_error

    identical = np.repeat(sim[:1], 3, axis=0)
    identical_b = np.repeat(b[:1], 3, axis=0)
    equal = [optimize(identical, a, identical_b, c) for c in range(3)]
    null_error = max(abs(r['score']) for r in equal)
    assert null_error < 1e-8, null_error

    # Shared high similarity cannot distinguish classes; the second row can.
    demo = np.array([[[.9, .9], [.8, .8]], [[.9, .9], [.2, .2]]])
    da = np.ones(2) / 2
    db = np.ones((2, 2)) / 2
    solved = optimize(demo, da, db, 0)
    assert solved['weights'][1] > da[1], solved['weights']
    assert np.all(np.diff(solved['trajectory']) > 0)
    assert solved['score'] > solved['trajectory'][0]
    report = dict(stage='synthetic solver checks only; no classification result',
                  inner_marginal_error=marginal, dual_gradient_error=gradient_error,
                  duplicate_split_mass_error=duplicate_error,
                  identical_classes_score_error=null_error,
                  shared_bright_demo_weights=solved['weights'].tolist(),
                  shared_bright_demo_initial=float(solved['trajectory'][0]),
                  shared_bright_demo_final=solved['score'])
    (Path(__file__).parent / 'solver_checks.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    checks()
