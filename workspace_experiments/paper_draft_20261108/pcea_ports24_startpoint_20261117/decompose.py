"""Descriptive decomposition of cached validation logits; no fitted predictor."""
from pathlib import Path
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'decomposition'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def load(path, ids=None, y=None):
    with np.load(path) as z:
        if ids is not None:
            assert np.array_equal(z['ids'], ids) and np.array_equal(z['label'], y)
        return z['ids'].copy(), z['label'].copy(), z['score'].astype(np.float64)


def margin(z, y):
    other = z.copy()
    other[np.arange(len(y)), y] = -np.inf
    return z[np.arange(len(y)), y] - other.max(1)


def decomposition(delta):
    # Remove softmax-invariant offsets before measuring shared class preference.
    offset = delta.mean(1, keepdims=True)
    centered = delta - offset
    bias = centered.mean(0, keepdims=True)
    residual = centered - bias
    energy = float(np.square(centered).sum())
    shared = float(len(delta) * np.square(bias).sum())
    variable = float(np.square(residual).sum())
    assert np.allclose(centered, bias + residual)
    assert np.isclose(energy, shared + variable, atol=1e-12)
    return bias, residual, dict(
        n=len(delta), class_bias=bias[0].tolist(),
        shared_energy_fraction=shared / energy if energy else 0.,
        sample_specific_energy_fraction=variable / energy if energy else 0.,
        centered_rms=float(np.sqrt(np.mean(centered ** 2))),
        residual_rms=float(np.sqrt(np.mean(residual ** 2))),
        softmax_invariant_offset_rms=float(np.sqrt(np.mean(offset ** 2))))


def main():
    OUT.mkdir(exist_ok=True)
    ids, y, initial = load(ROOT.parent / 'ports24_recovered_training_20261113/selected_s17/validation.npz')
    rows = read(Path('C:/Users/25791/AppData/Local/sar_ports24_fullscale_20261110/manifest.json'))
    available = np.load(ROOT.parent / 'ports24_sync_20261112/native_cache/available.npy', mmap_mode='r')[ids].copy()
    recovered = read(ROOT.parent / 'ports24_recovered_training_20261113/recovery_audit_local.json')
    available[[str(int(i)) in recovered for i in ids]] = 1
    dual = available.astype(bool).all(1)
    result, cases = {}, []
    for strength in (0., .1):
        folder = ROOT / 'fits' / f'cross_mean_s17_sp{strength:g}'
        _, _, on = load(folder / 'lastvalidation.npz', ids, y)
        _, _, off = load(folder / 'lastvalidation_off.npz', ids, y)
        _, _, full = load(ROOT / 'fits' / f'full_s17_sp{strength:g}/lastvalidation.npz', ids, y)
        comparisons = {'direct_on_off': (on, off), 'continuation': (on, initial),
                       'full_continuation': (full, initial),
                       'matched_training_difference': (on, full)}
        for name, (after, before) in comparisons.items():
            key = f'sp{strength:g}_{name}'
            delta = after - before
            bias, residual, stats = decomposition(delta[dual])
            stats['all_samples'] = decomposition(delta)[2]
            b, a, labels = before[dual], after[dual], y[dual]
            idx = np.arange(len(labels))
            competitor = b.copy()
            competitor[idx, labels] = -np.inf
            rival = competitor.argmax(1)
            fixed_delta = (a-b)[idx, labels] - (a-b)[idx, rival]
            bias_margin = (bias[0, labels] - bias[0, rival])
            residual_margin = residual[idx, labels] - residual[idx, rival]
            assert np.allclose(fixed_delta, bias_margin + residual_margin)
            wrong = b.argmax(1) != labels
            gap = -margin(b, labels)
            positive = wrong & (fixed_delta > 0)
            stats.update(
                prediction_changes=int((a.argmax(1) != b.argmax(1)).sum()),
                rescued=int(((a.argmax(1) == labels) & wrong).sum()),
                harmed=int(((a.argmax(1) != labels) & ~wrong).sum()),
                fixed_rival_margin_mean=float(fixed_delta.mean()),
                sample_specific_margin_mean=float(residual_margin.mean()),
                wrong_count=int(wrong.sum()), wrong_positive_correction=int(positive.sum()),
                positive_wrong_gap_fraction_median=float(np.median(fixed_delta[positive]/gap[positive])) if positive.any() else None,
                positive_wrong_gap_fraction_max=float(np.max(fixed_delta[positive]/gap[positive])) if positive.any() else None,
                classes={str(c): dict(n=int((labels == c).sum()),
                    margin_mean=float(fixed_delta[labels == c].mean()),
                    residual_margin_mean=float(residual_margin[labels == c].mean()))
                    for c in np.unique(labels)},
                ports={p: decomposition(delta[dual][np.array([rows[i]['port'] == p for i in ids[dual]])])[2]
                       for p in sorted({rows[i]['port'] for i in ids[dual]})})
            result[key] = stats
            for j, i in enumerate(np.flatnonzero(dual)):
                cases.append(dict(comparison=key, id=rows[ids[i]]['id'], port=rows[ids[i]]['port'],
                    label=int(y[i]), before_pred=int(before[i].argmax()), after_pred=int(after[i].argmax()),
                    rival=int(rival[j]), before_margin=float(margin(before[i:i+1], y[i:i+1])[0]),
                    after_margin=float(margin(after[i:i+1], y[i:i+1])[0]),
                    fixed_rival_change=float(fixed_delta[j]), shared_bias_change=float(bias_margin[j]),
                    sample_specific_change=float(residual_margin[j])))
    (OUT/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    with (OUT/'cases.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=cases[0]); w.writeheader(); w.writerows(cases)
    print(json.dumps({k: {a: v[a] for a in ('shared_energy_fraction', 'centered_rms',
        'residual_rms', 'prediction_changes', 'wrong_positive_correction',
        'positive_wrong_gap_fraction_median', 'positive_wrong_gap_fraction_max')}
        for k, v in result.items()}, indent=2))
    for row in cases:
        if str(row['id']) in ('21433', '35906', '31926', '23235', '35666'):
            print(json.dumps(row))


if __name__ == '__main__':
    main()
