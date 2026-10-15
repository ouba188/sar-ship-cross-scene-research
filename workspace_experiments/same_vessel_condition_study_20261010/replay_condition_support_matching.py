"""Replay versioned scores; legacy replay reproduces a known error, not correctness."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from condition_support_matching import ARMS, multipliers, incremental_factors, score_episode
from run_condition_support_matching import BASE, OUT, OLD, PARENT, SEEDS, load_npz


def main():
    meta = pd.read_csv(BASE/'observations.csv', usecols=['port', 'platform', 'pass_name', 'incidence_deg'])
    evidence = load_npz(OLD/'evidence_features.npz')
    feature = np.full((len(meta), 6), np.nan)
    feature[evidence['obs_ids']] = evidence['features']
    known = np.isfinite(feature).all(1)
    context = np.full(len(meta), np.nan)
    context[known] = np.logaddexp.reduce(feature[known, :3], axis=1)
    models = {a: load_npz(OUT/f'response_{a}.npz') for a in ('anchor', 'capacity', 'condition', 'sham')}
    gate = load_npz(OUT/'gate.npz')
    limits = json.loads((OUT/'LIMITS.json').read_text(encoding='utf-8'))
    protocol = json.loads((OUT/'PROTOCOL.json').read_text(encoding='utf-8'))
    formula = protocol.get('residual_formula', 'legacy_missing_minus_one')
    if formula not in ('signed_target_minus_prior_v2', 'legacy_missing_minus_one'):
        raise ValueError(f'Unknown residual formula: {formula}')
    legacy = formula == 'legacy_missing_minus_one'
    if legacy:
        print('LEGACY: reproducing the known missing-minus-one error; archive and old REPLAY.json stay unchanged.', flush=True)
    errors, counts = [], dict(scores=0, multipliers=0)
    for seed in SEEDS:
        bundle = load_npz(PARENT/f'full_{seed}_inputs.npz')
        prior, kernel = bundle['prior'].astype(float), bundle['kernel'].astype(float)
        selection = json.loads((OUT/f'SELECTION_FROZEN_{seed}.json').read_text(encoding='utf-8'))['chosen']
        for draw in range(3):
            saved = load_npz(OUT/f'{seed}_source_{draw}_predictions.npz')
            s, q = saved['support_obs_ids'], saved['query_obs_ids']
            factors, detail = multipliers(meta, s, q, feature[:, 5], context, models, gate, limits)
            if protocol.get('increment_only', False):
                factors = incremental_factors(factors)
            base, delta, op, _ = score_episode(prior, kernel, s, q, saved['support_labels'], factors)
            if legacy:
                present = np.bincount(saved['support_labels'], minlength=8) > 0
                residual = (2*np.eye(8)[saved['support_labels']]-prior[s])*present[None, :]
                delta = {arm: (op*(factor-1))@residual for arm, factor in factors.items()}
            np.testing.assert_array_equal(base, saved['baseline'])
            counts['scores'] += 1
            for arm in ARMS:
                np.testing.assert_array_equal(factors[arm], saved[arm+'_multiplier'])
                counts['multipliers'] += 1
                for dep, gain in [('selected', selection[arm]), ('forced1', 1.)]:
                    predicted = base+gain*delta[arm]
                    np.testing.assert_array_equal(predicted, saved[f'{arm}_{dep}'])
                    errors.append(float(np.max(np.abs(predicted-saved[f'{arm}_{dep}']))))
                    counts['scores'] += 1
    result = dict(**counts, maximum_error=max(errors), query_labels_read=False,
                  fitting=False, original_query_order=True, residual_formula=formula,
                  legacy_error_reproduced=legacy, replay_proves_semantic_correctness=False,
                  archive_directory=str(OUT))
    if not legacy:
        (OUT/'REPLAY.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    OUT = parser.parse_args().out
    main()
