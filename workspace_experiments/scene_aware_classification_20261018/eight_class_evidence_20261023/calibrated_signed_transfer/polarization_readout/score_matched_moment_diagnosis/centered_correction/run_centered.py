"""Authorized centered-distance correction, reusing the same frozen heads."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_diagnosis as replay
import analyze_diagnosis as analysis


def compare_versions():
    old = pd.read_csv(HERE.parent / 'paired_metrics.csv')
    new = pd.read_csv(HERE / 'paired_metrics.csv')
    key = ['role', 'seed', 'scope', 'comparison']
    old.merge(new, on=key, suffixes=('_uncentered', '_centered'), validate='one_to_one').to_csv(
        HERE / 'version_metrics.csv', index=False)
    raw = pd.read_csv(HERE.parent / 'predictions.csv')
    corrected = pd.read_csv(HERE / 'predictions.csv')
    key = ['fold', 'seed', 'role', 'instance_id']
    raw = raw.set_index(key).sort_index()
    corrected = corrected.set_index(key).loc[raw.index]
    np.testing.assert_array_equal(raw[[f'b{c}' for c in range(8)]], corrected[[f'b{c}' for c in range(8)]])
    np.testing.assert_array_equal(raw[[f'o{c}' for c in range(8)]], corrected[[f'o{c}' for c in range(8)]])
    rows = []
    for (role, seed), a in raw.groupby(level=['role', 'seed']):
        b = corrected.loc[a.index]
        rows.append(dict(role=role, seed=seed, n=len(a),
                         changed_donor_rows=int((a.donor_instance_id != b.donor_instance_id).sum()),
                         changed_matched_status=int((a.matched != b.matched).sum()),
                         changed_swapped_top1=int((a.swapped_prediction != b.swapped_prediction).sum()),
                         fixed_B_max_error=0, fixed_original_max_error=0))
    pd.DataFrame(rows).to_csv(HERE / 'version_checks.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def main():
    replay.HERE = HERE
    replay.main(centered=True)
    current = pd.read_csv(HERE / 'predictions.csv', dtype={'mmsi': str, 'donor_mmsi': str})
    prior = pd.read_csv(HERE.parent / 'centering_pair_changes.csv')
    key = ['fold', 'seed', 'role', 'instance_id']
    expected = prior.set_index(key).loc[pd.MultiIndex.from_frame(current[key])]
    np.testing.assert_array_equal(current.donor_instance_id, expected.centered_donor)
    analysis.HERE = HERE
    analysis.main()
    compare_versions()


if __name__ == '__main__':
    main()
