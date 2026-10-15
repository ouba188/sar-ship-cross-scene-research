"""Check that expanded reporting preserves the four previous CV tasks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    out = Path(__file__).resolve().parent
    old = out.parents[1] / 'target_domain_gap_20261012/support_cv_selection'
    original = pd.read_csv(old / 'summary.csv')
    current = pd.read_csv(out / 'summary.csv')
    current = current[current.stage == 'full']
    paired = original.merge(current, on=['task', 'scope', 'policy'],
                            suffixes=('_old', '_new'), validate='one_to_one')
    assert len(paired) == 4 * 2 * 6, len(paired)
    errors = {}
    for column in ('oa', 'ba', 'delta_oa', 'delta_ba'):
        left, right = paired[column + '_old'], paired[column + '_new']
        errors[column] = float(np.max(np.abs(left - right)))
        np.testing.assert_allclose(left, right, atol=1e-10, rtol=0)
    choices = pd.read_csv(out / 'choices.csv')
    old_choices = json.loads((old / 'SELECTION_FROZEN.json').read_text(encoding='utf-8'))['selected']
    count = 0
    for item in old_choices:
        row = choices[(choices.task == item['task']) & (choices.draw == item['draw'])]
        assert len(row) == 1
        for policy in ('cv_oa', 'cv_nll'):
            assert row.iloc[0][policy] == item[policy], (item, policy)
            count += 1
    assert count == 40, count
    complete = json.loads((out / 'COMPLETE.json').read_text(encoding='utf-8'))
    result = dict(previous_metric_rows=len(paired), previous_choices=count,
                  metric_max_error=errors, completion=complete)
    (out / 'REPLAY_VERIFIED.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
