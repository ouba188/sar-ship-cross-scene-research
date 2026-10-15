"""Reuse the original source model and verify matched support/budget before readout."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from run_f0921_transfer import ENCODER_SCOPE, freeze, evaluate, write_json

ORIGINAL = Path('E:/Hermes/paper_tracking/sensitivity/f0921_transfer_20261007')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    if (out/'summary.json').exists():
        raise ValueError('completed readout must not be overwritten')
    with (ORIGINAL/'source_model.pkl').open('rb') as handle:
        model = pickle.load(handle)
    with threadpool_limits(limits=4):
        freeze(out, model)
        with np.load(ORIGINAL/'policies.npz') as old, np.load(out/'policies.npz') as new:
            for key in ('sample_ids', 'support_rows', 'eval_masks', 'caps', 'seeds', 'arm_names'):
                np.testing.assert_array_equal(old[key], new[key], err_msg=key)
        write_json(out/'matched_protocol_check.json', dict(same_samples=True,
            same_supports=True, same_query_masks=True, same_budget_caps=True,
            source_model=str(ORIGINAL/'source_model.pkl'), source_model_refitted=False))
        evaluate(out, target_name='F0921 matched input-alignment development comparison',
                 encoder_scope=ENCODER_SCOPE)
    old = json.loads((ORIGINAL/'summary.json').read_text())
    new = json.loads((out/'summary.json').read_text())
    comparison = []
    for before, after in zip(old['means'], new['means']):
        assert before['arm'] == after['arm']
        comparison.append(dict(arm=before['arm'], old_oa=100*before['acc'],
            new_oa=100*after['acc'], delta_oa_pp=100*(after['acc']-before['acc']),
            old_ba=100*before['ba'], new_ba=100*after['ba'],
            delta_ba_pp=100*(after['ba']-before['ba'])))
    write_json(out/'input_alignment_comparison.json', comparison)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == '__main__':
    main()
