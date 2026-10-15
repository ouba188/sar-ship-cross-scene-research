"""Describe fixed pilot predictions without fitting or selecting another policy."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from budgeted_residual_handoff import score_policy
from run_opensarship2_transfer import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.out/'summary.json').read_text())
    diagnostics = {}
    for name in summary['arms']:
        arm = args.out/name
        labels = pd.read_csv(arm/'target_manifest.csv').class_id.to_numpy(dtype=int)
        rows = []
        with np.load(arm/'policies.npz') as archive:
            masks = archive['eval_masks']
            selected = archive['selected']
            base = archive['predictions'][:, 1]
            final = archive['predictions'][:, 2]
            expert = archive['expert_scores'].argmax(2)
        for seed in range(len(masks)):
            evaluation = masks[seed]
            recovered = evaluation & (base[seed] != labels) & (expert[seed] == labels)
            harmed = evaluation & (base[seed] == labels) & (expert[seed] != labels)
            full = score_policy(expert[seed], base[seed], labels, evaluation)
            actual = score_policy(final[seed], base[seed], labels, evaluation)
            base_accuracy = (base[seed, evaluation] == labels[evaluation]).mean()
            recovery_count = int((recovered & selected[seed]).sum())
            harm_count = int((harmed & selected[seed]).sum())
            np.testing.assert_allclose(actual['acc']-base_accuracy,
                (recovery_count-harm_count)/evaluation.sum(), atol=1e-12)
            rows.append(dict(seed=seed, full_expert_oa_pp=100*full['acc'],
                full_expert_ba_pp=100*full['ba'], all_query_recoveries=int(recovered.sum()),
                all_query_harms=int(harmed.sum()), admitted=int(selected[seed].sum()),
                selected_recoveries=recovery_count, selected_harms=harm_count,
                selected_recover_ba_pp=100*actual['recover_ba'],
                selected_harm_ba_pp=100*actual['harm_ba'],
                full_expert_class_recall=full['class_recall']))
        diagnostics[name] = dict(draws=rows, means={
            key: float(np.mean([row[key] for row in rows])) for key in rows[0]
            if key not in ('seed', 'full_expert_class_recall')})
    result = dict(arms=diagnostics,
        interpretation='Post-readout description only; full expert violates 5% deployment cap. '
                       'No query-label policy training or further arm selection is performed.')
    write_json(args.out/'diagnostics.json', result)
    print(json.dumps({name: value['means'] for name, value in diagnostics.items()}, indent=2))


if __name__ == '__main__':
    main()
