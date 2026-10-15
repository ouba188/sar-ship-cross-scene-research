"""Deploy the retained source geometry, without substituting refitted heads."""

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evaluate_neural_complement import freeze_arm
from run_opensarship2_transfer import evaluate, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    if not (out/'TRAINING_COMPLETE.json').exists():
        raise ValueError('all training must finish before readout')
    if (out/'summary.json').exists():
        raise ValueError('completed readout exists')
    protocol = json.loads((out/'protocol.json').read_text())
    previous, target = Path(protocol['previous']), Path(protocol['target'])
    with (out/'source_model.pkl').open('rb') as stream:
        model = pickle.load(stream)
    summaries = {}
    with threadpool_limits(limits=4):
        for name in protocol['arms']:
            folder = out/name
            shutil.copyfile(target/'target_manifest.csv', folder/'target_manifest.csv')
            freeze_arm(folder, target, model)
        for name in protocol['arms']:
            folder = out/name
            evaluate(folder, target_name=f'F0921 aligned residual expert: {name}',
                encoder_scope='Source-only episodic layer4 training. Source PCA/prior/gate frozen before '
                'training and retained at deployment. No encoder-level OOF claim. Inherited F0921 SSL exposure.')
            summaries[name] = json.loads((folder/'summary.json').read_text())
    comparisons = []
    for candidate, control, control_path in [
        ('residual_global', 'frozen_replay', previous/'frozen_replay'),
        ('residual_handoff', 'frozen_replay', previous/'frozen_replay'),
        ('residual_handoff', 'utility_tail', previous/'utility_tail'),
        ('residual_handoff', 'residual_global', out/'residual_global')]:
        left, right = pd.read_csv(out/candidate/'metrics.csv'), pd.read_csv(control_path/'metrics.csv')
        left = left[left.arm == 'visual_handoff5'].sort_values('seed')
        right = right[right.arm == 'visual_handoff5'].sort_values('seed')
        np.testing.assert_array_equal(left.seed, right.seed)
        differences = {key: (100*(left[key].to_numpy()-right[key].to_numpy())).tolist() for key in ('acc', 'ba')}
        comparisons.append(dict(candidate=candidate, control=control, draw_deltas_pp=differences,
            means_pp={key: float(np.mean(value)) for key, value in differences.items()},
            positive_draws={key: int(np.count_nonzero(np.asarray(value) > 0)) for key, value in differences.items()}))
    write_json(out/'summary.json', dict(arms=summaries, contrasts=comparisons,
        retained_training_head=True, no_posttraining_refit=True, independent_confirmation=False,
        query_readout_used_for_checkpoint_selection=False))
    print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == '__main__':
    main()
