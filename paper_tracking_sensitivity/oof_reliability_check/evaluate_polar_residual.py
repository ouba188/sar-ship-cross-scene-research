"""Fixed readout for paired pooling, competitor contrast and its inference ablation."""

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
        raise ValueError('all training/extraction must finish before query readout')
    if (out/'summary.json').exists():
        raise ValueError('completed readout exists')
    protocol = json.loads((out/'protocol.json').read_text())
    target = Path(protocol['target'])
    with (out/'source_model.pkl').open('rb') as stream:
        model = pickle.load(stream)
    summaries = {}
    with threadpool_limits(limits=4):
        for name in protocol['evaluation_arms']:
            folder = out/name
            shutil.copyfile(target/'target_manifest.csv', folder/'target_manifest.csv')
            freeze_arm(folder, target, model)
        if protocol.get('prior_preserving', False):
            with np.load(Path(protocol['original'])/'frozen_replay/policies.npz') as saved:
                reference = saved['prior_expert_scores']
            parity = {}
            for name in protocol['evaluation_arms']:
                with np.load(out/name/'policies.npz') as saved:
                    actual = saved['prior_expert_scores']
                np.testing.assert_allclose(actual, reference, atol=2e-4, rtol=2e-4)
                parity[name] = dict(max_abs_score_difference=float(np.abs(actual-reference).max()),
                    prior_prediction_changes=int(np.count_nonzero(actual.argmax(1) != reference.argmax(1))))
            write_json(out/'PRIOR_PRESERVATION.json', parity)
        for name in protocol['evaluation_arms']:
            folder = out/name
            evaluate(folder, target_name=f'F0921 paired expert information: {name}',
                encoder_scope='Source episodic layer4/paired residual training; retained source PCA/prior/gate. '
                    'Both VV and VH enter expert. Historical F0921 SSL exposure; no encoder-OOF claim.')
            summaries[name] = json.loads((folder/'summary.json').read_text())
    originals = Path(protocol['original'])
    aligned = Path(protocol['aligned'])
    comparisons = []
    for candidate, control_path in [
        ('polar_pooled', aligned/'residual_global'),
        ('polar_contrast', aligned/'residual_global'),
        ('polar_contrast', out/'polar_pooled'),
        ('polar_contrast', out/'contrast_no_spatial'),
        ('polar_pooled', originals/'frozen_replay'),
        ('polar_contrast', originals/'frozen_replay')]:
        left, right = pd.read_csv(out/candidate/'metrics.csv'), pd.read_csv(control_path/'metrics.csv')
        left = left[left.arm == 'visual_handoff5'].sort_values('seed')
        right = right[right.arm == 'visual_handoff5'].sort_values('seed')
        np.testing.assert_array_equal(left.seed, right.seed)
        delta = {key: (100*(left[key].to_numpy()-right[key].to_numpy())).tolist() for key in ('acc', 'ba')}
        comparisons.append(dict(candidate=candidate, control=control_path.name, draw_deltas_pp=delta,
            means_pp={key: float(np.mean(value)) for key, value in delta.items()},
            positive_draws={key: int(np.count_nonzero(np.asarray(value)>0)) for key, value in delta.items()}))
    write_json(out/'summary.json', dict(arms=summaries, contrasts=comparisons,
        retained_training_head=True, no_posttraining_refit=True, independent_confirmation=False,
        queried_target_for_checkpoint_selection=False, global_system_already_used_vv=True))
    print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == '__main__':
    main()
