"""Matched existing handoff downstream of the three frozen/trained representations."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from build_oof_predictions import load_manifest
from run_f0921_transfer import project_target_batches
from run_opensarship2_transfer import SOURCE, ARMS, write_json, fit_source, freeze_draw, evaluate


def freeze_arm(arm, target_dir, model):
    target = pd.read_csv(target_dir/'target_manifest.csv', keep_default_na=False)
    base = np.load(target_dir/'paired_r50.npy', mmap_mode='r')
    extra = np.load(arm/'target_extra.npy', mmap_mode='r')
    prior_b, prior_e, phi_b, phi_e = project_target_batches(model, base, extra)
    draws = []
    for seed in range(5):
        support = np.random.default_rng(seed).choice(len(target), 64, replace=False)
        draws.append(freeze_draw(prior_b, prior_e, phi_b, phi_e, model['gate'], support,
            target.iloc[support].class_id.to_numpy(dtype=int), target.target_manifest_row.to_numpy()))
    np.savez_compressed(arm/'policies.npz', **{k: np.stack([d[k] for d in draws]) for k in draws[0]},
        arm_names=np.asarray(ARMS), seeds=np.arange(5), sample_ids=target.sample_id.to_numpy(dtype=str),
        prior_base_scores=prior_b, prior_expert_scores=prior_e)
    with np.load(target_dir/'policies.npz') as old:
        np.testing.assert_array_equal(old['sample_ids'], target.sample_id.to_numpy(dtype=str))
        for key in ('support_rows', 'eval_masks', 'caps'):
            np.testing.assert_array_equal(old[key], np.stack([d[key] for d in draws]), err_msg=key)
        np.testing.assert_array_equal(old['predictions'][:, 1], np.stack([d['predictions'][1] for d in draws]))
    write_json(arm/'POLICIES_SAVED.json', dict(all_five_saved_before_readout=True,
        original_sample_support_query_budget_match=True, adapted_base_predictions_unchanged=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    if not (out/'TRAINING_COMPLETE.json').exists():
        raise ValueError('all representation arms must finish before target readout')
    if (out/'summary.json').exists():
        raise ValueError('completed experiment must not be overwritten')
    protocol = json.loads((out/'protocol.json').read_text())
    target = Path(protocol['target'])
    np.testing.assert_array_equal(np.load(out/'target_feature_rows.npy'), np.load(target/'feature_rows.npy'))
    manifest = load_manifest(SOURCE/'manifest.csv')
    rows = np.load(protocol['source_split'])['source_rows']
    base = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    scope = ('e206 target-unlabeled pretraining inherited. Supervised arms train layer4 only on '
             'identity-purged source labels. Source OOF is head-only conditional on these encoders, '
             'not encoder-level independence. No external target labels fine-tune the backbone.')
    summaries = {}
    with threadpool_limits(limits=4):
        for name in protocol['arms']:
            arm = out/name
            shutil.copyfile(target/'target_manifest.csv', arm/'target_manifest.csv')
            model = fit_source(arm, manifest, rows, base, np.load(arm/'source_extra.npy', mmap_mode='r'))
            freeze_arm(arm, target, model)
        # No arm's query readout can influence another arm's training or policy.
        for name in protocol['arms']:
            arm = out/name
            evaluate(arm, target_name=f'F0921 fixed-crop representation pilot: {name}', encoder_scope=scope)
            summaries[name] = json.loads((arm/'summary.json').read_text())
    contrasts = []
    for a, b in [('supervised', 'frozen_replay'), ('utility_tail', 'supervised'), ('utility_tail', 'frozen_replay')]:
        left = pd.read_csv(out/a/'metrics.csv')
        right = pd.read_csv(out/b/'metrics.csv')
        lm, rm = left[left.arm == 'visual_handoff5'], right[right.arm == 'visual_handoff5']
        differences = {key: (100*(lm[key].to_numpy()-rm[key].to_numpy())).tolist() for key in ('acc', 'ba')}
        contrasts.append(dict(candidate=a, control=b, draw_deltas_pp=differences,
                             means_pp={key: float(np.mean(v)) for key, v in differences.items()}))
    write_json(out/'summary.json', dict(arms=summaries, contrasts=contrasts,
        independent_confirmation=False, same_backbone_training_budget=True,
        full_attention_proposal_implemented=False, target_used_for_checkpoint_selection=False))
    print(json.dumps(contrasts, indent=2), flush=True)


if __name__ == '__main__':
    main()
