"""Freeze one structural follow-up, preserving the previous failed pilot."""

import argparse
import json
import shutil
from pathlib import Path

from run_opensarship2_transfer import write_json

PARENT = Path('E:/Hermes/paper_tracking/sensitivity/polar_residual_pilot_20261008')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        raise ValueError('follow-up protocol exists')
    protocol = json.loads((PARENT/'protocol.json').read_text())
    protocol.update(prior_preserving=True, vv_cache_root=str(PARENT), previous_polar=str(PARENT),
        hypothesis='Unconstrained new evidence damages retained source prior; preserve it exactly while exposing new coordinates to64support adaptation',
        residual='project learned residual into PCA-visible nullspace of source SSL classifier; original e206 layer4 frozen',
        projection='dz=(delta@PCA.T)/scale; dz_null=dz@(I-pinv(W_ssl)@W_ssl); delta_null=(dz_null*scale)@PCA',
        exploratory_followup=True,
        preparation='Reuse completed paired caches and source model/episodes; no additional extraction or source head refit')
    protocol['fixed_training']['backbone_lr'] = 0.
    protocol['fixed_training']['backbone'] = 'frozen original e206; only new residual/attention parameters train'
    write_json(out/'protocol.json', protocol)
    for name in ('source_model.pkl', 'deployment_geometry.npz', 'target_xs.npy', 'VV_CACHE_COMPLETE.json'):
        shutil.copyfile(PARENT/name, out/name)
    write_json(out/'PREPARATION_COMPLETE.json', dict(reused_from=str(PARENT), exploratory_followup=True,
        target_labels_for_training=0, prior_preserving=True))
    print('SUPPORT-VISIBLE POLAR PROTOCOL SAVED', flush=True)


if __name__ == '__main__':
    main()
