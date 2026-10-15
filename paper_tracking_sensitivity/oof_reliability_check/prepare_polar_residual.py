"""Freeze paired-information experiment without changing the deployment head."""

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from prepare_aligned_residual import geometry_arrays
from run_opensarship2_transfer import write_json

ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
ALIGNED = ROOT/'aligned_residual_pilot_20261008'
ORIGINAL = ROOT/'neural_complement_pilot_20261007'
TARGET = ROOT/'f0921_detector_recrop_20261007'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out/'PREPARATION_COMPLETE.json').exists():
        raise ValueError('completed paired experiment preparation exists')
    write_json(out/'protocol.json', dict(date='2026-10-08', target=str(TARGET),
        aligned=str(ALIGNED), original=str(ORIGINAL),
        training_arms=['polar_pooled', 'polar_contrast'],
        evaluation_arms=['polar_pooled', 'polar_contrast', 'contrast_no_spatial'],
        input='original VH plus co-indexed VV through shared e206 spatial encoder; normalized chip features, not physical polarimetric ratios',
        module='retained-prior competitor-directed difference-of-attention pooling on paired spatial feature differences',
        descriptor='mean(VV-VH) + sum((softmax(content+pair)-softmax(content-pair))*(VV-VH))',
        pair='top2 prior classes of unfused VH expert; query from exact frozen ridge class-difference direction in SSL feature space',
        residual='VH pooled + zero-initialized Linear512to512(descriptor)',
        comparison='pooled branch uses only mean(VV-VH), otherwise matched data/initialization/training',
        inference_ablation='same trained contrast checkpoint with spatial contrast set to zero; global VV-VH descriptor retained',
        fixed_training=dict(epochs=6, query_batch_size=128, support_size=64,
            seed=20261008, optimizer='AdamW', backbone_lr=1e-5, module_lr=1e-3,
            weight_decay=1e-4, clip=5., running_batchnorm='frozen', checkpoint='last fixed epoch',
            objective='source-class-balanced CE through actual64support adapted ridge expert; no handoff extraCE'),
        deployment='same pretraining PCA/prior/gate as aligned pilot; no head refit; same64labels and5%handoff',
        source_episodes=str(ALIGNED/'EPISODES_COMPLETE.json'),
        scientific_scope='dual-polarization attention is prior art; this is an implementation hypothesis, not novelty proof',
        information_scope='VV is newly available to expert branch; full system R50 already had VV/VH',
        target_scope='historical F0921, inherited e206 unlabeled pretraining, row-level support/query; no extra target labels',
        related_work='Shao et al., A Dual-Polarization Information-Guided Network for SAR Ship Classification, Remote Sensing15(8):2138,2023, doi10.3390/rs15082138'))
    shutil.copyfile(ALIGNED/'source_model.pkl', out/'source_model.pkl')
    with (out/'source_model.pkl').open('rb') as stream:
        model = pickle.load(stream)
    np.savez(out/'deployment_geometry.npz', **geometry_arrays(model))
    features = np.load(TARGET/'paired_r50.npy', mmap_mode='r')
    xs = np.lib.format.open_memmap(out/'target_xs.npy', mode='w+', dtype=np.float32, shape=(len(features), 128))
    with threadpool_limits(limits=4):
        for lo in range(0, len(features), 2048):
            hi = min(lo+2048, len(features))
            xs[lo:hi] = (model['pca'].transform(features[lo:hi])-model['center'])/model['scale']
    xs.flush()
    write_json(out/'PREPARATION_COMPLETE.json', dict(target_rows=len(features),
        source_geometry_retained=True, target_labels_read=False))
    print('POLAR PROTOCOL AND TARGET BASE GEOMETRY SAVED', flush=True)


if __name__ == '__main__':
    main()
