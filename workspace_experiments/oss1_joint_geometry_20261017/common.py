"""Shared paths and the existing public full-classifier protocol."""
import importlib.util
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='2'

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'scene_cross_dataset_redesign_20261014'
FULL=ROOT/'public_full_20261014'
MANIFEST=Path('E:/Hermes/paper_tracking/sensitivity/opensarship1_transfer_20261007/target_manifest.csv')
PIXELS=ROOT/'public_expansion/oss1_inputs128.npy'
sys.path.insert(0,'E:/Hermes/paper_tracking/sensitivity/oof_reliability_check')
spec=importlib.util.spec_from_file_location('public_full_reader',FULL/'run.py')
public=importlib.util.module_from_spec(spec)
spec.loader.exec_module(public)


def folds():
    return public.episode_paths('oss1')
