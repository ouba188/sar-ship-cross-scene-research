"""One-time source migration, not a runtime dependency.

Copy the already validated numerical operators without rewriting their math.
The generated module has only package-local and standard-library imports.
"""
import argparse
import ast
from pathlib import Path


def migrate(root, destination):
    original = root/'oss1_m123_replication_20261026/train_m3.py'
    source = original.read_text(encoding='utf-8-sig')
    names = {'write_json', 'metrics', 'exact_j', 'state', 'scale_fit',
             'transform', 'local_raw', 'gate_raw', 'JointHead', 'Gate',
             'balanced_weights', 'fit_head', 'Matcher', 'choose_alpha', 'fit_gate'}
    pieces = [ast.get_source_segment(source, n) for n in ast.parse(source).body
              if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    header = '''"""Frozen public-four-class training operators, migrated without math changes.

Provenance: oss1_m123_replication_20261026/train_m3.py.
No historical experiment directory is needed to import or execute this module.
"""
import json
from fractions import Fraction
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import softmax, logsumexp
from . import tokens
from .m3 import PairMetric, PolarHead, sinkhorn, regional_descriptors, pair_stats

C = 4


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def frozen_model(model, checkpoint, device):
    model.load_state_dict(checkpoint['state'])
    return model.to(device).eval().requires_grad_(False)
'''
    destination.write_text(header+'\n\n'+'\n\n\n'.join(pieces)+'\n', encoding='utf-8')
    alignment = (root/'m3_aligned_training_20261026/run.py').read_text(encoding='utf-8-sig')
    pieces = [ast.get_source_segment(alignment, n) for n in ast.parse(alignment).body
              if isinstance(n, (ast.FunctionDef, ast.ClassDef))
              and n.name in {'FrozenDecision', 'fit_metric'}]
    header = '''"""Accepted metric alignment against the actual frozen decision path.

Provenance: m3_aligned_training_20261026/run.py. Numerical code unchanged.
"""
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from . import train_ops as m3
from . import train_ops as tr
'''
    destination.with_name('training_alignment.py').write_text(
        header+'\n\n'+'\n\n\n'.join(pieces)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-root', required=True, type=Path)
    args = parser.parse_args()
    migrate(args.legacy_root, Path(__file__).resolve().parents[1]/'sar_m123/train_ops.py')
