"""Local paths and read-only reuse of the frozen pipeline."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')
import sys
sys.dont_write_bytecode = True
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_joint as run
sys.path.insert(0, str(HERE.parent / 'component_isolation'))
import run_components as source
parent, strong = run.parent, run.strong
sys.path.insert(0, str(parent.HERE))
import extract_moments as extract
SCORE = [f's{c}' for c in range(8)]
BASE = [f'b{c}' for c in range(8)]
ARM = 'two_sided_attention'


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def event(name, **values):
    with (HERE / 'stages.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(time_utc=now(), event=name, **values), ensure_ascii=False) + '\n')


def cache_load():
    with np.load(parent.HERE / 'signed_features.npz') as z:
        return {k: z[k] for k in ('instance_ids', 'reader_names', 'mean_pol', 'moments_pol', 'valid')}


def read_reference(role):
    roles = pd.read_csv(HERE.parent / 'predictions.csv', usecols=['role', 'arm'])
    keep = set(np.flatnonzero(roles.role.eq(role) & roles.arm.eq('additive')) + 1)
    return pd.read_csv(HERE.parent / 'predictions.csv',
        skiprows=lambda i: i != 0 and i not in keep, dtype={'mmsi': str}, float_precision='round_trip')


def restore(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = run.Head('additive').eval().requires_grad_(False)
    model.load_state_dict(ck['state'])
    return ck, model


def pooled(pred):
    rows = []
    for (role, arm, seed), g in pred.groupby(['role', 'arm', 'seed']):
        y = g.class_id.to_numpy()
        m = parent.assessment(y, g[SCORE].to_numpy())
        b = parent.assessment(y, g[BASE].to_numpy())
        rows.append(dict(role=role, arm=arm, seed=int(seed), **m,
            net=int(g.net.sum()), rescued=int(g.net.eq(1).sum()), harmed=int(g.net.eq(-1).sum()),
            unique_instances=g.instance_id.nunique(), unique_mmsi=g.mmsi.nunique(),
            **{f'delta_{k}_B': m[k]-b[k] for k in ['OA', 'BA', 'NLL', 'Brier']}))
    return pd.DataFrame(rows)


def frame(table, ix, valid, base, score, **meta):
    return source.predict_frame(table, ix, valid, base, score, meta)
