"""Frozen four-arm extension; no fitting or target-based model selection."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util
import json
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.special import softmax
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
LOCAL = BASE/'local_evidence_transport'
EXT = LOCAL/'external_confirmation'
COV = LOCAL/'frozen_joint_coverage'
UP = BASE/'regional_utility_pilot/upstream'
ANCHOR = BASE/'evidence_preserving_redesign/anchored_polar_adjustment_20261025'
PACKAGE = Path('E:/临时会话/sar_m123_standalone')
sys.path.insert(0, str(PACKAGE))
from sar_m123 import m3, tokens, frequency
SEEDS = (17, 29, 43)
ARMS = ('old_both', 'B_fixed', 'spatial', 'frequency')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def metrics(y, score):
    p = softmax(score, axis=1)
    return dict(n=len(y), OA=100*accuracy_score(y, score.argmax(1)),
                BA=100*balanced_accuracy_score(y, score.argmax(1)),
                NLL=log_loss(y, p, labels=np.arange(8)),
                Brier=float(np.square(p-np.eye(8)[y]).sum(1).mean()))


class Gate(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, 12), torch.nn.GELU(), torch.nn.Linear(12, 1))


@torch.inference_mode()
def classify(runtime, feat, local, refs, base, seed):
    model = runtime._load(seed)
    ck = model['ck']
    x = torch.from_numpy(m3.standardize(m3.local_features(local, base), ck['readout_scaler']))
    dv, dh = model['vv'](x).numpy(), model['vh'](x).numpy()
    regions = m3.regional_descriptors(feat, runtime.covariance_scaler)
    raw, available = m3.gate_features(regions, runtime.bank_regions, refs, base, dv, dh)
    anchor = ck['alpha_anchor']
    payload = dict(base=base, delta_vv=dv, delta_vh=dh, available=available, raw=raw, refs=refs)
    fixed = np.broadcast_to([anchor, 1-anchor], available.shape).astype(np.float32)*available
    payload['B_fixed'] = base+fixed[:, :1]*dv+fixed[:, 1:]*dh
    payload['B_fixed_weight'] = fixed
    for arm, dim in (('spatial', 41), ('frequency', 185)):
        gc = torch.load(ANCHOR/'models'/f'{arm}_{seed}.pt', map_location='cpu', weights_only=False)
        assert abs(float(gc['receipt']['alpha_anchor'])-anchor) < 1e-10
        gate = Gate(dim).eval()
        gate.load_state_dict(gc['state'])
        xx = torch.from_numpy(m3.standardize(raw, gc['scaler']))
        r = gate.net(xx[..., :dim]).mean((1, 2))
        alpha = (anchor+.25*r.tanh()).clamp(0, 1).numpy()
        alpha[~available.all(1)] = anchor
        weight = np.stack((alpha, 1-alpha), 1)*available
        payload[arm] = base+weight[:, :1]*dv+weight[:, 1:]*dh
        payload[arm+'_weight'] = weight
        assert abs(alpha-anchor).max() <= .250001
        assert not np.any(weight[~available])
    return payload


def development(runtime, source):
    joint = np.load(LOCAL/'spatial_frequency_joint_pooling/features.npz')
    energy = np.load(LOCAL/'frequency_conditioned_matching/features.npz')
    np.testing.assert_array_equal(joint['ids'], energy['ids'])
    ix = np.flatnonzero(source.iloc[joint['ids']].role.eq('evaluation').to_numpy())
    up = joint['ids'][ix]
    assert len(up) == 485
    feat = {k: joint[k][ix] for k in ('mass', 'mean', 'joint')}
    feat.update(e_image=energy['e_image'][ix], coverage=energy['coverage'][ix])
    checks = []
    for seed in SEEDS:
        old = np.load(LOCAL/'pair_conditioned_joint_metric/results'/f'pair_joint_{seed}.npz')
        np.testing.assert_array_equal(old['ids'], joint['ids'])
        base = np.load(UP/f'strategies_seed{seed}.npz')['logits'][up, 2]
        original = np.load(BASE/'evidence_preserving_redesign/query_polar_gate_20261025/results'/f'features_{seed}.npz')
        lookup = {int(v): i for i, v in enumerate(original['up_index'])}
        order = np.array([lookup[int(v)] for v in up])
        rawrefs = np.load(LOCAL/'scores_layer1.npz')['refs'][up]
        lookup = {int(v): i for i, v in enumerate(runtime.bank['source_index'])}
        refs = np.array([lookup[int(v)] for v in rawrefs.ravel()]).reshape(rawrefs.shape)
        result = classify(runtime, feat, old['local'][ix], refs, base, seed)
        err = float(abs(result['raw']-original['raw'][order]).max())
        assert err < 2e-4, ('gate_inputs', seed, err)
        for arm in ('B_fixed', 'spatial', 'frequency'):
            if arm == 'B_fixed':
                archive = np.load(BASE/'evidence_preserving_redesign/hybrid_polar_fusion_20261025/results'/f'hybrid_source_{seed}.npz')
            else:
                archive = np.load(ANCHOR/'results'/f'{arm}_{seed}.npz')
            lookup = {int(v): i for i, v in enumerate(archive['up_index'])}
            expected = archive['score'][[lookup[int(v)] for v in up]]
            error = float(abs(result[arm]-expected).max())
            assert error < 2e-5, (arm, seed, error)
            assert np.array_equal(result[arm].argmax(1), expected.argmax(1))
            checks.append(dict(seed=seed, arm=arm, max_score_error=error, max_raw_error=err))
        print('DEVELOPMENT_REPLAY', seed, checks[-1], flush=True)
    write_json(HERE/'replay_checks.json', checks)


def extract_energy(table, tok):
    path = HERE/'energy_external.npz'
    if path.exists():
        cache = np.load(path)
        np.testing.assert_array_equal(cache['instance_ids'].astype(str), table.instance_id.astype(str))
        return {k: cache[k].copy() for k in ('e_image', 'coverage')}
    ex = load('final_coverage_extract', COV/'extract.py')
    snapshots = {int(v['obs_id']): v for v in json.loads((ex.joint.geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    output = dict(e_image=np.zeros((len(table), 2, 20, 24), np.float32),
                  coverage=np.zeros((len(table), 2, 20, 3), np.float32))
    start = last = time.perf_counter()
    for j in range(len(table)):
        for p, pol in enumerate(('vv', 'vh')):
            if not tok['valid'][j, p]:
                continue
            pix, fp, g, placement = ex.joint.geo.native(table, snapshots, j, p)
            mask = tokens.slots(g, fp)
            np.testing.assert_allclose(mask.sum((-1, -2)), tok['mass'][j, p], rtol=1e-6, atol=1e-6)
            with Image.open(table.iloc[j]['image_'+pol]) as im:
                raw = np.asarray(im.convert('L'))
            np.testing.assert_array_equal(frequency.place(raw, placement), pix)
            out = frequency.extract(raw, mask, placement)
            for key in output:
                output[key][j, p] = out[key]
        if j == 0 or time.perf_counter()-last > 25:
            print('NATIVE_FREQUENCY', j+1, '/', len(table), 'seconds', round(time.perf_counter()-start, 1), flush=True)
            last = time.perf_counter()
    np.savez_compressed(path, **output, instance_ids=table.instance_id.to_numpy(str))
    write_json(HERE/'extraction.json', dict(n=len(table), seconds=time.perf_counter()-start,
        new_backbone_extraction=False, source='original native PNG before crop/pad; frozen kernels'))
    return output


def main():
    torch.set_num_threads(2)
    (HERE/'results').mkdir(exist_ok=True)
    runtime = m3.M3(PACKAGE/'assets')
    source = pd.read_csv(UP/'table.csv', dtype={'mmsi': str})
    development(runtime, source)
    table = pd.read_csv(EXT/'table.csv', dtype={'mmsi': str, 'instance_id': str})
    for key in ('obs_id', 'mmsi', 'pass_group'):
        assert not set(table[key]) & set(source[key]), key
    table.to_csv(HERE/'table.csv', index=False)
    tok = np.load(EXT/'tokens.npz')
    joint = np.load(COV/'joint_external.npz')
    np.testing.assert_array_equal(tok['instance_ids'].astype(str), table.instance_id)
    np.testing.assert_array_equal(joint['instance_ids'].astype(str), table.instance_id)
    feat = extract_energy(table, tok)
    mean = (tok['layer1']-runtime.projection['mean'])@runtime.projection['components'].T
    mean[tok['mass'] <= 0] = 0
    feat.update(mean=mean, joint=joint['joint'], mass=tok['mass'])
    plan = np.load(COV/'expanded_references.npz')
    refs = (np.arange(8)[None, :, None]*8+plan['nearest2']).astype(int)
    np.testing.assert_array_equal(runtime.bank['source_index'].reshape(8, 8), plan['bank'])
    x = tokens.unit(mean)
    np.testing.assert_array_equal(runtime.retrieve(dict(x=x, mass=tok['mass'])), refs)
    rows = []
    for seed in SEEDS:
        old = np.load(COV/'results'/f'expanded_nearest2_{seed}.npz')
        base = np.load(EXT/f'strategies_seed{seed}.npz')['logits'][:, 2].astype(np.float32)
        payload = classify(runtime, feat, old['both_local'], refs, base, seed)
        payload['old_both'] = old['both']
        np.savez_compressed(HERE/'results'/f'seed{seed}.npz', **payload, instance_ids=table.instance_id.to_numpy(str))
        for arm in ARMS:
            rows.append(dict(seed=seed, arm=arm, **metrics(table.class_id.to_numpy(int), payload[arm])))
        print('EVALUATED', seed, rows[-4:], flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE/'metrics.csv', index=False)
    frame.groupby('arm')[['OA', 'BA', 'NLL', 'Brier']].agg(['mean', 'std']).to_csv(HERE/'summary.csv')
    write_json(HERE/'evaluation_complete.json', dict(n=len(table), ships=int(table.mmsi.nunique()),
        passes=int(table.pass_group.nunique()), ports=int(table.port.nunique()),
        new_training=0, seeds=SEEDS, arms=ARMS, independent_blind_test=False))
    print(frame.groupby('arm')[['OA', 'BA', 'NLL', 'Brier']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
