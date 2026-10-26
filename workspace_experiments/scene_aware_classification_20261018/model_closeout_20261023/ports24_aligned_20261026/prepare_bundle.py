"""Reuse the fixed eight-class source roles and held acquisition coverage."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOCAL = ROOT/'local_evidence_transport'
UP = ROOT/'regional_utility_pilot/upstream'
EXT = LOCAL/'external_confirmation'
OLD = ROOT/'final_m3_coverage_20261026'
PACKAGE = Path('E:/临时会话/sar_m123_standalone')
sys.path.insert(0, str(PACKAGE))
from sar_m123.aligned import raw_features, summary_features
from sar_m123 import tokens


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def retrieve(x, mass, bank, table, query):
    b = tokens.body_descriptor(x[bank.ravel()], mass[bank.ravel()]).reshape(8, 8, 64)
    q = tokens.body_descriptor(x, mass)
    sim = np.einsum('nd,ckd->nck', q, b)
    sim[query[:, None, None] == table.mmsi.to_numpy(str)[bank]] = -np.inf
    rank = np.argsort(-sim, axis=-1)[..., :2]
    return np.take_along_axis(np.broadcast_to(bank, sim.shape), rank, axis=-1)


def main():
    source = pd.read_csv(UP/'table.csv', dtype={'mmsi': str})
    target = pd.read_csv(EXT/'table.csv', dtype={'mmsi': str})
    for field in ('obs_id', 'mmsi', 'pass_group'):
        assert not set(source[field]) & set(target[field]), field
    for row, cohort in ((source, 'development'), (target, 'coverage3075')):
        row['cohort'] = cohort
    st, tt = load(LOCAL/'tokens.npz'), load(EXT/'tokens.npz')
    np.testing.assert_array_equal(st['instance_ids'].astype(str), source.instance_id)
    np.testing.assert_array_equal(tt['instance_ids'].astype(str), target.instance_id)
    bank_old = load(LOCAL/'projection_layer1.npz')['bank'].astype(int)
    assert source.iloc[bank_old.ravel()].upstream_role.eq('fit_train').all()
    keep = ~source.upstream_role.eq('fit_excluded_mmsi').to_numpy()
    index = np.flatnonzero(keep)
    remap = np.full(len(source), -1, int)
    remap[index] = np.arange(len(index))
    bank = remap[bank_old]
    assert bank.min() >= 0
    frame = source.iloc[index].copy().reset_index(drop=True)
    frame['source_index'] = index
    mask = frame.role.eq('strategy_fit')
    frame.loc[mask, 'role'] = frame.loc[mask, 'upstream_role']
    truth = frame.copy()
    frame.loc[frame.role.eq('evaluation'), 'class_id'] = -1
    projection = load(PACKAGE/'assets/m3/projection.npz')
    joint = load(LOCAL/'spatial_frequency_joint_pooling/features.npz')
    freq = load(LOCAL/'frequency_conditioned_matching/features.npz')
    np.testing.assert_array_equal(joint['ids'], freq['ids'])
    # Higher-order matching features are needed only for bank/utility/selection/eval.
    # Other fit rows train the base using real raw/summary inputs, never these slots.
    n = len(source)
    feat = dict(mass=st['mass'], mean=(st['layer1']-projection['mean'])@projection['components'].T,
                joint=np.zeros((n, 2, 20, 96), np.float32),
                e_image=np.zeros((n, 2, 20, 24), np.float32),
                coverage=np.zeros((n, 2, 20, 3), np.float32))
    active = np.zeros(n, bool)
    active[joint['ids']] = True
    feat['joint'][joint['ids']] = joint['joint']
    feat['e_image'][freq['ids']] = freq['e_image']
    feat['coverage'][freq['ids']] = freq['coverage']
    feat['x'] = tokens.unit(feat['mean']).astype(np.float32)
    feat['x'][feat['mass'] <= 0] = 0
    required = source.role.ne('strategy_fit').to_numpy(copy=True)
    required[bank_old.ravel()] = True
    assert active[required].all()
    feat = {k: v[index] for k, v in feat.items()}
    refs = retrieve(feat['x'], feat['mass'], bank, frame, frame.mmsi.to_numpy(str))
    sc = load(PACKAGE/'assets/m3/covariance_scaler.npz')
    payload = dict(obs_id=frame.obs_id.to_numpy(), raw=raw_features(st, st['valid'])[index],
                   aux=summary_features(load(UP/'summaries.npz'))[index], bank=bank, refs=refs,
                   local_available=active[index], **feat,
                   **{'covariance_'+k: v for k, v in sc.items()})
    assert payload['raw'].shape == (len(frame), 1538)
    assert payload['aux'].shape == (len(frame), 176)
    out = HERE/'bundle'
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out/'table.csv', index=False)
    truth.to_csv(HERE/'source_truth.csv', index=False)
    np.savez_compressed(out/'features.npz', **payload)
    energy = load(OLD/'energy_external.npz')
    jext = load(LOCAL/'frozen_joint_coverage/joint_external.npz')
    for data in (energy, jext):
        np.testing.assert_array_equal(data['instance_ids'].astype(str), target.instance_id)
    ef = dict(mass=tt['mass'], mean=(tt['layer1']-projection['mean'])@projection['components'].T,
              joint=jext['joint'], e_image=energy['e_image'], coverage=energy['coverage'])
    ef['x'] = tokens.unit(ef['mean']).astype(np.float32)
    ef['x'][ef['mass'] <= 0] = 0
    target.to_csv(HERE/'target_truth.csv', index=False)
    target.role = 'evaluation'
    target.class_id = -1
    target.to_csv(out/'target_table.csv', index=False)
    allx, allm = np.concatenate((feat['x'], ef['x'])), np.concatenate((feat['mass'], ef['mass']))
    alltable = pd.concat((frame, target), ignore_index=True)
    er = retrieve(allx, allm, bank, alltable, alltable.mmsi.to_numpy(str))[len(frame):]
    oldrefs = load(LOCAL/'frozen_joint_coverage/expanded_references.npz')
    np.testing.assert_array_equal(er, bank[np.arange(8)[None, :, None], oldrefs['nearest2']])
    np.savez_compressed(out/'external.npz', obs_id=target.obs_id.to_numpy(), refs=er,
                        raw=raw_features(tt, tt['valid']),
                        aux=summary_features(load(EXT/'summaries.npz')), **ef)
    for seed in (17, 29, 43):
        ck = torch.load(PACKAGE/'assets/m3'/f'seed_{seed}.pt', map_location='cpu', weights_only=False)
        dest = HERE/'initial_assets/warm'/f'seed{seed}'
        dest.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state=ck['metric_state']), dest/'both_metric.pt')
        torch.save(ck['metric_scaler'], dest/'metric_scaler.pt')
    role_counts = truth.groupby(['role', 'class_id']).size().unstack(fill_value=0)
    role_counts.to_csv(HERE/'source_class_counts.csv')
    target_truth = pd.read_csv(HERE/'target_truth.csv')
    target_truth.groupby(['class_id', 'class_name']).agg(n=('obs_id', 'size'), ships=('mmsi', 'nunique'),
        passes=('pass_group', 'nunique'), ports=('port', 'nunique')).to_csv(HERE/'target_class_counts.csv')
    write(HERE/'preparation.json', dict(source_rows=len(frame), held_coverage=len(target),
        roles=frame.role.value_counts().to_dict(), classes=8, bank=64, refs_per_class=2,
        raw_dim=1538, aux_dim=176, excluded_internal_mmsi=int((~keep).sum()),
        source_held_overlap=False, all_eval_labels_masked=True,
        matching_feature_note='Not cached for unused base-only fit rows. No fitting/evaluation uses those rows for M3.',
        scope='Fixed historical source split and held acquisitions; not a new blind test or 24 LOPO retrainings.'))
    print(role_counts.to_string(), flush=True)
    print('PREPARED', len(frame), len(target), flush=True)


if __name__ == '__main__':
    main()
