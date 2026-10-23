"""Bounded descriptive source/external diagnosis; no fit or model selection."""
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
from scipy.special import softmax

HERE = Path(__file__).resolve().parent
EXTERNAL = HERE.parent
FROZEN = EXTERNAL.parent
DEP = FROZEN.parent
ROOT = DEP.parent
spec = importlib.util.spec_from_file_location('frozen_external', EXTERNAL / 'run_external.py')
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)
PAIRS, GROUPS = ext.dep.PAIRS, ext.dep.GROUPS
PAIR_NAMES = [GROUPS[g] + '__' + GROUPS[h] for g, h in PAIRS]


def describe(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n_valid=0, mean=np.nan, q10=np.nan, q25=np.nan, median=np.nan,
                    q75=np.nan, q90=np.nan, positive_fraction=np.nan)
    q = np.quantile(values, [.1, .25, .5, .75, .9])
    return dict(n_valid=len(values), mean=float(values.mean()), q10=q[0], q25=q[1],
                median=q[2], q75=q[3], q90=q[4], positive_fraction=float((values > 0).mean()))


def scopes(table, rival=None, ports=False):
    yield 'all', 'all', np.ones(len(table), bool)
    y = table.class_id.to_numpy()
    for c in range(8):
        yield 'class', str(c), y == c
    if ports:
        for port in sorted(table.port.unique()):
            yield 'port', str(port), table.port.to_numpy() == port
    if rival is not None:
        for c in range(8):
            for r in np.unique(rival[y == c]):
                yield 'class_rival', f'{c}:{r}', (y == c) & (rival == r)


def reference_diagnostics(u, raw_d):
    """Reference probabilities use the same valid/common reference sets as D."""
    n = len(u)
    region = {key: np.full((n, 6, 8), np.nan)
              for key in ('neff', 'max_share', 'available', 'top_reference')}
    pair = {key: np.full((n, 7, 8), np.nan)
            for key in ('same_top', 'joint_neff', 'joint_max_share', 'common_refs', 'overlap')}
    error = 0.
    for g in range(6):
        for c in range(8):
            rows = np.flatnonzero(np.isfinite(u[:, g, c]).any(-1))
            if not len(rows):
                continue
            values = u[rows, g, c]
            p = softmax(np.where(np.isfinite(values), values, -np.inf), axis=-1)
            region['neff'][rows, g, c] = 1 / (p*p).sum(-1)
            region['max_share'][rows, g, c] = p.max(-1)
            region['available'][rows, g, c] = np.isfinite(values).sum(-1)
            region['top_reference'][rows, g, c] = p.argmax(-1)
    for j, (g, h) in enumerate(PAIRS):
        for c in range(8):
            common = np.isfinite(u[:, g, c]) & np.isfinite(u[:, h, c])
            rows = np.flatnonzero(common.any(-1))
            if not len(rows):
                continue
            pg = softmax(np.where(common[rows], u[rows, g, c], -np.inf), axis=-1)
            ph = softmax(np.where(common[rows], u[rows, h, c], -np.inf), axis=-1)
            product = pg * ph
            overlap = product.sum(-1)
            joint = product / overlap[:, None]
            count = common[rows].sum(-1)
            pair['same_top'][rows, j, c] = pg.argmax(-1) == ph.argmax(-1)
            pair['joint_neff'][rows, j, c] = 1 / (joint*joint).sum(-1)
            pair['joint_max_share'][rows, j, c] = joint.max(-1)
            pair['common_refs'][rows, j, c] = count
            pair['overlap'][rows, j, c] = overlap
            error = max(error, float(np.max(np.abs(np.log(count*overlap) - raw_d[rows, c, j]))))
    assert error < 1e-10, error
    return region, pair, error


def main():
    tick = time.perf_counter()
    weights = np.load(FROZEN / 'weights.npz')
    source_table = pd.read_csv(ext.pool.UP / 'table.csv', dtype={'mmsi': str, 'instance_id': str})
    source = np.load(DEP / 'features.npz')
    source_idx = source['idx']
    source_rows = np.flatnonzero(source_table.iloc[source_idx].role.to_numpy() == 'evaluation')
    sources = [('source', source_table.iloc[source_idx[source_rows]].reset_index(drop=True), source,
                source_rows, source_idx[source_rows]),
               ('external', pd.read_csv(ext.EXT / 'table.csv', dtype={'mmsi': str, 'instance_id': str}),
                np.load(EXTERNAL / 'features.npz'), np.arange(3075), np.arange(3075))]
    gains, contribution_summary, ref_regions, ref_pairs, top_refs, sample_rows = [], [], [], [], [], []
    detail_rows, direction_rows, checks = [], [], []
    for cohort, table, data, rows, prediction_rows in sources:
        assert len(table) == (485 if cohort == 'source' else 3075)
        y = table.class_id.to_numpy(dtype=int)
        index = np.arange(len(y))
        u, raw = data['u'][rows], data['D_joint'][rows]
        features = data['joint'][rows]
        if cohort == 'source':
            features = features[..., 6:]
        valid = data['pair_valid'][rows]
        region, pair, identity_error = reference_diagnostics(u, raw)
        for seed in (17, 43):
            path = FROZEN / f'predictions_{seed}.npz' if cohort == 'source' else EXTERNAL / f'predictions_seed{seed}.npz'
            saved = np.load(path)
            anchor, joint = saved['anchor'][prediction_rows], saved['joint'][prediction_rows]
            competing = anchor.copy()
            competing[index, y] = -np.inf
            rival = competing.argmax(-1)
            w = weights[f'joint_{seed}']
            contributions = (features[index, y] - features[index, rival]) * w
            old_margin = anchor[index, y] - anchor[index, rival]
            new_margin = joint[index, y] - joint[index, rival]
            error = float(np.max(np.abs(contributions.sum(-1) - (new_margin-old_margin))))
            assert error < 5e-6, error
            checks.append(dict(cohort=cohort, seed=seed, margin_replay_error=error, D_identity_error=identity_error))
            ca, cj = anchor.argmax(-1) == y, joint.argmax(-1) == y
            for scope, key, select in scopes(table, ports=True):
                old, new = ext.dep.metrics(y[select], anchor[select]), ext.dep.metrics(y[select], joint[select])
                gains.append(dict(cohort=cohort, seed=seed, scope=scope, key=key, n=int(select.sum()),
                                  rescued=int((select & cj & ~ca).sum()), harmed=int((select & ~cj & ca).sum()),
                                  net=int(cj[select].sum()-ca[select].sum()),
                                  anchor_OA=old['OA'], joint_OA=new['OA'], delta_OA=new['OA']-old['OA'],
                                  anchor_BA=old['BA'], joint_BA=new['BA'], delta_BA=new['BA']-old['BA']))
            for scope, key, select in scopes(table, rival=rival, ports=True):
                for j in range(8):
                    values = contributions.sum(-1) if j == 7 else contributions[:, j]
                    contribution_summary.append(dict(cohort=cohort, seed=seed, scope=scope, key=key,
                                                     pair='total' if j == 7 else PAIR_NAMES[j],
                                                     n=int(select.sum()), **describe(values[select])))
            for i in range(len(y)):
                record = dict(cohort=cohort, seed=seed, local_index=i, table_index=int(prediction_rows[i]),
                              obs_id=int(table.iloc[i].obs_id), instance_id=table.iloc[i].instance_id,
                              port=table.iloc[i].port, true_class=int(y[i]), rival=int(rival[i]),
                              anchor_prediction=int(anchor[i].argmax()), joint_prediction=int(joint[i].argmax()),
                              anchor_margin=float(old_margin[i]), joint_margin=float(new_margin[i]),
                              delta_margin=float(contributions[i].sum()))
                sample_rows.append(record)
                for j, name in enumerate(PAIR_NAMES):
                    detail_rows.append(dict(**record, pair=name, coefficient=float(w[j]),
                                            contribution=float(contributions[i,j]),
                                            D_true=float(raw[i,y[i],j]), D_rival=float(raw[i,rival[i],j]),
                                            true_valid=bool(valid[i,y[i],j]), rival_valid=bool(valid[i,rival[i],j])))

            # Same true-class/rival strata separate relationship shifts from class mix.
            for scope, key, select in scopes(table, rival=rival):
                for j, name in enumerate(PAIR_NAMES):
                    tv, rv = valid[index, y, j], valid[index, rival, j]
                    both = select & tv & rv
                    for metric, values in [('D_true', raw[index, y, j]), ('D_rival', raw[index, rival, j]),
                                           ('D_gap', raw[index,y,j]-raw[index,rival,j]),
                                           ('contribution_both_valid', contributions[:,j])]:
                        direction_rows.append(dict(cohort=cohort, seed=seed, scope=scope, key=key, pair=name,
                                                   metric=metric, n=int(select.sum()), both_valid=int(both.sum()),
                                                   **describe(values[both])))
            for scope, key, select in scopes(table):
                for j, name in enumerate(GROUPS):
                    for metric in ('neff', 'max_share', 'available'):
                        tv, rv = region[metric][index,j,y], region[metric][index,j,rival]
                        for side, values in [('true', tv), ('rival', rv), ('difference', tv-rv)]:
                            ref_regions.append(dict(cohort=cohort, seed=seed, scope=scope, key=key, region=name,
                                                    metric=metric, side=side, n=int(select.sum()), **describe(values[select])))
                    for side, candidate in [('true',y), ('rival',rival)]:
                        tops = region['top_reference'][index,j,candidate]
                        for c in np.unique(candidate[select]):
                            selected = select & (candidate == c) & np.isfinite(tops)
                            for ref in range(8):
                                top_refs.append(dict(cohort=cohort, seed=seed, scope=scope, key=key, region=name,
                                                     side=side, candidate_class=int(c), reference=ref,
                                                     source_table_index=int(data['bank'][c,ref]),
                                                     n_valid=int(selected.sum()), count=int((selected & (tops==ref)).sum())))
                for j, name in enumerate(PAIR_NAMES):
                    for metric, array in pair.items():
                        tv, rv = array[index,j,y], array[index,j,rival]
                        for side, values in [('true',tv), ('rival',rv), ('difference',tv-rv)]:
                            ref_pairs.append(dict(cohort=cohort, seed=seed, scope=scope, key=key, pair=name,
                                                  metric=metric, side=side, n=int(select.sum()), **describe(values[select])))
        print(cohort, 'complete', len(table), 'D identity error', identity_error, flush=True)

    for name, records in [('gains',gains), ('contributions',contribution_summary), ('reference_regions',ref_regions),
                          ('reference_pairs',ref_pairs), ('top_reference_counts',top_refs), ('raw_D_direction',direction_rows),
                          ('samples',sample_rows), ('sample_contributions',detail_rows)]:
        pd.DataFrame(records).to_csv(HERE / f'{name}.csv', index=False)
    receipt = dict(status='complete', cohorts={'source_evaluation':485,'external':3075}, seeds=[17,43],
                   no_training=True, no_parameter_selection=True, no_images=True, no_git=True,
                   rival='highest anchor score among classes excluding the ground truth; frozen for decomposition',
                   reference_neff='1/sum(softmax(u)^2), not entropy count',
                   pair_probability='softmax(u_g+u_h) on common valid references',
                   quantiles=[.1,.25,.5,.75,.9], historical_external_not_blind=True,
                   checks=checks, seconds=time.perf_counter()-tick)
    (HERE/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(pd.DataFrame(gains).query("scope == 'all'").to_string(index=False),flush=True)
    print(pd.DataFrame(contribution_summary).query("scope == 'all'")[[
        'cohort','seed','pair','mean','median','q10','q90']].to_string(index=False),flush=True)


def bias_control():
    """Replace joint by source-utility mean only, or subtract that fixed mean."""
    tick = time.perf_counter()
    table = pd.read_csv(ext.pool.UP / 'table.csv', dtype={'mmsi': str, 'instance_id': str})
    source = np.load(DEP / 'features.npz')
    idx = source['idx']
    roles = table.iloc[idx].role.to_numpy()
    utility = np.flatnonzero(roles == 'utility_train')
    evaluation = np.flatnonzero(roles == 'evaluation')
    assert len(utility) == 644 and len(evaluation) == 485
    source_features = source['joint'][..., 6:]
    mean_cf = source_features[utility].mean(0)
    weights = np.load(FROZEN / 'weights.npz')
    external_table = pd.read_csv(ext.EXT / 'table.csv', dtype={'mmsi': str, 'instance_id': str})
    external = np.load(EXTERNAL / 'features.npz')
    datasets = [('source', table.iloc[idx[evaluation]].reset_index(drop=True), idx[evaluation], source_features[evaluation]),
                ('external', external_table, np.arange(len(external_table)), external['joint'])]
    result, gain_rows, shares, margins, constants, components = [], [], [], [], [], []
    replay_error, identity_error = 0., 0.
    for seed in (17, 29, 43):
        w = weights[f'joint_{seed}']
        raw_components = mean_cf*w
        centered_components = raw_components - raw_components.mean(0, keepdims=True)
        bias = raw_components.sum(-1)
        bias -= bias.mean()
        for c in range(8):
            constants.append(dict(seed=seed, class_id=c, bias=float(bias[c])))
            for j, name in enumerate(PAIR_NAMES):
                components.append(dict(seed=seed, class_id=c, pair=name, source_utility_mean=float(mean_cf[c,j]),
                                       coefficient=float(w[j]), centered_bias_component=float(centered_components[c,j])))
        predictions = dict(source_utility_mean=mean_cf, coefficient=w, bias=bias)
        for cohort, tab, rows, features in datasets:
            path = FROZEN/f'predictions_{seed}.npz' if cohort == 'source' else EXTERNAL/f'predictions_seed{seed}.npz'
            old = np.load(path)
            anchor, saved_joint = old['anchor'][rows], old['joint'][rows]
            with ext.torch.inference_mode():
                delta = ext.dep.delta_from(ext.torch.from_numpy(features), ext.torch.from_numpy(w)).numpy()
            remaining = delta-bias
            reconstructed = anchor+delta
            replay_error = max(replay_error, float(np.max(np.abs(reconstructed-saved_joint))))
            identity_error = max(identity_error, float(np.max(np.abs(delta-(bias+remaining)))))
            output = dict(anchor=anchor, actual=saved_joint, bias_only=anchor+bias, residual_only=anchor+remaining)
            predictions.update({f'{cohort}_{key}':value for key,value in output.items()})
            predictions[f'{cohort}_idx'] = rows
            y = tab.class_id.to_numpy(dtype=int)
            index = np.arange(len(y))
            competing = anchor.copy()
            competing[index,y] = -np.inf
            rival = competing.argmax(-1)
            actual_margin = delta[index,y]-delta[index,rival]
            bias_margin = bias[y]-bias[rival]
            residual_margin = actual_margin-bias_margin
            for scope,key,mask in scopes(tab, ports=True):
                base_correct = anchor.argmax(-1)==y
                for arm,score in output.items():
                    correct = score.argmax(-1)==y
                    result.append(dict(cohort=cohort, seed=seed, scope=scope, key=key, arm=arm,
                                       **ext.dep.metrics(y[mask],score[mask])))
                    gain_rows.append(dict(cohort=cohort, seed=seed, scope=scope,key=key,arm=arm,n=int(mask.sum()),
                                          rescued=int((mask&correct&~base_correct).sum()),harmed=int((mask&~correct&base_correct).sum()),
                                          net=int(correct[mask].sum()-base_correct[mask].sum())))
            for scope,key,mask in scopes(tab,rival=rival):
                dv = delta[mask].astype(float)
                bv = np.broadcast_to(bias,dv.shape).astype(float)
                denom = float((dv*dv).sum())
                md,mb = actual_margin[mask].astype(float),bias_margin[mask].astype(float)
                margin_denom = float((md*md).sum())
                shares.append(dict(cohort=cohort,seed=seed,scope=scope,key=key,n=int(mask.sum()),
                                   delta_rms=float(np.sqrt(np.mean(dv*dv))),bias_rms=float(np.sqrt(np.mean(bv*bv))),
                                   residual_rms=float(np.sqrt(np.mean((dv-bv)**2))),
                                   bias_energy_ratio=float((bv*bv).sum()/denom) if denom>0 else np.nan,
                                   bias_direction_projection=float((dv*bv).sum()/denom) if denom>0 else np.nan,
                                   margin_bias_energy_ratio=float((mb*mb).sum()/margin_denom) if margin_denom>0 else np.nan,
                                   margin_bias_projection=float((md*mb).sum()/margin_denom) if margin_denom>0 else np.nan))
                for component,values in [('actual',actual_margin),('bias',bias_margin),('residual',residual_margin)]:
                    margins.append(dict(cohort=cohort,seed=seed,scope=scope,key=key,component=component,
                                        n=int(mask.sum()),**describe(values[mask])))
        np.savez_compressed(HERE/f'bias_predictions_seed{seed}.npz',**predictions)
    # One combined closure check across both cohorts and all three frozen heads.
    assert max(replay_error,identity_error)<5e-6, (replay_error,identity_error)
    for name,records in [('bias_metrics',result),('bias_gains',gain_rows),('bias_shares',shares),
                         ('bias_margins',margins),('bias_constants',constants),('bias_components',components)]:
        pd.DataFrame(records).to_csv(HERE/f'{name}.csv',index=False)
    receipt=dict(status='complete',utility_count=644,mean_origin=str(DEP/'features.npz'),
                 no_training=True,no_selection=True,no_external_mean=True,seeds=[17,29,43],
                 new_parameter_count=0,replay_error=replay_error,identity_error=identity_error,
                 ratio_warning='energy ratio and projection are descriptive, may exceed one or be negative; zero delta is NA',
                 seconds=time.perf_counter()-tick)
    (HERE/'bias_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(pd.DataFrame(result).query("scope=='all'")[[
        'cohort','seed','arm','OA','BA','NLL','Brier']].to_string(index=False),flush=True)
    print(pd.DataFrame(shares).query("scope=='all'").to_string(index=False),flush=True)


if __name__ == '__main__':
    if '--bias-only' not in sys.argv:
        main()
    bias_control()
