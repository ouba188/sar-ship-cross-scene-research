"""Fixed-matcher M3 readout refit on anchored_raw_summary; no M2 fit."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import argparse
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
ROOT = NEW.parent
TRANSFER = ROOT / 'public_m123_cross_dataset_20261026'
REPAIR = TRANSFER / 'normalization_repair_20261026'
EVIDENCE = ROOT / 'paper_evidence_completion_20261026'
OLD = ROOT / 'readout_bottleneck_20261026/strong_base_refit'
spec = importlib.util.spec_from_file_location('previous_refit', OLD / 'run.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
m3, load, write = previous.m3, previous.load, previous.write
SEEDS = (17, 29, 43)
CANDIDATES = ('anchored_raw_summary', 'raw_summary')
ARMS = ('bare', 'joint', 'vv', 'vh', 'B_fixed', 'frequency')
CLASSES = ('bulk', 'general', 'chemical', 'container')
DIRECTIONS = (('oss1', 'oss2'), ('oss2', 'oss1'))


def ready():
    missing = [str(NEW / source / f'{side}_predictions.npz')
               for source, _ in DIRECTIONS for side in ('source', 'cross')
               if not (NEW / source / f'{side}_predictions.npz').exists()]
    if missing:
        print('WAITING_FOR_CANDIDATES', json.dumps(missing, ensure_ascii=False), flush=True)
        return False
    return True


def zero_check(model, x, base, name, checks, meta):
    with torch.inference_mode():
        delta = model(torch.as_tensor(x)).numpy()
    np.testing.assert_array_equal(delta, np.zeros_like(delta))
    np.testing.assert_array_equal(base + delta, base)
    checks.append(dict(**meta, check=f'{name}_zero_step_identity', max_error=0.0))


def train(candidate):
    if not ready():
        return False
    torch.set_num_threads(2)
    tick = time.perf_counter()
    dest = HERE / candidate
    dest.mkdir(exist_ok=True)
    receipts, checks = [], []
    write(dest / 'PROTOCOL.json', dict(
        candidate=candidate+'_seed{seed}', sources=['oss1', 'oss2'], seeds=SEEDS,
        interpretation=('preserves original s12 anchor' if candidate=='anchored_raw_summary'
                        else 'raw_summary replaces M2 scores; does not preserve original M1/M2 chain'),
        development_status=('predeclared downstream candidate' if candidate=='anchored_raw_summary'
                            else 'added after intermediate target results; development diagnostic, not blind test'),
        base='new candidate is s12; no M2 training',
        frozen='historical source-trained PairMetric; 32 references; local and non-base gate descriptors',
        trained='JointHead, VV, centeredVH, source alpha and frequency gate only',
        roles='utility_train for optimization/scalers; selection for checkpoint/alpha',
        head_steps=200, gate_steps=100, optimizer='AdamW lr=.003 weight_decay=.01',
        alpha_grid=[k / 10 for k in range(11)], checkpoint='every5steps; earliest strict exact-J optimum including0',
        target_labels='not loaded during fits; evaluate only after all six groups saved',
        scope='readout refit conditional on frozen matcher; not end-to-end M3 refit',
        base_predictions_oof=False, cpu_threads=2))
    for source, target in DIRECTIONS:
        direction = f'{source}-to-{target}'
        source_root = ROOT / f'{source}_m123_replication_20261026'
        table = pd.read_csv(source_root / 'stack_table.csv', dtype={'mmsi': str})
        y = table.class_id.to_numpy(int, copy=True)
        y[table.role.eq('evaluation')] = -1
        utility = np.flatnonzero(table.role.eq('utility_train'))
        selection = np.flatnonzero(table.role.eq('selection'))
        cross = pd.read_csv(TRANSFER / direction / 'target_table.csv',
                            usecols=['obs_id', 'mmsi', 'pass_group', 'role'], dtype={'mmsi': str})
        sbank, cbank = (load(NEW / source / f'{side}_predictions.npz') for side in ('source', 'cross'))
        np.testing.assert_array_equal(sbank['obs_id'], table.obs_id)
        np.testing.assert_array_equal(cbank['obs_id'], cross.obs_id)
        # The source repair cache inherits stack_table order and has no obs_id field.
        # Its source rows are independently recorded in the combined repair table.
        combined = pd.read_csv(REPAIR / direction / 'table.csv', dtype={'mmsi': str})
        source_rows = combined.loc[combined.dataset_side.eq('source')]
        np.testing.assert_array_equal(source_rows.original_obs_id, table.obs_id)
        for seed in SEEDS:
            meta = dict(source=source, seed=seed, candidate=candidate)
            out = dest / direction / f'seed{seed}'
            out.mkdir(parents=True, exist_ok=True)
            src = load(REPAIR / f'source_{source}/seed{seed}/predictions.npz')
            dst = load(REPAIR / direction / f'seed{seed}/predictions.npz')
            np.testing.assert_array_equal(dst['obs_id'], cross.obs_id)
            key = f'{candidate}_seed{seed}'
            sb, cb = sbank[key].astype(np.float32), cbank[key].astype(np.float32)
            base = np.concatenate([sb, cb])
            assert base.shape == (len(table) + len(cross), 4) and np.isfinite(base).all()
            local = np.concatenate([src['local'], dst['local']])
            available = np.concatenate([src['available'], dst['available']]).astype(np.float32)
            cached_gate = np.concatenate([src['gate_raw'], dst['gate_raw']])
            raw = m3.local_raw(local, base)
            centered = base - base.mean(1, keepdims=True)
            competitor = np.stack([np.max(np.delete(centered, c, axis=1), axis=1) for c in range(4)], 1)
            np.testing.assert_array_equal(raw[..., :12], local.reshape(len(base), 4, 12))
            np.testing.assert_allclose(raw[..., 12:], np.stack([centered, competitor, centered-competitor], -1), rtol=0, atol=0)
            scaler = m3.scale_fit(raw, utility)
            np.testing.assert_allclose(scaler['mu'], raw[utility].mean((0, 1)), rtol=0, atol=0)
            x = m3.transform(raw, scaler)
            torch.manual_seed(seed)
            joint = m3.JointHead()
            zero_check(joint, x, base, 'joint', checks, meta)
            dj, sj = m3.fit_head('joint_refit', joint, x, base, y, utility, selection, out, steps=200)
            null = m3.transform(np.zeros((1, 1, 15)), scaler).reshape(15)
            deltas, steps = [], []
            for pol, name in enumerate(('vv', 'vh_centered')):
                torch.manual_seed(seed)
                head = m3.PolarHead(pol, centered=bool(pol), null=null)
                torch.nn.init.zeros_(head.net[-1].weight)
                torch.nn.init.zeros_(head.net[-1].bias)
                zero_check(head, x, base, name, checks, meta)
                delta, step = m3.fit_head(name, head, x, base, y, utility, selection, out, steps=200)
                deltas.append(delta)
                steps.append(step)
            dv, dh = deltas
            alpha = m3.choose_alpha(base, dv, dh, available, y, selection)
            weights = available * np.array([alpha, 1-alpha], np.float32)
            fixed = base + weights[:, :1]*dv + weights[:, 1:]*dh
            gr = previous.gate_raw(cached_gate, base, dv, dh, available)
            np.testing.assert_array_equal(gr[..., 11:], cached_gate[..., 11:])
            np.testing.assert_allclose(gr[..., 0], centered, rtol=0, atol=0)
            np.testing.assert_array_equal(gr[..., 1], dv)
            np.testing.assert_array_equal(gr[..., 2], dh)
            for j, score in enumerate((base, base+dv, base+dh)):
                competitor = np.stack([np.max(np.delete(score, c, axis=1), axis=1) for c in range(4)], 1)
                np.testing.assert_allclose(gr[..., 3+j], score-competitor, rtol=0, atol=0)
                np.testing.assert_allclose(gr[..., 6+j], m3.softmax(score, axis=1), rtol=0, atol=0)
            np.testing.assert_array_equal(gr[..., 9:], np.concatenate([
                np.broadcast_to(available[:, None, :], (len(base), 4, 2)), cached_gate[..., 11:]], -1))
            gate_scaler = m3.scale_fit(gr, utility)
            torch.manual_seed(seed)
            initial_gate = m3.Gate(185, alpha)
            with torch.inference_mode():
                initial_weights = initial_gate(torch.tensor(m3.transform(gr, gate_scaler)), torch.tensor(available)).numpy()
            np.testing.assert_array_equal(initial_weights, weights)
            np.testing.assert_array_equal(base+initial_weights[:, :1]*dv+initial_weights[:, 1:]*dh, fixed)
            checks.append(dict(**meta, check='frequency_zero_step_equals_B_fixed', max_error=0.0))
            freq, fw, gs = m3.fit_gate('frequency', seed, gr, base, dv, dh, available,
                                       y, utility, selection, alpha, out)
            saved_gate_scaler = torch.load(out/'frequency.pt', map_location='cpu', weights_only=False)['scaler']
            for field in gate_scaler:
                np.testing.assert_array_equal(saved_gate_scaler[field], gate_scaler[field])
            np.testing.assert_allclose(freq, base+fw[:, :1]*dv+fw[:, 1:]*dh, rtol=0, atol=0)
            checks.append(dict(**meta, check='obsid_base_fields_and_scalers', max_error=0.0,
                               fixed_local=True, fixed_gate_descriptors=True, source_ids_via_combined_table=True))
            scores = dict(bare=base, s12=base, joint=base+dj,
                vv=base+dv*available[:, :1], vh=base+dh*available[:, 1:], B_fixed=fixed, frequency=freq)
            extra = dict(delta_joint=dj, delta_vv=dv, delta_vh=dh, available=available,
                         weight_frequency=fw, weight_B_fixed=weights, gate_raw=gr, local_raw=raw)
            for side, sl, tab in [('source', slice(0, len(table)), table), ('cross', slice(len(table), None), cross)]:
                np.savez_compressed(out/f'{side}_predictions.npz', obs_id=tab.obs_id.to_numpy(),
                                    **{k: v[sl] for k, v in scores.items()}, **{k: v[sl] for k, v in extra.items()})
            torch.save(scaler, out/'M3_readout_scaler.pt')
            receipt = dict(**meta, candidate_key=key, joint_step=sj, vv_step=steps[0], vh_step=steps[1],
                           alpha=alpha, frequency_step=gs, utility_n=len(utility), selection_n=len(selection),
                           source_n=len(table), cross_n=len(cross), M2_trained=False,
                           readout_active=int(scaler['active'].sum()), gate_active=int(gate_scaler['active'].sum()))
            receipts.append(receipt)
            write(out/'selection.json', receipt)
            print('SAVED', candidate, source, seed, flush=True)
    pd.DataFrame(receipts).to_csv(dest/'selections.csv', index=False)
    write(dest/'validation.json', checks)
    write(dest/'predictions_saved.json', dict(seconds=time.perf_counter()-tick,
        all_six_configurations_saved=True, target_metrics_read=False, frozen_matcher=True))
    return True


def evaluate(candidate):
    assert (HERE/'predictions_saved.json').exists()
    dest = HERE / candidate
    metrics, pairs, classes, composition = [], [], [], []
    for source, target in DIRECTIONS:
        direction = f'{source}-to-{target}'
        for side in ('source', 'cross'):
            tab = pd.read_csv(ROOT/f'{source}_m123_replication_20261026/stack_table.csv' if side=='source'
                              else TRANSFER/direction/'target_table.csv', dtype={'mmsi': str})
            for role, group in tab.groupby('role'):
                for c, part in group.groupby('class_id'):
                    composition.append(dict(candidate=candidate, source=source, side=side, role=role, class_id=int(c),
                        class_name=CLASSES[int(c)], n=len(part), ships=part.mmsi.nunique(), passes=part.pass_group.nunique()))
            old = load(EVIDENCE/f'classical_fit_only/{source}/{side}_predictions.npz')
            np.testing.assert_array_equal(old['obs_id'], tab.obs_id)
            for seed in SEEDS:
                z = load(dest/direction/f'seed{seed}/{side}_predictions.npz')
                np.testing.assert_array_equal(z['obs_id'], tab.obs_id)
                z['rawMLP'] = old[f'regional_mlp_seed{seed}']
                for role, ix in tab.groupby('role').groups.items():
                    ix = np.asarray(ix)
                    yy = tab.iloc[ix].class_id.to_numpy(int)
                    meta = dict(candidate=candidate, source=source, side=side, role=role, seed=seed)
                    for arm in (*ARMS, 'rawMLP'):
                        score = z[arm][ix]
                        metrics.append(dict(**meta, arm=arm, **m3.metrics(yy, score)))
                        for c in np.unique(yy):
                            take = yy==c
                            pred = score.argmax(1)
                            classes.append(dict(**meta, arm=arm, class_id=int(c), class_name=CLASSES[c],
                                n=int(take.sum()), recall=float((pred[take]==c).mean()),
                                predicted=int((pred==c).sum()), correct=int(((pred==c)&take).sum())))
                    for before, after in [('bare', 'joint'), ('bare', 'vv'), ('bare', 'vh'),
                                          ('bare', 'B_fixed'), ('bare', 'frequency'), ('B_fixed', 'frequency'),
                                          ('rawMLP', 'bare'), ('rawMLP', 'frequency')]:
                        a, b = z[before][ix].argmax(1)==yy, z[after][ix].argmax(1)==yy
                        for c in [None, *np.unique(yy).tolist()]:
                            take = np.ones(len(yy), bool) if c is None else yy==c
                            pairs.append(dict(**meta, before=before, after=after,
                                class_id=-1 if c is None else int(c), n=int(take.sum()),
                                rescue=int((b&~a&take).sum()), harm=int((a&~b&take).sum()),
                                CC=int((a&b&take).sum()), WW=int((~a&~b&take).sum()),
                                OA_delta_pp=float((b[take].astype(float)-a[take]).mean()*100),
                                BA_delta_pp=float(np.mean([(b[yy==k].astype(float)-a[yy==k]).mean()
                                    for k in np.unique(yy[take])])*100)))
    frame, paired, perclass = pd.DataFrame(metrics), pd.DataFrame(pairs), pd.DataFrame(classes)
    frame.to_csv(dest/'metrics.csv', index=False)
    frame.groupby(['candidate', 'source', 'side', 'role', 'arm']).agg(
        OA=('OA', 'mean'), BA=('BA', 'mean'), OA_std=('OA', 'std'), BA_std=('BA', 'std'),
        J=('J', 'mean'), NLL=('NLL', 'mean'), Brier=('Brier', 'mean'), n=('n', 'first'), seeds=('seed', 'nunique')
    ).reset_index().to_csv(dest/'summary.csv', index=False)
    paired.to_csv(dest/'paired.csv', index=False)
    paired.groupby(['candidate', 'source', 'side', 'role', 'before', 'after', 'class_id']).agg(
        OA_delta_pp=('OA_delta_pp', 'mean'), BA_delta_pp=('BA_delta_pp', 'mean'),
        OA_delta_std=('OA_delta_pp', 'std'), BA_delta_std=('BA_delta_pp', 'std'),
        rescue_mean=('rescue', 'mean'), harm_mean=('harm', 'mean'), CC_mean=('CC', 'mean'), WW_mean=('WW', 'mean')
    ).reset_index().to_csv(dest/'paired_summary.csv', index=False)
    perclass.to_csv(dest/'per_class.csv', index=False)
    perclass.groupby(['candidate', 'source', 'side', 'role', 'arm', 'class_id', 'class_name']).agg(
        recall=('recall', 'mean'), recall_std=('recall', 'std'), n=('n', 'first')
    ).reset_index().to_csv(dest/'per_class_summary.csv', index=False)
    pd.DataFrame(composition).to_csv(dest/'composition.csv', index=False)
    write(dest/'evaluation_complete.json', dict(all_predictions_saved_before_target_evaluation=True,
                                                configurations=6, arms=list(ARMS), rawMLP_comparator_only=True))
    summary = pd.read_csv(dest/'summary.csv')
    print(summary.loc[summary.role.eq('evaluation') & summary.side.eq('cross')].to_string(index=False), flush=True)


def train_all():
    if not ready():
        return False
    for candidate in CANDIDATES:
        train(candidate)
    pd.concat([pd.read_csv(HERE/c/'selections.csv') for c in CANDIDATES]).to_csv(HERE/'selections.csv', index=False)
    write(HERE/'validation.json', [r for c in CANDIDATES
        for r in json.loads((HERE/c/'validation.json').read_text(encoding='utf-8'))])
    write(HERE/'predictions_saved.json', dict(all_twelve_configurations_saved=True,
        target_metrics_read=False, candidates=list(CANDIDATES), raw_summary_is_posthoc_development_diagnostic=True))
    return True


def evaluate_all():
    for candidate in CANDIDATES:
        evaluate(candidate)
    for filename in ('metrics', 'summary', 'paired', 'paired_summary', 'per_class', 'per_class_summary', 'composition'):
        pd.concat([pd.read_csv(HERE/c/f'{filename}.csv') for c in CANDIDATES]).to_csv(HERE/f'{filename}.csv', index=False)
    write(HERE/'evaluation_complete.json', dict(configurations=12, prediction_barrier=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['ready', 'train', 'evaluate', 'all'], default='all')
    args = parser.parse_args()
    if args.stage == 'ready':
        ready()
    elif args.stage == 'evaluate':
        evaluate_all()
    elif train_all() and args.stage == 'all':
        evaluate_all()
