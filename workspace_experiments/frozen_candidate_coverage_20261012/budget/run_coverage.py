"""Frozen budget coverage. Only this directory is writable; no GPU or tuning."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import gc
import json
import pickle
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
STUDY = OUT.parents[1]/'same_vessel_condition_study_20261010'
ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(ROOT/'oof_reliability_check'))
from compact_fusion_source import SOURCE, EXTERNAL, CACHE, _expert_inputs
from unified_head_core import coefficients, late
from unified_head_run import base_kernel
from unified_head_expert import _conditions
from full_teacher_ops import numpy_phi
from case_completion_class_control import fit_control
from case_support_completion import COLS, predict, adapt_support
from evaluate_probability_layers import probability_rows

BUDGETS = (16, 64, 128)
DOSES = {16: 1., 64: .5, 128: .25}
OLD_PORTS = ('Los Angeles', 'Port Klang', 'Santos')
SEEDS = (31415, 27182)
TASKS = ROOT/'learned_frequency_residual_20261009/tasks'
PORT_GEOMETRY = ROOT/'protected_drift_activation_20261008/ports'


def archive(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def load_pickle(path):
    with path.open('rb') as stream:
        return pickle.load(stream)


def task_folders():
    return sorted(TASKS.glob('self24ports/*/PREPARED.json')) + [
        TASKS/d/'PREPARED.json' for d in ('oss2', 'f0921')]


def allowed_rows(ds, task):
    if ds == 'self24ports':
        return np.load(PORT_GEOMETRY/task/'source_rows.npy')
    return archive(ROOT/EXTERNAL[ds][0]/'source_split.npz')['source_rows']


def audit():
    meta = pd.read_csv(STUDY/'observations.csv')
    folds = archive(STUDY/'crossfit_evidence_20261011/source_folds.npz')
    ids = folds['fit_obs_ids']
    records = []
    for path in task_folders():
        info = read_json(path)
        ds, task = info['dataset'], info['task']
        rows = allowed_rows(ds, task)
        use = np.isin(meta.iloc[ids].manifest_row.to_numpy(), rows)
        counts = [meta.iloc[ids[use & (folds['assignment'] == k)]].mmsi.nunique()
                  for k in range(3)]
        with np.load(path.parent/'target.npz') as data:
            n, nq = len(data['rows']), int(data['eval_masks'].all(0).sum())
            pool = len(np.unique(data['support']))
        old = ds == 'self24ports' and task in OLD_PORTS
        reasons = []
        if not old:
            reasons.append('Task-specific source expert phi/OOF priors/full base components not prepared for this corrector cohort')
        if min(counts) < 160:
            reasons.append('Original fold has fewer than 128 support + 32 query vessels after target purge')
        records.append(dict(dataset=ds, task=task, target_n=n, common_query_n=nq,
            union_support_pool=pool, source_fit_n=int(use.sum()),
            fold0_ships=int(counts[0]), fold1_ships=int(counts[1]), fold2_ships=int(counts[2]),
            direct_frozen_weight_reuse=old, unchanged_episode_recipe_feasible=min(counts) >= 160,
            raw_simple_ready=pool >= 128 and nq > 0,
            initial_recipe='Task-specific refit, existing allowed source rows only',
            trained_candidates='cached own-task models' if old else 'not executed',
            blocker='; '.join(reasons)))
    records.append(dict(dataset='oss1', task='oss1', raw_simple_ready=False,
        direct_frozen_weight_reuse=False, unchanged_episode_recipe_feasible=False,
        blocker='No full learned-frequency task/base cache in this route. Compact OSS1 belongs to separate scope; do not substitute it.'))
    frame = pd.DataFrame(records)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT/'coverage_audit.csv', index=False)
    frozen = read_json(STUDY/'support_budget_fusion_select_20261012/FROZEN_CHOICE.json')
    write(OUT/'FROZEN_PROTOCOL.json', dict(budgets=BUDGETS, doses=DOSES,
        simple={str(b): frozen['choices'][str(b)]['simple_fusion_selected'] for b in BUDGETS},
        corrector='Original other-vessel same-class linear residual, fixed .1 trace Gram / dim ridge',
        initializer_fit='Original source-fit cohort intersect task allowed rows, multi-product vessels only; own task coordinates',
        training_candidates='Shared90 and isolated30perB preserved; no incompatible weight transplant',
        support='Original5x64 row samples, nested16/64/128; extra64 from original support union, seed124000+draw',
        query='Intersection of original five query masks, common to every budget and arm',
        identity_scope='Historical row-only support/query separation, not vessel-held target confirmation',
        historical_selection='Doses/simple choice frozen from previous development; some expanded ports contributed to historical hyperparameter selection',
        source_training='Outer task permitted source rows; target port/identities are never reintroduced',
        runtime=sys.executable, blas_threads=2, gpu=False,
        statistics='OA/BA,NLL,Brier,class-balanced risk,paired rescue/harm,draw composition; no universal-win requirement',
        feature_pipeline='Same stored geometry; no PCA/encoder refit. Original-port caches replayed before wider conclusions',
        full_coverage_claim=False))
    print(frame[['dataset', 'task', 'source_fit_n', 'fold0_ships', 'fold1_ships', 'fold2_ships',
                 'unchanged_episode_recipe_feasible']].to_string(index=False), flush=True)


class Features:
    def __init__(self, folder):
        self.folder = folder
        info = read_json(folder/'PREPARED.json')
        self.ds, self.task = info['dataset'], info['task']
        self.source_rows = allowed_rows(self.ds, self.task)
        self.raw_source = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
        self.index = pd.Index(self.source_rows)
        self.legacy = None
        if self.ds == 'self24ports':
            p = PORT_GEOMETRY/self.task
            self.geometry = load_pickle(p/'geometry.pkl')
            self.bank_source = np.load(p/'source_bank.npy', mmap_mode='r')
            self.bank_target = np.load(p/'target_bank.npy', mmap_mode='r')
            self.target_rows = np.load(p/'target_rows.npy')
            sf = ROOT/'evidence_aware_handoff_20261008/source_features'/self.task
            np.testing.assert_array_equal(np.load(sf/'rows.npy'), self.source_rows)
            self.extra_source = np.load(sf/'features.npy', mmap_mode='r')
            self.extra_target = np.load(ROOT/'module_completion_20261008/ports_projection'/self.task/'matched_joint/joint_extra.npy', mmap_mode='r')
            self.u = _conditions()
            self.head = load_pickle(ROOT/'direct_incidence_comparison_20261009/ports'/self.task/'direct_heads.pkl')['joint_expert']
            if self.task in OLD_PORTS:
                self.legacy = archive(STUDY/'full_teacher_integration_20261012'/self.task/'no_training/predictions.npz')
        else:
            self.joint = ROOT/EXTERNAL[self.ds][1]
            self.model = load_pickle(self.joint/'source_model.pkl')
            self.stats = archive(self.joint/'deployment_complement.npz')
            extra = self.joint/'joint/source_extra.npy'
            if extra.exists():
                self.extra_source = np.asarray(np.load(extra, mmap_mode='r')[self.source_rows])
            else:
                saved = archive(CACHE/f'{self.joint.name}_source_extra.npz')
                np.testing.assert_array_equal(saved['rows'], self.source_rows)
                self.extra_source = saved['features']
            self.extra_target = np.load(self.joint/'joint/target_extra.npy', mmap_mode='r')
            target = Path(read_json(self.joint/'protocol.json')['target'])
            self.target_manifest = target/'target_manifest.csv'
            if self.ds == 'oss2':
                self.raw_target = archive(target/'features.npz')['paired_r50']
            else:
                self.raw_target = np.load(target/'paired_r50.npy', mmap_mode='r')
            self.target_rows = np.arange(len(self.raw_target))

    def external(self, raw, extra):
        # Reuse the deployed view operator; preserve its FP32 coordinate casts.
        from evaluate_joint_residual import views
        p, retained, discarded = views(raw, extra, self.model, self.stats, True)
        return np.c_[retained/np.sqrt(2), discarded/np.sqrt(2), np.ones(len(raw))], p

    def source_phi(self, rows):
        ix = self.index.get_indexer(rows)
        assert (ix >= 0).all(), 'Target-purged row must not be restored to source fitting'
        if self.ds == 'self24ports':
            return numpy_phi(self.bank_source[ix], self.raw_source[rows], self.extra_source[ix],
                             self.u[rows], self.geometry)
        return self.external(self.raw_source[rows], self.extra_source[ix])[0]

    def target(self, ix):
        if self.legacy is not None:
            np.testing.assert_array_equal(self.legacy['rows'], self.target_rows)
            return self.legacy['phi'][ix], self.legacy['prior'][ix]
        if self.ds == 'self24ports':
            rows, bank = self.target_rows[ix], self.bank_target[ix]
            phi = numpy_phi(bank, self.raw_source[rows], self.extra_target[ix], self.u[rows], self.geometry)
            p = np.c_[bank, self.u[rows]]@self.head['coef']+self.head['intercept']
            return phi, p
        return self.external(self.raw_target[ix], self.extra_target[ix])


def fit_initializer(features, dest, meta, folds):
    ids = folds['fit_obs_ids']
    ids = ids[np.isin(meta.iloc[ids].manifest_row.to_numpy(), features.source_rows)]
    mm = meta.iloc[ids].reset_index(drop=True)
    multi = mm.groupby('mmsi')['product'].transform('nunique').to_numpy() > 1
    ids, mm = ids[multi], mm.loc[multi].reset_index(drop=True)
    if features.ds == 'self24ports':
        assert not (mm.port == features.task).any(), 'Target port enters corrector training'
    phi = features.source_phi(mm.manifest_row.to_numpy())
    start = time.perf_counter()
    fitted = fit_control(phi[:, COLS], mm)
    elapsed = time.perf_counter()-start
    replay = None
    if features.ds == 'self24ports' and features.task in OLD_PORTS:
        old = archive(STUDY/'case_completion_class_control_20261012'/features.task/'final_model.npz')
        replay = max(float(abs(fitted[k]-old[k]).max()) for k in old)
        # Inspect discrepancies rather than silently substituting cross-task models.
        for key in old:
            np.testing.assert_allclose(fitted[key], old[key], rtol=1e-8, atol=1e-8,
                                       err_msg='Original initializer replay: '+key)
        fitted = old
    np.savez_compressed(dest/'initial.npz', **fitted)
    mm.assign(obs_id=ids).to_csv(dest/'initial_training_rows.csv', index=False)
    record = dict(rows=len(mm), ships=int(mm.mmsi.nunique()),
        class_ships=mm.groupby('class_id').mmsi.nunique().to_dict(),
        all_rows_allowed=bool(np.isin(mm.manifest_row, features.source_rows).all()),
        dimension=len(COLS), seconds=elapsed, original_initializer_max_error=replay,
        parameters_in_this_task_coordinates=True, outside_source_training=False)
    write(dest/'INITIAL_FIT.json', record)
    return fitted


def base_block(data, embedding, q, s, eta):
    joined = np.r_[s, q]
    sub = {k: data[k][joined] for k in ('phi', 'mean', 'conditions')}
    return base_kernel(sub, embedding[joined].astype(float), np.arange(len(s)), eta)


def freeze_task(path):
    started = time.perf_counter()
    folder, info = path.parent, read_json(path)
    ds, task = info['dataset'], info['task']
    dest = OUT/'tasks'/ds/task
    if (dest/'PREDICTIONS_FROZEN.json').exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    data, emb = archive(folder/'target.npz'), archive(folder/'selective_embedding.npz')
    np.testing.assert_array_equal(data['rows'], emb['rows'])
    meta = pd.read_csv(STUDY/'observations.csv')
    folds = archive(STUDY/'crossfit_evidence_20261011/source_folds.npz')
    features = Features(folder)
    np.testing.assert_array_equal(data['rows'], features.target_rows)
    initializer = fit_initializer(features, dest, meta, folds)
    manifest = pd.read_csv(SOURCE/'manifest.csv' if ds == 'self24ports' else features.target_manifest)
    target_meta = manifest.iloc[data['rows']].reset_index(drop=True)
    q = np.flatnonzero(data['eval_masks'].all(0))
    pool = np.unique(data['support'])
    np.testing.assert_array_equal(pool, np.flatnonzero(~data['eval_masks'].all(0)))
    assert len(pool) >= 128 and len(q) > 0
    # Store only support labels before prediction; query labels enter evaluate().
    choices = read_json(OUT/'FROZEN_PROTOCOL.json')['simple']
    replay_errors = []
    for draw, old_s in enumerate(data['support']):
        rng = np.random.default_rng(124000+draw)
        order = np.r_[rng.permutation(old_s), rng.choice(np.setdiff1d(pool, old_s), 64, replace=False)]
        assert len(np.unique(order)) == 128 and not np.intersect1d(order, q).size
        for budget in BUDGETS:
            s = order[:budget]
            sy = target_meta.iloc[s].class_id.to_numpy(int)
            sp, prior_s = features.target(s)
            models = {'initial': initializer}
            if ds == 'self24ports' and task in OLD_PORTS:
                for seed in SEEDS:
                    models[f'task_{seed}'] = archive(STUDY/'support_task_loss_20261012/models'/task/str(seed)/'task.npz')
                    models[f'isolated_{seed}'] = archive(STUDY/'support_budget_isolation_20261012/models'/task/str(seed)/f'{budget}.npz')
                    models[f'reconstruction_{seed}'] = archive(STUDY/'support_task_loss_20261012/models'/task/str(seed)/'reconstruction.npz')
            support_phi = {'raw': sp}
            for arm, model in models.items():
                delta = predict(model, sp[:, COLS], np.zeros(budget, int), 'image_history')
                support_phi[arm] = adapt_support(sp, delta, DOSES[budget])
            alpha = {a: coefficients(x@x.T, prior_s, sy, budget/8.) for a, x in support_phi.items()}
            bss = base_block(data, emb['embedding'], np.empty(0, int), s, info['eta'])
            ab = coefficients(bss, data['prior'][s], sy, budget/8.)
            values = {a: np.empty((len(q), 8)) for a in ('base', 'simple', *support_phi)}
            for lo in range(0, len(q), 2048):
                qi = q[lo:lo+2048]
                qp, pq = features.target(qi)
                kb = base_block(data, emb['embedding'], qi, s, info['eta'])[budget:]
                values['base'][lo:lo+len(qi)] = data['prior'][qi]+kb@ab
                for a, x in support_phi.items():
                    values[a][lo:lo+len(qi)] = pq+(qp@x.T)@alpha[a]
                choice = choices[str(budget)]
                scale = 1.-choice['intervention'] if choice['arm'] == 'simple' else 1.
                values['simple'][lo:lo+len(qi)] = pq+scale*(values['raw'][lo:lo+len(qi)]-pq)
            if budget == 64:
                expected = data['expert'][draw][q]
                err = float(abs(values['raw']-expected).max())
                np.testing.assert_allclose(values['raw'], expected, rtol=1e-6, atol=3e-5)
                np.testing.assert_array_equal(values['raw'].argmax(1), expected.argmax(1))
                replay_errors.append(dict(draw=draw, expert_max_error=err))
            saved = dict(query_ids=data['rows'][q], support_ids=data['rows'][s], support_labels=sy,
                         base=values['base'])
            for a in values:
                if a != 'base':
                    saved['expert_'+a] = values[a]
                    saved['fusion_'+a] = late(values['base'], values[a])
            if ds == 'self24ports' and task in OLD_PORTS:
                old = archive(STUDY/'support_budget_isolation_20261012/target'/task/str(SEEDS[0])/f'{draw}_{budget}.npz')
                for key in ('support_ids', 'query_ids'):
                    np.testing.assert_array_equal(saved[key], old[key])
                for arm in ('raw', 'initial', 'simple'):
                    np.testing.assert_allclose(saved['fusion_'+arm], old['fusion_'+arm], rtol=1e-7, atol=1e-7)
                for arm in ('task', 'isolated', 'reconstruction'):
                    np.testing.assert_allclose(saved['fusion_'+arm+'_'+str(SEEDS[0])], old['fusion_'+arm], rtol=1e-7, atol=1e-7)
            np.savez_compressed(dest/f'{draw}_{budget}.npz', **saved)
        print('FROZEN', ds, task, 'draw', draw, flush=True)
    write(dest/'PREDICTIONS_FROZEN.json', dict(dataset=ds, task=task, query_n=len(q),
        prediction_units=15, budgets=BUDGETS, original_B64_replay=replay_errors,
        initializer_refit=True, new_shared_or_isolated_fits=0,
        shared_isolated_coverage='old task only' if task in OLD_PORTS else 'not run: see coverage audit',
        target_manifest=str(SOURCE/'manifest.csv' if ds == 'self24ports' else features.target_manifest),
        query_labels_used_for_prediction=False, seconds=time.perf_counter()-started))
    print('TASK COMPLETE', ds, task, round(time.perf_counter()-started, 2), flush=True)
    del features, data, emb
    gc.collect()


def evaluate():
    records, paired, classes, supports = [], [], [], []
    receipts = {}
    # Prefer separately frozen strong external versions and newly trained arms.
    for root in ('tasks', 'strong_external', 'trained_predictions'):
        for receipt in sorted((OUT/root).glob('*/*/PREDICTIONS_FROZEN.json')):
            item = read_json(receipt)
            receipts[(item['dataset'], item['task'])] = receipt
    for receipt in receipts.values():
        info = read_json(receipt)
        table = pd.read_csv(info['target_manifest'])
        for budget in BUDGETS:
            for draw in range(5):
                z = archive(receipt.parent/f'{draw}_{budget}.npz')
                qm, sm = table.iloc[z['query_ids']], table.iloc[z['support_ids']]
                y = qm.class_id.to_numpy(int)
                _, inv, counts = np.unique(y, return_inverse=True, return_counts=True)
                w = 1./(len(counts)*counts[inv])
                ident = dict(dataset=info['dataset'], task=info['task'], budget=budget, draw=draw)
                srow = dict(**ident, n_query=len(y), class_counts=json.dumps(sm.class_id.value_counts().sort_index().to_dict()))
                ship_key = next((k for k in ('mmsi', 'mmsi_or_ship_id') if k in table), None)
                if ship_key:
                    srow.update(ship_field=ship_key, known_support_ships=int(sm[ship_key].nunique()),
                        query_support_ship_overlap=float((qm[ship_key].notna() & qm[ship_key].isin(sm[ship_key].dropna())).mean()))
                supports.append(srow)
                for layer in ('expert', 'fusion'):
                    kind = 'logits' if layer == 'expert' else 'probabilities'
                    ps = {a.removeprefix(layer+'_'): probability_rows(z[a], y, kind)
                          for a in z if a.startswith(layer+'_')}
                    metrics = {}
                    for arm, p in ps.items():
                        m = dict(oa=100*p['correct'].mean(), ba=100*w@p['correct'],
                            nll=float(p['nll'].mean()), brier=float(p['brier'].mean()),
                            class_nll=float(w@p['nll']), class_brier=float(w@p['brier']))
                        metrics[arm] = m
                        records.append(dict(**ident, layer=layer, arm=arm, **m))
                        for c in np.unique(y):
                            ix = y == c
                            classes.append(dict(**ident, layer=layer, arm=arm, class_id=int(c),
                                n=int(ix.sum()), support_n=int((sm.class_id == c).sum()),
                                recall=100*p['correct'][ix].mean(), brier=float(p['brier'][ix].mean())))
                    for arm in ps:
                        if arm == 'raw':
                            continue
                        refs = ['raw'] if arm in ('initial', 'simple') else ['raw', 'initial', 'simple']
                        if arm.startswith('isolated_'):
                            refs.append('task_'+arm.split('_')[-1])
                        for ref in refs:
                            p, r = ps[arm], ps[ref]
                            rescue = 100*w@(~r['correct'] & p['correct'])
                            harm = 100*w@(r['correct'] & ~p['correct'])
                            dba = metrics[arm]['ba']-metrics[ref]['ba']
                            np.testing.assert_allclose(rescue-harm, dba, atol=1e-10, rtol=0)
                            paired.append(dict(**ident, layer=layer, arm=arm, reference=ref,
                                delta_ba=dba, delta_oa=metrics[arm]['oa']-metrics[ref]['oa'],
                                rescue_ba=rescue, harm_ba=harm,
                                brier_gain=metrics[ref]['brier']-metrics[arm]['brier'],
                                class_brier_gain=metrics[ref]['class_brier']-metrics[arm]['class_brier']))
    if not records:
        raise ValueError('No completed predictions')
    frame, pairs = pd.DataFrame(records), pd.DataFrame(paired)
    for name, table in [('metrics', frame), ('paired', pairs), ('classes', pd.DataFrame(classes)),
                        ('supports', pd.DataFrame(supports))]:
        table.to_csv(OUT/(name+'.csv'), index=False)
    measures = ['oa', 'ba', 'nll', 'brier', 'class_nll', 'class_brier']
    frame.groupby(['dataset', 'task', 'budget', 'layer', 'arm'])[measures].mean().reset_index().to_csv(OUT/'task_summary.csv', index=False)
    grouped = frame.groupby(['dataset', 'budget', 'layer', 'arm'])
    summary = grouped[measures].mean().join(grouped.task.nunique().rename('task_count')).reset_index()
    summary.to_csv(OUT/'dataset_summary.csv', index=False)
    pairs.groupby(['dataset', 'task', 'budget', 'layer', 'arm', 'reference']).agg(
        delta_ba=('delta_ba', 'mean'), sd_ba=('delta_ba', 'std'),
        delta_oa=('delta_oa', 'mean'), class_brier_gain=('class_brier_gain', 'mean'),
        wins=('delta_ba', lambda x: int((x > 1e-10).sum())),
        losses=('delta_ba', lambda x: int((x < -1e-10).sum()))).reset_index().to_csv(OUT/'paired_task_summary.csv', index=False)
    write(OUT/'EVALUATION_COMPLETE.json', dict(status='partial_candidate_coverage',
        tasks=frame[['dataset', 'task']].drop_duplicates().to_dict('records'),
        rows=len(frame), trained_correction_expansion_complete=len(list((OUT/'trained_predictions').glob('*/*/PREDICTIONS_FROZEN.json'))) == 12,
        evaluated_receipts=[str(p) for p in receipts.values()],
        no_target_selection=True, protocol_notes='Different arm coverage must not be compared across different task sets'))
    print(frame[(frame.layer == 'fusion') & frame.arm.isin(['raw', 'initial', 'simple'])].groupby(
        ['dataset', 'budget', 'arm'])[['oa', 'ba']].mean().to_string(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['audit', 'freeze', 'evaluate'])
    parser.add_argument('--task', action='append')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.stage == 'audit':
            audit()
        elif args.stage == 'freeze':
            for path in task_folders():
                if args.task and read_json(path)['task'] not in args.task:
                    continue
                freeze_task(path)
        else:
            evaluate()
