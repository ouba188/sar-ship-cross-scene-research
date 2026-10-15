"""Read out existing source-trained architectures under current frozen heads."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
import time
import zipfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_learned_frequency import ROOT, SOURCE, DATASETS, load_npz
from frozen_comparison import folders, write, ba
from unified_head_core import coefficients, late
from frequency_band_core import old_kernel

OUT = ROOT/'completion_training_20261010'
REFERENCE = ROOT/'unified_head_20261010'
ARMS = ('baseline', 'unsplit', 'fixed', 'selective', 'selective_uniform')
READOUTS = ('base', 'fusion', 'unified')
EDGES = (
    ('unsplit_added', 'baseline', 'unsplit'),
    ('fixed_added', 'baseline', 'fixed'),
    ('selective_added', 'baseline', 'selective'),
    ('split_vs_unsplit', 'unsplit', 'fixed'),
    ('learned_vs_fixed', 'fixed', 'selective'),
    ('learned_gate_removed', 'selective', 'selective_uniform'),
    ('selective_training_uniform_readout', 'fixed', 'selective_uniform'),
)


def metadata(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    return info, f"{info['dataset']}/{info['task']}"


def protocol():
    write(OUT/'PROTOCOL.json', dict(date='2026-10-10', arms=ARMS, readouts=READOUTS,
        contrasts=EDGES, tasks=25, draws_per_task=5, target_labels=64, ridge=8.,
        frequency_weight=.25, fusion_weight=.5, eta='Original source-selected per task',
        neural_training='Reuse 75 real source-trained checkpoints, no new neural training',
        epochs=6, seed=20261009, beta_source=.25,
        baseline='Current full base minus frequency, same current expert',
        uniform='Remove gate only after selective training; not a fourth trained architecture',
        frozen='Source prior, geometry, C4, conditions, expert, support draws, evaluation masks',
        interpretation='Current readout of source-trained architectures; not end-to-end joint-head training',
        selection='None; all arms fixed before new query-label evaluation',
        recovered_f0921_incidence_used=False, independent_confirmation=False,
        thread_limit=2, gpu_required=False))


def inventory():
    import torch
    from learned_frequency_model import FrequencyResidual

    torch.set_num_threads(2)
    records, episodes = [], []
    for folder in folders():
        info, suffix = metadata(folder)
        with np.load(folder/'target.npz') as saved:
            allowed, target_rows = saved['source_rows'], saved['rows']
        for path in sorted((folder/'episodes').glob('*.npz')):
            with np.load(path) as ep:
                rows, support, query = ep['rows'], ep['support'], ep['query']
                if np.setdiff1d(rows, allowed).size:
                    raise ValueError(f'Source episode outside permitted rows: {path}')
                if len(support) != 64 or np.intersect1d(support, query).size:
                    raise ValueError(f'Invalid source split: {path}')
                if info['dataset'] == 'self24ports' and np.intersect1d(rows, target_rows).size:
                    raise ValueError(f'Outer target leakage: {path}')
                episodes.append(dict(dataset=info['dataset'], task=info['task'],
                    episode=path.stem, source_rows=len(rows), supports=len(support), queries=len(query)))
        for arm in ARMS[1:4]:
            checkpoint = torch.load(folder/f'{arm}.pt', map_location='cpu', weights_only=True)
            log = json.loads((folder/f'{arm}_training.json').read_text())
            np.testing.assert_array_equal(checkpoint['source_rows'], allowed)
            assert checkpoint['mode'] == arm and checkpoint['epochs'] == 6
            assert checkpoint['seed'] == 20261009 and checkpoint['beta'] == .25
            assert checkpoint['eta'] == info['eta']
            assert log['source_only'] and log['final_epoch_used']
            assert not log['target_checkpoint_selection'] and len(log['epochs']) == 6
            assert log['first_gradient_norm'] > 0
            model = FrequencyResidual(arm)
            model.load_state_dict(checkpoint['state'], strict=True)
            parameters = sum(p.numel() for p in model.parameters())
            assert parameters == log['parameters']
            records.append(dict(dataset=info['dataset'], task=info['task'], arm=arm,
                checkpoint=str(folder/f'{arm}.pt'), embedding=str(folder/f'{arm}_embedding.npz'),
                source_rows=len(allowed), epochs=6, seed=20261009, parameters=parameters,
                source_queue_exact=True, final_epoch=True, target_selection=False,
                first_gradient=log['first_gradient_norm'],
                training_seconds=log['epochs'][-1]['seconds']))
    assert len(records) == 75
    pd.DataFrame(records).to_csv(OUT/'checkpoint_inventory.csv', index=False)
    pd.DataFrame(episodes).to_csv(OUT/'source_episode_inventory.csv', index=False)
    write(OUT/'INVENTORY_COMPLETE.json', dict(checkpoints=75, tasks=25,
        source_episodes=len(episodes), all_source_queues_exact=True,
        all_checkpoints_strict_load=True, new_training_needed=False, gpu_minutes=0))
    print('INVENTORY: 75 real checkpoints, all source queues exact, no GPU needed', flush=True)


class ZipRows:
    """Sequential row reads from existing compressed arrays, without extraction."""

    def __init__(self, path, key, width=None):
        self.zip = zipfile.ZipFile(path)
        self.stream = self.zip.open(key+'.npy')
        version = np.lib.format.read_magic(self.stream)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        self.shape, fortran, self.dtype = reader(self.stream)
        assert not fortran and not self.dtype.hasobject
        self.tail = (width,) if width is not None else self.shape[1:]
        self.width = int(np.prod(self.tail))
        self.offset = self.stream.tell()

    def read(self, lo, hi):
        self.stream.seek(self.offset+lo*self.width*self.dtype.itemsize)
        raw = self.stream.read((hi-lo)*self.width*self.dtype.itemsize)
        return np.frombuffer(raw, self.dtype).reshape((hi-lo,)+self.tail)

    def take(self, indices):
        result = np.empty((len(indices),)+self.tail, self.dtype)
        for i in np.argsort(indices):
            result[i] = self.read(int(indices[i]), int(indices[i])+1)[0]
        return result

    def close(self):
        self.stream.close()
        self.zip.close()


class ExpertBlocks:
    """Same deployed expert operations, restricted to one query block."""

    def __init__(self, folder, data):
        from unified_head_expert import ExpertContext, _conditions, _pickle, PORT_REFERENCE, EPISODES
        self.context = c = ExpertContext(folder)
        self.rows = data['rows']
        self.port = c.dataset == 'self24ports'
        if self.port:
            self.raw = c.raw
            self.bank = np.load(c.parent/'target_bank.npy', mmap_mode='r')
            self.extra = np.load(EPISODES/c.task/'matched_joint/joint_extra.npy', mmap_mode='r')
            self.u = _conditions()[self.rows]
            self.head = _pickle(PORT_REFERENCE/c.task/'direct_heads.pkl')['joint_expert']
            np.testing.assert_array_equal(np.load(c.parent/'target_rows.npy'), self.rows)
        else:
            protocol = json.loads((c.joint.parent/'protocol.json').read_text())
            target = Path(protocol['target'])
            if c.dataset == 'oss2':
                with np.load(target/'features.npz') as saved:
                    self.raw = saved['paired_r50']
            else:
                self.raw = np.load(target/'paired_r50.npy', mmap_mode='r')
            self.extra = np.load(c.joint/'target_extra.npy', mmap_mode='r')
            np.testing.assert_array_equal(self.rows, np.arange(len(self.raw)))

    def views(self, indices):
        c = self.context
        if self.port:
            from run_protected_drift_activation import make_views
            bank = self.bank[indices]
            views = make_views(bank, self.raw[self.rows[indices]], self.extra[indices], c.geometry)
            common = views['common'][True]
            prior = np.c_[bank, self.u[indices]]@self.head['coef']+self.head['intercept']
            return prior, common['retained'], common['complement'], views['stable'], self.u[indices]
        from evaluate_joint_residual import views
        prior, retained, discarded = views(self.raw[indices], self.extra[indices], c.model, c.stats, True)
        return prior, retained, discarded

    def set_reference(self, support):
        self.reference = self.views(support)
        return self.kernel(self.reference)

    def kernel(self, values):
        r = self.reference
        if self.port:
            kernel = .5*(values[1]@r[1].T+values[2]@r[2].T)+1.
            kernel += .25*values[3]@r[3].T
            kernel += .25*(values[4]@r[4].T)/2.
        else:
            kernel = .5*(values[1]@r[1].T)+.5*(values[2]@r[2].T)+1.
        return values[0], kernel


def predict_task(folder):

    info, suffix = metadata(folder)
    dest = OUT/'targets'/suffix
    if (dest/'COMPLETE.json').exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    with np.load(folder/'target.npz') as saved:
        data = {k: saved[k] for k in ('rows', 'ids', 'support', 'support_labels', 'eval_masks')}
    current = load_npz(REFERENCE/'targets'/suffix/'policies.npz')
    with np.load(folder/'policies.npz') as saved:
        previous = {k: saved[k] for k in ('modes', 'base_predictions', 'query_rows', 'sample_ids', 'support_rows', 'eval_masks')}
    for key in ('rows', 'ids', 'support', 'eval_masks'):
        np.testing.assert_array_equal(data[key], current[key])
    for key, old in [('rows', 'query_rows'), ('ids', 'sample_ids'),
                     ('support', 'support_rows'), ('eval_masks', 'eval_masks')]:
        np.testing.assert_array_equal(data[key], previous[old])
    gate_records = []
    for arm in ARMS[1:]:
        with np.load(folder/f'{arm}_embedding.npz') as z:
            for key in ('rows', 'ids'):
                np.testing.assert_array_equal(data[key], z[key])
            weights = z['weights']
            for i, band in enumerate(('LL', 'LH', 'HL', 'HH')):
                gate_records.append(dict(dataset=info['dataset'], task=info['task'], arm=arm,
                    band=band, mean=float(weights[:, i].mean()), std=float(weights[:, i].std())))
    context = ExpertBlocks(folder, data)
    predictions = np.empty((5, 3, len(ARMS), len(data['rows'])), dtype=np.int8)
    checks = []
    started = time.perf_counter()
    with ExitStack() as stack:
        def stream(path, key, width=None):
            obj = ZipRows(path, key, width)
            stack.callback(obj.close)
            return obj
        inputs = {k: stream(folder/'target.npz', k) for k in ('phi', 'prior', 'mean', 'conditions')}
        embeddings = {arm: stream(folder/f'{arm}_embedding.npz', 'embedding') for arm in ARMS[1:]}
        expert_saved = stream(folder/'target.npz', 'expert', 8)
        old_scores = {arm: stream(folder/'policies.npz', 'scores', 8) for arm in ARMS}
        reference = np.unique(data['support'])
        support_inputs = {k: v.take(reference) for k, v in inputs.items()}
        support_z = {k: v.take(reference).astype(float) for k, v in embeddings.items()}
        n = len(data['rows'])
        for draw, support in enumerate(data['support']):
            labels = data['support_labels'][draw]
            if len(np.unique(support)) != 64 or data['eval_masks'][draw, support].any():
                raise ValueError('Support/query mismatch')
            positions = pd.Index(reference).get_indexer(support)
            sd = {k: v[positions] for k, v in support_inputs.items()}
            sz = {k: v[positions] for k, v in support_z.items()}
            pe, ke = context.set_reference(support)
            ae = coefficients(ke, pe, labels, 8.)
            core, cond = old_kernel(sd, np.arange(64), info['eta'])
            alphas, errors = {}, dict.fromkeys(ARMS, 0.)
            for arm in ARMS:
                kb = core if arm == 'baseline' else core+.25*(sz[arm]@sz[arm].T)*cond
                alphas[arm] = (coefficients(kb, sd['prior'], labels, 8.),
                    coefficients(.5*kb+.5*ke, .5*sd['prior']+.5*pe, labels, 8.))
            for lo in range(0, n, 512):
                hi = min(lo+512, n)
                block = {k: v.read(lo, hi) for k, v in inputs.items()}
                cond = np.exp(-np.sum((block['conditions'][:, None]-sd['conditions'][None])**2, axis=2)/.5)
                core = block['phi']@sd['phi'].T+info['eta']*(block['mean']@sd['mean'].T)*cond
                pe, ke = context.kernel(context.views(np.arange(lo, hi)))
                expert = pe+ke@ae
                np.testing.assert_allclose(expert, expert_saved.read(draw*n+lo, draw*n+hi), atol=3e-5, rtol=1e-6)
                np.testing.assert_array_equal(expert.argmax(1), current['predictions'][
                    draw, list(current['modes']).index('expert'), lo:hi])
                for ai, arm in enumerate(ARMS):
                    kb = core
                    if arm != 'baseline':
                        z = embeddings[arm].read(lo, hi).astype(float)
                        np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1., atol=2e-6)
                        kb = core+.25*(z@sz[arm].T)*cond
                    ab, au = alphas[arm]
                    base = block['prior']+kb@ab
                    unified = .5*block['prior']+.5*pe+(.5*kb+.5*ke)@au
                    for ri, values in enumerate((base, late(base, expert), unified)):
                        predictions[draw, ri, ai, lo:hi] = values.argmax(1)
                    oi = list(previous['modes']).index(arm)
                    np.testing.assert_array_equal(predictions[draw, 0, ai, lo:hi], previous['base_predictions'][draw, oi, lo:hi])
                    offset = (draw*len(previous['modes'])+oi)*n
                    saved = old_scores[arm].read(offset+lo, offset+hi)
                    errors[arm] = max(errors[arm], float(np.max(np.abs(base-saved))))
                    np.testing.assert_allclose(base, saved, atol=3e-5, rtol=1e-6)
                if lo % 32768 == 0:
                    print(f'BLOCK {suffix} draw {draw+1}: {hi}/{n}', flush=True)
            for arm in ARMS:
                checks.append(dict(draw=draw, arm=arm, old_base_prediction_exact=True,
                    old_base_score_max_error=errors[arm]))
            for ri, mode in enumerate(('base', 'late_softmax_fixed', 'unified_fixed')):
                np.testing.assert_array_equal(predictions[draw, ri, ARMS.index('selective')],
                    current['predictions'][draw, list(current['modes']).index(mode)])
            print(f'{suffix} draw {draw+1}/5: trained architectures and full replay exact', flush=True)
    np.savez_compressed(dest/'policies.npz', predictions=predictions,
        arms=np.array(ARMS), readouts=np.array(READOUTS), rows=data['rows'], ids=data['ids'],
        support=data['support'], eval_masks=data['eval_masks'])
    pd.DataFrame(checks).to_csv(dest/'replay_checks.csv', index=False)
    pd.DataFrame(gate_records).to_csv(dest/'gate_statistics.csv', index=False)
    write(dest/'COMPLETE.json', dict(dataset=info['dataset'], task=info['task'],
        seconds=time.perf_counter()-started, current_full_replay_exact=True,
        all_old_base_arms_replay_exact=True, source_trained_embeddings_reused=True))


def report():
    tasks = folders()
    assert len(tasks) == 25
    assert all((OUT/'targets'/metadata(f)[1]/'COMPLETE.json').exists() for f in tasks)
    write(OUT/'ALL_PREDICTIONS_SAVED.json', dict(tasks=25, target_selection=False))
    metrics, pairs, classes, recalls = [], [], [], []
    for folder in tasks:
        info, suffix = metadata(folder)
        p = load_npz(OUT/'targets'/suffix/'policies.npz')
        manifest, idcol = ((SOURCE/'manifest.csv', 'instance_id') if info['dataset'] == 'self24ports'
                          else (ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', 'sample_id'))
        frame = pd.read_csv(manifest, usecols=[idcol, 'class_id'], keep_default_na=False).iloc[p['rows']]
        np.testing.assert_array_equal(frame[idcol].to_numpy(str), p['ids'])
        truth = frame.class_id.to_numpy()
        common = dict(dataset=info['dataset'], task=info['task'])
        for draw, ev in enumerate(p['eval_masks']):
            y = truth[ev]
            present = np.unique(y)
            for ri, readout in enumerate(READOUTS):
                for ai, arm in enumerate(ARMS):
                    pred = p['predictions'][draw, ri, ai, ev]
                    metrics.append(dict(**common, draw=draw, readout=readout, arm=arm,
                                        oa=100*np.mean(pred == y), ba=ba(y, pred), n=len(y)))
                    for c in present:
                        mask = y == c
                        recalls.append(dict(**common, draw=draw, readout=readout, arm=arm,
                            class_id=int(c), n=int(mask.sum()), recall=100*np.mean(pred[mask] == c)))
                for name, before, after in EDGES:
                    old = p['predictions'][draw, ri, ARMS.index(before), ev]
                    new = p['predictions'][draw, ri, ARMS.index(after), ev]
                    oc, nc = old == y, new == y
                    pairs.append(dict(**common, draw=draw, readout=readout, contrast=name,
                        delta_oa=100*np.mean(nc.astype(float)-oc), delta_ba=ba(y, new)-ba(y, old),
                        rescued=int(np.sum(nc & ~oc)), harmed=int(np.sum(oc & ~nc))))
                    for c in present:
                        mask = y == c
                        delta = 100*np.mean(nc[mask].astype(float)-oc[mask])
                        classes.append(dict(**common, draw=draw, readout=readout, contrast=name,
                            class_id=int(c), n=int(mask.sum()), delta_recall=delta,
                            ba_contribution=delta/len(present),
                            rescued=int(np.sum(nc[mask] & ~oc[mask])), harmed=int(np.sum(oc[mask] & ~nc[mask]))))
    m, p, cl, rc = map(pd.DataFrame, (metrics, pairs, classes, recalls))
    for name, table in [('metrics', m), ('paired_metrics', p), ('class_effects', cl), ('class_recalls', rc)]:
        table.to_csv(OUT/f'{name}.csv', index=False)
    summary = m.groupby(['dataset', 'task', 'readout', 'arm'])[['oa', 'ba']].mean().groupby(
        ['dataset', 'readout', 'arm']).mean().reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    draws = p.groupby(['dataset', 'draw', 'readout', 'contrast'])[['delta_oa', 'delta_ba']].mean().reset_index()
    draws.to_csv(OUT/'draw_effects.csv', index=False)
    effects = draws.groupby(['dataset', 'readout', 'contrast']).agg(
        delta_oa=('delta_oa', 'mean'), delta_ba=('delta_ba', 'mean'),
        oa_positive=('delta_oa', lambda x: int((x > 1e-10).sum())),
        ba_positive=('delta_ba', lambda x: int((x > 1e-10).sum()))).reset_index()
    effects.to_csv(OUT/'effects.csv', index=False)
    counts = m.groupby('dataset').task.nunique()
    cs = cl.groupby(['dataset', 'readout', 'contrast', 'class_id']).agg(
        ba_contribution=('ba_contribution', 'sum'), rescued=('rescued', 'sum'), harmed=('harmed', 'sum')).reset_index()
    cs['ba_contribution'] /= cs.dataset.map(counts)*5
    cs.to_csv(OUT/'class_ba_contributions.csv', index=False)
    np.testing.assert_allclose(cs.groupby(['dataset', 'readout', 'contrast']).ba_contribution.sum().sort_index(),
        effects.set_index(['dataset', 'readout', 'contrast']).delta_ba.sort_index(), atol=1e-10)
    reference = pd.read_csv(REFERENCE/'summary.csv')
    for readout, mode in zip(READOUTS, ('base', 'late_softmax_fixed', 'unified_fixed')):
        actual = summary[(summary.readout == readout) & (summary.arm == 'selective')].sort_values('dataset')
        expected = reference[reference['mode'] == mode].sort_values('dataset')
        np.testing.assert_allclose(actual[['oa', 'ba']], expected[['oa', 'ba']], atol=1e-10, rtol=0)
    write(OUT/'COMPLETE.json', dict(tasks=25, draws=125, trained_architectures=3,
        reused_checkpoints=75, new_neural_training=False, metric_rows=len(m),
        current_full_replay_exact=True, class_ba_accounting_exact=True, no_target_selection=True))
    print(effects[effects.readout == 'fusion'].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('inventory', 'predict', 'report', 'all'), default='all')
    parser.add_argument('--tasks', nargs='+')
    args = parser.parse_args()
    protocol()
    if args.stage in ('inventory', 'all'):
        inventory()
    if args.stage in ('predict', 'all'):
        for folder in folders():
            if not args.tasks or metadata(folder)[0]['task'] in args.tasks:
                predict_task(folder)
    if args.stage in ('report', 'all'):
        report()


if __name__ == '__main__':
    main()
