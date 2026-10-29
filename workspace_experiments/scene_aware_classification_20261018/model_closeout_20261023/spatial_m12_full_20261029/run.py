"""Complete matched M123 training: own-teacher selection, adaptation, expansion."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = HERE.parents[2]
RELEASE = WORK/'sar_m123_current_20261027'
FIT = ROOT/'spatial_m12_refit_20261029'
sys.path.insert(0, str(WORK/'sar_m123_standalone'))
from sar_m123.current.data import read_manifest, prepare_record, batch, pair_mask, FREQUENCY_KEYS
from sar_m123.current.interventions import variants, select_teacher, select_records
from sar_m123.current.model import load_model, setup, reference_bank, forward_batch, LIVE
from sar_m123.current.training import train

ARMS = ('legacy', 'spatial')
SEEDS = (17, 29, 43)
KEYS = [(arm, seed) for seed in SEEDS for arm in ARMS]
BATCH = 16


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def load_rows():
    rows = {name: read_manifest(RELEASE/f'manifests/{file}.json') for name, file in
            (('fit', 'source_fit'), ('adapt', 'source_adapt'),
             ('development', 'development'), ('references', 'references'))}
    rows['expanded'] = read_manifest(ROOT/'m2_preadd_expansion_20261028/query_records.json')
    assert [len(rows[k]) for k in ('fit', 'adapt', 'development', 'expanded', 'references')] == [1205, 136, 160, 3075, 16]
    for name, group in rows.items():
        assert len({r['id'] for r in group}) == len(group), name
    # Source fit/adapt and historical development/expansion are distinct cohorts.
    for left, right in (('fit', 'adapt'), ('fit', 'development'), ('adapt', 'development'),
                        ('fit', 'expanded'), ('adapt', 'expanded'), ('development', 'expanded')):
        for key in ('id', 'mmsi', 'pass_group'):
            overlap = {str(r[key]) for r in rows[left]} & {str(r[key]) for r in rows[right]}
            assert not overlap, (left, right, key, sorted(overlap))
    write(HERE/'cohorts.json', rows)
    return rows


def checkpoint(arm, seed, stage):
    return (FIT if stage == 'before' else HERE)/'training'/f'{arm}_s{seed}'/'model.pt'


def save_scores(scope, stage, arm, seed, scores, records):
    folder = HERE/'predictions'
    folder.mkdir(exist_ok=True)
    assert np.isfinite(scores).all()
    np.savez_compressed(folder/f'{scope}_{stage}_{arm}_s{seed}.npz', score=scores,
        ids=np.array([str(r['id']) for r in records]), y=np.array([r['label'] for r in records]))


def from_prefix(net, values, prefix, bank, legal):
    return net.forward_from(prefix, **{k: v for k, v in values.items() if k != 'images'},
                            reference=bank, pair_valid=legal)


def check_prefix(models):
    first = next(iter(models.values())).state_dict()
    names = [k for k in first if k.startswith('backbone.model.') and
             not k.startswith(('backbone.model.layer3.', 'backbone.model.layer4.', 'backbone.model.fc.'))]
    for name, model in models.items():
        state = model.state_dict()
        assert all(torch.equal(first[k], state[k]) for k in names), ('shared_prefix', name)


@torch.no_grad()
def select(rows):
    paths = {key: HERE/'selection'/f'{key[0]}_s{key[1]}.json' for key in KEYS}
    assert not any(p.exists() for p in paths.values()), 'Selection already exists; do not overwrite.'
    models = {key: load_model(checkpoint(*key, 'before'), 'cuda') for key in KEYS}
    check_prefix(models)
    first = next(iter(models.values()))
    bank = reference_bank(first, rows['references'], 'cuda')
    results = {key: [] for key in KEYS}
    tick = time.perf_counter()
    for index, record in enumerate(rows['adapt']):
        items, meta = variants(record)
        scores = {key: [] for key in KEYS}
        for start in range(0, len(items), 8):
            values = batch(items[start:start+8], 'cuda')
            prefix = first.prefix(values['images'], stop=2)
            legal = pair_mask([record]*len(values['images']), rows['references'], 'cuda')
            for key, net in models.items():
                scores[key].append(from_prefix(net, values, prefix, bank, legal).cpu().numpy())
        y = record['label']
        for key, net in models.items():
            all_scores = np.concatenate(scores[key])
            choice = select_teacher(all_scores, meta, y)
            altered = {k: v.copy() for k, v in items[choice['chosen_index']].items()}
            for frequency_key in FREQUENCY_KEYS:
                altered[frequency_key] = items[0][frequency_key]
            changed = forward_batch(net, batch([altered], 'cuda'), bank,
                pair_mask([record], rows['references'], 'cuda'))[0].cpu()
            original = torch.tensor(all_scores[0])
            def margin(z):
                competitors = z.clone()
                competitors[y] = -torch.inf
                return z[y]-competitors.max()
            useful = choice['useful'] and bool((changed.log_softmax(0)[y] > original.log_softmax(0)[y])
                                               & (margin(changed) > margin(original)))
            results[key].append(dict(id=record['id'], label=y, useful=useful,
                original=original.tolist(), altered=changed.tolist(), choice=choice,
                selected_intervention=meta[choice['chosen_index']]))
        if index % 16 == 0 or index+1 == len(rows['adapt']):
            print('SELECT', index+1, len(rows['adapt']), round(time.perf_counter()-tick, 1), flush=True)
    # One complete record per architecture verifies the batched orchestration
    # against the existing native selection API, including its second confirmation.
    checks = {}
    for arm in ARMS:
        expected = select_records(models[arm, 17], rows['adapt'][:1], rows['references'], 'cuda')[0]
        actual = results[arm, 17][0]
        assert expected['choice'] == actual['choice'] and expected['useful'] == actual['useful']
        error = max(float(np.max(abs(np.array(actual[k])-expected[k]))) for k in ('original', 'altered'))
        assert error < 1e-4, (arm, error)
        checks[arm] = error
    for key, values in results.items():
        write(paths[key], values)
    write(HERE/'selection_check.json', dict(native_api_errors=checks, source_count=136,
        own_teacher=True, explicit_frequency_confirmation='original fields retained',
        seconds=time.perf_counter()-tick, counts={f'{a}_s{s}': sum(x['useful'] for x in v)
                                               for (a, s), v in results.items()}))


def adapt(rows):
    for arm, seed in KEYS:
        folder = HERE/'training'/f'{arm}_s{seed}'
        assert not (folder/'model.pt').exists(), f'Already trained: {folder}'
        selected = json.loads((HERE/'selection'/f'{arm}_s{seed}.json').read_text(encoding='utf-8'))
        net = load_model(checkpoint(arm, seed, 'before'), 'cuda')
        fixed = {k: v.detach().cpu().clone() for k, v in net.state_dict().items() if k not in LIVE}
        initial_live = {k: v.detach().cpu().clone() for k, v in net.state_dict().items() if k in LIVE}
        tick = time.perf_counter()
        receipt = train(net, rows['adapt'], rows['references'], folder, 'cuda', stage='adapt',
                        selection=selected, seed=seed, epochs=6, batch_size=16)
        assert receipt['steps'] == 54 and receipt['completed_epochs'] == 6
        assert all(torch.equal(net.state_dict()[k].cpu(), v) for k, v in fixed.items())
        updates = {k: float((net.state_dict()[k].cpu()-v).norm()) for k, v in initial_live.items()}
        assert all(v > 0 for v in updates.values())
        del fixed, initial_live
        net.requires_grad_(False)
        bank = reference_bank(net, rows['references'], 'cuda')
        qrows = rows['development'][:16]
        values = batch([prepare_record(r) for r in qrows], 'cuda')
        legal = pair_mask(qrows, rows['references'], 'cuda')
        with torch.no_grad():
            score = forward_batch(net, values, bank, legal).cpu()
        restored = load_model(folder/'model.pt', 'cuda')
        with torch.no_grad():
            replay = forward_batch(restored, values, bank, legal).cpu()
        error = (score-replay).abs().max().item()
        assert error < 1e-5, error
        receipt.update(arm=arm, seconds=time.perf_counter()-tick, frozen_unchanged=True,
                       live_updates=updates, save_replay_max_error=error,
                       source_fit_checkpoint=str(checkpoint(arm, seed, 'before')))
        write(folder/'training.json', receipt)
        print('ADAPT_COMPLETE', arm, seed, receipt['selected_count'], round(receipt['seconds'], 1), flush=True)
        del net, restored, values, bank
        torch.cuda.empty_cache()


@torch.no_grad()
def evaluate(rows):
    models = {(stage, arm, seed): load_model(checkpoint(arm, seed, stage), 'cuda')
              for stage in ('before', 'after') for arm, seed in KEYS}
    check_prefix(models)
    first = next(iter(models.values()))
    bank = reference_bank(first, rows['references'], 'cuda')
    tick = time.perf_counter()
    replay_errors = {}
    for scope in ('development', 'expanded'):
        output = {key: [] for key in models}
        for start in range(0, len(rows[scope]), BATCH):
            qrows = rows[scope][start:start+BATCH]
            values = batch([prepare_record(r) for r in qrows], 'cuda')
            prefix = first.prefix(values['images'], stop=2)
            legal = pair_mask(qrows, rows['references'], 'cuda')
            for key, net in models.items():
                scores = from_prefix(net, values, prefix, bank, legal).cpu().numpy()
                if scope == 'development' and start == 0:
                    if key[0] == 'before':
                        with np.load(FIT/'predictions'/f'{key[1]}_s{key[2]}.npz') as saved:
                            assert saved['ids'][:len(qrows)].tolist() == [r['id'] for r in qrows]
                            expected = saved['score'][:len(qrows)]
                    else:
                        expected = forward_batch(net, values, bank, legal).cpu().numpy()
                    error = float(abs(expected-scores).max())
                    assert error < 1e-4, (key, error)
                    replay_errors['_'.join(map(str, key))] = error
                output[key].append(scores)
            if start % 256 == 0 or start+BATCH >= len(rows[scope]):
                print('EVAL', scope, min(start+BATCH, len(rows[scope])), len(rows[scope]),
                      round(time.perf_counter()-tick, 1), flush=True)
        for (stage, arm, seed), scores in output.items():
            save_scores(scope, stage, arm, seed, np.concatenate(scores), rows[scope])
    write(HERE/'evaluation_check.json', dict(replay_errors=replay_errors,
        development=160, expanded=3075, models=12, seconds=time.perf_counter()-tick,
        labels_used_for_inference=False, default_checkpoint_changed=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('select', 'adapt', 'evaluate', 'all'))
    args = parser.parse_args()
    setup('cuda')
    rows = load_rows()
    for command in ('select', 'adapt', 'evaluate'):
        if args.command in (command, 'all'):
            globals()[command](rows)
            torch.cuda.empty_cache()
    print('COMPLETE', args.command, flush=True)


if __name__ == '__main__':
    main()
