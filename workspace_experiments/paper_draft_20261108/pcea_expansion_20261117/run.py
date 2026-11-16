"""Bounded validation tuning and frozen confirmation on an exposed held-out split.

Importing this module does not import torch or construct a model. Run phases
sequentially; do not run tune concurrently with an external oldtrain.fit job.
"""
import argparse
import csv
import gc
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT.parent / 'pcea_coordinate_repair_20261117'
ARMS = ('coordinated', 'common')
SEEDS = (17, 29, 43)
CONFIGS = {
    'base': dict(ccrel_value=1e-5, other_trainable=1e-3),
    'slow': dict(ccrel_value=1e-5, other_trainable=3e-4),
    'reader_slow': dict(ccrel_value=3e-6, other_trainable=1e-3),
}
EVALUATION_NOTE = ('evaluation378 has historical research exposure; held-out split / '
                   'frozen confirmation, not a new blind test; no evaluation selection')
RECEIPT_NOTE = ('Inherited trainable_scope prose contains literal default rates. '
                'Use learning_rates and parameter_groups for the actual rates; '
                'the inherited receipt is not modified.')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding='utf-8')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def destination(arm, seed, config):
    if seed == 17:
        parent = MODEL_ROOT if config == 'base' else ROOT / 'tuning' / config
    else:
        parent = ROOT / 'confirm'
    return parent / f'{arm}_s{seed}'


def describe(arm, seed, config):
    dest = destination(arm, seed, config)
    receipt = read(dest / 'receipt.json')
    assert receipt['arm'] == arm and receipt['seed'] == seed
    assert (receipt['source'], receipt['validation'], receipt['references'],
            receipt['epochs']) == (719, 228, 32, 4)
    assert receipt['learning_rates'] == CONFIGS[config], dest
    for name in ('best.pt', 'validation.npz'):
        if not (dest / name).is_file():
            raise FileNotFoundError(dest / name)
    best = receipt['best']
    metrics = best['metrics']
    return dict(arm=arm, seed=seed, config=config, learning_rates=CONFIGS[config],
                epoch=best['epoch'], metrics=metrics,
                key=[metrics['ba'], metrics['oa'], -metrics['nll']],
                checkpoint=str((dest / 'best.pt').resolve()),
                validation_predictions=str((dest / 'validation.npz').resolve()),
                receipt=str((dest / 'receipt.json').resolve()),
                reused_coordinate_base=seed == 17 and config == 'base',
                receipt_note=RECEIPT_NOTE)


class Runtime:
    """Load the existing implementation only when a forward or fit is needed."""

    def __init__(self):
        self.model = self.trainer = self.inputs = None

    def get(self):
        if self.model is None:
            self.model = load('expansion_coordinate_model', MODEL_ROOT / 'model.py')
            self.trainer = load('expansion_coordinate_train', MODEL_ROOT / 'train.py')
            self.inputs = self.model.full.Inputs()
            for split in ('small', 'validation'):
                self.inputs.prepare_geometry(split)
            groups = self.inputs.groups
            assert tuple(len(groups[s]) for s in ('small', 'validation', 'evaluation',
                                                 'references')) == (719, 228, 378, 32)
            assert set(groups['references']).issubset(groups['small'])
            # Inputs already asserts ship/pass/id separation among the three splits.
        return self.model, self.trainer, self.inputs


def fit_one(runtime, arm, seed, config):
    dest = destination(arm, seed, config)
    if (dest / 'receipt.json').exists():
        return describe(arm, seed, config)
    if seed == 17 and config == 'base':
        raise FileNotFoundError('Complete old base run required: ' + str(dest))
    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError(f'Incomplete run at {dest}; let the active fit finish first')
    model, trainer, inputs = runtime.get()
    names = ('ROOT', 'SEED', 'READER_LR', 'OTHER_LR')
    old = [getattr(trainer, name) for name in names]
    rates = CONFIGS[config]
    try:
        trainer.ROOT, trainer.SEED = dest.parent, seed
        trainer.READER_LR = rates['ccrel_value']
        trainer.OTHER_LR = rates['other_trainable']
        # model.ROOT stays unchanged: calibration caches remain seed-specific.
        trainer.fit(model, inputs, arm)
    finally:
        for name, value in zip(names, old):
            setattr(trainer, name, value)
    return describe(arm, seed, config)


def tune(runtime):
    path = ROOT / 'selection.json'
    if path.exists():
        print('REUSE_FROZEN_SELECTION', path, flush=True)
        return read(path)
    candidates, selected = {}, {}
    for arm in ARMS:
        candidates[arm] = [fit_one(runtime, arm, 17, config) for config in CONFIGS]
        selected[arm] = max(candidates[arm], key=lambda row: row['key'])
    selection = dict(status='frozen', selection_seed=17, selection_split='validation228',
                     rule='lexicographic BA/OA/-NLL; epoch0 eligible; ties use config order',
                     config_order=list(CONFIGS), configs=CONFIGS, arms=selected,
                     candidates=candidates, evaluation_note=EVALUATION_NOTE,
                     source_samples=719, references=32, epochs=4)
    write(path, selection)
    write(ROOT / 'selected_checkpoints.json', {'17': selected})
    print('SELECTION_FROZEN', {a: selected[a]['config'] for a in ARMS}, flush=True)
    return selection


def frozen_selection():
    selection = read(ROOT / 'selection.json')
    assert selection['status'] == 'frozen'
    return selection


def confirm(runtime):
    selection = frozen_selection()
    selected = {'17': selection['arms']}
    for seed in (29, 43):
        selected[str(seed)] = {}
        for arm in ARMS:
            config = selection['arms'][arm]['config']
            selected[str(seed)][arm] = fit_one(runtime, arm, seed, config)
            write(ROOT / 'selected_checkpoints.json', selected)
    return selected


def paired(np, labels, baseline, scores):
    a, b = baseline.argmax(1), scores.argmax(1)
    good_a, good_b = a == labels, b == labels
    rescue = int((~good_a & good_b).sum())
    harm = int((good_a & ~good_b).sum())
    return dict(rescue=rescue, harm=harm, net=rescue-harm,
                stable_good=int((good_a & good_b).sum()),
                stable_bad=int((~good_a & ~good_b).sum()),
                prediction_diff=int((a != b).sum()),
                delta_oa=100. * (rescue-harm) / len(labels))


def replay(np, actual, expected):
    error = float(np.max(np.abs(actual-expected)))
    differences = int((actual.argmax(1) != expected.argmax(1)).sum())
    assert error < 1e-3 and differences == 0, (error, differences)
    return dict(max_abs_error=error, prediction_diff=differences)


def evaluate(runtime):
    selection = frozen_selection()
    selected = {str(seed): {arm: describe(arm, seed, selection['arms'][arm]['config'])
                           for arm in ARMS} for seed in SEEDS}
    # All six checkpoints must exist before the evaluation split is loaded.
    write(ROOT / 'selected_checkpoints.json', selected)
    model, _, inputs = runtime.get()
    np, torch, full = model.np, model.torch, model.full
    inputs.load_split('evaluation')
    inputs.prepare_geometry('evaluation')
    ids = inputs.groups['evaluation']
    labels = inputs.y[ids]
    reports, table = {}, []
    for seed in SEEDS:
        output = ROOT / f'evaluation_s{seed}.npz'
        report_path = ROOT / f'evaluation_s{seed}.json'
        if output.exists() and report_path.exists():
            report = read(report_path)
            assert report['selected'] == selected[str(seed)]
            reports[str(seed)] = report
            print('SKIP_COMPLETE_EVALUATION', seed, flush=True)
            continue
        scores, checks = {}, {}
        for arm in ARMS:
            entry = selected[str(seed)][arm]
            net = model.Model(seed, arm)
            net.load_state_dict(torch.load(entry['checkpoint'], map_location='cuda',
                                           weights_only=False), strict=True)
            assert bool(net.source_ready.item())
            net.accelerate()
            val = inputs.groups['validation']
            current = full.predict(net, inputs, val, 'validation')
            with np.load(entry['validation_predictions'], allow_pickle=False) as saved:
                assert np.array_equal(val, saved['ids'])
                checks[arm + '_validation'] = replay(np, current, saved['score'])
            scores[arm] = full.predict(net, inputs, ids, 'evaluation')
            net.exchange.disabled = True
            disabled = full.predict(net, inputs, ids, 'evaluation')
            with np.load(full.ROOT / f'selected_s{seed}/evaluation.npz',
                         allow_pickle=False) as original:
                assert np.array_equal(ids, original['ids'])
                checks[arm + '_off_original_full'] = replay(np, disabled, original['score'])
            if 'initial_full' in scores:
                checks['both_arms_off'] = replay(np, disabled, scores['initial_full'])
            else:
                scores['initial_full'] = disabled
            del net
            gc.collect()
            torch.cuda.empty_cache()
        metrics = {name: full.metrics(labels, score) for name, score in scores.items()}
        comparisons = {}
        for name, left, right in (('coordinated_vs_initial', 'initial_full', 'coordinated'),
                                  ('common_vs_initial', 'initial_full', 'common'),
                                  ('coordinated_vs_common', 'common', 'coordinated')):
            comparisons[name] = paired(np, labels, scores[left], scores[right])
            comparisons[name]['delta_ba'] = metrics[right]['ba'] - metrics[left]['ba']
        report = dict(seed=seed, n=len(ids), split=EVALUATION_NOTE,
                      selected=selected[str(seed)], metrics=metrics,
                      paired=comparisons, replay=checks)
        np.savez_compressed(output, ids=ids, label=labels, **scores)
        write(report_path, report)
        reports[str(seed)] = report
        print('EVALUATED', seed, metrics, comparisons, flush=True)
    for seed, report in reports.items():
        for arm, metrics in report['metrics'].items():
            entry = selected[seed].get(arm, {})
            table.append(dict(seed=int(seed), arm=arm, config=entry.get('config', ''),
                              epoch=entry.get('epoch', ''), checkpoint=entry.get('checkpoint', ''),
                              **{k: metrics[k] for k in ('oa', 'ba', 'nll')}))
    write(ROOT / 'evaluation_metrics.json', dict(note=EVALUATION_NOTE, seeds=reports))
    with (ROOT / 'evaluation_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('tune', 'confirm', 'evaluate', 'all'), required=True)
    args = parser.parse_args()
    runtime = Runtime()
    if args.phase in ('tune', 'all'):
        tune(runtime)
    if args.phase in ('confirm', 'all'):
        confirm(runtime)
    if args.phase in ('evaluate', 'all'):
        evaluate(runtime)


if __name__ == '__main__':
    main()
