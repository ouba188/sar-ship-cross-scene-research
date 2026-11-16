"""Joint-reader validation diagnostics. Receipt summaries/self-tests do not import torch."""
import argparse
import ast
import csv
import gc
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

BASE_ROOT = Path(__file__).resolve().parent
ROOT = BASE_ROOT
ARMS = ('joint', 'joint_fine', 'joint_fine_gate', 'independent_fine')
SPEC = importlib.util.spec_from_file_location(
    'joint_analysis_helpers', ROOT.parent/'pcea_readout_repair_20261116/analysis.py')
prior = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prior)
np = prior.np


def configured_arms():
    run_config = ROOT/'run_config.json'
    if run_config.exists():
        return tuple(prior.read(run_config)['planned_arms'])
    path = BASE_ROOT/'model.py'
    if path.exists():
        for node in ast.parse(path.read_text(encoding='utf-8-sig')).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ARMS' for t in node.targets):
                return tuple(ast.literal_eval(node.value))
    return ARMS


def summary_contents(arms, receipts):
    rows = []
    text = ['# Joint reader / PCEA validation summary', '',
            'Existing OSS2 validation, seed17; not an independent test or a new whole architecture.',
            'The trained read projection is a separate copy. Fixed-weight withdrawals diagnose paths, not retrained ablations.',
            'Best includes epoch0. Last is reported separately and is not a selected improvement.', '',
            '| Arm | Best epoch | Initial OA / BA | Best OA / BA | Last OA / BA |',
            '|---|---:|---:|---:|---:|---:|']
    fields = ('arm', 'status', 'endpoint', 'epoch', 'oa', 'ba', 'macro_f1', 'nll', 'seconds', 'trainable')
    for arm in arms:
        if arm not in receipts:
            rows.append(dict(arm=arm, status='pending'))
            text.append(f'| {arm} | pending | pending | pending | pending |')
            continue
        r = receipts[arm]
        initial, best, last = r['initial'], r['best']['metrics'], r['last']
        text.append(f"| {arm} | {r['best']['epoch']} | {initial['oa']:.3f} / {initial['ba']:.3f} | "
                    f"{best['oa']:.3f} / {best['ba']:.3f} | {last['oa']:.3f} / {last['ba']:.3f} |")
        for endpoint, epoch, metric in (('initial', 0, initial), ('best', r['best']['epoch'], best),
                                        ('last', r['epochs'], last)):
            rows.append(dict(arm=arm, status='complete', endpoint=endpoint, epoch=epoch,
                             **{k: metric.get(k, '') for k in ('oa', 'ba', 'macro_f1', 'nll')},
                             seconds=r.get('seconds', ''), trainable=r.get('trainable', '')))
    text += ['', 'Detailed outputs: analysis_{arm}.json, analysis_{arm}_scores.npz, gallery_{arm}/.',
             'Best epoch0 automatically selects trained last for diagnosis, not as a validation-selected gain.',
             'All last diagnostics use a _last suffix and do not overwrite best diagnostics.',
             'Controls use the same best/last endpoint policy, with each actual selected epoch recorded.',
             'No source recalibration, fitting, or independent-test evaluation occurs in this analysis.']
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return '\n'.join(text)+'\n', stream.getvalue()


def write_summary(arms, receipts):
    markdown, table = summary_contents(arms, receipts)
    for path, content in ((ROOT/'RESULTS.md', markdown), (ROOT/'RESULTS.csv', table)):
        if not path.exists() or path.read_text(encoding='utf-8') != content:
            path.write_text(content, encoding='utf-8')


def switched_prediction(owner, flag, predict):
    original = getattr(owner, flag)
    setattr(owner, flag, True)
    try:
        return predict()
    finally:
        setattr(owner, flag, original)


def withdraw(net, predict, with_fine):
    scores = dict(branch_off=switched_prediction(net.exchange, 'disabled', predict),
                  no_product=switched_prediction(net.exchange, 'no_product', predict))
    if with_fine:
        scores['fine_off'] = switched_prediction(net.ccrel, 'no_fine', predict)
        scores['residual_off'] = scores['fine_off']
    return scores


def diagnostic_checkpoint(receipt, requested):
    fallback = requested == 'best' and receipt['best']['epoch'] == 0
    endpoint = 'last' if fallback else requested
    epoch = receipt['epochs'] if endpoint == 'last' else receipt['best']['epoch']
    return endpoint, int(epoch), fallback


def score_difference(actual, other):
    delta = actual.astype(float)-other.astype(float)
    centered = delta-delta.mean(1, keepdims=True)
    return dict(prediction_diff=int((actual.argmax(1) != other.argmax(1)).sum()),
                max_abs=float(abs(delta).max()), rms=float(np.sqrt(np.mean(delta**2))),
                centered_rms=float(np.sqrt(np.mean(centered**2))),
                direction='factual minus counterfactual; same trained head')


def collect_trace(model, inputs, net, ids, reference):
    keys = ('tokens', 'effective_cells', 'active', 'alpha', 'beta', 'input_rms', 'update_rms')
    arrays = {k: [] for k in (*keys, 'fine_residual_rms')}
    batches = []
    for start in range(0, len(ids), 12):
        part = ids[start:start+12]
        z, _ = net(inputs.data(part, 'validation'), [inputs.rows[i] for i in part], reference, inputs.reference_rows)
        batches.append(z.cpu().numpy())
        for key in keys:
            value = net.exchange.trace[key]
            if value is not None:
                arrays[key].append(value.cpu().numpy())
        arrays['fine_residual_rms'].append(net.ccrel.trace['fine_residual_rms'].cpu().numpy())
    return np.concatenate(batches), {k: np.concatenate(v) for k, v in arrays.items() if v}


def with_initial_reader(net, evaluate):
    reader = net.ccrel.value
    snapshot = {k: v.detach().clone() for k, v in reader.state_dict().items()}
    try:
        reader.load_state_dict(net.net.guide.value.state_dict(), strict=True)
        return evaluate()
    finally:
        reader.load_state_dict(snapshot, strict=True)


def token_mean_drift(trained, restored, mean, scale, active):
    # Both views use checkpoint SOURCE moments, never re-estimated validation moments.
    z = (trained.astype(float)-mean)/scale
    z0 = (restored.astype(float)-mean)/scale
    means, means0 = np.zeros(z.shape[1:]), np.zeros(z.shape[1:])
    regions = {}
    counts = []
    for r, name in enumerate(prior.REGIONS):
        valid = active[:, r].astype(bool)
        counts.append(int(valid.sum()))
        region = dict(n=int(valid.sum()), polarizations={})
        for p, pol in enumerate(('VV', 'VH')):
            if not valid.any():
                region['polarizations'][pol] = dict(n=0)
                continue
            a, b = z[valid, p, r], z0[valid, p, r]
            ma, mb = a.mean(0), b.mean(0)
            means[p, r], means0[p, r] = ma, mb
            region['polarizations'][pol] = dict(
                n=len(a), trained_mean_rms=float(np.sqrt(np.mean(ma**2))),
                initial_reader_mean_rms=float(np.sqrt(np.mean(mb**2))),
                mean_drift_rms=float(np.sqrt(np.mean((ma-mb)**2))),
                paired_standardized_change_rms=float(np.sqrt(np.mean((a-b)**2))),
                trained_mean_by_slot_rms=np.sqrt(np.mean(ma**2, axis=-1)).tolist(),
                initial_reader_mean_by_slot_rms=np.sqrt(np.mean(mb**2, axis=-1)).tolist())
        regions[name] = region
    report = dict(regions=regions, eligibility='same actual dual-valid active region for both reader states',
                  note='Unweighted validation token means in fixed initial-source units. Restoring only ccrel.value retains the trained fine residual, PCEA, and normalizer; not initial-model replay or a training ablation.')
    arrays = dict(standardized_token_mean_trained=means, standardized_token_mean_initial_reader=means0,
                  standardized_token_mean_drift=means-means0, token_mean_region_counts=np.asarray(counts))
    return report, arrays


def replay_check(actual, expected, label):
    error = float(np.abs(actual-expected).max())
    top1 = bool(np.array_equal(actual.argmax(1), expected.argmax(1)))
    if not np.isfinite(actual).all() or error >= 1e-3 or not top1:
        raise AssertionError(f'{label}: replay error={error}, top1_identical={top1}')
    return dict(max_abs_error=error, top1_identical=top1,
                bitwise_equal=bool(np.array_equal(actual, expected)), tolerance=1e-3)


def select_cases(y, scores, arm, ids):
    selected, counts = prior.choose_cases(y, scores, arm, ids)
    if arm == 'joint_fine':
        rival = prior.fixed_rival(scores['initial_full'], y)
        delta = prior.margin(scores[arm], y, rival)-prior.margin(scores['initial_full'], y, rival)
        for control in ('joint', 'joint_fine_gate', 'independent_fine', 'residual_off'):
            if control not in scores:
                continue
            mask = (scores[arm].argmax(1) == y) & (scores[control].argmax(1) != y)
            name = 'better_than_'+control
            counts[name] = int(mask.sum())
            indices = np.flatnonzero(mask)
            if len(indices):
                order = np.lexsort((ids[indices], abs(delta[indices])))
                selected.append((name, int(indices[order[len(order)//2]])))
    return selected, counts


def analyze_arm(model, inputs, arm, receipts, checkpoint='best', restore_initial_reader=False):
    torch, full = model.torch, model.full
    ids = inputs.groups['validation']; y = inputs.y[ids]
    receipt = receipts[arm]
    requested_checkpoint = checkpoint
    checkpoint, epoch, fallback = diagnostic_checkpoint(receipt, checkpoint)
    filename = 'validation.npz' if checkpoint == 'best' else 'last_validation.npz'
    initial_path = full.ROOT/'selected_s17/validation.npz'
    scores = dict(initial_full=prior.load_score(initial_path, ids, y))
    selected_checkpoints = {}
    for name, r in receipts.items():
        scores[name] = prior.load_score(ROOT/f'{name}_s17'/filename, ids, y)
        selected_checkpoints[name] = dict(checkpoint=checkpoint,
                                         epoch=r['best']['epoch'] if checkpoint == 'best' else r['epochs'])
    net = model.Model(17, arm)
    net.load_state_dict(torch.load(ROOT/f'{arm}_s17'/(checkpoint+'.pt'),
                                  map_location='cpu', weights_only=True, mmap=True), strict=True)
    net.eval().requires_grad_(False)
    net.exchange.disabled = False
    net.exchange.no_product = False
    net.ccrel.no_fine = False
    try:
        with torch.inference_mode():
            reference = net.reference(inputs.data(inputs.groups['references'], 'small'))
            actual, arrays = collect_trace(model, inputs, net, ids, reference)
            checks = dict(checkpoint=replay_check(actual, scores[arm], arm))
            scores[arm] = actual
            predict = lambda: full.predict(net, inputs, ids, 'validation')
            withdrawn = withdraw(net, predict, with_fine=bool(net.ccrel.use_fine))
            scores.update(withdrawn)
            checks['branch_off_initial_full'] = replay_check(scores['branch_off'], scores['initial_full'], 'branch off')
            reader_report = dict(executed=False, option='--restore-initial-reader')
            if restore_initial_reader:
                reader_scores, restored_arrays = with_initial_reader(
                    net, lambda: collect_trace(model, inputs, net, ids, reference))
                scores['initial_reader'] = reader_scores
                reader_report, means = token_mean_drift(
                    arrays['tokens'], restored_arrays['tokens'],
                    net.exchange.mean.detach().cpu().numpy(), net.exchange.scale.detach().cpu().numpy(),
                    arrays['active'])
                if not np.array_equal(arrays['active'], restored_arrays['active']):
                    raise AssertionError('Restoring projection changed region eligibility')
                reader_report.update(executed=True, score_effect=score_difference(actual, reader_scores),
                                     factual_vs_restored=prior.paired(y, actual, reader_scores))
                arrays.update(means)
                arrays['tokens_initial_reader'] = restored_arrays['tokens']
                arrays['fine_residual_rms_initial_reader'] = restored_arrays['fine_residual_rms']
            active = arrays['active'].astype(bool).any(1)
            rival = prior.fixed_rival(scores['initial_full'], y)
            selected, counts = select_cases(y, scores, arm, ids)
            withdrawals = {}
            for name, z in withdrawn.items():
                d = prior.margin(actual, y, rival)-prior.margin(z, y, rival)
                withdrawals[name] = dict(
                    factual_vs_withdrawn=prior.paired(y, actual, z),
                    score_effect=score_difference(actual, z),
                    fixed_initial_rival_margin_effect=prior.stats(d),
                    absolute_margin_effect=prior.stats(abs(d)),
                    class_correct_change=[int(((y == k) & (actual.argmax(1) == y)).sum()-
                                              ((y == k) & (z.argmax(1) == y)).sum()) for k in range(4)])
            report = dict(arm=arm, seed=17, checkpoint=checkpoint, epoch=int(epoch),
                          requested_checkpoint=requested_checkpoint, best_epoch=int(receipt['best']['epoch']),
                          trained_last_fallback=fallback,
                          trained_checkpoint=bool(epoch > 0), last_diagnostic=checkpoint == 'last',
                          split='existing OSS2 validation; not independent test',
                          scope='Read projection/conditional relation and PCEA path study; not a new whole architecture.',
                          run_root=str(ROOT), model_root=str(BASE_ROOT),
                          initial_full=str(initial_path), control_checkpoints=selected_checkpoints,
                          pending_arms=[name for name in configured_arms() if name not in receipts],
                          replay=checks, metrics={name: prior.metrics(y, z) for name, z in scores.items()},
                          versus_initial={name: prior.paired(y, z, scores['initial_full']) for name, z in scores.items()},
                          withdrawals=withdrawals,
                          constant_shift=dict(all_queries=prior.constant_shift(actual, scores['branch_off'], y),
                                              active_queries=prior.constant_shift(actual, scores['branch_off'], y, active)),
                          trace=prior.trace_diagnostics(arrays, inputs.available[ids].astype(bool)),
                          fine_residual_rms=dict(
                              available_polarizations=prior.stats(arrays['fine_residual_rms'][inputs.available[ids].astype(bool)]),
                              active_queries=prior.stats(arrays['fine_residual_rms'][active]),
                              per_pol={pol: prior.stats(arrays['fine_residual_rms'][inputs.available[ids, p].astype(bool), p])
                                       for p, pol in enumerate(('VV', 'VH'))}),
                          initial_reader_counterfactual=reader_report,
                          state_counts=counts,
                          notes=['Best epoch0 triggers trained-last diagnosis, explicitly labeled; it is not a selected improvement.',
                                 'branch_off disables the exchange; the trained reader is a separate copy, so original Full must replay.',
                                 'no_product disables only the exchange product at fixed weights.',
                                 'fine_off sets ccrel.no_fine=True; residual_off is an identical compatibility alias, not another forward.',
                                 'All withdrawal margins fix the initial Full rival. These are inference path effects, not training ablations.',
                                 'Validation-fitted mean shifts are descriptive and do not prove a physical mechanism.'])
            suffix = arm if checkpoint == 'best' else arm+'_last'
            folder = ROOT/f'gallery_{suffix}'
            cases = prior.make_gallery(model, inputs, net, reference, ids, y, scores, arm,
                                       checkpoint, selected, counts, folder)
            report['cases'] = dict(count=len(cases), file=str(folder/'cases.json'), gallery=str(folder/'index.html'),
                                   unavailable_states=[name for name in ('rescued', 'harmed', 'stable_correct', 'stable_wrong') if not counts[name]])
            np.savez_compressed(ROOT/f'analysis_{suffix}_scores.npz', ids=ids, label=y, **scores)
            np.savez_compressed(ROOT/f'analysis_{suffix}_traces.npz', ids=ids, label=y, **arrays)
            prior.write(ROOT/f'analysis_{suffix}.json', report)
            print(json.dumps(dict(arm=arm, checkpoint=checkpoint, epoch=epoch, metrics=report['metrics'][arm],
                                  cases=len(cases), replay_error=checks['checkpoint']['max_abs_error'])), flush=True)
            return report
    finally:
        del net
        gc.collect()


def self_test():
    from types import SimpleNamespace
    prior.self_test()
    net = SimpleNamespace(exchange=SimpleNamespace(disabled=False, no_product=False), ccrel=SimpleNamespace(no_fine=False))
    observed = []
    def predict():
        state = (net.exchange.disabled, net.exchange.no_product, net.ccrel.no_fine)
        observed.append(state)
        return state
    results = withdraw(net, predict, True)
    assert observed == [(True, False, False), (False, True, False), (False, False, True)]
    assert set(results) == {'branch_off', 'no_product', 'fine_off', 'residual_off'}
    assert results['fine_off'] is results['residual_off']
    assert predict() == (False, False, False)
    def fail():
        raise RuntimeError('intentional withdrawal test')
    try:
        switched_prediction(net.ccrel, 'no_fine', fail)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Expected injected failure')
    assert not net.ccrel.no_fine
    m = dict(oa=50., ba=50., nll=1.)
    receipt = dict(initial=m, best=dict(epoch=0, metrics=m), epochs=4, last=m, seconds=1, trainable=4)
    markdown, table = summary_contents(ARMS, {'joint': receipt})
    assert 'joint_fine | pending' in markdown
    assert len(list(csv.DictReader(io.StringIO(table)))) == 6
    assert diagnostic_checkpoint(receipt, 'best') == ('last', 4, True)
    assert diagnostic_checkpoint(receipt, 'last') == ('last', 4, False)
    assert diagnostic_checkpoint(dict(receipt, best=dict(epoch=2, metrics=m)), 'best') == ('best', 2, False)
    y = np.array([0, 1, 0, 1]); ids = np.arange(4)
    base = np.array([[0., 1.], [0., 2.], [3., 0.], [2., 0.]])
    actual = np.array([[2., 1.], [3., 2.], [4., 0.], [3., 0.]])
    selected, counts = select_cases(y, dict(initial_full=base, joint_fine=actual, residual_off=base), 'joint_fine', ids)
    assert counts['better_than_residual_off'] == 1
    assert ('better_than_residual_off', 0) in selected
    assert replay_check(base, base, 'toy')['bitwise_equal']
    effect = score_difference(actual, base)
    assert effect['prediction_diff'] == 2 and effect['max_abs'] == 3
    assert np.isclose(effect['rms'], np.sqrt(15/8))
    old_tokens = np.ones((3, 2, 3, 5, 4))
    trained_tokens = old_tokens+2
    eligible = np.array([[True, True, False], [True, False, False], [False, False, False]])
    drift, arrays = token_mean_drift(trained_tokens, old_tokens, np.ones((2, 3, 5, 4)),
                                    np.full((2, 3, 5, 4), 2.), eligible)
    assert drift['regions']['body']['polarizations']['VV']['mean_drift_rms'] == 1
    assert drift['regions']['side']['polarizations']['VV']['n'] == 0
    assert arrays['token_mean_region_counts'].tolist() == [2, 1, 0]
    # Test state restoration on success AND exception without importing torch.
    class Weight:
        def __init__(self, value): self.value = value
        def detach(self): return self
        def clone(self): return Weight(self.value)
    class Reader:
        def __init__(self, value): self.weight = Weight(value)
        def state_dict(self): return {'weight': self.weight}
        def load_state_dict(self, state, strict=True): self.weight = state['weight'].clone()
    net.ccrel.value = Reader(7)
    net.net = SimpleNamespace(guide=SimpleNamespace(value=Reader(2)))
    assert with_initial_reader(net, lambda: net.ccrel.value.weight.value) == 2
    assert net.ccrel.value.weight.value == 7
    try:
        with_initial_reader(net, fail)
    except RuntimeError:
        pass
    assert net.ccrel.value.weight.value == 7
    assert 'torch' not in sys.modules
    print('JOINT_SELF_TEST_OK: withdrawal flags, best0 fallback, score differences, source-unit token drift, reader restoration, four-state cases; no torch/GPU/file writes.')


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=BASE_ROOT,
                        help='Checkpoint/receipt/output root; model imports and source calibration stay in the base directory.')
    parser.add_argument('--arms', '--arm', nargs='+', default=['joint', 'joint_fine'])
    parser.add_argument('--checkpoint', choices=('best', 'last'), default='best')
    parser.add_argument('--restore-initial-reader', action='store_true',
                        help='Counterfactual: restore only ccrel.value from frozen guide.value; retain trained fine residual/PCEA and source statistics.')
    parser.add_argument('--summary-only', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    ROOT = args.run_root.expanduser().resolve()
    if not ROOT.is_dir():
        parser.error(f'Run root does not exist: {ROOT}')
    arms = configured_arms()
    if any(a not in arms for a in args.arms):
        parser.error(f'Expected arms from {arms}')
    receipts = {a: prior.read(ROOT/f'{a}_s17/receipt.json') for a in arms if (ROOT/f'{a}_s17/receipt.json').exists()}
    write_summary(arms, receipts)
    if args.summary_only:
        print(f'{len(receipts)}/{len(arms)} completed receipts; no model imported.')
        return
    ready = [a for a in args.arms if a in receipts]
    pending = [a for a in args.arms if a not in receipts]
    if pending:
        print('Pending requested arms; no fabricated analysis: '+', '.join(pending), flush=True)
    if not ready:
        return
    import model
    model.torch.set_num_threads(1)
    inputs = model.full.Inputs()
    for split in ('small', 'validation'):
        if not (model.full.ROOT/'geometry_cache'/split/'complete.json').exists():
            raise FileNotFoundError(f'Missing existing {split} geometry; analysis does not recalibrate or build caches.')
        inputs.prepare_geometry(split)
    for arm in ready:
        analyze_arm(model, inputs, arm, receipts, args.checkpoint, args.restore_initial_reader)


if __name__ == '__main__':
    main()
