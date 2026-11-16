"""Receipt summaries and fixed-checkpoint diagnostics; --self-test never imports torch."""
import argparse
import ast
import csv
import gc
import html
import io
import json
import os
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np

ROOT = Path(__file__).resolve().parent
TARGET = 'fine_center_product'
NAMES = ('Bulk', 'General cargo', 'Chemical/product tanker', 'Container')
REGIONS = ('body', 'ends', 'side')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def arms_from_source():
    # Read the current arm list without importing the GPU model for receipt-only work.
    for node in ast.parse((ROOT/'model.py').read_text(encoding='utf-8-sig')).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ARMS' for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise ValueError('model.py has no literal ARMS assignment')


def stats(values):
    a = np.asarray(values, dtype=float).reshape(-1)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(n=0, mean=None, median=None, p90=None, max=None)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p90=float(np.quantile(a, .9)), max=float(a.max()))


def metrics(y, score):
    y = np.asarray(y, dtype=int)
    pred = score.argmax(1)
    cm = np.zeros((score.shape[1], score.shape[1]), dtype=int)
    np.add.at(cm, (y, pred), 1)
    present = cm.sum(1) > 0
    recall = np.divide(cm.diagonal(), cm.sum(1), out=np.zeros(len(cm), float), where=present)
    return dict(n=len(y), oa=float(100*(pred == y).mean()), ba=float(100*recall[present].mean()),
                confusion=cm.tolist())


def fixed_rival(score, y):
    other = score.copy()
    other[np.arange(len(y)), y] = -np.inf
    return other.argmax(1)


def margin(score, y, rival=None):
    if rival is None:
        rival = fixed_rival(score, y)
    return score[np.arange(len(y)), y]-score[np.arange(len(y)), rival]


def paired(y, score, baseline):
    a, b = score.argmax(1) == y, baseline.argmax(1) == y
    return dict(rescued=int((a & ~b).sum()), harmed=int((~a & b).sum()),
                stable_correct=int((a & b).sum()), stable_wrong=int((~a & ~b).sum()),
                prediction_changes=int((score.argmax(1) != baseline.argmax(1)).sum()))


def constant_shift(actual, baseline, y, eligible=None):
    if eligible is None:
        eligible = np.ones(len(y), bool)
    delta = actual.astype(float)-baseline.astype(float)
    delta -= delta.mean(1, keepdims=True)
    if not eligible.any():
        return dict(n=0, mean_centered_logits=None)
    selected = delta[eligible]
    constant = selected.mean(0)
    total_energy = float(np.square(selected).sum(1).mean())
    constant_energy = float(np.square(constant).sum())
    approximation = baseline.astype(float).copy()
    approximation[eligible] += constant
    changes = eligible & (actual.argmax(1) != baseline.argmax(1))
    return dict(n=int(eligible.sum()), mean_centered_logits=constant.tolist(),
                mean_centered_logit_energy=constant_energy, mean_squared_change=total_energy,
                residual_squared_change=float(np.square(selected-constant).sum(1).mean()),
                fraction_squared_change=constant_energy/total_energy if total_energy > 0 else None,
                matched_predictions=int(((actual.argmax(1) == approximation.argmax(1)) & eligible).sum()),
                changed_predictions=int(changes.sum()),
                matched_changed_predictions=int(((actual.argmax(1) == approximation.argmax(1)) & changes).sum()),
                approximation_metrics=metrics(y, approximation),
                note='Validation-fitted descriptive mean; not an independent trained calibration control.')


def choose_cases(y, scores, arm, ids):
    base, actual = scores['initial_full'], scores[arm]
    rival = fixed_rival(base, y)
    change = margin(actual, y, rival)-margin(base, y, rival)
    a, b = actual.argmax(1) == y, base.argmax(1) == y
    states = dict(rescued=a & ~b, harmed=~a & b, stable_correct=a & b, stable_wrong=~a & ~b)
    selected = []
    counts = {k: int(v.sum()) for k, v in states.items()}
    for state, mask in states.items():
        candidates = np.flatnonzero(mask)
        if len(candidates):
            order = np.lexsort((ids[candidates], abs(change[candidates])))
            selected.append((state, int(candidates[order[len(order)//2]])))
    if arm == TARGET:
        for control in ('fine_center_gate', 'fine_independent', 'bias_control'):
            if control in scores:
                mask = a & (scores[control].argmax(1) != y)
                counts['better_than_'+control] = int(mask.sum())
                candidates = np.flatnonzero(mask)
                if len(candidates):
                    order = np.lexsort((ids[candidates], abs(change[candidates])))
                    selected.append(('better_than_'+control, int(candidates[order[len(order)//2]])))
    if arm == 'fine_std':
        prior_harm = np.flatnonzero(ids == 925)
        if len(prior_harm):
            selected.append(('previous_harm_recheck', int(prior_harm[0])))
    return selected, counts


def load_score(path, ids, y):
    with np.load(path, allow_pickle=False) as data:
        got = np.asarray(data['ids'], dtype=int)
        if len(np.unique(got)) != len(ids) or set(got) != set(ids):
            raise ValueError(f'Incomplete or different cohort: {path}')
        lookup = {int(v): i for i, v in enumerate(got)}
        order = [lookup[int(v)] for v in ids]
        score = np.asarray(data['score'][order], dtype=np.float64)
        for key in ('y', 'label'):
            if key in data and not np.array_equal(data[key][order], y):
                raise ValueError(f'Label mismatch: {path}')
    if score.shape != (len(ids), 4) or not np.isfinite(score).all():
        raise ValueError(f'Invalid scores: {path}')
    return score


def receipt_summary(arms, receipts):
    records = []
    for arm in arms:
        if arm not in receipts:
            records.append(dict(arm=arm, state='pending', endpoint='', epoch='', oa='', ba='',
                                macro_f1='', nll='', seconds='', trainable=''))
            continue
        r = receipts[arm]
        for endpoint, epoch, m in (('initial', 0, r['initial']),
                                  ('best', r['best']['epoch'], r['best']['metrics']),
                                  ('last', r['epochs'], r['last'])):
            records.append(dict(arm=arm, state='complete', endpoint=endpoint, epoch=epoch,
                                **{k: m.get(k, '') for k in ('oa', 'ba', 'macro_f1', 'nll')},
                                seconds=r['seconds'], trainable=r['trainable']))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(records)
    text = ['# PCEA readout repair: validation receipt summary', '',
            'OSS2 existing validation, seed17; not an independent test. Epoch0 remains eligible.',
            'Best and last are listed separately. A last-epoch diagnostic is not a selected gain.',
            'bias_control is the source-trained four-class bias control with matched dual-valid eligibility and training budget.', '',
            '| Arm | Best epoch | Best OA / BA | Last OA / BA |', '|---|---:|---:|---:|']
    for arm in arms:
        if arm not in receipts:
            text.append(f'| {arm} | pending | pending | pending |')
            continue
        r = receipts[arm]; b, last = r['best']['metrics'], r['last']
        text.append(f"| {arm} | {r['best']['epoch']} | {b['oa']:.3f} / {b['ba']:.3f} | {last['oa']:.3f} / {last['ba']:.3f} |")
    text += ['', 'Fixed-weight diagnostics are saved separately as analysis_{arm}.json and gallery_{arm}/.',
             'All comparison arms use the same best/last policy selected for that diagnostic arm.',
             'Branch withdrawal and no_product probe trained computation paths, not retrained ablations.',
             'Case selection and the fitted constant-shift approximation are descriptive validation analyses.']
    for path, content in ((ROOT/'RESULTS.csv', stream.getvalue()), (ROOT/'RESULTS.md', '\n'.join(text)+'\n')):
        if not path.exists() or path.read_text(encoding='utf-8') != content:
            path.write_text(content, encoding='utf-8')


def trace_diagnostics(arrays, available):
    active = arrays['active'].astype(bool)
    tokens = arrays['tokens']
    difference = np.sqrt(np.mean((tokens[:, :, :, 1:]-tokens[:, :, :, :1])**2, axis=(-2, -1)))
    magnitude = np.sqrt(np.mean(tokens[:, :, :, 0]**2, axis=-1))
    relative = difference/np.maximum(magnitude, 1e-8)
    regions = {}
    alpha_std = []
    for r, name in enumerate(REGIONS):
        entry = dict(active_samples=int(active[:, r].sum()), polarizations={})
        for p, pol in enumerate(('VV', 'VH')):
            valid = available[:, p] & (arrays['effective_cells'][:, p, r] > 0)
            entry['polarizations'][pol] = dict(
                effective_cells=stats(arrays['effective_cells'][valid, p, r]),
                class_vs_uniform_slot_rms=stats(difference[valid, p, r]),
                class_vs_uniform_slot_relative_rms=stats(relative[valid, p, r]))
            pairs = [np.sqrt(np.mean((tokens[valid, p, r, i]-tokens[valid, p, r, j])**2, axis=-1))
                     for i in range(1, 5) for j in range(i+1, 5)]
            entry['polarizations'][pol]['class_slot_pairwise_rms'] = stats(np.concatenate(pairs))
        if 'alpha' in arrays:
            a = arrays['alpha'][active[:, r], r]
            std = a.std(0) if len(a) > 1 else None
            entry['alpha_across_samples'] = dict(n=len(a), std_by_slot_and_pol=None if std is None else std.tolist())
            if std is not None:
                alpha_std.extend(std.flatten().tolist())
        regions[name] = entry
    ratio = arrays['update_rms']/np.maximum(arrays['input_rms'], 1e-8)
    return dict(regions=regions, alpha_between_query_std=stats(alpha_std),
                active_samples_any=int(active.any(1).sum()),
                update_to_input_rms=stats(ratio[available]),
                slot_definition='Raw shallow reader tokens: slot0 uniform, slots1-4 candidate-class responses; no truth-selected slot.',
                eligibility='Reader statistics: available polarization and nonempty region. Alpha: actual dual-valid active region.')


def make_gallery(model, inputs, net, reference, ids, y, scores, arm, endpoint, selected, counts, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype'] = 'none'
    torch, full = model.torch, model.full
    folder.mkdir(exist_ok=True)
    rival = fixed_rival(scores['initial_full'], y)
    records = []
    with torch.inference_mode():
        for state, j in selected:
            idx = int(ids[j]); row = inputs.rows[idx]
            z, _ = net(inputs.data([idx], 'validation'), [row], reference, inputs.reference_rows)
            if not np.allclose(z.cpu().numpy()[0], scores[arm][j], atol=1e-3, rtol=0):
                raise AssertionError('Single-case replay differs from validation replay')
            trace = net.exchange.trace
            delta = trace['delta'][0].square().mean(1).sqrt().cpu().numpy()
            legal = full.staged.task.legal([row]+inputs.reference_rows, np.array([0]),
                                         np.arange(1, 1+len(inputs.reference_rows)))[0]
            legal = legal.cpu().numpy() if torch.is_tensor(legal) else np.asarray(legal)
            references = []
            fig, axes = plt.subplots(2, 4, figsize=(12, 7))
            for p, pol in enumerate(('VV', 'VH')):
                for ax in axes[p]:
                    ax.axis('off')
                axes[p, 0].imshow(inputs.pixels[idx, p], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                suffix = '' if inputs.available[idx, p] else ' (unavailable to model)'
                axes[p, 0].set_title(pol+' native160'+suffix, fontsize=8)
                axes[p, 1].imshow(delta[p], cmap='magma', vmin=0, vmax=max(float(delta.max()), 1e-8), interpolation='nearest')
                axes[p, 1].set_title('Writeback RMS, 10x10', fontsize=9)
                for column, klass in ((2, int(y[j])), (3, int(rival[j]))):
                    candidates = [int(i) for i, ok in zip(inputs.groups['references'], legal)
                                  if ok and inputs.y[i] == klass and inputs.available[i, p]]
                    ref = candidates[0] if candidates and inputs.available[idx, p] else None
                    role = 'Same class' if column == 2 else 'Initial Full rival'
                    if ref is None:
                        axes[p, column].text(.5, .5, role+'\nNo usable legal reference', ha='center', va='center', fontsize=8)
                    else:
                        rr = inputs.rows[ref]
                        axes[p, column].imshow(inputs.pixels[ref, p], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                        axes[p, column].set_title(role+'\n'+rr['id'], fontsize=7)
                        references.append(dict(polarization=pol, role=role, index=ref, id=rr['id'], label=int(rr['label']),
                                               mmsi=rr['mmsi'], pass_group=rr['pass_group'], source_paths=rr.get('source_paths'),
                                               legal=True, selection='first legal class member with this polarization available'))
            details = {name: dict(prediction=int(s[j].argmax()), predicted_class=NAMES[int(s[j].argmax())],
                                  score=s[j].tolist(), fixed_initial_rival_margin=float(margin(s, y, rival)[j]),
                                  own_maxrival_margin=float(margin(s, y)[j])) for name, s in scores.items()}
            fig.suptitle(f"{row['id']} | {NAMES[int(y[j])]} | {state}\n{arm}, seed17, {endpoint}; existing validation\n"
                         f"Initial margin {details['initial_full']['fixed_initial_rival_margin']:+.4f}; "
                         f"arm {details[arm]['fixed_initial_rival_margin']:+.4f}; "
                         f"off {details['branch_off']['fixed_initial_rival_margin']:+.4f}; "
                         f"no product {details['no_product']['fixed_initial_rival_margin']:+.4f}", fontsize=10)
            fig.text(.5, .025, 'Fixed display 0-255; native crop/pad. Legal reference examples are not causal influence rankings.', ha='center', fontsize=8)
            fig.subplots_adjust(left=.02, right=.98, top=.78, bottom=.07, wspace=.18, hspace=.35)
            stem = f'{state}_{idx}'
            fig.savefig(folder/(stem+'.png'), dpi=150)
            fig.savefig(folder/(stem+'.svg'))
            plt.close(fig)
            records.append(dict(id=row['id'], index=idx, seed=17, arm=arm, checkpoint=endpoint,
                                split='validation', state=state, label=int(y[j]), mmsi=row['mmsi'],
                                scene_id=row.get('scene_id'), pass_group=row['pass_group'],
                                fixed_rival=int(rival[j]), source_paths=row.get('source_paths'),
                                available=inputs.available[idx].tolist(), predictions=details, references=references,
                                delta_rms=delta.tolist(), active=trace['active'][0].cpu().tolist(),
                                alpha=None if trace['alpha'] is None else trace['alpha'][0].cpu().tolist(),
                                beta=None if trace['beta'] is None else trace['beta'][0].cpu().tolist(),
                                image=stem+'.png', svg=stem+'.svg'))
    write(folder/'cases.json', dict(arm=arm, checkpoint=endpoint, state_counts=counts, cases=records,
                                  selection='Median absolute fixed-initial-rival margin change within each state; ties by manifest index. No state is fabricated.'))
    page = ['<!doctype html><meta charset="utf-8"><title>PCEA readout cases</title>',
            '<style>body{font:15px system-ui;max-width:1300px;margin:24px auto}img{max-width:100%}section{border-top:1px solid #aaa;padding:20px 0}pre{white-space:pre-wrap}</style>',
            '<h1>'+html.escape(arm)+' / '+endpoint+'</h1>',
            '<p>Existing validation, seed17. Fixed-weight withdrawals are not retrained ablations. Last-epoch diagnostics are not validation-selected gains. Missing outcome states remain absent.</p>']
    for r in records:
        text = {k: v for k, v in r.items() if k not in ('delta_rms', 'alpha', 'beta')}
        page.append(f'<section><h2>{html.escape(r["id"])} / {r["state"]}</h2><img src="{r["image"]}"><p><a href="{r["svg"]}">SVG</a></p><pre>'+html.escape(json.dumps(text, ensure_ascii=False, indent=2))+'</pre></section>')
    (folder/'index.html').write_text('\n'.join(page), encoding='utf-8')
    return records


def analyze_arm(model, inputs, arm, receipts):
    torch, full = model.torch, model.full
    receipt = receipts[arm]
    endpoint = 'last' if receipt['best']['epoch'] == 0 else 'best'
    diagnostic_epoch = receipt['epochs'] if endpoint == 'last' else receipt['best']['epoch']
    ids = inputs.groups['validation']; y = inputs.y[ids]
    initial_path = full.ROOT/'selected_s17/validation.npz'
    scores = dict(initial_full=load_score(initial_path, ids, y))
    chosen = {}
    for name, r in receipts.items():
        filename = 'last_validation.npz' if endpoint == 'last' else 'validation.npz'
        scores[name] = load_score(ROOT/f'{name}_s17'/filename, ids, y)
        chosen[name] = dict(checkpoint=endpoint, epoch=r['epochs'] if endpoint == 'last' else r['best']['epoch'])
    selected, counts = choose_cases(y, scores, arm, ids)
    net = model.Model(17, arm)
    net.load_state_dict(torch.load(ROOT/f'{arm}_s17'/(endpoint+'.pt'), map_location='cpu', weights_only=True, mmap=True), strict=True)
    net.eval().requires_grad_(False)
    arrays = {k: [] for k in ('tokens', 'effective_cells', 'active', 'alpha', 'beta', 'input_rms', 'update_rms')}
    batches = []
    try:
        with torch.inference_mode():
            reference = net.reference(inputs.data(inputs.groups['references'], 'small'))
            for start in range(0, len(ids), 12):
                part = ids[start:start+12]
                z, _ = net(inputs.data(part, 'validation'), [inputs.rows[i] for i in part], reference, inputs.reference_rows)
                batches.append(z.cpu().numpy())
                for key in arrays:
                    value = net.exchange.trace[key]
                    if value is not None:
                        arrays[key].append(value.cpu().numpy())
            actual = np.concatenate(batches)
            expected = scores[arm]
            error = float(abs(actual-expected).max())
            if error >= 1e-3 or not np.array_equal(actual.argmax(1), expected.argmax(1)):
                raise AssertionError(f'{arm}: stored checkpoint replay failed, max error {error}')
            scores[arm] = actual
            net.exchange.disabled = True
            try:
                scores['branch_off'] = full.predict(net, inputs, ids, 'validation')
            finally:
                net.exchange.disabled = False
            net.exchange.no_product = True
            try:
                scores['no_product'] = full.predict(net, inputs, ids, 'validation')
            finally:
                net.exchange.no_product = False
            off_error = float(abs(scores['branch_off']-scores['initial_full']).max())
            if off_error >= 1e-3 or not np.array_equal(scores['branch_off'].argmax(1), scores['initial_full'].argmax(1)):
                raise AssertionError(f'Branch-off does not recover frozen initial Full: {off_error}')
            arrays = {k: np.concatenate(v) for k, v in arrays.items() if v}
            active = arrays['active'].astype(bool).any(1)
            report = dict(arm=arm, seed=17, split='existing validation; not independent test', checkpoint=endpoint,
                          diagnostic_epoch=diagnostic_epoch, best_epoch=int(receipt['best']['epoch']),
                          last_fallback=endpoint == 'last',
                          selection='Target best unless epoch0; then last for all comparison arms, diagnostic only.',
                          control_checkpoints=chosen, pending_arms=[a for a in model.ARMS if a not in receipts],
                          replay=dict(max_abs_error=error, bitwise_equal=bool(np.array_equal(actual, expected)),
                                      top1_identical=True, tolerance=1e-3, branch_off_initial_error=off_error),
                          metrics={name: metrics(y, z) for name, z in scores.items()},
                          versus_initial={name: paired(y, z, scores['initial_full']) for name, z in scores.items()},
                          withdrawals={name: paired(y, actual, scores[name]) for name in ('branch_off', 'no_product')},
                          constant_shift=dict(all_queries=constant_shift(actual, scores['branch_off'], y),
                                              active_queries=constant_shift(actual, scores['branch_off'], y, active)),
                          trace=trace_diagnostics(arrays, inputs.available[ids].astype(bool)),
                          state_counts=counts,
                          no_product_note='exchange.no_product=True at unchanged weights; center subtraction is recomputed. Gate/independent/bias controls may be unaffected by design.',
                          interpretation='A constant shift fitted on validation is descriptive. bias_control is separately source-trained. Neither fixed-weight withdrawal nor scalar trace statistics prove a physical mechanism.')
            folder = ROOT/f'gallery_{arm}'
            cases = make_gallery(model, inputs, net, reference, ids, y, scores, arm, endpoint, selected, counts, folder)
            report['cases'] = dict(count=len(cases), file=str(folder/'cases.json'), gallery=str(folder/'index.html'))
            np.savez_compressed(ROOT/f'analysis_{arm}_scores.npz', ids=ids, label=y, **scores)
            write(ROOT/f'analysis_{arm}.json', report)
            print(json.dumps(dict(arm=arm, checkpoint=endpoint, metrics=report['metrics'][arm],
                                  cases=len(cases), replay_error=error), ensure_ascii=False), flush=True)
            return report
    finally:
        del net
        gc.collect()


def self_test():
    y = np.array([0, 1, 0, 1])
    base = np.array([[0., 1.], [0., 2.], [3., 0.], [2., 0.]])
    actual = np.array([[2., 1.], [3., 2.], [4., 0.], [3., 0.]])
    selected, counts = choose_cases(y, {'initial_full': base, 'candidate': actual}, 'candidate', np.arange(4))
    assert counts == dict(rescued=1, harmed=1, stable_correct=1, stable_wrong=1)
    assert [i for _, i in selected] == [0, 1, 2, 3]
    shifted = base+np.array([.25, -.25])+np.arange(4)[:, None]
    report = constant_shift(shifted, base, y)
    assert np.isclose(report['fraction_squared_change'], 1)
    assert report['matched_predictions'] == 4
    assert np.isclose(report['residual_squared_change'], 0)
    assert constant_shift(base, base, y)['fraction_squared_change'] is None
    np.testing.assert_allclose(margin(base, y), [-1, 2, 3, -2])
    assert metrics(y, base)['oa'] == 50
    tokens = np.ones((3, 2, 3, 5, 2))
    tokens[:, :, :, 1:] += np.arange(1, 5)[None, None, None, :, None]
    arrays = dict(tokens=tokens, effective_cells=np.full((3, 2, 3), 2.),
                  active=np.array([[True]*3, [True]*3, [False]*3]),
                  alpha=np.stack([np.full((3, 5, 2), v) for v in (.2, .6, .9)]),
                  input_rms=np.ones((3, 2)), update_rms=np.full((3, 2), .1))
    available = np.array([[True, True], [True, True], [True, False]])
    trace = trace_diagnostics(arrays, available)
    assert trace['regions']['body']['polarizations']['VH']['effective_cells']['n'] == 2
    assert np.isclose(trace['alpha_between_query_std']['mean'], .2)
    assert np.isclose(trace['regions']['body']['polarizations']['VV']['class_vs_uniform_slot_rms']['mean'], np.sqrt(7.5))
    assert 'torch' not in __import__('sys').modules
    print('SELF_TEST_OK: outcome selection, fixed margins, centered shift energy, zero-effect denominator, eligible trace/alpha/slot statistics; no torch import.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', default=TARGET)
    parser.add_argument('--summary-only', action='store_true', help='CPU receipt aggregation only; no torch import.')
    parser.add_argument('--self-test', action='store_true', help='Pure NumPy tests; no model, GPU, or files written.')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    arms = arms_from_source()
    if args.arm not in arms:
        parser.error(f'Unknown arm {args.arm}; expected one of {arms}')
    receipts = {a: read(ROOT/f'{a}_s17/receipt.json') for a in arms if (ROOT/f'{a}_s17/receipt.json').exists()}
    receipt_summary(arms, receipts)
    if args.summary_only:
        print(f'Receipts complete: {len(receipts)}/{len(arms)}; no model imported.')
        return
    if args.arm not in receipts:
        print(f'{args.arm} is unfinished; receipt summary only. No model imported or diagnosis fabricated.')
        return
    # Explicitly importing here keeps receipt summaries and self-tests GPU-free.
    import model
    model.torch.set_num_threads(1)
    inputs = model.full.Inputs()
    for split in ('small', 'validation'):
        if not (model.full.ROOT/'geometry_cache'/split/'complete.json').exists():
            raise FileNotFoundError(f'Missing existing geometry cache for {split}; analysis does not rebuild it.')
        inputs.prepare_geometry(split)
    report = analyze_arm(model, inputs, args.arm, receipts)
    if args.arm.startswith('fine_center') and not (report['state_counts']['rescued'] or report['state_counts']['harmed']):
        if 'fine_std' in receipts:
            analyze_arm(model, inputs, 'fine_std', receipts)
        else:
            print('fine_std receipt pending; its separate case diagnosis has not run.')


if __name__ == '__main__':
    main()
