"""CPU cache summary by default; --diagnose explicitly enables model forwards.

Examples:
  python analysis.py --arm affine_frozen --seeds 17 29 43
  python analysis.py --split validation --seeds 17 --last
  python analysis.py --diagnose --split validation --arm affine_frozen --seeds 17
The candidate is specified, never selected using evaluation results.
"""
import argparse
import csv
import gc
import html
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'pcea_expansion_20261117'
FULL = ROOT.parent / 'full_finetune_20261111'
MANIFEST = ROOT.parent.parent / 'm1_coverage_20261101/manifest.json'
NAMES = ('Bulk', 'General cargo', 'Chemical/product tanker', 'Container')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The same CPU-only helper used by expansion/analysis.py; its main is not run.
helper = load('conditional_readout_helper', ROOT.parent / 'pcea_readout_repair_20261116/analysis.py')
read, write = helper.read, helper.write


def metrics(y, z):
    result = helper.metrics(y, z)
    a = z.astype(float) - z.max(1, keepdims=True)
    result['nll'] = float((np.log(np.exp(a).sum(1))-a[np.arange(len(y)), y]).mean())
    return result


def csv_write(path, rows):
    if rows:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def endpoint(entry, last):
    checkpoint = Path(entry['checkpoint'])
    if last:
        return checkpoint.parent / 'last.pt', checkpoint.parent / 'lastvalidation.npz'
    return checkpoint, Path(entry.get('validation_predictions', checkpoint.parent / 'validation.npz'))


def cached(seed, split, last, selected, manifest):
    entries = selected.get(str(seed), {})
    paths = {}
    for arm, entry in entries.items():
        paths[arm] = (endpoint(entry, last)[1] if split == 'validation' else
                      ROOT / 'evaluation' / f'{arm}_s{seed}' / 'predictions.npz')
    present = {arm: path for arm, path in paths.items() if path.exists()}
    if not present:
        return None
    with np.load(next(iter(present.values())), allow_pickle=False) as data:
        ids, y = data['ids'].astype(int), data['label'].astype(int)
    assert np.array_equal(y, [manifest[i]['label'] for i in ids])
    scores = {arm: helper.load_score(path, ids, y) for arm, path in present.items()}
    baseline = FULL / f'selected_s{seed}' / f'{split}.npz'
    if baseline.exists():
        scores['initial_full'] = helper.load_score(baseline, ids, y)
    old_path = OLD / 'selected_checkpoints.json'
    if split == 'validation' and old_path.exists():
        old_entry = read(old_path).get(str(seed), {}).get('coordinated')
        if old_entry:
            path = Path(old_entry.get('validation_predictions', Path(old_entry['checkpoint']).parent / 'validation.npz'))
            if path.exists():
                scores['old_broadcast'] = helper.load_score(path, ids, y)
    elif split == 'evaluation' and (OLD / f'evaluation_s{seed}.npz').exists():
        with np.load(OLD / f'evaluation_s{seed}.npz', allow_pickle=False) as data:
            lookup = {int(i): j for j, i in enumerate(data['ids'])}
            assert set(lookup) == set(ids)
            order = [lookup[int(i)] for i in ids]
            assert np.array_equal(data['label'][order], y)
            scores['old_broadcast'] = data['coordinated'][order].copy()
    missing = [arm for arm in ('head_only', 'broadcast_head', 'affine_frozen', 'affine_head') if arm not in scores]
    return ids, y, scores, missing


def gallery(model, inputs, net, ids, y, scores, arm, split, folder, manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype'] = 'none'
    baseline = 'head_only' if 'head_only' in scores and arm != 'head_only' else 'initial_full'
    if baseline not in scores:
        raise ValueError('No real head_only or initial_full baseline for gallery')
    aliases = dict(scores, initial_full=scores[baseline])
    chosen, counts = helper.choose_cases(y, aliases, arm, ids)
    rival = helper.fixed_rival(scores[baseline], y)
    change = helper.margin(scores[arm], y, rival)-helper.margin(scores[baseline], y, rival)
    for state, j in (('margin_gain', int(np.argmax(change))), ('margin_loss', int(np.argmin(change)))):
        if j not in [i for _, i in chosen]:
            chosen.append((state, j))
    for j, idx in enumerate(ids):
        if manifest[idx]['id'].endswith('_000414') and j not in [i for _, i in chosen]:
            chosen.append(('shared_000414', j))
    folder.mkdir(parents=True, exist_ok=True)
    torch = model.torch
    net.eval()
    with torch.inference_mode():
        ref = net.reference(inputs.data(inputs.groups['references'], 'small'))
    cases = []
    for state, j in chosen:
        idx = int(ids[j]); row = manifest[idx]
        with torch.inference_mode():
            z, _ = net(inputs.data([idx], split), [row], ref, inputs.reference_rows)
        assert np.allclose(z.cpu().numpy()[0], scores[arm][j], atol=1e-3, rtol=0)
        trace = net.exchange.trace
        delta = trace['delta'][0].square().mean(1).sqrt().cpu().numpy()
        if net.exchange.disabled:
            delta = np.zeros_like(delta)
        assert delta.shape == (2, 10, 10), delta.shape
        legal = model.full.staged.task.legal([row]+inputs.reference_rows, np.array([0]),
                                           np.arange(1, 1+len(inputs.reference_rows)))[0]
        legal = legal.cpu().numpy() if torch.is_tensor(legal) else np.asarray(legal)
        fig, axes = plt.subplots(2, 4, figsize=(12, 7))
        references = []
        for p, pol in enumerate(('VV', 'VH')):
            for ax in axes[p]:
                ax.axis('off')
            axes[p, 0].imshow(inputs.pixels[idx, p], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            axes[p, 0].set_title(pol+' native160'+(' unavailable' if not inputs.available[idx, p] else ''), fontsize=9)
            axes[p, 1].imshow(delta[p], cmap='magma', vmin=0, vmax=max(float(delta.max()), 1e-8), interpolation='nearest')
            axes[p, 1].set_title('Applied writeback RMS 10x10', fontsize=9)
            for col, klass in ((2, int(y[j])), (3, int(rival[j]))):
                usable = [int(i) for i, ok in zip(inputs.groups['references'], legal)
                          if ok and inputs.y[i] == klass and inputs.available[i, p]]
                role = 'True class' if col == 2 else baseline+' rival'
                if usable and inputs.available[idx, p]:
                    ri = usable[0]; rr = manifest[ri]
                    axes[p, col].imshow(inputs.pixels[ri, p], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                    axes[p, col].set_title(role+' '+rr['id'], fontsize=8)
                    references.append(dict(index=ri, id=rr['id'], role=role, polarization=pol,
                                           legal=True, source_paths=rr.get('source_paths')))
                else:
                    axes[p, col].text(.05, .5, 'No legal available reference', fontsize=8)
        details = {name: dict(prediction=int(z[j].argmax()), score=z[j].tolist(),
                             margin=float(helper.margin(z, y)[j]),
                             fixed_rival_margin=float(helper.margin(z, y, rival)[j])) for name, z in scores.items()}
        text = ' | '.join(f'{name}: {details[name]["fixed_rival_margin"]:+.3f}'
                          for name in dict.fromkeys((baseline, arm, 'old_broadcast')) if name in details)
        fig.suptitle(f'{row["id"]} | {NAMES[int(y[j])]} | {state}\n{split}; outcomes relative to {baseline}\n{text}', fontsize=10)
        fig.text(.5, .025, 'Native crop/pad; fixed 0-255. Legal examples, not influence rankings. Trained head retained.', ha='center', fontsize=8)
        fig.subplots_adjust(left=.02, right=.98, top=.8, bottom=.07, wspace=.18, hspace=.35)
        stem = f'{state}_{idx}'
        fig.savefig(folder / (stem+'.png'), dpi=120)
        fig.savefig(folder / (stem+'.svg'))
        plt.close(fig)
        cases.append(dict(id=row['id'], index=idx, state=state, label=int(y[j]),
                          mmsi=row['mmsi'], pass_group=row['pass_group'], source_paths=row.get('source_paths'),
                          available=inputs.available[idx].tolist(), predictions=details, references=references,
                          active=trace['active'][0].cpu().tolist(), image=stem+'.png', svg=stem+'.svg',
                          delta_display_max=float(delta.max())))
    write(folder / 'cases.json', dict(baseline=baseline, counts=counts, cases=cases,
          note='Four outcome states use median abs fixed-rival margin change; extrema and 000414 added when present.'))
    page = ['<!doctype html><meta charset="utf-8"><title>Conditional modulation</title>',
            '<style>body{font:15px sans-serif;margin:24px}img{max-width:100%}pre{white-space:pre-wrap}</style>',
            f'<h1>{html.escape(folder.name)}</h1><p>Trained head retained. Fixed-weight interventions are not retraining ablations.</p>']
    for case in cases:
        page.append(f'<h2>{case["id"]} / {case["state"]}</h2><img src="{case["image"]}"><pre>'+
                    html.escape(json.dumps(case, ensure_ascii=False, indent=2))+'</pre>')
    (folder / 'index.html').write_text('\n'.join(page), encoding='utf-8')


def diagnose(args, selected, manifest, bundles):
    model = load('conditional_diagnostic_model', ROOT / 'model.py')
    torch, full = model.torch, model.full
    inputs = full.Inputs()
    for split in dict.fromkeys(('small', 'validation', args.split)):
        inputs.load_split(split)
        inputs.prepare_geometry(split)
    for seed, (ids, y, scores, missing) in bundles.items():
        if args.arm not in scores or args.arm not in selected.get(str(seed), {}):
            print('SKIP_MISSING_CANDIDATE', seed, args.arm, flush=True)
            continue
        checkpoint, _ = endpoint(selected[str(seed)][args.arm], args.last)
        net = model.Model(seed, args.arm)
        net.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        net.eval().requires_grad_(False)
        net.accelerate()
        actual = full.predict(net, inputs, ids, args.split)
        error = float(abs(actual-scores[args.arm]).max())
        assert error < 1e-3 and np.array_equal(actual.argmax(1), scores[args.arm].argmax(1)), error
        z = dict(scores)
        skipped = []
        for flag, name in (('disabled', 'branch_off'), ('no_scale', 'no_scale'),
                           ('no_product', 'no_product'), ('force_common', 'force_common')):
            if not hasattr(net.exchange, flag):
                skipped.append(flag)
                continue
            original = getattr(net.exchange, flag)
            setattr(net.exchange, flag, True)
            try:
                z[name] = full.predict(net, inputs, ids, args.split)
            finally:
                setattr(net.exchange, flag, original)
        stem = f'{args.split}_{args.arm}_s{seed}_{"last" if args.last else "best"}'
        np.savez_compressed(ROOT / ('diagnostics_'+stem+'.npz'), ids=ids, label=y, **z)
        write(ROOT / ('diagnostics_'+stem+'.json'), dict(checkpoint=str(checkpoint), replay_error=error,
              metrics={k: metrics(y, value) for k, value in z.items()},
              paired_vs_candidate={k: helper.paired(y, value, actual) for k, value in z.items()},
              skipped_flags=skipped, note='All withdrawals retain trained head; branch_off need not equal original Full.'))
        gallery(model, inputs, net, ids, y, z, args.arm, args.split, ROOT / ('gallery_'+stem), manifest)
        del net
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('validation', 'evaluation'), default='evaluation')
    parser.add_argument('--arm', default='affine_frozen', help='Predeclared candidate, not chosen by evaluation')
    parser.add_argument('--seeds', type=int, nargs='+', default=[17, 29, 43])
    parser.add_argument('--last', action='store_true', help='Validation-only final-weight diagnostic, not selected result')
    parser.add_argument('--diagnose', action='store_true', help='Explicit GPU forwards and native gallery')
    args = parser.parse_args()
    if args.last and args.split != 'validation':
        parser.error('--last is validation-only; evaluation uses frozen selected checkpoints')
    selected, manifest = read(ROOT / 'selected_checkpoints.json'), read(MANIFEST)
    records, pairs, grouped, fits, bundles, absent = [], [], [], [], {}, {}
    for seed in args.seeds:
        data = cached(seed, args.split, args.last, selected, manifest)
        if data is None:
            absent[str(seed)] = 'No predictions for this split/endpoint'
            continue
        bundles[seed] = data
        ids, y, scores, missing = data
        absent[str(seed)] = missing
        masks = [('overall', 'all', np.ones(len(ids), bool))]
        masks += [('class', NAMES[k], y == k) for k in range(4) if (y == k).any()]
        passes = np.array([manifest[i]['pass_group'] for i in ids])
        masks += [('pass', p, passes == p) for p in sorted(set(passes))]
        for name, z in scores.items():
            m = metrics(y, z)
            records.append(dict(seed=seed, arm=name, **{k: m[k] for k in ('n', 'oa', 'ba', 'nll')}))
            for kind, value, ix in masks[1:]:
                gm = metrics(y[ix], z[ix])
                grouped.append(dict(seed=seed, arm=name, group=kind, value=value,
                                    **{k: gm[k] for k in ('n', 'oa', 'ba', 'nll')}))
        if args.arm in scores:
            for base in ('head_only', 'broadcast_head', 'old_broadcast', 'initial_full'):
                if base not in scores or base == args.arm:
                    continue
                for kind, value, ix in masks:
                    p = helper.paired(y[ix], scores[args.arm][ix], scores[base][ix])
                    pairs.append(dict(seed=seed, arm=args.arm, baseline=base, group=kind, value=value,
                                      n=int(ix.sum()), net=p['rescued']-p['harmed'], **p))
    # Count physical receipts, not aliases in selection; head_only reused fits count once.
    for parent in ('tuning', 'confirm'):
        for path in sorted((ROOT / parent).glob('**/receipt.json')):
            r = read(path)
            fits.append(dict(arm=r['arm'], seed=r['seed'], config=r['config'], trainable=r['trainable'],
                             steps=r['steps'], epochs=r['epochs'], seconds=r.get('seconds'),
                             best_epoch=r['best']['epoch'], best_oa=r['best']['metrics']['oa'],
                             best_ba=r['best']['metrics']['ba'], last_oa=r['last']['oa'], last_ba=r['last']['ba'],
                             receipt=str(path)))
    summary = {}
    for arm in sorted({r['arm'] for r in records}):
        rows = [r for r in records if r['arm'] == arm]
        summary[arm] = dict(seeds=[r['seed'] for r in rows], **{
            key: dict(mean=float(np.mean([r[key] for r in rows])),
                      std=float(np.std([r[key] for r in rows], ddof=1)) if len(rows) > 1 else None)
            for key in ('oa', 'ba', 'nll')})
    stem = f'{args.split}_{args.arm}_{"last" if args.last else "best"}'
    write(ROOT / ('analysis_'+stem+'.json'), dict(candidate=args.arm, split=args.split,
          endpoint='last diagnostic, not selected' if args.last else 'validation-selected best',
          summary=summary, records=records, paired=pairs, groups=grouped, missing=absent,
          physical_training_fits=len(fits), fits=fits,
          note='Same observations across seeds, not independent samples. Partial arms allowed. Historical evaluation exposure; not blind test.'))
    for suffix, rows in (('metrics', records), ('groups', grouped), ('paired', pairs), ('fits', fits)):
        csv_write(ROOT / (stem+'_'+suffix+'.csv'), rows)
    text = ['# Conditional modulation summary', '', f'Split: {args.split}; candidate: {args.arm}; endpoint: {"LAST DIAGNOSTIC, NOT SELECTED" if args.last else "validation-selected best"}.',
            '', '| Arm | Seeds | OA mean | BA mean | NLL mean |', '|---|---|---:|---:|---:|']
    for arm, row in summary.items():
        text.append(f'| {arm} | {row["seeds"]} | {row["oa"]["mean"]:.3f} | {row["ba"]["mean"]:.3f} | {row["nll"]["mean"]:.6f} |')
    text += ['', 'Missing predictions (not replaced by another arm): '+json.dumps(absent),
             f'Physical completed fits: {len(fits)}; see fits CSV for trainable counts and budgets.',
             'Old broadcast uses its frozen best, including when current arms are last-weight diagnostics.',
             'Repeated seeds do not multiply independent observations. Evaluation has historical exposure.',
             'Gallery, when requested, describes image evidence; physical mechanisms are not established by writeback maps.']
    (ROOT / ('RESULTS_'+stem+'.md')).write_text('\n'.join(text)+'\n', encoding='utf-8')
    print('CPU_SUMMARY', stem, 'seeds', list(bundles), 'missing', absent, flush=True)
    if args.diagnose:
        diagnose(args, selected, manifest, bundles)


if __name__ == '__main__':
    main()
