"""Frozen expansion summaries, controlled withdrawals and native-chip review."""
from pathlib import Path
import csv
import gc
import html
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
SEEDS = (17, 29, 43)
NAMES = ('Bulk', 'General cargo', 'Chemical/product tanker', 'Container')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def gallery(model, helper, inputs, net, ref, ids, y, scores, seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype'] = 'none'
    np, torch = model.np, model.torch
    chosen, counts = helper.choose_cases(y, scores, 'coordinated', ids)
    wrong = np.flatnonzero(scores['initial_full'].argmax(1) != y)
    improvement = helper.margin(scores['coordinated'], y)-helper.margin(scores['initial_full'], y)
    if len(wrong):
        for state, j in (('largest_error_margin_gain', int(wrong[np.argmax(improvement[wrong])])),
                         ('largest_error_margin_loss', int(wrong[np.argmin(improvement[wrong])]))):
            if j not in [i for _, i in chosen]:
                chosen.append((state, j))
    need = (scores['coordinated'].argmax(1) == y) & (scores['common_replacement'].argmax(1) != y)
    candidates = np.flatnonzero(need)
    if len(candidates):
        j = int(candidates[len(candidates)//2])
        if j not in [i for _, i in chosen]:
            chosen.append(('needs_query_content', j))
    folder = ROOT/f'gallery_s{seed}'
    folder.mkdir(exist_ok=True)
    rival = helper.fixed_rival(scores['initial_full'], y)
    cases = []
    for state, j in chosen:
        idx = int(ids[j]); row = inputs.rows[idx]
        with torch.inference_mode():
            score, _ = net(inputs.data([idx], 'evaluation'), [row], ref, inputs.reference_rows)
        assert np.allclose(score.cpu().numpy()[0], scores['coordinated'][j], atol=1e-3, rtol=0)
        trace = net.exchange.trace
        delta = trace['delta'][0].square().mean(1).sqrt().cpu().numpy()
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
            axes[p, 1].set_title('Actual writeback RMS, 10x10', fontsize=8)
            for column, klass in ((2, int(y[j])), (3, int(rival[j]))):
                available = [int(i) for i, ok in zip(inputs.groups['references'], legal)
                             if ok and inputs.y[i] == klass and inputs.available[i, p]]
                selected = available[0] if available and inputs.available[idx, p] else None
                role = 'Same class' if column == 2 else 'Initial rival'
                if selected is None:
                    axes[p, column].text(.5, .5, 'No usable legal reference', ha='center', fontsize=8)
                else:
                    rr = inputs.rows[selected]
                    axes[p, column].imshow(inputs.pixels[selected, p], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                    axes[p, column].set_title(role+' '+rr['id'], fontsize=8)
                    references.append(dict(index=selected, id=rr['id'], role=role, polarization=pol,
                                           source_paths=rr.get('source_paths'), legal=True))
        details = {k: dict(prediction=int(z[j].argmax()), score=z[j].tolist(),
                          margin=float(helper.margin(z, y)[j]),
                          fixed_rival_margin=float(helper.margin(z, y, rival)[j])) for k, z in scores.items()}
        subtitle = ' | '.join(f'{k}: {details[k]["margin"]:+.3f}' for k in
                              ('initial_full', 'coordinated', 'common', 'no_product'))
        fig.suptitle(f'{row["id"]} | {NAMES[int(y[j])]} | {state} | seed{seed}\nFrozen expansion378; {row["pass_group"]}\n{subtitle}', fontsize=9)
        fig.text(.5, .025, 'Fixed display 0-255; native crop/pad. References are legal examples, not influence rankings.', ha='center', fontsize=8)
        fig.subplots_adjust(left=.02, right=.98, top=.8, bottom=.07, wspace=.18, hspace=.35)
        stem = f'{state}_{idx}'
        fig.savefig(folder/(stem+'.png'), dpi=120)
        fig.savefig(folder/(stem+'.svg'))
        plt.close(fig)
        cases.append(dict(id=row['id'], index=idx, seed=seed, state=state, label=int(y[j]),
                          mmsi=row['mmsi'], pass_group=row['pass_group'], source_paths=row.get('source_paths'),
                          available=inputs.available[idx].tolist(), predictions=details, references=references,
                          active=trace['active'][0].cpu().tolist(), image=stem+'.png', svg=stem+'.svg'))
    write(folder/'cases.json', dict(counts=counts, cases=cases,
          selection='Median absolute fixed-rival margin change within each outcome state; largest error-margin gain/loss; additional query-needed case when present.'))
    page = ['<!doctype html><meta charset="utf-8"><title>PCEA expansion cases</title>',
            '<style>body{font:15px system-ui;max-width:1450px;margin:24px auto}img{max-width:100%}pre{white-space:pre-wrap}section{border-top:1px solid #aaa}</style>',
            f'<h1>Frozen expansion378 / seed{seed}</h1>',
            '<p>Prior historical exposure; no tuning on this round\'s evaluation. Fixed-weight withdrawal is not retraining.</p>']
    for case in cases:
        page.append(f'<section><h2>{case["id"]} / {case["state"]}</h2><img src="{case["image"]}"><pre>'+
                    html.escape(json.dumps(case, ensure_ascii=False, indent=2))+'</pre></section>')
    (folder/'index.html').write_text('\n'.join(page), encoding='utf-8')
    return cases


def main():
    model = load('expansion_coordinate_model', ROOT.parent/'pcea_coordinate_repair_20261117/model.py')
    helper = load('expansion_readout_analysis', ROOT.parent/'pcea_readout_repair_20261116/analysis.py')
    np, torch, full = model.np, model.torch, model.full
    inputs = full.Inputs()
    for split in ('small', 'validation', 'evaluation'):
        inputs.load_split(split)
        inputs.prepare_geometry(split)
    ids = inputs.groups['evaluation']; y = inputs.y[ids]
    records, diagnostics, all_cases, differences, development = [], {}, [], [], []
    selected = read(ROOT/'selected_checkpoints.json')
    for seed in SEEDS:
        val = inputs.groups['validation']
        with np.load(full.ROOT/f'selected_s{seed}/validation.npz') as data:
            assert np.array_equal(data['ids'], val)
            baseline_val = data['score'].copy()
        for arm in ('coordinated', 'common'):
            entry = selected[str(seed)][arm]
            with np.load(entry['validation_predictions']) as data:
                assert np.array_equal(data['ids'], val)
                z = data['score'].copy()
            development.append(dict(seed=seed, arm=arm, config=entry['config'], epoch=entry['epoch'],
                baseline=full.metrics(inputs.y[val], baseline_val), metrics=full.metrics(inputs.y[val], z),
                paired=helper.paired(inputs.y[val], z, baseline_val)))
        with np.load(ROOT/f'evaluation_s{seed}.npz') as data:
            assert np.array_equal(data['ids'], ids) and np.array_equal(data['label'], y)
            scores = {k: data[k].copy() for k in ('initial_full', 'coordinated', 'common')}
        for arm in ('initial_full', 'coordinated', 'common'):
            m = full.metrics(y, scores[arm]); paired = helper.paired(y, scores[arm], scores['initial_full'])
            records.append(dict(seed=seed, arm=arm, **{k:m[k] for k in ('oa','ba','macro_f1','nll')}, **paired))
        # run.py supplies a manifest of selected checkpoint paths; no evaluation-based selection.
        checkpoint = Path(selected[str(seed)]['coordinated']['checkpoint'])
        net = model.Model(seed, 'coordinated')
        net.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True))
        net.eval().requires_grad_(False)
        with torch.inference_mode():
            prediction = lambda: full.predict(net, inputs, ids, 'evaluation')
            actual = prediction()
            assert np.allclose(actual, scores['coordinated'], atol=1e-3, rtol=0)
            assert np.array_equal(actual.argmax(1), scores['coordinated'].argmax(1))
            for flag, name in (('disabled', 'branch_off'), ('no_product', 'no_product'),
                               ('force_common', 'common_replacement')):
                setattr(net.exchange, flag, True)
                try:
                    scores[name] = prediction()
                finally:
                    setattr(net.exchange, flag, False)
            assert np.allclose(scores['branch_off'], scores['initial_full'], atol=1e-3, rtol=0)
            assert np.array_equal(scores['branch_off'].argmax(1), scores['initial_full'].argmax(1))
            ref = net.reference(inputs.data(inputs.groups['references'], 'small'))
            cases = gallery(model, helper, inputs, net, ref, ids, y, scores, seed)
        all_cases.extend(cases)
        grouped = {}
        for field in ('label', 'pass_group', 'polarization'):
            grouped[field] = {}
            values = (np.where(inputs.available[ids].astype(bool).all(1), 'dual', 'single')
                      if field == 'polarization' else np.array([str(inputs.rows[i][field]) for i in ids]))
            for value in sorted(set(values)):
                ix = values == value
                grouped[field][value] = dict(n=int(ix.sum()),
                    metrics={k:helper.metrics(y[ix], scores[k][ix]) for k in ('initial_full','coordinated','common')},
                    paired=helper.paired(y[ix], scores['coordinated'][ix], scores['initial_full'][ix]))
        single = ~inputs.available[ids].astype(bool).all(1)
        assert np.allclose(scores['coordinated'][single], scores['initial_full'][single], atol=1e-3, rtol=0)
        before, after = (helper.margin(scores[k], y) for k in ('initial_full', 'coordinated'))
        wrong = before < 0
        diagnostics[str(seed)] = dict(metrics={k:helper.metrics(y,z) for k,z in scores.items()},
            paired={k:helper.paired(y,scores['coordinated'],z) for k,z in scores.items()}, groups=grouped,
            margin_analysis=dict(original_wrong=int(wrong.sum()),
                wrong_within_01=int((wrong & (before > -.1)).sum()),
                wrong_within_05=int((wrong & (before > -.5)).sum()),
                wrong_improved=int((wrong & (after > before+1e-5)).sum()),
                wrong_worsened=int((wrong & (after < before-1e-5)).sum()),
                error_initial_margin=helper.stats(before[wrong]),
                error_margin_change=helper.stats((after-before)[wrong]),
                all_margin_change=helper.stats(after-before)),
            note='Comparisons to withdrawals show fixed-weight dependency, not independently retrained mechanism gains.')
        np.savez_compressed(ROOT/f'diagnostics_s{seed}.npz', ids=ids, label=y, **scores)
        differences.append((scores['coordinated'].argmax(1)==y).astype(float)-(scores['initial_full'].argmax(1)==y))
        print('DIAGNOSTICS', seed, diagnostics[str(seed)]['paired'], flush=True)
        del net, ref
        gc.collect(); torch.cuda.empty_cache()
    # Cluster ships, keeping the three seed outcomes paired within each sampled ship.
    ships = np.array([str(inputs.rows[i]['mmsi']) for i in ids])
    unique = sorted(set(ships)); groups = [np.flatnonzero(ships==s) for s in unique]
    delta = np.stack(differences).mean(0)
    rng = np.random.default_rng(20261117)
    bootstrap = []
    for _ in range(2000):
        ix = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        bootstrap.append(float(100*delta[ix].mean()))
    summary = {arm:{key:dict(mean=float(np.mean([r[key] for r in records if r['arm']==arm])),
                                std=float(np.std([r[key] for r in records if r['arm']==arm],ddof=1)))
                    for key in ('oa','ba','macro_f1','nll')} for arm in ('initial_full','coordinated','common')}
    write(ROOT/'analysis.json', dict(records=records, summary=summary, diagnostics=diagnostics, development=development,
          oa_cluster_bootstrap95=np.quantile(bootstrap,[.025,.975]).tolist(),
          unique_evaluation_ships=len(unique), evaluation_observations=len(ids), seeds=list(SEEDS),
          note='Bootstrap is descriptive paired ship-resampling for this fixed historical cohort, not new-scene uncertainty.'))
    with (ROOT/'RESULTS.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    text=['# Frozen OSS2 expansion', '', '| Seed | Arm | OA | BA | Rescue | Harm |', '|---|---|---:|---:|---:|---:|']
    for r in records:
        text.append(f'|{r["seed"]}|{r["arm"]}|{r["oa"]:.3f}|{r["ba"]:.3f}|{r["rescued"]}|{r["harmed"]}|')
    text += ['', 'Mean and sample standard deviation across three seeds:', '', '| Arm | OA | BA |','|---|---:|---:|']
    for arm,s in summary.items():
        text.append(f'|{arm}|{s["oa"]["mean"]:.3f} +/- {s["oa"]["std"]:.3f}|{s["ba"]["mean"]:.3f} +/- {s["ba"]["std"]:.3f}|')
    text += ['', 'No tuning on expansion results; historical exposure prevents a new-blind-test claim.',
             'The same 378 observations are repeated across seeds, not 1134 independent examples.']
    (ROOT/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    write(ROOT/'cases_all.json', all_cases)


if __name__ == '__main__':
    main()
