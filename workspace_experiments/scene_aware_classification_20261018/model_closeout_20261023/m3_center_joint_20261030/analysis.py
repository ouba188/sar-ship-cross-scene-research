"""Development-only CPU analysis. Reuse coverage metrics, pairing and native panels."""
import argparse
from collections import Counter
import html
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = HERE.parents[2]
OUT = HERE / 'analysis'
ARMS = ('raw_joint', 'center_joint')
SEEDS = (17, 29, 43)
STATES = ('rescued', 'harmed', 'CC', 'WW')
spec = importlib.util.spec_from_file_location('joint_coverage_cpu', ROOT / 'm3_reference_coverage_20261030/analysis.py')
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)
prior = coverage.prior
read, dump, metric, events = coverage.read, coverage.dump, coverage.metric, coverage.events
METRICS = coverage.METRICS


def write_csv(path, rows):
    if rows:
        coverage.csv_file(path, rows)
    else:
        path.write_text('', encoding='utf-8')


def manifest_dir(dataset):
    return (WORK / 'sar_m123_current_20261027/manifests' if dataset == 'private'
            else ROOT / 'final_oss2_replication_20261029/manifests')


def load_dataset(dataset, requested):
    manifest = manifest_dir(dataset)
    group = prior.rows(manifest / ('development.json' if dataset == 'private' else 'validation.json'))
    classes = prior.PRIVATE if dataset == 'private' else prior.PUBLIC
    scores, receipts, pending, seeds = {}, {}, [], []
    for seed in requested:
        paths = [HERE / dataset / folder / f'{arm}_s{seed}.{ext}'
                 for arm in ARMS for folder, ext in (('training', 'json'), ('predictions', 'npz'))]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            pending.extend(missing)
            continue
        for arm in ARMS:
            receipt = read(HERE / dataset / 'training' / f'{arm}_s{seed}.json')
            assert (receipt['dataset'], receipt['arm'], receipt['seed']) == (dataset, arm, seed)
            assert receipt['query_count'] == len(group) and not receipt['evaluation_used']
            assert receipt['steps'] == (54 if dataset == 'private' else 96)
            assert receipt['epochs'] == 6 and receipt['lr'] == 3e-5
            assert receipt['frozen_unchanged']
            assert max(receipt['restore_error'], receipt['native_error'], receipt['teacher_error']) < 1e-4
            pack = coverage.load_npz(HERE / dataset / 'predictions' / f'{arm}_s{seed}.npz', group, classes)
            assert set(pack) == {'score', 'baseline'}
            y = np.array([r['label'] for r in group])
            for field, phase in (('score', 'after'), ('baseline', 'baseline')):
                actual = metric(pack[field], y)
                assert all(abs(actual[k] - receipt[phase][k.lower()]) < 1e-6 for k in METRICS)
            if ('saved_baseline', seed) in scores:
                assert np.array_equal(scores['saved_baseline', seed], pack['baseline'])
                other = receipts['raw_joint', seed]
                assert receipt['before'] == other['before']
                assert receipt['source_count'] == other['source_count']
            scores['saved_baseline', seed] = pack['baseline']
            scores[arm, seed] = pack['score']
            receipts[arm, seed] = receipt
        seeds.append(seed)
    return dict(dataset=dataset, group=group, classes=classes, scores=scores, receipts=receipts,
                seeds=seeds, requested=list(requested), pending=pending,
                refs=prior.rows(manifest / 'references.json'), y=np.array([r['label'] for r in group]))


def source_composition(records, classes, seeds, teachers):
    """Mirror adapt class_weights and the batch-normalized CE/KL coefficient."""
    labels = np.array([r['label'] for r in records])
    counts = Counter(labels.tolist())
    weights = np.array([len(records) / (len(counts) * counts[c]) for c in labels], dtype=np.float32)
    normalized = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        exposure = np.zeros(len(records), dtype=float)
        for _ in range(6):
            for ix in np.array_split(rng.permutation(len(records)), range(16, len(records), 16)):
                exposure[ix] += weights[ix] / weights[ix].sum(dtype=np.float32)
        normalized[seed] = exposure
    rows = []
    for c, name in enumerate(classes):
        ix = labels == c
        cohort = [r for r in records if r['label'] == c]
        unique = lambda key: len({str(r[key]) for r in cohort if r.get(key) not in (None, '')})
        row = dict(label=c, class_name=name, source_samples=len(cohort), source_mmsi=unique('mmsi'),
                   source_passes=unique('pass_group'), source_ports=unique('port'),
                   missing_mmsi=sum(r.get('mmsi') in (None, '') for r in cohort),
                   missing_pass=sum(r.get('pass_group') in (None, '') for r in cohort),
                   weight_per_sample=float(weights[ix][0]) if ix.any() else None,
                   weight_sum=float(weights[ix].sum(dtype=np.float64)),
                   sample_share=float(ix.mean()), weight_share=float(weights[ix].sum(dtype=np.float64)/weights.sum(dtype=np.float64)))
        for seed in seeds:
            t = teachers[seed]
            correct = np.array([int(np.argmax(x['original'])) == r['label'] for r, x in zip(records, t)])
            useful = np.array([bool(x['useful']) for x in t])
            row[f's{seed}_ce_normalized_mass_6epochs'] = float(normalized[seed][ix].sum())
            row[f's{seed}_ce_normalized_share'] = float(normalized[seed][ix].sum()/normalized[seed].sum())
            row[f's{seed}_teacher_correct'] = int((ix & correct).sum())
            row[f's{seed}_useful'] = int((ix & useful).sum())
            row[f's{seed}_kl_eligible_mass_6epochs'] = float(normalized[seed][ix & correct].sum())
        rows.append(row)
    return rows


def source_rows(d):
    records = prior.rows(manifest_dir(d['dataset']) / 'source_adapt.json')
    teachers = {}
    for seed in d['seeds']:
        path = (ROOT / 'spatial_m12_full_20261029/selection' / f'spatial_s{seed}.json'
                if d['dataset'] == 'private' else ROOT / 'final_oss2_replication_20261029/teachers' / f'full_s{seed}.json')
        teachers[seed] = read(path)
        assert [(str(t['id']), t['label']) for t in teachers[seed]] == [(r['id'], r['label']) for r in records]
        assert all(d['receipts'][arm, seed]['source_count'] == len(records) for arm in ARMS)
    return source_composition(records, d['classes'], d['seeds'], teachers)


def paired_tables(d):
    scores, y, group = d['scores'], d['y'], d['group']
    endpoints, paired, tradeoffs, all_events = [], [], [], []
    for seed in d['seeds']:
        for arm in ('saved_baseline', *ARMS):
            endpoints.append(dict(arm=arm, phase='saved' if arm == 'saved_baseline' else 'after', seed=seed,
                                  n=len(y), **metric(scores[arm, seed], y)))
        for arm in ARMS:
            endpoints.append(dict(arm=arm, phase='sourcefit_before', seed=seed, n=len(y),
                                  **{k:d['receipts'][arm, seed]['before'][k.lower()] for k in METRICS}))
        for arm, reference in (('raw_joint', 'saved_baseline'), ('center_joint', 'saved_baseline'), ('center_joint', 'raw_joint')):
            old, new = scores[reference, seed], scores[arm, seed]
            es = events(old, new, group, seed)
            saved = scores['saved_baseline', seed].argmax(1)
            for i, e in enumerate(es):
                e.update(arm=arm, reference=reference, saved_prediction=int(saved[i]), saved_correct=bool(saved[i] == y[i]),
                         mmsi=group[i].get('mmsi'), pass_group=group[i].get('pass_group'), port=group[i].get('port'))
            all_events.extend(es)
            a, b = metric(old, y), metric(new, y)
            counts = Counter(e['state'] for e in es)
            paired.append(dict(arm=arm, reference=reference, seed=seed,
                **{'delta_'+k:b[k]-a[k] for k in METRICS}, **{k:counts[k] for k in STATES},
                top1_changes=sum(e['changed'] for e in es), wrong_to_wrong_changes=sum(e['changed'] and e['state']=='WW' for e in es),
                rescue_saved_wrong=sum(e['state']=='rescued' and not e['saved_correct'] for e in es),
                rescue_saved_correct=sum(e['state']=='rescued' and e['saved_correct'] for e in es),
                harm_saved_wrong=sum(e['state']=='harmed' and not e['saved_correct'] for e in es),
                harm_saved_correct=sum(e['state']=='harmed' and e['saved_correct'] for e in es),
                max_abs_logits=float(np.abs(new-old).max())))
            for c, name in enumerate(d['classes']):
                ix = y == c
                subset = [e for e in es if e['label'] == c]
                sc = Counter(e['state'] for e in subset)
                a, b = (metric(old[ix], y[ix]), metric(new[ix], y[ix])) if ix.any() else ({}, {})
                tradeoffs.append(dict(arm=arm, reference=reference, seed=seed, label=c, class_name=name,
                    queries=int(ix.sum()), old_recall=a.get('OA'), new_recall=b.get('OA'),
                    old_predicted_count=int((old.argmax(1)==c).sum()), new_predicted_count=int((new.argmax(1)==c).sum()),
                    delta_predicted_count=int((new.argmax(1)==c).sum()-(old.argmax(1)==c).sum()),
                    delta_recall_pp=b['OA']-a['OA'] if ix.any() else None,
                    delta_NLL=b['NLL']-a['NLL'] if ix.any() else None,
                    delta_Brier=b['Brier']-a['Brier'] if ix.any() else None,
                    **{k:sc[k] for k in STATES}, top1_changes=sum(e['changed'] for e in subset)))
    return endpoints, paired, tradeoffs, all_events


def summarize(d):
    out = OUT / d['dataset']
    out.mkdir(parents=True, exist_ok=True)
    endpoints, paired, tradeoffs, es = paired_tables(d)
    source = source_rows(d)
    by_label = {r['label']:r for r in source}
    tradeoffs = [dict(by_label[r['label']], **r) for r in tradeoffs]
    means = []
    for arm, reference in dict.fromkeys((r['arm'], r['reference']) for r in paired):
        rows = [r for r in paired if (r['arm'],r['reference']) == (arm,reference)]
        selected = [e for e in es if (e['arm'],e['reference']) == (arm,reference)]
        row = dict(arm=arm, reference=reference, orders=len(rows), seeds=d['seeds'])
        for k in METRICS:
            values = [r['delta_'+k] for r in rows]
            row['delta_'+k] = float(np.mean(values))
            row['sd_'+k] = float(np.std(values, ddof=1)) if len(values)>1 else None
        for state in STATES:
            subset = [e for e in selected if e['state']==state]
            row[state+'_events'] = len(subset)
            row[state+'_queries'] = len({e['id'] for e in subset})
        row['top1_change_events'] = sum(r['top1_changes'] for r in rows)
        row['top1_changed_queries'] = len({e['id'] for e in selected if e['changed']})
        means.append(row)
    training = []
    for (arm, seed), r in d['receipts'].items():
        names, grads = set(r['trainable_names']), r['gradients']
        training.append(dict(arm=arm, seed=seed, steps=r['steps'],
            selected_without_first_step_gradient=sorted(names-set(grads)),
            first_step_nonzero_gradient_names=sorted(n for n,g in grads.items() if g>0),
            first_step_zero_gradient_names=sorted(n for n,g in grads.items() if g==0),
            gradient_scope='First step only; nonzero gradient is not proof of a causal contribution.',
            before=r['before'], baseline=r['baseline'], after=r['after'],
            delta_after_before={k:r['after'][k.lower()]-r['before'][k.lower()] for k in METRICS}))
    summary = dict(dataset=d['dataset'], scope='development', seeds=d['seeds'], requested=d['requested'],
        paired_three_orders_complete=set(d['seeds'])==set(SEEDS), pending=d['pending'], query_count=len(d['group']),
        endpoint_rows=endpoints, paired_means=means, training=training,
        source_adapt=source, final_test_manifests_read=False, final_test_predictions_read=False,
        notes=['raw_joint and center_joint both start at sourcefit and use original six-epoch adaptation with LIVE plus M3 value project/output/gain at 3e-5.',
               'saved_baseline is the saved original complete model, not the sourcefit before endpoint. Before is receipt-level only; no sample-level before predictions exist here.',
               'BA averages present classes. Orders share queries/initialization; repeated query events are not independent ships.',
               'Source adapt weights use float32 N/(present_classes*class_count); per-batch CE/KL normalizes by sum(weights). Normalized mass is coefficient exposure, not gradient size or a causal explanation.',
               'Teacher-correct counts determine KL eligibility. Useful flags are reported separately; their counts alone do not determine margin-loss gradients.',
               'Selected reader gains without gradients are not counted as training contributions. No new final-test data is read.'])
    summary['endpoint_means'] = [dict(arm=arm,phase=phase,orders=len(d['seeds']),
        **{k:float(np.mean([r[k] for r in endpoints if (r['arm'],r['phase'])==(arm,phase)])) for k in METRICS})
        for arm,phase in dict.fromkeys((r['arm'],r['phase']) for r in endpoints)]
    dump(out / 'summary.json', summary)
    for name, rows in (('metrics', endpoints), ('paired', paired), ('paired_means', means), ('class_tradeoffs', tradeoffs),
                       ('source_adapt', source), ('events', es), ('changed_queries', [e for e in es if e['changed']])):
        write_csv(out / (name+'.csv'), rows)
    lines = [f"# {d['dataset']} development", '', f"Completed paired orders: {d['seeds']}; three-order complete: {summary['paired_three_orders_complete']}."]
    prior.table(lines, ['Arm','Phase','Seed',*METRICS], [[r['arm'],r['phase'],r['seed'],*[f'{r[k]:.6f}' for k in METRICS]] for r in endpoints])
    prior.table(lines, ['Candidate - reference','Seed','Delta OA','Delta BA','Rescue','Harm','All top1 changes','WW changes'],
                [[r['arm']+' - '+r['reference'],r['seed'],r['delta_OA'],r['delta_BA'],r['rescued'],r['harmed'],r['top1_changes'],r['wrong_to_wrong_changes']] for r in paired])
    lines += ['', *summary['notes']]
    (out / 'SUMMARY.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return summary


def transplant_summary(d):
    folder = HERE/d['dataset']/'transplant'
    rows, pending, completed = [], [], []
    for seed in d['seeds']:
        receipt_path = folder/f'results_s{seed}.json'
        if not receipt_path.is_file():
            pending.append(str(receipt_path))
            continue
        report = read(receipt_path)
        assert not report['evaluation_used']
        variants = {}
        baseline = d['scores']['saved_baseline',seed]
        assert all(abs(metric(baseline,d['y'])[k]-report['baseline'][k.lower()])<1e-6 for k in METRICS)
        for arm in ARMS:
            for component in ('m3_only','live_only'):
                name = arm+'_'+component
                pack = coverage.load_npz(folder/f'{name}_s{seed}.npz',d['group'],d['classes'])
                assert np.array_equal(pack['baseline'],baseline)
                score = variants[name] = pack['score']
                actual = metric(score,d['y'])
                rec = report['variants'][name]
                assert all(abs(actual[k]-rec['after'][k.lower()])<1e-6 for k in METRICS)
                for ref,z in (('saved_baseline',baseline),(arm,d['scores'][arm,seed])):
                    es = events(z,score,d['group'],seed)
                    counts = Counter(e['state'] for e in es)
                    if ref=='saved_baseline':
                        assert (counts['rescued'],counts['harmed'],sum(e['changed'] for e in es)) == (rec['rescued'],rec['harmed'],rec['changed'])
                    old = metric(z,d['y'])
                    rows.append(dict(arm=arm,component=component,reference=ref,seed=seed,**actual,
                        **{'delta_'+k:actual[k]-old[k] for k in METRICS},
                        **{k:counts[k] for k in STATES},top1_changes=sum(e['changed'] for e in es),
                        wrong_to_wrong_changes=sum(e['changed'] and e['state']=='WW' for e in es)))
        for component in ('m3_only','live_only'):
            old,new = variants['raw_joint_'+component],variants['center_joint_'+component]
            es = events(old,new,d['group'],seed)
            counts = Counter(e['state'] for e in es)
            a,b = metric(old,d['y']),metric(new,d['y'])
            rows.append(dict(arm='center_joint',component=component,reference='raw_joint_'+component,seed=seed,**b,
                **{'delta_'+k:b[k]-a[k] for k in METRICS},**{k:counts[k] for k in STATES},
                top1_changes=sum(e['changed'] for e in es),wrong_to_wrong_changes=sum(e['changed'] and e['state']=='WW' for e in es)))
        completed.append(seed)
    means = []
    for arm,component,reference in dict.fromkeys((r['arm'],r['component'],r['reference']) for r in rows):
        selected = [r for r in rows if (r['arm'],r['component'],r['reference'])==(arm,component,reference)]
        means.append(dict(arm=arm,component=component,reference=reference,orders=len(selected),
            **{k:float(np.mean([r[k] for r in selected])) for k in (*METRICS,*('delta_'+k for k in METRICS))},
            rescued_events=sum(r['rescued'] for r in selected),harmed_events=sum(r['harmed'] for r in selected),
            top1_change_events=sum(r['top1_changes'] for r in selected)))
    out = OUT/d['dataset']
    result = dict(seeds=completed,pending=pending,paired_three_orders_complete=set(completed)==set(SEEDS),means=means,
        scope='development weight-transplant diagnostic; no retraining or final-test access',
        limitation='M3-only includes its trained M3 tensors and centering state on the saved full model; LIVE-only replaces LIVE on that model. Effects need not add under nonlinear joint inference.')
    dump(out/'transplant_summary.json',result)
    write_csv(out/'transplant.csv',rows)
    write_csv(out/'transplant_means.csv',means)
    lines = ['# Development weight transplants','',f'Completed orders: {completed}.', '', result['limitation']]
    prior.table(lines,['Arm','Component','Reference','Orders','Delta OA','Delta BA','Rescue/harm events'],
        [[r['arm'],r['component'],r['reference'],r['orders'],r['delta_OA'],r['delta_BA'],f"{r['rescued_events']}/{r['harmed_events']}"] for r in means])
    (out/'TRANSPLANT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return dict(seeds=completed,pending=pending,paired_three_orders_complete=result['paired_three_orders_complete'])


def select_cases(d, reference):
    pool = [e for s in d['seeds'] for e in events(d['scores'][reference,s], d['scores']['center_joint',s], d['group'],s)]
    selected, absent, used = [], [], set()
    for state in STATES:
        rows = [e for e in pool if e['state']==state]
        frequency = Counter(e['id'] for e in rows)
        rows.sort(key=lambda e:(-frequency[e['id']], -abs(e['margin_change']), e['id'], e['seed']))
        event = next((e for e in rows if e['id'] not in used), None)
        if event is None:
            absent.append(dict(state=state, state_events=len(rows), reason='No distinct query in this state.'))
            continue
        used.add(event['id'])
        selected.append(dict(event, reference=reference, state_orders=frequency[event['id']]))
    return selected, absent


def gallery(d, reference):
    """Adapt coverage.gallery selection/panels; each comparator has its own gallery."""
    helper = coverage.module('joint_native_panel', ROOT / 'spatial_m12_refit_20261029/analysis.py')
    helper.CLASSES = tuple(d['classes'])
    native = coverage.module('joint_native_input', WORK / 'sar_m123_standalone/sar_m123/input.py')
    folder = OUT / d['dataset'] / ('four_state_cases' if reference=='saved_baseline' else 'incremental_cases')
    folder.mkdir(parents=True, exist_ok=True)
    chosen, absent = select_cases(d, reference)
    lookup = {r['id']:i for i,r in enumerate(d['group'])}
    cases, sources, blocks = [], [], []
    for selected in chosen:
        i = lookup[selected['id']]
        q = d['group'][i]
        truth, rival = q['label'], selected['rival']
        case = dict(selected, dataset=d['dataset'], scope='development', visual_status='not_reviewed',
                    class_name=d['classes'][truth], rival_name=d['classes'][rival],
                    mmsi=q.get('mmsi'), pass_group=q.get('pass_group'), port=q.get('port'), orders=[], panels={})
        for seed in d['seeds']:
            row = dict(seed=seed)
            for arm in ('saved_baseline', *ARMS):
                z = d['scores'][arm,seed][i]
                row[arm] = dict(prediction=int(z.argmax()), logits=z.tolist(), truth_rival_margin=float(z[truth]-z[rival]))
            for ref in ('saved_baseline', 'raw_joint'):
                row['state_vs_'+ref] = prior.state(row[ref]['prediction'],row['center_joint']['prediction'],truth)
            case['orders'].append(row)
        legal = [r for r in d['refs'] if helper.eligible(q,r)]
        case['all_legal_reference_ids'] = [r['id'] for r in legal]
        for pol in ('vv','vh'):
            available = lambda r: bool(r.get(pol) and r[pol].get('alignment_valid', True))
            own = [r for r in legal if r['label']==truth and available(r)]
            other = [r for r in legal if r['label']==rival and available(r)]
            shown = own[:2]+other[:2]
            members = [q,*shown]
            roles = ['QUERY']+['TRUE REF']*len(own[:2])+['RIVAL REF']*len(other[:2])
            name = f"{selected['state']}_{q['id']}_{pol}.png"
            receipts = helper.panel(members,roles,pol,folder/name,native)
            query_usable = receipts[0]['available']
            for r, entry in zip(members,receipts):
                entry.update(case_id=q['id'], panel=name, label=r['label'], class_name=d['classes'][r['label']],
                             pair_legal=entry['role']!='QUERY', usable_with_query=bool(query_usable and entry['available']))
            sources.extend(receipts)
            case['panels'][pol] = dict(file=name, same_class_available=len(own), rival_available=len(other),
                                       shown_reference_ids=[r['id'] for r in shown])
        cases.append(case)
        blocks += ['<h2>'+html.escape(f"{selected['state']}: {q['id']} / center_joint vs {reference}")+'</h2>',
                   '<pre>'+html.escape(json.dumps(case,ensure_ascii=False,indent=2))+'</pre>']
        blocks += [f'<h3>{p.upper()}</h3><img src="{case["panels"][p]["file"]}">' for p in ('vv','vh')]
    dump(folder/'cases.json',cases)
    dump(folder/'native_sources.json',sources)
    dump(folder/'selection.json',dict(reference=reference,seeds=d['seeds'],absent=absent,visual_status='not_reviewed',
        rule='Independent comparator gallery; one distinct query/state, frequency then absolute fixed-rival margin change, ID, seed. No random sampling or fallback comparator.',
        references='Actual fixed reference bank; exclude identical ID/MMSI. First two same/rival class views in bank order. Labels are posthoc display only, not attention ranking.',
        limitation='Outcome-selected examples are not prevalence or causal evidence. No new post-adaptation relation trace is available.'))
    page = '<!doctype html><meta charset="utf-8"><title>M3 joint development cases</title><style>body{font:14px Arial;margin:24px}pre{white-space:pre-wrap}img{image-rendering:pixelated;max-width:none}</style>'
    page += '<h1>'+html.escape(d['dataset']+' center_joint vs '+reference)+'</h1><p>VV/VH separate; original pixels, exact crop/pad, separate OBB overlay. Rendered, not yet visually reviewed.</p>'
    (folder/'index.html').write_text(page+''.join(blocks),encoding='utf-8')
    return dict(reference=reference,cases=len(cases),panels=2*len(cases),absent=absent,status='rendered_not_reviewed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('private','oss2','all'),default='all')
    parser.add_argument('--seeds',type=int,nargs='+',choices=SEEDS,default=list(SEEDS))
    parser.add_argument('--gallery',action='store_true')
    parser.add_argument('--transplant',action='store_true')
    args = parser.parse_args()
    status = []
    for dataset in (('private','oss2') if args.dataset=='all' else (args.dataset,)):
        d = load_dataset(dataset,sorted(set(args.seeds)))
        item = dict(dataset=dataset,seeds=d['seeds'],pending=d['pending'])
        if d['seeds']:
            summary = summarize(d)
            item['paired_three_orders_complete'] = summary['paired_three_orders_complete']
            if args.gallery:
                item['galleries'] = [gallery(d,ref) for ref in ('saved_baseline','raw_joint')]
            if args.transplant:
                item['transplant'] = transplant_summary(d)
        status.append(item)
    assert 'torch' not in sys.modules, 'CPU analysis must not import model runtime'
    OUT.mkdir(parents=True,exist_ok=True)
    receipt = dict(status=status,cpu_only=True,torch_imported=False,final_test_manifests_read=False,final_test_predictions_read=False)
    dump(OUT/f"run_{args.dataset}_receipt.json",receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
