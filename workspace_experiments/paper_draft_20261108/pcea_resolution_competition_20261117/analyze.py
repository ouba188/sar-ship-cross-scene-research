"""CPU summaries and optional same-weight causal-path interventions."""
import argparse
import csv
import gc
import html
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'pcea_conditional_modulation_20261117'
FULL = ROOT.parent/'full_finetune_20261111'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


helper = load('resolution_analysis_helper', ROOT.parent/'pcea_readout_repair_20261116/analysis.py')
read, write = helper.read, helper.write


def cached(seed):
    full = np.load(FULL/f'selected_s{seed}/validation.npz')
    ids, y = full['ids'], full['label']
    scores = {'full': full['score']}
    previous = read(OLD/'selected_checkpoints.json')[str(seed)]['affine_frozen']
    scores['pooled'] = helper.load_score(previous['validation_predictions'], ids, y)
    entries = read(ROOT/'selected_checkpoints.json').get(str(seed), {})
    for arm, entry in entries.items():
        scores[arm] = helper.load_score(entry['validation_predictions'], ids, y)
    return ids, y, scores, entries


def summarize():
    selected = read(ROOT/'selected_checkpoints.json')
    rows, cases, teacher_conditions = [], {}, {}
    manifest = read(ROOT.parent.parent/'m1_coverage_20261101/manifest.json')
    for key in selected:
        seed = int(key)
        ids, y, scores, entries = cached(seed)
        bank = np.load(ROOT/f'teacher_s{seed}.npz')
        teacher = bank['validation'][:, 0]
        conditions = []
        for split in ('small', 'validation'):
            ti = bank[split+'_ids']
            ty = np.array([manifest[i]['label'] for i in ti])
            dual = np.array([manifest[i]['vv'] is not None and manifest[i]['vh'] is not None for i in ti])
            for code in range(bank[split].shape[1]):
                tz = bank[split][:,code]
                tp = tz.argmax(1)
                tm = helper.margin(tz,ty)
                conditions.append(dict(split=split, code=code, n=len(ty), dual=int(dual.sum()),
                    wrong=int((tp!=ty).sum()), dual_wrong=int(((tp!=ty)&dual).sum()),
                    near_boundary=int((abs(tm)<.5).sum()), median_margin=float(np.median(tm)),
                    metrics=helper.metrics(ty,tz)))
        teacher_conditions[key] = conditions
        top2 = np.argsort(-teacher, axis=1)[:, :2]
        covered = (top2 == y[:, None]).any(1)
        polar = np.array([manifest[i]['vv'] is not None and manifest[i]['vh'] is not None for i in ids])
        cases[key] = dict(top2_true_coverage=int(covered.sum()), n=len(y), dual=int(polar.sum()),
            true_excluded=int((~covered & polar).sum()), predictions={})
        for arm, z in scores.items():
            metrics = helper.metrics(y, z)
            for control in ('full', 'pooled'):
                pair = helper.paired(y, z, scores[control])
                rows.append(dict(seed=seed, arm=arm, control=control, **{k:v for k,v in metrics.items() if k!='confusion'}, **pair,
                    epoch=entries.get(arm, {}).get('epoch'), config=entries.get(arm, {}).get('config')))
            changes = np.flatnonzero(z.argmax(1) != scores['pooled'].argmax(1))
            cases[key]['predictions'][arm] = [dict(index=int(ids[j]), id=manifest[ids[j]]['id'], label=int(y[j]),
                full=int(scores['full'][j].argmax()), pooled=int(scores['pooled'][j].argmax()), candidate=int(z[j].argmax()),
                margin=float(helper.margin(z, y)[j]), top2=top2[j].tolist(), dual=bool(polar[j])) for j in changes]
    write(ROOT/'summary.json', dict(rows=rows, cases=cases, teacher_conditions=teacher_conditions))
    with (ROOT/'metrics.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(rows=rows, cases=cases), ensure_ascii=False), flush=True)


def diagnose(arms, seed, last=False):
    run = load('resolution_diagnose_run', ROOT/'run.py')
    model, inputs = run.runtime()
    run.teacher(model, inputs, seed)
    torch, full = model.torch, model.full
    ids, y, scores, entries = cached(seed)
    if last:
        for arm in arms:
            entry = dict(entries[arm])
            parent = Path(entry['checkpoint']).parent
            entry['checkpoint'] = str(parent/'last.pt')
            entry['validation_predictions'] = str(parent/'lastvalidation.npz')
            entries[arm] = entry
            scores[arm] = helper.load_score(entry['validation_predictions'], ids, y)
    result = {}
    for arm in arms:
        entry = entries[arm]
        net = model.Model(seed, arm)
        net.load_state_dict(torch.load(entry['checkpoint'], map_location='cpu', weights_only=True))
        net.eval().requires_grad_(False)
        net.accelerate()
        traces = []
        def capture(module, args, output):
            traces.append({k: module.trace[k].cpu().numpy() for k in
                ('active', 'effective_cells', 'detail_rms', 'input_rms', 'update_rms', 'tokens')})
        handle = net.exchange.register_forward_hook(capture)
        actual = full.predict(net, inputs, ids, 'validation')
        handle.remove()
        trace_arrays = {k:np.concatenate([t[k] for t in traces]) for k in traces[0]}
        run.runner.replay(model, ids, actual, entry['validation_predictions'])
        variations = {'actual': actual}
        for flag in ('no_detail', 'no_competition', 'shuffle_detail', 'no_product', 'disabled'):
            setattr(net.exchange, flag, True)
            variations[flag] = full.predict(net, inputs, ids, 'validation')
            setattr(net.exchange, flag, False)
        rows = []
        for name, z in variations.items():
            rows.append(dict(variant=name, metrics=helper.metrics(y, z),
                paired_to_actual=helper.paired(y, z, actual), paired_to_full=helper.paired(y, z, scores['full']),
                max_logit_change=float(abs(z-actual).max())))
        suffix = '_last' if last else ''
        np.savez_compressed(ROOT/f'diagnostics_{arm}_s{seed}{suffix}.npz', ids=ids, label=y, **variations)
        eligible = trace_arrays['active'].any(1)
        body = trace_arrays['tokens'][:, :, 0]
        bi, pi = np.arange(len(ids))[:,None], np.arange(2)[None,:]
        rival = helper.fixed_rival(scores['full'],y)
        class_difference = body[bi,pi,(y+1)[:,None]]-body[bi,pi,(rival+1)[:,None]]
        contrast_ratio = np.sqrt(np.mean(class_difference**2,-1))/np.maximum(np.sqrt(np.mean(body[:,:,0]**2,-1)),1e-8)
        trace_summary = dict(eligible=int(eligible.sum()),
            detail_rms=helper.stats(trace_arrays['detail_rms'][eligible]),
            true_rival_body_summary_difference=helper.stats(contrast_ratio[eligible]),
            update_to_input=helper.stats((trace_arrays['update_rms']/np.maximum(trace_arrays['input_rms'],1e-8))[eligible]),
            effective_cells=helper.stats(trace_arrays['effective_cells'][eligible]),
            common_shift=helper.constant_shift(actual,scores['full'],y,eligible))
        result[arm] = dict(variants=rows, trace_summary=trace_summary)
        np.savez_compressed(ROOT/f'trace_{arm}_s{seed}{suffix}.npz', ids=ids, **trace_arrays)
        make_gallery(model, inputs, net, ids, y, scores, arm, variations, seed, suffix)
        del net
        gc.collect()
        torch.cuda.empty_cache()
    write(ROOT/f'diagnostics_s{seed}{"_last" if last else ""}.json', result)
    print(json.dumps(result), flush=True)


def make_gallery(model, inputs, net, ids, y, scores, arm, variants, seed, suffix=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype'] = 'none'
    folder = ROOT/f'gallery_{arm}_s{seed}{suffix}'
    folder.mkdir(exist_ok=True)
    aliases = dict(scores, initial_full=scores['pooled'])
    chosen, counts = helper.choose_cases(y, aliases, arm, ids)
    # Prefer a persistent error whose teacher top2 contains the truth for the mechanism question.
    teacher = inputs.teacher['validation'][:, 0].cpu().numpy()
    top2 = np.argsort(-teacher, axis=1)[:, :2]
    eligible = np.array([bool(inputs.available[i].all()) for i in ids])
    change = helper.margin(scores[arm],y)-helper.margin(scores['pooled'],y)
    active_ids = np.flatnonzero(eligible)
    additions = [('max_margin_gain', int(active_ids[np.argmax(change[active_ids])])),
                 ('max_margin_loss', int(active_ids[np.argmin(change[active_ids])]))]
    known = [j for j,i in enumerate(ids) if inputs.rows[i]['id'].endswith('_000414')]
    additions += [('shared_rescue_000414',j) for j in known]
    for state,j in additions:
        if j not in [q for _,q in chosen]: chosen.append((state,j))
    need = np.flatnonzero(eligible & (scores[arm].argmax(1) != y) & (top2 == y[:,None]).any(1))
    if len(need):
        j = int(need[np.argmin(abs(helper.margin(scores[arm], y)[need]))])
        if j not in [q for _,q in chosen]: chosen.append(('near_error_top2', j))
    records = []
    with model.torch.inference_mode():
        ref = net.reference(inputs.data(inputs.groups['references'], 'small'))
    for state, j in chosen:
        idx, row = int(ids[j]), inputs.rows[ids[j]]
        with model.torch.inference_mode():
            z, _ = net(inputs.data([idx], 'validation'), [row], ref, inputs.reference_rows)
        assert np.max(abs(z.cpu().numpy()[0]-scores[arm][j])) < 1e-3
        trace = net.exchange.trace
        delta = trace['delta'][0].square().mean(1).sqrt().cpu().numpy()
        legal = model.full.staged.task.legal([row]+inputs.reference_rows, np.array([0]), np.arange(1,33))[0]
        legal = legal.cpu().numpy() if model.torch.is_tensor(legal) else np.asarray(legal)
        rival = int(helper.fixed_rival(scores['pooled'], y)[j])
        fig, axes = plt.subplots(2, 4, figsize=(12, 7))
        refs = []
        for p, pol in enumerate(('VV', 'VH')):
            for ax in axes[p]: ax.axis('off')
            axes[p,0].imshow(inputs.pixels[idx,p], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            axes[p,0].set_title(pol+' native160', fontsize=9)
            axes[p,1].imshow(delta[p], cmap='magma', vmin=0, vmax=max(delta.max(),1e-8), interpolation='nearest')
            axes[p,1].set_title('Writeback RMS 10x10', fontsize=9)
            for col, klass in ((2,int(y[j])),(3,rival)):
                usable = [int(i) for i,ok in zip(inputs.groups['references'],legal) if ok and inputs.y[i]==klass and inputs.available[i,p]]
                if usable:
                    ri=usable[0]
                    axes[p,col].imshow(inputs.pixels[ri,p], cmap='gray', vmin=0,vmax=255,interpolation='nearest')
                    axes[p,col].set_title(('True' if col==2 else 'Rival')+' '+inputs.rows[ri]['id'],fontsize=7)
                    refs.append(dict(index=ri,id=inputs.rows[ri]['id'],polarization=pol,role='true' if col==2 else 'rival'))
        details={name:dict(prediction=int(z[j].argmax()),margin=float(helper.margin(z,y)[j]),logits=z[j].tolist()) for name,z in dict(scores,**variants).items()}
        fig.suptitle(f'{row["id"]} | {helper.NAMES[int(y[j])]} | {arm} seed{seed} | {state}\n'+
                     ' / '.join(f'{k}: {details[k]["margin"]:+.3f}' for k in ('full','pooled',arm)),fontsize=10)
        fig.text(.5,.025,'Native crop/pad, no image scaling. Reference examples are legal, not influence rankings.',ha='center',fontsize=8)
        fig.subplots_adjust(left=.02,right=.98,top=.82,bottom=.07,wspace=.15,hspace=.3)
        name=f'{state}_{idx}'
        fig.savefig(folder/(name+'.png'),dpi=120)
        fig.savefig(folder/(name+'.svg'))
        plt.close(fig)
        records.append(dict(id=row['id'],index=idx,label=int(y[j]),state=state,top2=top2[j].tolist(),
            active=trace['active'][0].cpu().tolist(),detail_rms=trace.get('detail_rms',model.torch.zeros(1,2))[0].cpu().tolist(),
            source_paths=row.get('source_paths'),references=refs,scores=details,image=name+'.png'))
    write(folder/'cases_local.json',dict(counts=counts,cases=records))
    page='<meta charset="utf-8"><title>PCEA paired diagnostics</title><style>body{font:15px sans-serif;margin:24px}img{max-width:100%}</style>'
    for case in records: page+=f'<h2>{html.escape(case["id"])} / {case["state"]}</h2><img src="{case["image"]}">'
    (folder/'index.html').write_text(page,encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--diagnose',action='store_true')
    parser.add_argument('--arms',nargs='+',default=['both'])
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--last',action='store_true')
    args=parser.parse_args()
    summarize()
    if args.diagnose: diagnose(args.arms,args.seed,args.last)
