"""Evaluate fixed source-selected gates and inspect actual source/query images."""
import gc
import csv
import json
import numpy as np
import torch
import gated_run as run
from check_causal_work import scene_rows, stratified
from unified_model import late


@torch.no_grad()
def evaluate():
    inputs, _, inner = run.base.runtime('unified')
    old = run.ROOT/'causal_work_check_20261119'
    selection = np.load(old/'selection_local.npz')['validation']
    baseline_schedule = np.load(old/'baseline_s17/schedule_local.npz')
    records, probes = [], []
    rows = scene_rows()
    rr = rows[358:]
    rd = late.prepare(rr)
    for arm in run.ARMS:
        folder = run.OUT/f'{arm}_s17'
        assert (folder/'receipt.json').exists()
        schedule = np.load(folder/'schedule_local.npz')
        assert all(np.array_equal(schedule[k], baseline_schedule[k]) for k in ('ids','codes','references'))
        assert np.array_equal(schedule['validation'], selection)
        torch.manual_seed(17)
        net = run.make_model().eval()
        net.load_state_dict(torch.load(folder/'best.pt', map_location='cuda', weights_only=True))
        scores = run.predict(net, inputs, inner)
        np.savez_compressed(run.OUT/f'{arm}_inner_local.npz', ids=inner, labels=inputs.y[inner], scores=scores)
        for name, pick in [('selection',np.isin(inner, selection)), ('remaining_inner',~np.isin(inner, selection)),
                           ('all_inner',np.ones(len(inner),dtype=bool))]:
            metrics = run.base.m.full.metrics(inputs.y[inner][pick], scores[pick])
            records.append(dict(arm=arm, cohort=name, n=int(pick.sum()), **metrics))
        ref = net.reference(rd)
        scene_scores = []
        for first in range(0,358,8):
            batch = rows[first:min(first+8,358)]
            z, _ = net(late.prepare(batch), batch, ref, rr)
            scene_scores.append(z.cpu().numpy())
        scene_scores = np.concatenate(scene_scores)
        labels = np.array([r['label'] for r in rows[:358]])
        np.savez_compressed(run.OUT/f'{arm}_scene_local.npz', scores=scene_scores, labels=labels)
        for name, cut in [('development160',slice(0,160)), ('development198',slice(160,358))]:
            metrics = run.base.m.full.metrics(labels[cut],scene_scores[cut])
            records.append(dict(arm=arm, cohort=name, n=len(labels[cut]), **metrics))
            print('SCENE', arm, name, metrics, flush=True)
        # Held-source swaps, with both learned and identity fusion: decreased KL
        # alone can mean the fusion ignored donors rather than improved content.
        ids = stratified(selection, inputs.y, 8, 11919)
        data = inputs.data(ids, 'small')
        ref = net.reference(inputs.data(inputs.groups['references'], 'small'))
        query = [inputs.rows[i] for i in ids]
        net.capture_split = True
        clean, _ = net(data, query, ref, inputs.reference_rows)
        trace = net.split_trace
        valid_pol = data['available'].bool().cpu()
        gate = trace['gate'][valid_pol].flatten()
        fusion_state = {k:v.clone() for k,v in net.disentangler.fusions.state_dict().items()}
        record = dict(arm=arm, n=len(ids), gate_mean=float(gate.mean()), gate_min=float(gate.min()),
            gate_max=float(gate.max()), gate_gt099=float((gate>.99).float().mean()),
            sample_gate_std=float(trace['gate'].std(0).mean()),
            remainder_to_content_energy=float(trace['remainder'][valid_pol].norm()/trace['content'][valid_pol].norm()),
            swaps=[], donor_probability_variation={})
        net.capture_split = False
        for kind in ('learned','identity'):
            donor_predictions = []
            if kind == 'identity':
                for fusion in net.disentangler.fusions:
                    fusion.weight.zero_()
                    eye = torch.eye(256,device='cuda')
                    fusion.weight[:,:256,0,0].copy_(eye)
                    fusion.weight[:,256:,0,0].copy_(eye)
            for seed in range(4):
                donor, eligible = run.donor_permutation(data['available'],torch.Generator().manual_seed(810+seed))
                changed, _ = net.forward_variant(data,query,ref,inputs.reference_rows,'swap',donor)
                donor_predictions.append(changed.softmax(-1).cpu())
                loss, stats = run.weighted_consistency(changed,clean,eligible.to('cuda'))
                record['swaps'].append(dict(fusion=kind, draw=seed, weighted_kl=float(loss),
                    flips=int((changed.argmax(-1)!=clean.argmax(-1)).sum()), **stats))
            record['donor_probability_variation'][kind] = float(torch.stack(donor_predictions).var(0,unbiased=False).mean())
        net.disentangler.fusions.load_state_dict(fusion_state)
        record['donor_weight_norm'] = [float(f.weight[:,256:].norm()) for f in net.disentangler.fusions]
        probes.append(record)
        run.write(run.OUT/'results.json', records)
        run.write(run.OUT/'gate_swap_probe.json', probes)
        del net, ref, clean, changed, trace, fusion_state
        gc.collect()
        torch.cuda.empty_cache()


def gallery():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    rows = scene_rows()
    baseline = np.load(run.ROOT/'causal_work_check_20261119/baseline_scene_local.npz')
    ce = np.load(run.OUT/'gate_ce_scene_local.npz')
    full = np.load(run.OUT/'gate_uwcl_scene_local.npz')
    assert np.array_equal(baseline['labels'], full['labels'])
    y = full['labels']
    a, b, c = [v['scores'].argmax(1)==y for v in (baseline,ce,full)]
    conditions = dict(rescue=~a&c, harm=a&~c, correct=a&c, wrong=~a&~c,
                      consistency_rescue=~b&c, consistency_harm=b&~c)
    run.write(run.OUT/'flip_counts.json', {k:int(v.sum()) for k,v in conditions.items()})
    records, used = [], set()
    for kind, condition in conditions.items():
        picks = [int(i) for i in np.flatnonzero(condition) if int(i) not in used][:1]
        for idx in picks:
            used.add(idx)
            row = rows[idx]
            rival = int(np.where(np.arange(8)==y[idx],-np.inf,full['scores'][idx]).argmax())
            same = next(i for i in range(358,374) if rows[i]['label']==y[idx])
            other = next(i for i in range(358,374) if rows[i]['label']==rival)
            availability = late.prepare([rows[i] for i in (idx,same,other)])['available'].cpu().tolist()
            record = dict(kind=kind,id=row['id'],port=row['port'],label=int(y[idx]),
                scores={k:v['scores'][idx].tolist() for k,v in [('baseline',baseline),('gate_ce',ce),('gate_uwcl',full)]},
                references=[rows[same]['id'],rows[other]['id']], available=availability,
                raw_paths={pol:row[pol]['image'] if row.get(pol) else None for pol in ('vv','vh')})
            records.append(record)
            fig, axes = plt.subplots(2,3,figsize=(9,6),layout='constrained')
            for p,pol in enumerate(('vv','vh')):
                for col,i in enumerate((idx,same,other)):
                    if rows[i].get(pol):
                        with Image.open(rows[i][pol]['image']) as im:
                            axes[p,col].imshow(np.asarray(im),cmap='gray',interpolation='nearest')
                            axes[p,col].set_xlabel(f'{im.width} x {im.height} native PNG')
                    else:
                        axes[p,col].text(.5,.5,'Unavailable',ha='center')
                    suffix = '' if availability[col][p] else ' [not used]'
                    axes[p,col].set_title(f"{pol}: {('Query','Same-class reference','Rival-class reference')[col]}{suffix}")
                    axes[p,col].set_xticks([]); axes[p,col].set_yticks([])
            pred = [int(v['scores'][idx].argmax()) for v in (baseline,ce,full)]
            fig.suptitle(f"{kind}: {row['id']} / {row['port']} / true {y[idx]}\nBaseline / Gate CE / Gate UWCL: {pred}")
            fig.savefig(run.OUT/f'case_{len(records)-1:02d}_local.png',dpi=125)
            plt.close(fig)
    run.write(run.OUT/'cases_local.json',records)


def summarize():
    old = run.ROOT/'causal_work_check_20261119'
    records = json.loads((run.OUT/'results.json').read_text(encoding='utf-8'))
    for name in ('inner_results.json','scene_results.json'):
        records.extend(json.loads((old/name).read_text(encoding='utf-8')))
    fields = ['arm','cohort','oa','ba','macro_f1','nll']
    with (run.OUT/'comparison.csv').open('w',encoding='utf-8',newline='') as f:
        writer = csv.DictWriter(f,fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(records)
    arrays = {a:np.load(run.OUT/f'{a}_scene_local.npz') for a in run.ARMS}
    arrays['baseline'] = np.load(old/'baseline_scene_local.npz')
    labels = arrays['baseline']['labels']
    result = []
    for cohort, cut in [('development160',slice(0,160)),('development198',slice(160,358))]:
        y = labels[cut]
        for control in ('baseline','gate_ce'):
            before = arrays[control]['scores'][cut].argmax(1)==y
            after = arrays['gate_uwcl']['scores'][cut].argmax(1)==y
            result.append(dict(cohort=cohort,control=control,n=len(y),rescue=int((~before&after).sum()),
                harm=int((before&~after).sum()),net=int(after.sum()-before.sum())))
    run.write(run.OUT/'paired_results.json',result)
    fields = ['cohort','class','n','baseline_correct','gate_ce_correct','gate_uwcl_correct']
    with (run.OUT/'per_class.csv').open('w',encoding='utf-8',newline='') as f:
        writer = csv.DictWriter(f,fields)
        writer.writeheader()
        for cohort,cut in [('development160',slice(0,160)),('development198',slice(160,358))]:
            y=labels[cut]
            for c in range(8):
                row = dict(cohort=cohort,**{'class':c},n=int((y==c).sum()))
                for arm,array in arrays.items():
                    row[arm+'_correct']=int(((array['scores'][cut].argmax(1)==y)&(y==c)).sum())
                writer.writerow(row)


if __name__ == '__main__':
    torch.set_num_threads(4)
    evaluate()
    gallery()
    summarize()
