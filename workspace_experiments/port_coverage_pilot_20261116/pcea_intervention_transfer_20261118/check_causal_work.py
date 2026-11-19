"""Matched, bounded learning check; no target-guided model selection."""
import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import causal_run as run

OUT = run.ROOT/'causal_work_check_20261119'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def stratified(ids, labels, per_class, seed):
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.permutation(ids[labels[ids] == c])[:per_class]
                                  for c in range(8)]))


def train(arm):
    inputs, fit, inner = run.base.runtime('unified')
    validation = stratified(inner, inputs.y, 32, 20261119)
    args = SimpleNamespace(phase='train', arm=arm, seed=17, steps=120, batch=8,
        keep_ratio=.5, warmup=30, ramp=30, swap_weight=1., bias_weight=.5, eval_every=30)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'selection_local.npz', validation=validation, all_inner=inner)
    previous = None
    diagnostics = []
    original = run.backward

    def measured(net, *pos, **kw):
        nonlocal previous
        mask = net.disentangler.mask().detach().cpu()
        changed = 0 if previous is None else int((mask != previous).sum())
        previous = mask.clone()
        clean, stats = original(net, *pos, **kw)
        norms = {g['name']: sum(float(p.grad.square().sum()) for p in g['params']
                               if p.grad is not None)**.5 for g in net.optimizer_groups()}
        ids = pos[1]
        diagnostics.append(dict(step=len(diagnostics)+1, changed_mask_entries=changed,
            gradient_norm=norms, train_batch_oa=float((clean.argmax(-1).cpu().numpy() == inputs.y[ids]).mean()),
            **stats))
        if len(diagnostics) % 10 == 0:
            write(OUT/f'{arm}_diagnostics.json', diagnostics)
            print('PROGRESS', arm, len(diagnostics), 'CE', stats['ce'], 'mask_changes', changed, flush=True)
        return clean, stats

    with patch.object(run.base, 'runtime', return_value=(inputs, fit, validation)), \
         patch.object(run, 'OUT', OUT), patch.object(run, 'backward', measured):
        run.train(args)
    write(OUT/f'{arm}_diagnostics.json', diagnostics)


def gradients(net, inputs, ids):
    groups = net.optimizer_groups()
    stored, losses = {}, {}
    clean, aux, data = run.forward(net, inputs, ids, 0)
    y = torch.as_tensor(inputs.y[ids], device='cuda')
    w = torch.as_tensor(inputs.weights[ids], device='cuda')
    target = clean.detach()
    donor, eligible = run.donor_permutation(data['available'], torch.Generator().manual_seed(912))
    for kind in ('ce', 'swap', 'bias'):
        net.zero_grad(set_to_none=True)
        if kind == 'ce':
            loss = run.classification(clean, aux, y, w, data['available'])
        elif kind == 'swap':
            z, _, d = run.forward(net, inputs, ids, 0, 'swap', donor)
            loss = run.swap_consistency(z, target, eligible.to('cuda'))
        else:
            z, _, d = run.forward(net, inputs, ids, 0, 'bias')
            loss = .5*run.bias_prior_loss(z, net.source_prior, d['available'].bool())
        loss.backward()
        losses[kind] = float(loss.detach())
        stored[kind] = {g['name']: torch.cat([
            (torch.zeros_like(p) if p.grad is None else p.grad).flatten().detach().cpu()
            for p in g['params']]) for g in groups}
    result = dict(losses=losses, eligible=int(eligible.sum()), groups={})
    for g in stored['ce']:
        a, b, c = (stored[k][g] for k in ('ce','swap','bias'))
        result['groups'][g] = dict(ce_norm=float(a.norm()), swap_norm=float(b.norm()),
            weighted_bias_norm=float(c.norm()),
            ce_swap_cos=float(torch.nn.functional.cosine_similarity(a,b,dim=0)),
            ce_bias_cos=float(torch.nn.functional.cosine_similarity(a,c,dim=0)))
    net.zero_grad(set_to_none=True)
    return result


@torch.no_grad()
def evaluate_inner():
    inputs, fit, inner = run.base.runtime('unified')
    chosen = np.load(OUT/'selection_local.npz')['validation']
    unseen = np.setdiff1d(inner, chosen)
    schedules = [np.load(OUT/f'{a}_s17'/'schedule_local.npz') for a in run.ARMS]
    for other in schedules[1:]:
        assert all(np.array_equal(schedules[0][k],other[k]) for k in ('ids','codes','references'))
    def describe(ids):
        rows=[inputs.rows[i] for i in ids]
        return dict(n=len(ids),classes=np.bincount(inputs.y[ids],minlength=8).tolist(),
                    vessels=len({r['mmsi'] for r in rows}),ports=len({r['port'] for r in rows}),
                    passages=len({'_'.join(r['pass_group'].split('_')[:2]) for r in rows}))
    write(OUT/'data_scope.json',dict(fit_pool=describe(fit),trained=describe(schedules[0]['ids']),
        selection=describe(chosen),remaining_inner=describe(unseen),references=describe(schedules[0]['references']),
        identical_orders_augmentations_references=True,historical_development=True))
    records = []
    for arm in run.ARMS:
        folder = OUT/f'{arm}_s17'
        assert (folder/'receipt.json').exists()
        net = run.make_model(1. if arm == 'baseline' else .5).eval()
        net.load_state_dict(torch.load(folder/'best.pt', map_location='cuda', weights_only=True))
        scores = run.predict(net, inputs, inner)
        np.savez_compressed(OUT/f'{arm}_inner_local.npz', ids=inner, labels=inputs.y[inner], scores=scores)
        mask = net.disentangler.mask().detach().cpu().numpy()
        np.save(OUT/f'{arm}_mask_local.npy', mask)
        for name, ids in [('selection',chosen),('remaining_inner',unseen),('all_inner',inner)]:
            select = np.isin(inner,ids)
            metrics = run.base.m.full.metrics(inputs.y[inner][select],scores[select])
            records.append(dict(arm=arm, cohort=name, n=int(select.sum()), **metrics))
            print('EVAL',arm,name,metrics,flush=True)
        del net
        gc.collect()
        torch.cuda.empty_cache()
    write(OUT/'inner_results.json',records)


def probe():
    inputs,fit,_ = run.base.runtime('unified')
    ids = stratified(np.setdiff1d(fit,inputs.groups['references']),inputs.y,1,1119)
    result = {}
    for stage in ('initial','split_ce','causal'):
        net = run.make_model(.5).eval()
        run.set_source_prior(net,inputs,fit)
        if stage != 'initial':
            net.load_state_dict(torch.load(OUT/f'{stage}_s17'/'best.pt',map_location='cuda',weights_only=True))
        result[stage] = gradients(net,inputs,ids)
        write(OUT/'gradient_probe.json',result)
        print('PROBE',stage,result[stage],flush=True)
        del net
        gc.collect()
        torch.cuda.empty_cache()


@torch.no_grad()
def pooled_features(net, inputs, ids):
    values = {k: [] for k in ('causal','bias')}
    net.capture_split = True
    for first in range(0,len(ids),8):
        data = inputs.data(ids[first:first+8],'small')
        net.extract(data,True)
        trace = net.split_trace
        valid = torch.nn.functional.adaptive_avg_pool2d(data['valid'].flatten(0,1), (40,40))
        valid = valid.reshape(-1,2,1,40,40).cpu()*data['available'].cpu()[:,:,None,None,None]
        for k in values:
            pooled = (trace[k]*valid).sum((-1,-2))/valid.sum((-1,-2)).clamp_min(1)
            values[k].append(pooled.flatten(1).numpy())
    net.capture_split = False
    return {k: np.concatenate(v) for k,v in values.items()}


def feature_probe():
    inputs,fit,_ = run.base.runtime('unified')
    fit = stratified(np.setdiff1d(fit,inputs.groups['references']),inputs.y,64,9119)
    validation = np.load(OUT/'selection_local.npz')['validation']
    records = []
    for stage in ('initial','split_ce','causal'):
        net = run.make_model(.5).eval()
        if stage != 'initial':
            net.load_state_dict(torch.load(OUT/f'{stage}_s17'/'best.pt',map_location='cuda',weights_only=True))
        a = pooled_features(net,inputs,fit)
        b = pooled_features(net,inputs,validation)
        for branch in a:
            scale = a[branch].std(0)
            active = scale > 1e-6
            mean = a[branch][:,active].mean(0)
            x = (a[branch][:,active]-mean)/scale[active]
            q = (b[branch][:,active]-mean)/scale[active]
            x = np.column_stack((x,np.ones(len(x)))).astype(np.float64)
            q = np.column_stack((q,np.ones(len(q)))).astype(np.float64)
            target = np.eye(8)[inputs.y[fit]]
            # Fixed regularizer for every arm; no validation tuning of the probe.
            penalty = np.eye(x.shape[1])*10.
            penalty[-1,-1] = 0.
            coef = np.linalg.solve(x.T@x+penalty,x.T@target)
            metrics = run.base.m.full.metrics(inputs.y[validation],q@coef)
            records.append(dict(stage=stage,branch=branch,fit=len(fit),validation=len(validation),
                active_dimensions=int(active.sum()),**metrics))
            print('FEATURE_PROBE',stage,branch,metrics['oa'],metrics['ba'],flush=True)
        del net
        gc.collect()
        torch.cuda.empty_cache()
    write(OUT/'feature_probe.json',records)


def scene_rows():
    from unified_model import late
    store = Path('C:/Users/25791/AppData/Local/sar_ports24_fullscale_20261110')
    original = json.loads((store/'manifest.json').read_text(encoding='utf-8'))
    groups = json.loads((store/'groups.json').read_text(encoding='utf-8'))
    expansion = json.loads((run.base.PAPER/'pcea_class_local_expansion_20261118/evaluation_manifest_local.json').read_text(encoding='utf-8'))
    recover = late.base.native.old.old.recovered.recovery.recover
    rows = [recover(original[i])[0] for i in groups['validation']]+expansion[:198]+expansion[198:214]
    assert len(rows)==374
    assert [r['id'] for r in rows[358:]]==[original[i]['id'] for i in groups['references']]
    return rows


@torch.no_grad()
def scene_evaluate():
    from unified_model import late
    rows = scene_rows()
    reference_rows = rows[358:]
    y = np.array([r['label'] for r in rows[:358]])
    reference_data = late.prepare(reference_rows)
    records = []
    # No refitting or reselection after seeing these historical development cohorts.
    for arm in run.ARMS:
        net = run.make_model(1. if arm=='baseline' else .5).eval()
        net.load_state_dict(torch.load(OUT/f'{arm}_s17'/'best.pt',map_location='cuda',weights_only=True))
        reference = net.reference(reference_data)
        scores=[]
        for first in range(0,358,8):
            part=rows[first:min(first+8,358)]
            z,_=net(late.prepare(part),part,reference,reference_rows)
            scores.append(z.cpu().numpy())
        scores=np.concatenate(scores)
        np.savez_compressed(OUT/f'{arm}_scene_local.npz',scores=scores,labels=y,
                            ids=np.array([r['id'] for r in rows[:358]]))
        for name,section in [('development160',slice(0,160)),('development198',slice(160,358))]:
            metrics=run.base.m.full.metrics(y[section],scores[section])
            records.append(dict(arm=arm,cohort=name,**metrics))
            print('SCENE',arm,name,metrics['oa'],metrics['ba'],flush=True)
        del net,reference
        gc.collect()
        torch.cuda.empty_cache()
    write(OUT/'scene_results.json',records)


@torch.no_grad()
def swap_probe():
    inputs,fit,inner=run.base.runtime('unified')
    ids=stratified(inner,inputs.y,8,719)
    records=[]
    for arm in ('split_ce','causal'):
        net=run.make_model(.5).eval()
        net.load_state_dict(torch.load(OUT/f'{arm}_s17'/'best.pt',map_location='cuda',weights_only=True))
        net.capture_split=True
        z,_,data=run.forward(net,inputs,ids,0)
        # Four fixed donor draws; report overlap and nontrivial injection alongside stability.
        for seed in range(4):
            donors,eligible=run.donor_permutation(data['available'],torch.Generator().manual_seed(700+seed))
            changed,_,_=run.forward(net,inputs,ids,0,'swap',donors)
            mask=eligible.numpy()
            p=z.softmax(-1); q=changed.softmax(-1)
            trace=net.split_trace
            injection=trace['output']-trace['causal']
            energy=injection.square().flatten(1).sum(1)
            causal_energy=trace['causal'].square().flatten(1).sum(1).clamp_min(1e-12)
            records.append(dict(arm=arm,seed=seed,n=len(ids),eligible=int(eligible.sum()),
                injection_energy_ratio=float((energy/causal_energy)[eligible].mean()),
                cross_class_donor_fraction=float((inputs.y[ids][mask]!=inputs.y[ids[donors.numpy()]][mask]).mean()),
                prediction_flips=int(((z.argmax(-1)!=changed.argmax(-1)).cpu().numpy()&mask).sum()),
                probability_mse=float((p-q).square().mean(-1)[eligible.to('cuda')].mean()),
                clean_oa=float((z.argmax(-1).cpu().numpy()==inputs.y[ids]).mean()),
                swapped_oa=float((changed.argmax(-1).cpu().numpy()==inputs.y[ids]).mean())))
        biased,_,_=run.forward(net,inputs,ids,0,'bias')
        entropy=float(-(net.source_prior*net.source_prior.clamp_min(1e-12).log()).sum())
        records.append(dict(arm=arm,branch='bias',n=len(ids),
            oa=float((biased.argmax(-1).cpu().numpy()==inputs.y[ids]).mean()),
            prior_cross_entropy=float(run.bias_prior_loss(biased,net.source_prior,data['available'].bool())),
            prior_entropy=entropy))
        del net
        gc.collect()
        torch.cuda.empty_cache()
    write(OUT/'swap_probe.json',records)


@torch.no_grad()
def gallery():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from unified_model import late
    rows=scene_rows()
    arrays={arm:np.load(OUT/f'{arm}_scene_local.npz') for arm in run.ARMS}
    y=arrays['baseline']['labels']
    old=arrays['baseline']['scores']; new=arrays['causal']['scores']
    ok0=old.argmax(1)==y; ok1=new.argmax(1)==y
    conditions={'rescue':~ok0&ok1,'harm':ok0&~ok1,'correct':ok0&ok1,'wrong':~ok0&~ok1}
    chosen=[]
    for name,condition in conditions.items():
        # First cases in fixed cohort ordering, not chosen for persuasive appearance.
        chosen.extend((int(i),name) for i in np.flatnonzero(condition)[:2])
    split_ok=arrays['split_ce']['scores'].argmax(1)==y
    for name,condition in [('constraint_rescue',~split_ok&ok1),('constraint_harm',split_ok&~ok1)]:
        candidates=[int(i) for i in np.flatnonzero(condition) if i not in {j for j,_ in chosen}]
        chosen.extend((i,name) for i in candidates[:1])
    net=run.make_model(.5).eval()
    net.load_state_dict(torch.load(OUT/'causal_s17/best.pt',map_location='cuda',weights_only=True))
    records=[]
    for j,(idx,kind) in enumerate(chosen):
        row=rows[idx]
        net.capture_split=True
        data=late.prepare([row])
        net.extract(data,True)
        availability=data['available'][0].cpu().tolist()
        trace=net.split_trace
        rival=int(np.where(np.arange(8)==y[idx],-np.inf,new[idx]).argmax())
        same=next(i for i in range(358,374) if rows[i]['label']==y[idx])
        other=next(i for i in range(358,374) if rows[i]['label']==rival)
        record=dict(id=row['id'],index=idx,kind=kind,port=row.get('port'),label=int(y[idx]),available=availability,
            baseline=old[idx].tolist(),causal=new[idx].tolist(),split_ce=arrays['split_ce']['scores'][idx].tolist(),
            raw_paths={p:row[p]['image'] if row.get(p) else None for p in ('vv','vh')},
            references=[rows[same]['id'],rows[other]['id']])
        records.append(record)
        fig,axes=plt.subplots(2,5,figsize=(13,6.8),layout='constrained')
        for p,pol in enumerate(('vv','vh')):
            for col,i in ((0,idx),(3,same),(4,other)):
                if rows[i].get(pol):
                    with Image.open(rows[i][pol]['image']) as im:
                        axes[p,col].imshow(np.asarray(im),cmap='gray',interpolation='nearest')
                        axes[p,col].set_xlabel(f'{im.width}x{im.height} native pixels')
            energies=[np.log1p(trace[k][0,p].square().mean(0).sqrt().numpy()) for k in ('causal','bias')]
            for col,energy in enumerate(energies,1):
                if availability[p]:
                    axes[p,col].imshow(energy,cmap='magma',interpolation='nearest',vmin=0,vmax=max(e.max() for e in energies))
                else:
                    axes[p,col].text(.5,.5,'Unavailable: not used',ha='center',va='center',transform=axes[p,col].transAxes)
            for col,title in enumerate(('Query PNG','Retained energy','Complement energy','Same-class reference','Rival-class reference')):
                axes[p,col].set_title(f'{pol} {title}',fontsize=9)
                axes[p,col].set_xticks([]); axes[p,col].set_yticks([])
        fig.suptitle(f"{kind}: {row['id']} / {row.get('port')} / true {y[idx]}\nBaseline {old[idx].argmax()} -> causal {new[idx].argmax()}; energy is not causal attribution")
        fig.savefig(OUT/f'case_{j:02d}_local.png',dpi=125)
        plt.close(fig)
    write(OUT/'cases_local.json',records)
    write(OUT/'flip_counts.json',dict(baseline_comparison={name:int(c.sum()) for name,c in conditions.items()},
        split_ce_comparison=dict(rescue=int((~split_ok&ok1).sum()),harm=int((split_ok&~ok1).sum()))))


if __name__ == '__main__':
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=(*run.ARMS,'evaluate','probe','feature_probe','scene','swap_probe','gallery','post'))
    args = parser.parse_args()
    if args.action == 'post':
        for function in (evaluate_inner, scene_evaluate, probe, feature_probe, swap_probe, gallery):
            print('START',function.__name__,flush=True)
            function()
            gc.collect()
            torch.cuda.empty_cache()
    elif args.action == 'evaluate':
        evaluate_inner()
    elif args.action == 'probe':
        probe()
    elif args.action == 'feature_probe':
        feature_probe()
    elif args.action == 'scene':
        scene_evaluate()
    elif args.action == 'swap_probe':
        swap_probe()
    elif args.action == 'gallery':
        gallery()
    else:
        train(args.action)
