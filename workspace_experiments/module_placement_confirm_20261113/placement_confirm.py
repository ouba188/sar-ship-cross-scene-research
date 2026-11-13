"""Confirm two CRFB positions after a shared intervention-trained initialization."""
import argparse
import gc
import json
import shutil
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import placement_study as pilot
from placement_model import make_model

bundle = pilot.bundle
ROOT = Path(__file__).resolve().parent
OUT = ROOT/'runs'/'placement_confirm'
COMMON = ROOT/'placement_common'
VARIANTS = ('current', 'm1_middle')
SEEDS = (17, 29, 43)


def clear(ft):
    gc.collect()
    if ft.DEVICE == 'cuda':
        ft.torch.cuda.empty_cache()


def common(ft, seed, variant):
    model = make_model(ft, seed, variant)
    state = ft.torch.load(COMMON/f'seed{seed}'/'reference.pt',
                          map_location='cpu', weights_only=True)
    model.net.load_state_dict(state, strict=True)
    model.net.phase = 1
    model.eval().requires_grad_(False)
    return model


def check_history(inputs, seed):
    folder = COMMON/f'seed{seed}'
    targets = bundle.read(folder/'teacher.json')
    receipt = bundle.read(folder/'intervention_receipt.json')
    adapt = inputs.groups['source_adapt'].tolist()
    assert [r['index'] for r in targets] == adapt
    assert all(str(t['id']) == str(inputs.rows[i]['id']) and
               t['label'] == inputs.rows[i]['label'] for i,t in zip(adapt,targets))
    assert sum(t['useful'] for t in targets) == receipt['selected_count']
    return dict(seed=seed, source_adapt=len(adapt), selected=receipt['selected_count'],
                inherited_steps=receipt['steps'], intervention_retrained=False,
                checkpoint='reference.pt', m1_active_in_common_stages=False)


def full_trainable(model):
    model.requires_grad_(False)
    net = model.net
    net.phase, net.fixed_delta = 3, None
    bb = net.base.backbone.model
    bb.requires_grad_(True)
    bb.fc.requires_grad_(False)
    for module in (net.m1, net.base.internal_m2, net.base.deep_output, net.matcher,
                   net.guide.value, net.guide.output, net.guide.head):
        module.requires_grad_(True)
    model.capture_readout = False
    # Eager execution is used by both arms; the mathematical operator is unchanged.
    model.accelerate = lambda: None
    return model


def smoke(ft, inputs):
    """Catch wrong stage gates, source mapping, cached inputs and broken gradients."""
    folder = OUT/'smoke'
    folder.mkdir(parents=True, exist_ok=True)
    report = {'histories': [check_history(inputs,s) for s in SEEDS], 'variants': {}}
    ids = inputs.groups['validation'][:4]
    rid = inputs.groups['references']
    qrows = [inputs.rows[i] for i in ids]
    qdata = inputs.data(ids, 'validation')
    rdata = inputs.data(rid, 'small')
    baseline = None
    for name in VARIANTS:
        model = common(ft, 17, name)
        with ft.torch.no_grad():
            reference = model.reference(rdata)
            scores, aux = model(qdata,qrows,reference,inputs.reference_rows)
            assert aux is None and all(t[0]!='m1' for t in model.trace)
            expected = ft.np.load(COMMON/'seed17'/'reference_validation.npz')
            lookup = {int(i):j for j,i in enumerate(expected['ids'])}
            wanted = expected['score'][[lookup[int(i)] for i in ids]]
            error = float(abs(scores.cpu().numpy()-wanted).max())
            assert error < 1e-3, (name,error)
            if baseline is None:
                baseline = scores.detach().clone()
            else:
                assert (scores-baseline).abs().max() < 1e-4
            a,c = model.extract(qdata,True)
            model.net.phase = 2
            live = model(qdata,qrows,reference,inputs.reference_rows)[0]
            cached = model.finish(a,c,qrows,reference,inputs.reference_rows)[0]
            assert (live-cached).abs().max() < 1e-5
        ft.staged.configure(model.net,2)
        names = sorted(n for n,p in model.net.named_parameters() if p.requires_grad)
        assert 'base.backbone.model.layer2.3.conv3.weight' in names
        assert 'base.backbone.model.layer2.2.conv3.weight' not in names
        model.net.zero_grad(set_to_none=True)
        z,aux = model.finish(a,c,qrows,reference,inputs.reference_rows)
        y = ft.torch.tensor(inputs.y[ids],device=ft.DEVICE)
        loss = ft.F.cross_entropy(z,y)+.02*ft.F.cross_entropy(aux.flatten(0,1),y.repeat_interleave(2))
        loss.backward()
        assert any(p.grad is not None and p.grad.norm()>0 for p in model.net.m1.parameters())
        del a,c,reference,live,cached,z,aux,loss,scores
        model.zero_grad(set_to_none=True)
        full_trainable(model).train()
        train = inputs.groups['small'][:ft.MICRO]
        loss,z = ft.loss_for(model,inputs,train,1)
        loss.backward()
        prefixes = ('net.m1.','net.base.internal_m2.','net.matcher.',
                    'net.base.backbone.model.conv1.')
        gradients = {pre:sum(float(p.grad.norm()) for n,p in model.named_parameters()
                             if n.startswith(pre) and p.grad is not None) for pre in prefixes}
        assert all(v>0 and ft.np.isfinite(v) for v in gradients.values())
        assert all(p.grad is None for p in model.anchor_conv.parameters())
        report['variants'][name] = dict(common_replay_error=error,
            feedback_trainable=names,full_gradient_norms=gradients,
            trace=model.trace,microbatch=ft.MICRO,
            peak_gib=ft.torch.cuda.max_memory_allocated()/2**30 if ft.DEVICE=='cuda' else None)
        print('CONFIRM_SMOKE',name,json.dumps(report['variants'][name]),flush=True)
        del model,loss,z
        clear(ft)
    assert report['variants']['current']['feedback_trainable'] == report['variants']['m1_middle']['feedback_trainable']
    bundle.save(folder/f'{ft.DEVICE}.json',report)


def cache_inputs(ft, model, inputs):
    """Fixed prefix for staged fitting only; discarded before full-backbone tuning."""
    ids = ft.np.r_[inputs.groups['small'],inputs.groups['validation']]
    a,c = {},{}
    with ft.torch.no_grad():
        for split in ('small','validation'):
            chosen = inputs.groups[split]
            for start in range(0,len(chosen),12):
                ix = chosen[start:start+12]
                aa,cc = model.extract(inputs.data(ix,split),True)
                cc['early_identity'] = cc['early_identity'].reshape(len(ix),2,512,20,20)
                for dest,src in ((a,aa),(c,cc)):
                    for key,value in src.items():
                        if key not in dest:
                            dest[key] = ft.torch.empty((len(inputs.rows),)+value.shape[1:],dtype=value.dtype)
                        dest[key][ix] = value.detach().cpu()
    print('STAGED_CACHE',len(ids),'records',flush=True)
    return a,c


def fit_stages(ft, inputs, name, seed, dest):
    staged = ft.staged
    model = common(ft,seed,name)
    model.capture_readout = True
    a,c = cache_inputs(ft,model,inputs)
    original_tensor = staged.task.tensor
    def tensor(value,ids):
        if ft.torch.is_tensor(value):
            return value[ft.torch.as_tensor(ids,device=value.device)].to(ft.DEVICE)
        return original_tensor(value,ids)
    staged.task.tensor = tensor
    def forward(net,aa,cc,rows,ids,rid,m3_on=True):
        query = {k:tensor(v,ids) for k,v in aa.items()}
        local = {k:tensor(v,ids) for k,v in cc.items()}
        local['early_identity'] = local['early_identity'].flatten(0,1)
        ref = {k:tensor(aa[k],rid) for k in ('desc','mass','available')}
        return model.finish(query,local,[rows[i] for i in ids],ref,[rows[i] for i in rid],m3_on)
    model.net.forward = MethodType(forward,model.net)
    try:
        protocol = bundle.read(COMMON/'protocol.json')
        for cfg in protocol['phases']:
            if cfg['phase'] not in (2,3):
                continue
            phase_dir = dest/cfg['name']
            bundle.save(OUT/'status.json',dict(state='staged_training',variant=name,seed=seed,phase=cfg['name']))
            if (phase_dir/'receipt.json').exists():
                model.net.load_state_dict(ft.torch.load(phase_dir/'final.pt',map_location='cpu',weights_only=True))
                staged.configure(model.net,cfg['phase'])
            else:
                staged.train_phase(model.net,cfg,seed,dest,inputs.rows,inputs.groups,a,c,1.)
            if name=='current':
                old=ft.np.load(COMMON/f'seed{seed}'/f'{cfg["name"]}_validation.npz')
                new=ft.np.load(phase_dir/'validation.npz')
                assert ft.np.array_equal(old['ids'],new['ids'])
                bundle.save(phase_dir/'historical_replay.json',dict(
                    max_logit_error=float(abs(old['score']-new['score']).max()),
                    top1_equal=bool(ft.np.array_equal(old['score'].argmax(1),new['score'].argmax(1))),
                    note='Training arithmetic may differ across CPU/GPU and live/cache implementations.'))
    finally:
        staged.task.tensor = original_tensor
        del model.net.forward
        del model,a,c
        clear(ft)


def fit_full(ft, inputs, name, seed, dest):
    """Reuse the actual full-tuning optimizer, stopping rule and checkpoint selector."""
    stage = dest/'readout'
    init_root = dest/'initialization'
    prior = init_root/'staged_ablation_20261109'/f'full_s{seed}_lr1'
    prior.mkdir(parents=True,exist_ok=True)
    shutil.copy2(stage/'validation.npz',prior/'validation.npz')
    def factory(s,scope):
        assert s==seed and scope=='full'
        model = make_model(ft,s,name)
        model.net.load_state_dict(ft.torch.load(stage/'final.pt',map_location='cpu',weights_only=True))
        return full_trainable(model)
    old_factory,old_paper,old_root,old_config = ft.LiveFull,ft.PAPER,ft.ROOT,ft.configuration
    def configuration(scope,lr,s):
        result = old_config(scope,lr,s)
        result.update(initialization=f'{name}/seed{s}: common intervention-trained reference checkpoint + position-specific feedback/readout',
            intervention='inherited identical source-only teacher selection and original-margin protection; not newly selected during fine-tuning',
            variant=name,cuda_graph=False)
        return result
    ft.LiveFull,ft.PAPER,ft.ROOT,ft.configuration = factory,init_root,dest,configuration
    target = dest/'finetune'
    try:
        bundle.save(OUT/'status.json',dict(state='full_finetune',variant=name,seed=seed,
                                          progress=str(target/'curve.json'),test_scored=False))
        ft.fit(inputs,'full',1e-4,seed,target)
    finally:
        ft.LiveFull,ft.PAPER,ft.ROOT,ft.configuration = old_factory,old_paper,old_root,old_config


def complete_evaluation(ft,inputs):
    # No evaluation predictions are generated until all six fits have completed.
    assert all((OUT/f'{name}_s{seed}'/'finetune'/'receipt.json').exists()
               for seed in SEEDS for name in VARIANTS)
    inputs.load_split('evaluation')
    inputs.prepare_geometry('evaluation')
    result=[]
    for seed in SEEDS:
        for name in VARIANTS:
            folder=OUT/f'{name}_s{seed}'/'finetune'
            model=make_model(ft,seed,name)
            model.load_state_dict(ft.torch.load(folder/'best.pt',map_location='cpu',weights_only=True))
            for split in ('validation','evaluation'):
                ids=inputs.groups[split]
                scores=ft.predict(model,inputs,ids,split)
                if split=='validation':
                    saved=ft.np.load(folder/'validation.npz')
                    error=float(abs(scores-saved['score']).max())
                    assert error<1e-3,(name,seed,error)
                else:
                    ft.np.savez_compressed(folder/'evaluation.npz',ids=ids,score=scores,label=inputs.y[ids])
                    m=ft.metrics(inputs.y[ids],scores)
                    bundle.save(folder/'evaluation.json',m)
                    result.append(dict(seed=seed,variant=name,**m))
            del model
            clear(ft)
        base=OUT/f'current_s{seed}'/'finetune'/'validation.npz'
        pilot.case_summary(ft,inputs,OUT/f'm1_middle_s{seed}'/'finetune',base)
        test_base=base.with_name('evaluation.npz')
        pilot.case_summary(ft,inputs,OUT/f'm1_middle_s{seed}'/'finetune',test_base,
                           'evaluation.npz','fixed-position held-out test after all six fits')
    bundle.save(OUT/'test_summary.json',result)
    bundle.save(OUT/'status.json',dict(state='training_and_evaluation_complete',
        test_scored=True,formal_model_changed=False,visual_review='pending',
        scope='OSS2 two positions, full intervention-history recipe, three paired seeds'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('smoke','train'))
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args=p.parse_args()
    ports,ft=bundle.runtime(args.device)
    ft.MICRO=12
    bundle.verify('oss2')
    inputs=bundle.inputs_for(SimpleNamespace(dataset='oss2'),ports,ft)
    OUT.mkdir(parents=True,exist_ok=True)
    if args.action=='smoke':
        return smoke(ft,inputs)
    assert args.device=='cuda'
    assert (OUT/'smoke'/'cuda.json').exists(),'CUDA smoke must pass before launching.'
    configuration=dict(variants=VARIANTS,seeds=SEEDS,dataset='oss2',
        shared_stages=['carrier','intervention','reference'],
        repeated_stages=['feedback','readout','full_finetune'],
        feedback_conv='layer2[3].conv3 for BOTH positions',
        source_only_intervention_inherited=True,formal_model_changed=False,
        test_policy='score both fixed positions only after all six fits',
        finetune=dict(max_epochs=30,min_epochs=12,patience=8,batch=24,micro=ft.MICRO,
                      backbone_lr=1e-4,module_lr=1e-4))
    cfg=OUT/'protocol.json'
    if cfg.exists():
        assert bundle.read(cfg)==json.loads(json.dumps(configuration))
    bundle.save(cfg,configuration)
    for seed in SEEDS:
        history=check_history(inputs,seed)
        for name in VARIANTS:
            dest=OUT/f'{name}_s{seed}'
            dest.mkdir(parents=True,exist_ok=True)
            bundle.save(dest/'inherited_intervention.json',history)
            if not (dest/'readout'/'receipt.json').exists():
                fit_stages(ft,inputs,name,seed,dest)
            if not (dest/'finetune'/'receipt.json').exists():
                fit_full(ft,inputs,name,seed,dest)
    complete_evaluation(ft,inputs)


if __name__=='__main__':
    main()
