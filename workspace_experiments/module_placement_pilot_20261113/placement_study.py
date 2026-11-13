"""Paired within-stage position pilot; validation-only selection, no test access."""
import argparse
import gc
import json
import importlib.util
import math
import time
from pathlib import Path
from types import SimpleNamespace

from placement_model import POSITIONS, make_model

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('portable_bundle', ROOT/'run.py')
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)
OUT = ROOT/'runs'/'placement_pilot'


def save(path, value):
    bundle.save(path, value)


def batch(ft, inputs, ids, split, code=0):
    return inputs.data(ids, split, code)


def evaluate(ft, model, inputs, ids, split):
    model.eval()
    values=[]
    with ft.torch.no_grad():
        reference=model.reference(batch(ft,inputs,inputs.groups['references'],'small'))
        for start in range(0,len(ids),4):
            chosen=ids[start:start+4]
            z,_=model(batch(ft,inputs,chosen,split),[inputs.rows[i] for i in chosen],
                      reference,inputs.reference_rows)
            values.append(z.cpu().numpy())
    return ft.np.concatenate(values)


def metrics(ft, labels, score):
    classes=score.shape[1]
    cm=ft.np.zeros((classes,classes),dtype=int)
    ft.np.add.at(cm,(labels,score.argmax(1)),1)
    present=cm.sum(1)>0
    return dict(oa=float((labels==score.argmax(1)).mean()*100),
                ba=float((cm.diagonal()[present]/cm.sum(1)[present]).mean()*100),
                confusion=cm.tolist())


def smoke(args,ports,ft):
    rows,groups=bundle.records(args.dataset)
    ids=groups['validation'][:2]
    qrows=[rows[i] for i in ids]
    rrows=[rows[i] for i in groups['references']]
    qdata,rdata=ft.cost.prepare(qrows),ft.cost.prepare(rrows)
    classes=4 if args.dataset=='oss2' else 8
    original=ft.LiveFull(args.seed,'full') if classes==4 else ports.LiveFull(args.seed)
    weight=ROOT/'checkpoints'/args.dataset/f'seed{args.seed}'/'best.pt'
    original.load_state_dict(ft.torch.load(weight,map_location='cpu',weights_only=True))
    candidate=make_model(ft,args.seed,'current',classes)
    candidate.load_state_dict(original.state_dict(),strict=True)
    original.eval();candidate.eval()
    with ft.torch.no_grad():
        a=original(qdata,qrows,original.reference(rdata),rrows)[0]
        b=candidate(qdata,qrows,candidate.reference(rdata),rrows)[0]
    error=float((a-b).abs().max())
    assert error<1e-4,error
    del original,candidate,a,b
    gc.collect()
    if args.device=='cuda':ft.torch.cuda.empty_cache()
    result={'current_original_logit_error':error,'variants':{},'training':False}
    count=None
    initial_state=None
    for name in POSITIONS:
        model=make_model(ft,args.seed,name,classes)
        # Tensor comparison checks common initialization, not file fingerprints.
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if initial_state is None:initial_state=state
        else:assert all(ft.torch.equal(state[k],v) for k,v in initial_state.items()),name
        del state
        params=sum(p.numel() for p in model.parameters() if p.requires_grad)
        if count is None:count=params
        assert params==count
        model.train()
        z,aux=model(qdata,qrows,model.reference(rdata),rrows)
        y=ft.torch.tensor([r['label'] for r in qrows],device=ft.DEVICE)
        loss=ft.F.cross_entropy(z,y)+.02*ft.F.cross_entropy(aux.flatten(0,1),y.repeat_interleave(2))
        loss.backward()
        prefixes=('net.m1.','net.base.internal_m2.','net.matcher.','net.base.backbone.model.conv1.')
        gradients={pre:sum(float(p.grad.norm()) for n,p in model.named_parameters()
                           if n.startswith(pre) and p.grad is not None) for pre in prefixes}
        assert all(v>0 and math.isfinite(v) for v in gradients.values()),(name,gradients)
        assert sorted((t[0],t[1]) for t in model.trace)==sorted(zip(('m1','m2','m3'),POSITIONS[name]))
        result['variants'][name]=dict(trainable=params,trace=model.trace,gradients=gradients)
        print('SMOKE',name,result['variants'][name],flush=True)
        del model,z,aux,loss
        gc.collect()
        if args.device=='cuda':ft.torch.cuda.empty_cache()
    save(OUT/args.dataset/f'smoke_s{args.seed}_{args.device}.json',result)


def case_summary(ft,inputs,folder,reference_file):
    """Save paired validation cases and source-chip paths for subsequent visual review."""
    baseline=ft.np.load(reference_file)
    candidate=ft.np.load(folder/'validation.npz')
    ids=candidate['ids']
    assert ft.np.array_equal(ids,baseline['ids'])
    labels=inputs.y[ids]
    b,c=baseline['score'].argmax(1),candidate['score'].argmax(1)
    kinds={'rescued':(b!=labels)&(c==labels),'harmed':(b==labels)&(c!=labels),
           'stable_correct':(b==labels)&(c==labels),'stable_wrong':(b!=labels)&(c!=labels)}
    chosen=[]
    for kind,mask in kinds.items():
        for j in ft.np.flatnonzero(mask)[:2]:
            row=inputs.rows[int(ids[j])]
            chosen.append(dict(kind=kind,index=int(ids[j]),id=row['id'],label=int(labels[j]),
                baseline_prediction=int(b[j]),candidate_prediction=int(c[j]),
                baseline_score=baseline['score'][j].tolist(),candidate_score=candidate['score'][j].tolist(),
                chips={p:None if row.get(p) is None else row[p]['image'] for p in ('vv','vh')}))
    save(folder/'paired_cases.json',dict(counts={k:int(v.sum()) for k,v in kinds.items()},
        examples=chosen,reference=str(reference_file),review='pending original-chip visual review',
        scope='validation, not independent test evidence'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    fig,axes=plt.subplots(len(chosen),2,figsize=(7,3*len(chosen)),squeeze=False)
    for i,entry in enumerate(chosen):
        for p,pol in enumerate(('vv','vh')):
            ax=axes[i,p]
            path=entry['chips'][pol]
            if path:
                with Image.open(path) as image:ax.imshow(ft.np.asarray(image),cmap='gray',interpolation='nearest')
            ax.set_title(f"{entry['kind']} id={entry['id']} {pol.upper()}\ny={entry['label']} {entry['baseline_prediction']}->{entry['candidate_prediction']}")
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(folder/'paired_native_chips.png',dpi=120)
    plt.close(fig)


def fit(args,ft,inputs,name):
    dest=OUT/args.dataset/f'{name}_s{args.seed}'
    dest.mkdir(parents=True,exist_ok=True)
    config=dict(dataset=args.dataset,seed=args.seed,variant=name,positions=POSITIONS[name],
        epochs=args.epochs,batch=24,microbatch=4,backbone_lr=1e-4,module_lr=1e-4,
        initialization='ImageNet backbone; identical fresh task operators/head per seed; no staged checkpoint',
        loss='class/vessel-balanced CE + 0.02 polarization-availability-masked M1 auxiliary CE',
        augmentation='paired native-grid rot90/flip, same order and codes across variants',
        selection='highest validation BA, then OA; fixed epochs; test not accessed',
        intervention_training=False,frequency_branches=False,
        scope='common-initialization position pilot; not replacement for current full training recipe',
        test_pixels_may_be_preprocessed=True,test_scored=False,test_labels_used_for_training_or_selection=False)
    if (dest/'config.json').exists():
        previous=bundle.read(dest/'config.json')
        if previous != json.loads(json.dumps(config)):
            raise ValueError(f'Existing run configuration differs: {dest}; use a separate output directory.')
    if (dest/'receipt.json').exists():
        print('SKIP_COMPLETED',name,flush=True)
        return
    save(dest/'config.json',config)
    model=make_model(ft,args.seed,name,4 if args.dataset=='oss2' else 8)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=ft.torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
    batches=ft.np.random.default_rng(args.seed)
    curve=[];best=(-1.,-1.);first=0;tick=time.perf_counter()
    resume=dest/'resume.pt'
    if resume.exists():
        state=ft.torch.load(resume,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        batches.bit_generator.state=state['rng']
        curve=state['curve'];best=tuple(state['best']);first=state['epoch']
        assert state['epochs_max']==args.epochs
        del state
    train=inputs.groups['small']
    for epoch in range(first,args.epochs):
        model.train()
        schedule=min(1.,(epoch+1)/2)*(.05+.95*.5*(1+math.cos(math.pi*epoch/args.epochs)))
        for group in optimizer.param_groups:group['lr']=1e-4*schedule
        order=batches.permutation(train)
        running=0.
        for start in range(0,len(order),24):
            ids=order[start:start+24]
            code=int(batches.integers(8))
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0,len(ids),4):
                part=ids[offset:offset+4]
                ref=model.reference(batch(ft,inputs,inputs.groups['references'],'small'))
                data=batch(ft,inputs,part,'small',code)
                z,aux=model(data,[inputs.rows[i] for i in part],ref,inputs.reference_rows)
                y=ft.torch.as_tensor(inputs.y[part],device=ft.DEVICE)
                w=ft.torch.as_tensor(inputs.weights[part],device=ft.DEVICE)
                ce=ft.F.cross_entropy(z,y,reduction='none')
                ac=ft.F.cross_entropy(aux.flatten(0,1),y.repeat_interleave(2),reduction='none').reshape(len(part),2)
                av=data['available']
                loss=((ce+.02*(ac*av).sum(1)/av.sum(1).clamp_min(1))*w).sum()/len(ids)
                assert ft.torch.isfinite(loss)
                loss.backward()
                running+=float(loss.detach())*len(ids)
                del ref,data,z,aux,loss
            ft.torch.nn.utils.clip_grad_norm_(params,5.)
            optimizer.step()
        ids=inputs.groups['validation']
        score=evaluate(ft,model,inputs,ids,'validation')
        report=metrics(ft,inputs.y[ids],score)
        curve.append(dict(epoch=epoch+1,loss=running/len(train),lr=optimizer.param_groups[0]['lr'],**report))
        key=(report['ba'],report['oa'])
        if key>best:
            best=key
            ft.torch.save(model.state_dict(),dest/'best.pt')
            ft.np.savez_compressed(dest/'validation.npz',ids=ids,score=score,label=inputs.y[ids])
        save(dest/'curve.json',curve)
        ft.torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            rng=batches.bit_generator.state,curve=curve,best=best,epoch=epoch+1,epochs_max=args.epochs),resume)
        save(OUT/'status.json',dict(state='training',dataset=args.dataset,variant=name,seed=args.seed,
                                  epoch=epoch+1,epochs=args.epochs,validation=report,test_scored=False))
        print('EPOCH',name,epoch+1,json.dumps(report),flush=True)
    save(dest/'receipt.json',dict(complete=True,config=config,validation_best_ba=best[0],
        validation_best_oa=best[1],seconds=time.perf_counter()-tick,test_scored=False,
        trainable=sum(p.numel() for p in params),epochs=len(curve),visual_review='pending'))
    base=OUT/args.dataset/f'current_s{args.seed}'/'validation.npz'
    if base.exists():case_summary(ft,inputs,dest,base)
    del model,optimizer,params
    gc.collect();ft.torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('smoke','train'))
    parser.add_argument('--dataset',choices=('oss2','ports24'),default='oss2')
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--variants',nargs='+',choices=list(POSITIONS),default=list(POSITIONS))
    args=parser.parse_args()
    bundle.verify(args.dataset)
    ports,ft=bundle.runtime(args.device)
    if args.action=='smoke':return smoke(args,ports,ft)
    if args.device!='cuda':raise ValueError('Training must run on the GPU server.')
    checked=OUT/args.dataset/f'smoke_s{args.seed}_cuda.json'
    if not checked.exists():raise FileNotFoundError('Run placement_study.py smoke first.')
    inputs=bundle.inputs_for(SimpleNamespace(dataset=args.dataset),ports,ft)
    for name in args.variants:fit(args,ft,inputs,name)
    save(OUT/'status.json',dict(state='pilot_complete',dataset=args.dataset,seed=args.seed,
        variants=args.variants,test_scored=False,visual_review='pending',formal_model_changed=False))


if __name__=='__main__':main()
