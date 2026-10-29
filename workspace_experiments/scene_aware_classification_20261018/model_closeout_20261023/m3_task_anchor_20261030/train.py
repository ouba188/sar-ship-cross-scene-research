"""Same-budget M3 adaptation to task-conditioned comparison anchors."""
import importlib.util
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('task_anchor_probe',HERE/'probe.py')
probe=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=probe
spec.loader.exec_module(probe)
previous,PREVIOUS,torch,np=probe.previous,probe.PREVIOUS,probe.torch,probe.np
attach,source_relations=probe.attach,probe.source_relations


def restore(path):
    blob=torch.load(path,map_location='cpu',weights_only=True)
    net=attach(previous.original(blob['dataset'],blob['seed']))
    net.load_state_dict(blob['state'],strict=True)
    return net.eval().requires_grad_(False)


def main(dataset,seeds):
    previous.setup()
    cache,(_,source,queries,refs)=previous.data(dataset)
    bank=cache.take(refs)
    y=np.array([r['label'] for r in queries]); sy=np.array([r['label'] for r in source])
    lookup={str(r['id']):i for i,r in enumerate(source)}
    for seed in seeds:
        out=HERE/dataset/'training';out.mkdir(parents=True,exist_ok=True)
        path=out/f'task_s{seed}.json'
        if path.exists():raise FileExistsError(path)
        net=previous.original(dataset,seed)
        baseline=cache.predict(net,queries,refs,bank)
        teacher=cache.predict(net,source,refs,bank)
        with np.load(PREVIOUS/dataset/'predictions'/f'ce_s{seed}.npz') as old:
            assert np.array_equal(baseline,old['baseline'])
            assert np.array_equal(old['ids'],[str(r['id']) for r in queries])
        attach(net)
        frozen_prediction=cache.predict(net,queries,refs,bank)
        schedule_path=PREVIOUS/dataset/'exposures'/f'schedule_s{seed}.json'
        schedule=[np.array([lookup[str(k)] for k in group]) for group in previous.json.loads(schedule_path.read_text())]
        selected=previous.trainable(net)
        names={n for n,p in selected}
        fixed={n:v.detach().cpu().clone() for n,v in net.state_dict().items() if n not in names}
        opt=torch.optim.AdamW([p for n,p in selected],lr=1e-4,weight_decay=.01)
        curve=[]; start=previous.time.perf_counter()
        for step,ix in enumerate(schedule):
            group=[source[i] for i in ix]
            labels=torch.tensor(sy[ix],device='cuda')
            target=torch.tensor(teacher[ix],device='cuda')
            opt.zero_grad(set_to_none=True)
            z=cache.score(net,group,refs,bank)
            ce=previous.F.cross_entropy(z,labels)
            kl=previous.F.kl_div(z.log_softmax(1),target.log_softmax(1),log_target=True,reduction='none').sum(1)
            keep=(kl*target.argmax(1).eq(labels)).mean()
            loss=ce+.1*keep
            assert torch.isfinite(loss)
            loss.backward()
            if step==0:
                grads={n:float(p.grad.norm()) for n,p in selected if p.grad is not None}
                assert any(v>0 for v in grads.values())
            opt.step()
            curve.append(dict(step=step+1,ce=float(ce.detach()),keep=float(keep.detach())))
            if (step+1)%16==0:print('TASK_ANCHOR_TRAIN',dataset,seed,curve[-1],flush=True)
        assert all(torch.equal(net.state_dict()[n].detach().cpu(),v) for n,v in fixed.items())
        net.requires_grad_(False)
        score=cache.predict(net,queries,refs,bank)
        source_score=cache.predict(net,source,refs,bank)
        relations=source_relations(net,cache,source,refs,bank)
        pred=HERE/dataset/'predictions';pred.mkdir(parents=True,exist_ok=True)
        np.savez(pred/f'task_s{seed}.npz',score=score,baseline=baseline,frozen=frozen_prediction,y=y,ids=np.array([str(r['id']) for r in queries]))
        np.savez(pred/f'task_source_s{seed}.npz',score=source_score,baseline=teacher,y=sy,ids=np.array([str(r['id']) for r in source]))
        torch.save(dict(dataset=dataset,seed=seed,state={n:v.detach().cpu() for n,v in net.state_dict().items()}),out/f'task_s{seed}.pt')
        report=dict(dataset=dataset,seed=seed,arm='task',source_count=len(source),query_count=len(queries),
            reference_count=len(refs),reference_ids=[str(r['id']) for r in refs],steps=len(schedule),batch_size=16,lr=1e-4,
            trainable_names=sorted(names),trainable_parameters=sum(p.numel() for n,p in selected),frozen_unchanged=True,
            baseline=previous.metrics(baseline,y),frozen=previous.metrics(frozen_prediction,y),after=previous.metrics(score,y),
            source_before=previous.metrics(teacher,sy),source_after=previous.metrics(source_score,sy),
            source_relations=relations['summary'],gradients=grads,curve=curve,seconds=previous.time.perf_counter()-start,evaluation_used=False)
        del net,opt,selected,fixed,z,loss,ce,keep
        net=restore(out/f'task_s{seed}.pt')
        small=queries[:4]
        report['restore_error']=float(abs(cache.predict(net,small,refs,bank)-score[:4]).max())
        q=previous.batch([previous.prepare_record(r) for r in small],'cuda')
        r=previous.batch([previous.prepare_record(v) for v in refs],'cuda')
        with torch.no_grad():
            r['prefix']=net.prefix(r['images'],stop=2)
            native=previous.forward_batch(net,q,r,previous.pair_mask(small,refs,'cuda')).cpu().numpy()
        report['native_error']=float(abs(native-score[:4]).max())
        assert max(report['restore_error'],report['native_error'])<1e-4
        previous.write(path,report)
        print('COMPLETE_TASK_ANCHOR',dataset,seed,report['after'],report['source_relations'],flush=True)
        del net,q,r


if __name__=='__main__':
    p=previous.argparse.ArgumentParser();p.add_argument('--dataset',choices=['private','oss2'],required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[17])
    a=p.parse_args();main(a.dataset,a.seeds)
