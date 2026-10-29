"""Frozen M3-output joint-adaptation repair on OSS2; no test-driven selection."""
from public import *
from sar_m123.current.model import training_policy,LIVE
from sar_m123.current.training import adaptation_loss


def main():
    setup()
    rows={k:read_manifest(PUB/'manifests'/f'{k}.json') for k in ('source_adapt','validation','evaluation','references')}
    cache=Cache(rows)
    refs=rows['references']
    bank=cache.take(refs)
    records=rows['source_adapt']
    weights=class_weights(records,'adapt')
    for seed in (17,29,43):
        teachers=json.loads((PUB/'teachers'/f'full_s{seed}.json').read_text())
        assert [t['id'] for t in teachers]==[r['id'] for r in records]
        for arm in ('control','joint_low'):
            out=HERE/'oss2_joint/training'/f'{arm}_s{seed}'
            if (out/'training.json').exists():
                continue
            net=api.load(PUB/'training'/f'full_s{seed}/fit/model.pt','cuda')
            training_policy(net,'adapt')
            for n,p in net.named_parameters():
                if arm=='joint_low' and n.startswith('m3.') and '.output.' in n:
                    p.requires_grad_(True)
            trainable=[p for p in net.parameters() if p.requires_grad]
            fixed={n:p.detach().cpu().clone() for n,p in net.named_parameters() if not p.requires_grad}
            optimizer=torch.optim.AdamW(trainable,lr=3e-5,weight_decay=.01)
            rng=np.random.default_rng(seed)
            tick=time.perf_counter()
            curve=[]
            steps=0
            for epoch in range(1,7):
                terms=[]
                for ix in np.array_split(rng.permutation(len(records)),range(16,len(records),16)):
                    q=[records[i] for i in ix]
                    y=torch.tensor([r['label'] for r in q],device='cuda')
                    w=torch.as_tensor(weights[ix],device='cuda')
                    target=torch.tensor([teachers[i]['original'] for i in ix],device='cuda')
                    useful=torch.tensor([teachers[i]['useful'] for i in ix],device='cuda')
                    optimizer.zero_grad(set_to_none=True)
                    score=cache.score(net,q,refs,bank)
                    if steps==0:
                        teacher_error=float((score.detach()-target).abs().max())
                        assert teacher_error<1e-4
                    loss,t=adaptation_loss(score,y,target,w,useful,.3683066666126251)
                    assert torch.isfinite(loss)
                    loss.backward()
                    optimizer.step()
                    steps+=1
                    terms.append({k:float(v.detach()) for k,v in t.items()})
                curve.append(dict(epoch=epoch,**{k:float(np.mean([t[k] for t in terms])) for k in terms[0]}))
                print('PUBLIC_JOINT_EPOCH',arm,seed,curve[-1],flush=True)
            assert steps==96
            assert all(torch.equal(dict(net.named_parameters())[n].detach().cpu(),v) for n,v in fixed.items())
            net.requires_grad_(False)
            api.save(net,out/'model.pt',m3_repair=arm,seed=seed)
            receipt=dict(arm=arm,seed=seed,steps=steps,epochs=6,lr=3e-5,source_count=len(records),
                trainable=sum(p.numel() for p in trainable),curve=curve,teacher_error=teacher_error,
                frozen_unchanged=True,test_selection=False,selection_reason='Frozen private development joint_low, not selected on OSS2 evaluation')
            pred=HERE/'oss2_joint/predictions'
            pred.mkdir(parents=True,exist_ok=True)
            for scope in ('validation','evaluation'):
                z=cache.predict(net,rows[scope],refs,bank)
                yy=np.array([r['label'] for r in rows[scope]])
                ids=np.array([r['id'] for r in rows[scope]])
                baseline=None
                if scope=='evaluation':
                    with np.load(PUB/'predictions'/f'evaluation_full_adapt_s{seed}.npz') as f:
                        assert np.array_equal(f['ids'],ids) and np.array_equal(f['y'],yy)
                        baseline=f['score']
                    receipt['original_evaluation']=metrics(baseline,yy)
                    receipt['baseline_prediction_maxerror']=float(abs(z-baseline).max()) if arm=='control' else None
                np.savez(pred/f'{scope}_{arm}_s{seed}.npz',score=z,y=yy,ids=ids,
                    **({'baseline':baseline} if baseline is not None else {}))
                receipt[scope]=metrics(z,yy)
            sample=rows['validation'][:4]
            expected=cache.predict(net,sample,refs,bank)
            values=batch([prepare_record(r) for r in sample],'cuda')
            native_refs=batch([prepare_record(r) for r in refs],'cuda')
            with torch.no_grad():
                native_refs['prefix']=net.prefix(native_refs['images'],stop=2)
                native=forward_batch(net,values,native_refs,pair_mask(sample,refs,'cuda')).cpu().numpy()
            receipt['native_error']=float(abs(native-expected).max())
            assert receipt['native_error']<1e-4
            del net,optimizer,fixed,trainable,values,native_refs,score,loss
            restored=api.load(out/'model.pt','cuda')
            replay=cache.predict(restored,sample,refs,bank)
            receipt['replay_error']=float(abs(replay-expected).max())
            assert receipt['replay_error']<1e-4
            receipt['seconds']=time.perf_counter()-tick
            write(out/'training.json',receipt)
            print('PUBLIC_JOINT_COMPLETE',arm,seed,receipt['evaluation'],flush=True)
            del restored


if __name__=='__main__':
    main()
