"""Repair the adaptation interface, not an extra endpoint continuation."""
from run import *
from sar_m123.current.model import training_policy, LIVE, save_model
from sar_m123.current.training import adaptation_loss


def main(seeds,arms):
    setup()
    rows,arrays,indices,bank=refit.load()
    add_adapt(rows,arrays,indices)
    records=rows['adapt']
    w=class_weights(records,'adapt')
    y=np.array([r['label'] for r in rows['development']])
    for seed in seeds:
        teachers=json.loads((ROOT/'spatial_m12_full_20261029/selection'/f'spatial_s{seed}.json').read_text())
        assert [str(t['id']) for t in teachers]==[str(r['id']) for r in records]
        for arm in arms:
            out=HERE/'joint_adaptation/training'/f'{arm}_s{seed}'
            if (out/'training.json').exists():
                continue
            net=load_model(ROOT/'spatial_m12_refit_20261029/training'/f'spatial_s{seed}/model.pt','cuda')
            training_policy(net,'adapt')
            extra=[]
            if arm!='control':
                for n,p in net.named_parameters():
                    if n.startswith('m3.') and '.output.' in n:
                        p.requires_grad_(True)
                        extra.append(p)
            fixed={n:p.detach().cpu().clone() for n,p in net.named_parameters() if not p.requires_grad}
            groups=[dict(params=[p for n,p in net.named_parameters() if n in LIVE],lr=3e-5)]
            if extra:
                groups.append(dict(params=extra,lr=3e-5 if arm=='joint_low' else 1e-4))
            optimizer=torch.optim.AdamW(groups,weight_decay=.01)
            rng=np.random.default_rng(seed)
            tick=time.perf_counter()
            curve=[]
            steps=0
            for epoch in range(1,7):
                terms=[]
                for ix in np.array_split(rng.permutation(len(records)),range(16,len(records),16)):
                    q=[records[i] for i in ix]
                    labels=torch.tensor([r['label'] for r in q],device='cuda')
                    weight=torch.as_tensor(w[ix],device='cuda')
                    target=torch.tensor([teachers[i]['original'] for i in ix],device='cuda')
                    useful=torch.tensor([teachers[i]['useful'] for i in ix],device='cuda')
                    optimizer.zero_grad(set_to_none=True)
                    score=refit.score(net,arrays,indices['adapt'][ix],q,rows['references'],bank)
                    if steps==0:
                        teacher_error=float((score.detach()-target).abs().max())
                        assert teacher_error<1e-4,teacher_error
                    loss,t=adaptation_loss(score,labels,target,weight,useful,.3683066666126251)
                    assert torch.isfinite(loss)
                    loss.backward()
                    if steps==0:
                        gradients={n:float(p.grad.norm()) for n,p in net.named_parameters() if p.requires_grad and p.grad is not None}
                    optimizer.step()
                    steps+=1
                    terms.append({k:float(v.detach()) for k,v in t.items()})
                curve.append(dict(epoch=epoch,**{k:float(np.mean([t[k] for t in terms])) for k in terms[0]}))
                print('JOINT_EPOCH',arm,seed,curve[-1],flush=True)
            assert all(torch.equal(dict(net.named_parameters())[n].detach().cpu(),p) for n,p in fixed.items())
            net.requires_grad_(False)
            after=refit.predict(net,arrays,indices['development'],rows['development'],rows['references'],bank)
            save_model(net,out/'model.pt',repair=arm,seed=seed)
            pred=HERE/'joint_adaptation/predictions'
            pred.mkdir(parents=True,exist_ok=True)
            with np.load(ROOT/'spatial_m12_full_20261029/predictions'/f'development_after_spatial_s{seed}.npz') as f:
                baseline=f['score']
            np.savez(pred/f'{arm}_s{seed}.npz',score=after,baseline=baseline,y=y,ids=np.array([str(r['id']) for r in rows['development']]))
            receipt=dict(arm=arm,seed=seed,steps=steps,epochs=6,source_count=len(records),
                before=metrics(baseline,y),after=metrics(after,y),curve=curve,teacher_error=teacher_error,
                nonselected_parameters_unchanged=True,gradients=gradients,seconds=time.perf_counter()-tick,
                added_trainable=sum(p.numel() for p in extra),baseline_prediction_maxerror=float(abs(after-baseline).max()) if arm=='control' else None)
            del optimizer,fixed,score,loss,net,extra,groups
            restored=load_model(out/'model.pt','cuda')
            replay=refit.predict(restored,arrays,indices['development'][:4],rows['development'][:4],rows['references'],bank)
            receipt['replay_error']=float(abs(replay-after[:4]).max())
            assert receipt['replay_error']<1e-4
            write(out/'training.json',receipt)
            print('JOINT_COMPLETE',arm,seed,receipt['after'],receipt['baseline_prediction_maxerror'],flush=True)
            del restored


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,nargs='+',default=[17])
    parser.add_argument('--arms',nargs='+',choices=['control','joint_low','joint_high'],default=['control','joint_low','joint_high'])
    args=parser.parse_args()
    main(args.seeds,args.arms)
