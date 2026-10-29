"""Matched M3 value-path adaptation after a source-centered comparison change."""
from center import *
from sar_m123.current.training import adaptation_loss


def load_candidate(path):
    blob=torch.load(path,map_location='cpu',weights_only=True)
    net=(SpatialM12ResNet50(None,'pre3x3','local',8) if blob['dataset']=='private'
         else api.AblationModel('full',4))
    if blob['arm']=='center':
        for p,old in enumerate(net.m3[0]):
            m=CenteredMatch(dynamic=old.dynamic,frequency=old.frequency,temperature=old.temperature)
            for i in range(5):
                m.register_buffer(f'center_{i}',torch.zeros_like(blob['state'][f'm3.0.{p}.center_{i}']))
            m.centered=True
            net.m3[0][p]=m
    net.load_state_dict(blob['state'],strict=True)
    return net.cuda().eval().requires_grad_(False)


def main(dataset,seeds):
    setup()
    if dataset=='private':
        cache=PrivateCache()
        add_adapt(cache.rows,cache.arrays,cache.indices)
        cache.lookup.update({str(r['id']):int(i) for r,i in zip(cache.rows['adapt'],cache.indices['adapt'])})
        queries=cache.rows['development'];refs=cache.rows['references'];train=cache.rows['adapt']
    else:
        groups={k:read_manifest(PUB/'manifests'/f'{k}.json') for k in ('source_adapt','validation','references')}
        cache=Cache(groups);queries=groups['validation'];refs=groups['references'];train=groups['source_adapt']
    bank=cache.take(refs);y=np.array([r['label'] for r in queries]);weights=class_weights(train,'adapt')
    out=RUN_ROOT/dataset/'trained_center';out.mkdir(parents=True,exist_ok=True)
    for seed in seeds:
        path=(ROOT/'spatial_m12_full_20261029/training'/f'spatial_s{seed}/model.pt' if dataset=='private'
            else PUB/'training'/f'full_s{seed}/adapt/model.pt')
        original=load_model(path,'cuda') if dataset=='private' else api.load(path,'cuda')
        baseline=cache.predict(original,queries,refs,bank)
        teacher=cache.predict(original,train,refs,bank)
        initial_state={n:v.cpu().clone() for n,v in original.state_dict().items()}
        del original
        centers=torch.load(RUN_ROOT/dataset/'center'/f'centers_s{seed}.pt',map_location='cuda',weights_only=True)
        for arm in ('control','center'):
            net=load_model(path,'cuda') if dataset=='private' else api.load(path,'cuda')
            if arm=='center':
                for p,old in enumerate(net.m3[0]):
                    m=CenteredMatch(dynamic=old.dynamic,frequency=old.frequency,temperature=old.temperature).cuda()
                    m.load_state_dict(old.state_dict(),strict=True)
                    for i in range(5):m.register_buffer(f'center_{i}',centers[f'{p}_{i}'])
                    m.centered=True;net.m3[0][p]=m
            net.eval().requires_grad_(False)
            selected=[]
            for n,p in net.named_parameters():
                if n.startswith('m3.') and ('.output.' in n or '.project.' in n or n.endswith('.gain')):
                    p.requires_grad_(True);selected.append((n,p))
            optimizer=torch.optim.AdamW([p for _,p in selected],lr=1e-4,weight_decay=.01)
            rng=np.random.default_rng(seed);curve=[];steps=0;tick=time.perf_counter()
            for epoch in range(1,7):
                losses=[]
                for ix in np.array_split(rng.permutation(len(train)),range(24,len(train),24)):
                    q=[train[i] for i in ix];label=torch.tensor([r['label'] for r in q],device='cuda')
                    w=torch.tensor(weights[ix],device='cuda');target=torch.tensor(teacher[ix],device='cuda')
                    optimizer.zero_grad(set_to_none=True)
                    z=cache.score(net,q,refs,bank)
                    loss,_=adaptation_loss(z,label,target,w,torch.zeros_like(w),0.)
                    assert torch.isfinite(loss);loss.backward()
                    if steps==0:grad={n:float(p.grad.norm()) for n,p in selected if p.grad is not None}
                    optimizer.step();steps+=1;losses.append(float(loss.detach()))
                curve.append(dict(epoch=epoch,loss=float(np.mean(losses))))
                print('CENTER_TRAIN',dataset,arm,seed,curve[-1],flush=True)
            train_names={n for n,p in selected}
            assert all(torch.equal(p.detach().cpu(),initial_state[n]) for n,p in net.named_parameters() if n not in train_names)
            net.requires_grad_(False)
            score=cache.predict(net,queries,refs,bank)
            np.savez(out/f'{arm}_s{seed}.npz',score=score,baseline=baseline,y=y,ids=np.array([str(r['id']) for r in queries]))
            torch.save(dict(state={n:v.detach().cpu() for n,v in net.state_dict().items()},arm=arm,dataset=dataset),out/f'{arm}_s{seed}.pt')
            report=dict(arm=arm,seed=seed,steps=steps,source_count=len(train),query_count=len(queries),lr=1e-4,
                trainable_names=sorted(train_names),frozen_parameters_unchanged=True,gradients=grad,curve=curve,
                baseline=metrics(baseline,y),after=metrics(score,y),seconds=time.perf_counter()-tick)
            # Restore all state into this explicitly configured architecture and replay actual queries.
            saved=torch.load(out/f'{arm}_s{seed}.pt',map_location='cuda',weights_only=True)
            net.load_state_dict(saved['state'],strict=True)
            replay=cache.predict(net,queries[:4],refs,bank)
            report['saved_state_replay_error']=float(abs(replay-score[:4]).max())
            assert report['saved_state_replay_error']<1e-4
            write(out/f'{arm}_s{seed}.json',report)
            print('TRAINED_RESULT',dataset,arm,seed,report['after'],flush=True)
            del optimizer,net,selected,loss,z,saved


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['private','oss2'],required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[17]);args=parser.parse_args();main(args.dataset,args.seeds)
