"""OSS2 source-validation check of the fixed protected-output repair."""
from run import *
PUB=ROOT/'final_oss2_replication_20261029'
spec=importlib.util.spec_from_file_location('delta_public_api',ROOT/'final_module_ablations_20261029/model.py')
api=importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
from sar_m123.current.data import SHAPES


class Cache:
    def __init__(self,groups):
        records={r['id']:r for group in groups.values() for r in group}
        self.lookup={key:i for i,key in enumerate(records)}
        ix=np.array([r['indices']['table_index'] for r in records.values()])
        self.values={}
        for k in sorted((set(SHAPES)-{'images'})|{'prefix'}):
            a=np.load(PUB/'cache'/f'{k}.npy',mmap_mode='r')
            t=torch.empty((len(ix),*a.shape[1:]),device='cuda')
            for first in range(0,len(ix),16):
                t[first:first+16].copy_(torch.as_tensor(np.array(a[ix[first:first+16]],copy=True),device='cuda'))
            self.values[k]=t

    def take(self,rows):
        ix=torch.tensor([self.lookup[r['id']] for r in rows],device='cuda')
        return {k:v.index_select(0,ix) for k,v in self.values.items()}

    def score(self,net,queries,refs,bank):
        return net.forward_from(**self.take(queries),reference=bank,pair_valid=pair_mask(queries,refs,'cuda'))

    @torch.no_grad()
    def predict(self,net,queries,refs,bank):
        return np.concatenate([self.score(net,queries[i:i+24],refs,bank).cpu().numpy() for i in range(0,len(queries),24)])


def main(seeds):
    setup()
    rows={k:read_manifest(PUB/'manifests'/f'{k}.json') for k in ('source_fit','source_adapt','validation','references')}
    # No evaluation or selection manifests/labels are loaded in this pilot.
    cache=Cache(rows)
    refs=rows['references']
    bank=cache.take(refs)
    train=rows['source_adapt']
    weight=class_weights(train,'adapt')
    y=np.array([r['label'] for r in rows['validation']])
    chosen=[r for c in range(4) for r in [r for r in rows['source_fit'] if r['label']==c][:8]]
    for seed in seeds:
        path=PUB/'training'/f'full_s{seed}/adapt/model.pt'
        net=api.load(path,'cuda')
        teacher=cache.predict(net,train,refs,bank)
        before=cache.predict(net,rows['validation'],refs,bank)
        del net
        for arm in ('total','control'):
            out=HERE/'oss2_validation/training'/f'{arm}_s{seed}'
            if (out/'training.json').exists():
                continue
            net=api.load(path,'cuda')
            scales=[]
            if arm=='total':
                attach(net,arm)
                totals=[0.,0.]
                count=[0,0]
                with torch.no_grad():
                    for first in range(0,len(chosen),16):
                        cache.score(net,chosen[first:first+16],refs,bank)
                        for p,m in enumerate(net.m3[0]):
                            totals[p]+=float(m.diagnostic['selected_square'])
                            count[p]+=m.diagnostic['selected_numel']
                    for p,m in enumerate(net.m3[0]):
                        scale=(totals[p]/count[p])**.5
                        m.extra_scale.fill_(scale)
                        scales.append(scale)
            initial=cache.predict(net,rows['validation'][:4],refs,bank)
            initial_error=float(abs(initial-before[:4]).max())
            assert initial_error<1e-4
            params=configure(net,arm)
            fixed={n:p.detach().cpu().clone() for n,p in net.named_parameters() if not p.requires_grad}
            optimizer=torch.optim.AdamW(params,lr=.0001,weight_decay=.01)
            rng=np.random.default_rng(seed)
            curve=[]
            tick=time.perf_counter()
            for epoch in range(1,7):
                terms=[]
                for ix in np.array_split(rng.permutation(len(train)),range(24,len(train),24)):
                    group=[train[i] for i in ix]
                    labels=torch.tensor([r['label'] for r in group],device='cuda')
                    w=torch.as_tensor(weight[ix],device='cuda')
                    target=torch.as_tensor(teacher[ix],device='cuda')
                    optimizer.zero_grad(set_to_none=True)
                    z=cache.score(net,group,refs,bank)
                    ce=(w*F.cross_entropy(z,labels,reduction='none')).mean()
                    kl=F.kl_div(z.log_softmax(1),target.log_softmax(1),log_target=True,reduction='none').sum(1)
                    keep=(kl*target.argmax(1).eq(labels)*w).mean()
                    loss=ce+.1*keep
                    assert torch.isfinite(loss)
                    loss.backward()
                    optimizer.step()
                    terms.append([float(ce.detach()),float(keep.detach())])
                curve.append(dict(epoch=epoch,ce=float(np.mean(terms,0)[0]),keep=float(np.mean(terms,0)[1])))
                print('PUBLIC_EPOCH',arm,seed,curve[-1],flush=True)
            assert all(torch.equal(dict(net.named_parameters())[n].detach().cpu(),v) for n,v in fixed.items())
            net.requires_grad_(False)
            after=cache.predict(net,rows['validation'],refs,bank)
            out.mkdir(parents=True,exist_ok=True)
            torch.save(dict(repair=arm,classes=4,state={n:p.detach().cpu() for n,p in net.state_dict().items()}),out/'model.pt')
            pred=HERE/'oss2_validation/predictions'
            pred.mkdir(parents=True,exist_ok=True)
            np.savez(pred/f'{arm}_s{seed}.npz',score=after,baseline=before,y=y,ids=np.array([r['id'] for r in rows['validation']]))
            q=rows['validation'][:4]
            values=batch([prepare_record(r) for r in q],'cuda')
            native_bank=batch([prepare_record(r) for r in refs],'cuda')
            with torch.no_grad():
                native_bank['prefix']=net.prefix(native_bank['images'],stop=2)
                native=forward_batch(net,values,native_bank,pair_mask(q,refs,'cuda')).cpu().numpy()
            native_error=float(abs(native-after[:4]).max())
            assert native_error<1e-4
            receipt=dict(arm=arm,seed=seed,source_count=len(train),validation_count=len(y),reference_count=len(refs),
                epochs=6,steps=66,lr=.0001,scales=scales,curve=curve,initial_error=initial_error,native_error=native_error,
                before=metrics(before,y),after=metrics(after,y),source_baseline=metrics(teacher,np.array([r['label'] for r in train])),
                nonselected_parameters_unchanged=True,seconds=time.perf_counter()-tick,evaluation_used=False)
            del net,params,optimizer,fixed,values,native_bank,z,loss
            saved=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
            restored=api.AblationModel('full',4).cuda().eval()
            if arm=='total':
                attach(restored,arm)
            restored.load_state_dict(saved['state'],strict=True)
            restored.requires_grad_(False)
            replay=cache.predict(restored,q,refs,bank)
            receipt['replay_error']=float(abs(replay-after[:4]).max())
            assert receipt['replay_error']<1e-4
            write(out/'training.json',receipt)
            print('PUBLIC_COMPLETE',arm,seed,receipt['before'],receipt['after'],flush=True)
            del saved,restored


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,nargs='+',default=[17])
    main(parser.parse_args().seeds)
