"""Source-centered comparison only; preserve M3 transport/value/writeback."""
from probe import *
from sar_m123.selective_matching import SelectiveReferenceMatch


class CenteredMatch(SelectiveReferenceMatch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.centered=False
        self.accumulating=False
        self.source_weight=None
        self.sums=None
        self.counts=None

    def descriptors(self, anchors, local, slots, local_slots, valid, energy, local_energy):
        with torch.no_grad():
            deep=self.reader3(anchors.detach(),slots,valid,energy)
            fine=self.reader2(local.detach(),local_slots,valid,local_energy)
            raw=[deep['mean'],fine['mean']]+list(fine['bands'].unbind(2))
            masses=[deep['mass'],fine['mass'],fine['mass'],fine['mass'],fine['mass']]
            if self.accumulating:
                if self.sums is None:
                    self.sums=[torch.zeros_like(v[0],dtype=torch.float64) for v in raw]
                    self.counts=[torch.zeros_like(m[0],dtype=torch.float64) for m in masses]
                for i,(v,m) in enumerate(zip(raw,masses)):
                    w=(m>0).double()*self.source_weight[:,None]
                    self.sums[i]+=(v.double()*w[...,None]).sum(0)
                    self.counts[i]+=w.sum(0)
            result=[]
            for i,v in enumerate(raw):
                if self.centered:
                    v=v-getattr(self,f'center_{i}')
                result.append(F.normalize(v,dim=-1)*(masses[i]>0)[...,None])
        return result,deep['mass']


@torch.no_grad()
def attach_center(net,cache,records):
    from collections import Counter
    counts=Counter(str(r['mmsi']) for r in records)
    for pol,old in enumerate(net.m3[0]):
        new=CenteredMatch(dynamic=old.dynamic,frequency=old.frequency,temperature=old.temperature).cuda()
        new.load_state_dict(old.state_dict(),strict=True)
        new.accumulating=True
        net.m3[0][pol]=new
    for start in range(0,len(records),16):
        group=records[start:start+16];data=cache.take(group)
        b,p=data['prefix'].shape[:2]
        deep=protected_deep(net,data['prefix'])
        for pol,m in enumerate(net.m3[0]):
            m.source_weight=torch.tensor([1/counts[str(r['mmsi'])] for r in group],device='cuda')*data['available'][:,pol]
            m.descriptors(deep[:,pol],data['prefix'][:,pol],data['slot_masks'][:,pol],
                data['local_slot_masks'][:,pol],data['valid'][:,pol],data['frequency'][:,pol],data['local_frequency'][:,pol])
    stats=[]
    for m in net.m3[0]:
        m.accumulating=False
        stats.append([])
        for i,(s,c) in enumerate(zip(m.sums,m.counts)):
            center=(s/c.clamp_min(1)[...,None]).float()
            m.register_buffer(f'center_{i}',center)
            stats[-1].append(dict(stream=i,shape=list(center.shape),norm=float(center.norm()),empty_slots=int((c==0).sum())))
        m.sums=m.counts=m.source_weight=None
    net.requires_grad_(False)
    return stats


@torch.no_grad()
def relation(net,cache,queries,refs,bank):
    out=[]
    for first in range(0,len(queries),16):
        q=queries[first:first+16]
        _,detail=net.forward_from(**cache.take(q),reference=bank,pair_valid=pair_mask(q,refs,'cuda'),return_details=True)
        same=torch.tensor([[r['label']==s['label'] for s in refs] for r in q],device='cuda')
        for pol,d in enumerate(detail['operators']['m3_0']):
            w=d['reference_weights'];live=d['reference_valid'].any(1)
            ent=-(w*w.clamp_min(1e-20).log()).sum(1).mean(1)
            mass=(w*same[...,None]).sum(1).mean(1)
            for j,r in enumerate(q):
                if live[j]:out.append(dict(id=str(r['id']),pol=pol,entropy=float(ent[j]),same_mass=float(mass[j])))
    return out


def main(dataset,seeds):
    setup()
    if dataset=='private':
        cache=PrivateCache();fit=cache.rows['fit'];queries=cache.rows['development'];refs=cache.rows['references']
    else:
        groups={k:read_manifest(PUB/'manifests'/f'{k}.json') for k in ('source_fit','validation','references')}
        cache=Cache(groups);fit=groups['source_fit'];queries=groups['validation'];refs=groups['references']
    out=RUN_ROOT/dataset/'center';out.mkdir(parents=True,exist_ok=True)
    y=np.array([r['label'] for r in queries]);bank=cache.take(refs)
    for seed in seeds:
        path=(ROOT/'spatial_m12_full_20261029/training'/f'spatial_s{seed}/model.pt' if dataset=='private'
            else PUB/'training'/f'full_s{seed}/adapt/model.pt')
        net=load_model(path,'cuda') if dataset=='private' else api.load(path,'cuda')
        sample=cache.take(fit[:2])
        valid=sample['valid']*sample['available'][...,None,None,None]
        rv=bank['valid']*bank['available'][...,None,None,None]
        with torch.no_grad():
            paths=net.stage3_paths(sample['prefix'],valid,sample['available'],sample['field2'],sample['field3'],
                bank,rv,sample['slot_masks'],sample['frequency'],{},[])
            reference_anchor=paths[3]
            new_anchor=protected_deep(net,sample['prefix'])
            old_anchor=net.backbone.model.layer3(sample['prefix'].flatten(0,1)).reshape_as(new_anchor)
            anchor_error=float((reference_anchor-new_anchor).abs().max())
            old_anchor_error=float((reference_anchor-old_anchor).abs().max())
        assert anchor_error<1e-6,anchor_error
        original=cache.predict(net,queries,refs,bank)
        statistics=attach_center(net,cache,fit)
        initial=cache.predict(net,queries[:4],refs,bank)
        error=float(abs(initial-original[:4]).max());assert error<1e-4,error
        report={'initial_error':error,'calibration':statistics,'original':metrics(original,y),
            'actual_anchor_error':anchor_error,'previous_live_anchor_error':old_anchor_error}
        report['raw_relation']=relation(net,cache,queries,refs,bank)
        for m in net.m3[0]:m.centered=True
        score=cache.predict(net,queries,refs,bank)
        report['centered']=metrics(score,y)
        report['centered_relation']=relation(net,cache,queries,refs,bank)
        np.savez(out/f'centered_s{seed}.npz',score=score,baseline=original,y=y,ids=np.array([str(r['id']) for r in queries]))
        torch.save({f'{p}_{i}':getattr(m,f'center_{i}').cpu() for p,m in enumerate(net.m3[0]) for i in range(5)},out/f'centers_s{seed}.pt')
        write(out/f'results_s{seed}.json',report)
        print('CENTER_RESULT',dataset,seed,report['original'],report['centered'],flush=True)
        del net


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['private','oss2'],required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[17])
    args=parser.parse_args();main(args.dataset,args.seeds)
