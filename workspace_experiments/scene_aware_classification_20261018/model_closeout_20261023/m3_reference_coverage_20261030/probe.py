"""Source-selected reference coverage; no final evaluation cohort is loaded."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent/'m3_reference_delta_20261030'
sys.path.insert(0, str(PREVIOUS))
from public import *
from sar_m123.pool_replacement import regional_mean
HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE/'anchor_corrected'


@torch.no_grad()
def protected_deep(net,prefix):
    b,p=prefix.shape[:2]
    z=prefix.flatten(0,1)
    for block in list(net.backbone.model.layer3.children())[:-1]:
        z=block(z)
    return net.anchor_last(z).reshape(b,p,1024,10,10)


class PrivateCache:
    def __init__(self):
        self.rows,self.arrays,self.indices,_=refit.load()
        self.lookup={str(r['id']):int(i) for k in self.rows for r,i in zip(self.rows[k],self.indices[k])}

    def take(self, records):
        values,extra=refit.get_batch(self.arrays,[self.lookup[str(r['id'])] for r in records])
        return dict(zip(('prefix','masks','valid','available'),values),**extra)

    def score(self, net, queries, refs, bank, details=False):
        return net.forward_from(**self.take(queries),reference=bank,
            pair_valid=pair_mask(queries,refs,'cuda'),return_details=details)

    @torch.no_grad()
    def predict(self, net, queries, refs, bank):
        return np.concatenate([self.score(net,queries[i:i+16],refs,bank).cpu().numpy() for i in range(0,len(queries),16)])


@torch.no_grad()
def descriptors(net, cache, records):
    vectors=[]
    for start in range(0,len(records),16):
        data=cache.take(records[start:start+16])
        b,p=data['prefix'].shape[:2]
        deep=protected_deep(net,data['prefix'])
        cols=[]
        for pol,reader in enumerate(net.m3[0]):
            ds,mass=reader.descriptors(deep[:,pol],data['prefix'][:,pol],
                data['slot_masks'][:,pol],data['local_slot_masks'][:,pol],
                data['valid'][:,pol],data['frequency'][:,pol],data['local_frequency'][:,pol])
            # All five comparison streams enter with equal norm, not raw channel count.
            for d in ds:
                weight=mass/mass.sum(1,keepdim=True).clamp_min(1e-8)
                v=F.normalize((d*weight[...,None]).sum(1),dim=-1)
                cols.append(v*data['available'][:,pol,None])
        vectors.append(F.normalize(torch.cat(cols,-1),dim=-1).cpu().numpy())
    return np.concatenate(vectors)


def select(records, features, per_class, method):
    rng=np.random.default_rng(1030)
    chosen=[]
    reports=[]
    y=np.array([r['label'] for r in records])
    for c in sorted(set(y)):
        ix=np.flatnonzero(y==c)
        ships=[str(records[i]['mmsi']) for i in ix]
        counts={s:ships.count(s) for s in set(ships)}
        weight=np.array([1/counts[s] for s in ships]);weight/=weight.sum()
        distance=np.maximum(0,1-features[ix]@features[ix].T)
        selected=[]; nearest=np.full(len(ix),2.)
        for _ in range(per_class):
            eligible=np.array([j for j,s in enumerate(ships) if s not in {ships[k] for k in selected}])
            if len(eligible)==0:
                break
            if method=='random':
                j=int(rng.choice(eligible))
            else:
                risk=(np.minimum(nearest[:,None],distance[:,eligible])*weight[:,None]).sum(0)
                j=int(eligible[risk.argmin()])
            selected.append(j);nearest=np.minimum(nearest,distance[:,j])
        assert len(selected)==per_class,(c,len(selected),per_class)
        chosen.extend(ix[selected].tolist())
        reports.append(dict(label=int(c),source_count=len(ix),ships=len(counts),coverage=float(nearest@weight)))
    return [records[i] for i in chosen], reports


def coverage(records, features, references):
    lookup={str(r['id']):i for i,r in enumerate(records)}
    values=[]
    for c in sorted({r['label'] for r in records}):
        a=[i for i,r in enumerate(records) if r['label']==c]
        b=[lookup[str(r['id'])] for r in references if r['label']==c]
        dist=np.maximum(0,1-features[a]@features[b].T).min(1)
        ships=[str(records[i]['mmsi']) for i in a]
        w=np.array([1/ships.count(s) for s in ships]);w/=w.sum()
        values.append(dict(label=c,coverage=float(dist@w)))
    return values


def main(dataset,seeds):
    setup()
    if dataset=='private':
        cache=PrivateCache();fit=cache.rows['fit'];queries=cache.rows['development'];original=cache.rows['references']
    else:
        groups={k:read_manifest(PUB/'manifests'/f'{k}.json') for k in ('source_fit','validation','references')}
        cache=Cache(groups);fit=groups['source_fit'];queries=groups['validation'];original=groups['references']
    y=np.array([r['label'] for r in queries])
    classes=sorted({r['label'] for r in fit})
    k=len(original)//len(classes)
    out=RUN_ROOT/dataset;out.mkdir(parents=True,exist_ok=True)
    for seed in seeds:
        path=(ROOT/'spatial_m12_full_20261029/training'/f'spatial_s{seed}/model.pt' if dataset=='private'
              else PUB/'training'/f'full_s{seed}/adapt/model.pt')
        net=load_model(path,'cuda') if dataset=='private' else api.load(path,'cuda')
        net.requires_grad_(False)
        feature=descriptors(net,cache,fit)
        banks={'original':original};diagnostics={'original':coverage(fit,feature,original)}
        for n in (k,2*k):
            for method in ('random','coverage'):
                key=f'{method}_{n}'
                banks[key],diagnostics[key]=select(fit,feature,n,method)
        output={}
        for name,refs in banks.items():
            assert all(str(r['id']) in {str(x['id']) for x in fit} for r in refs)
            bank=cache.take(refs)
            tick=time.perf_counter();z=cache.predict(net,queries,refs,bank)
            np.savez(out/f'{name}_s{seed}.npz',score=z,y=y,ids=np.array([str(r['id']) for r in queries]))
            output[name]=dict(metrics=metrics(z,y),reference_count=len(refs),coverage=diagnostics[name],
                reference_ids=[str(r['id']) for r in refs],seconds=time.perf_counter()-tick)
            print('BANK_RESULT',dataset,seed,name,output[name]['metrics'],flush=True)
        write(out/f'results_s{seed}.json',output)
        del net,bank


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['private','oss2'],required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[17])
    args=parser.parse_args();main(args.dataset,args.seeds)
