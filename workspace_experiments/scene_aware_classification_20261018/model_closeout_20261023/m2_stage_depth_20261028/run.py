"""Matched M2 depth comparison within layer2; one pre-add operator per view."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'m2_residual_placement_20261028'
spec = importlib.util.spec_from_file_location('depth_previous', OLD/'run.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
np, torch, legacy = previous.np, previous.torch, previous.legacy
RELEASE, setup = previous.RELEASE, previous.setup
read_manifest, pair_mask = previous.read_manifest, previous.pair_mask
Layer2BlockM2 = previous.previous.Layer2BlockM2
ARMS = dict(baseline=None, stage='stage', block1=1, block2=2, block3=3)
SEEDS = (17,29,43)


def write(name, value):
    (ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def construct(state, location, seed):
    torch.manual_seed(903)
    if isinstance(location,int):
        return Layer2BlockM2(state,'preadd',block_index=location).cuda().eval()
    return Layer2BlockM2(state,location or 'none').cuda().eval()


class Cache(previous.Cache):
    def __init__(self,name):
        super().__init__(name)
        for i in (1,2):
            self.arrays[f'prefix2_{i}'] = np.load(ROOT/'cache'/f'{name}_prefix2_{i}.npy',mmap_mode='r')


@torch.no_grad()
def build_prefix(state,records):
    model=construct(state,None,17)
    (ROOT/'cache').mkdir(exist_ok=True)
    errors={}
    for name,rows in records.items():
        old=previous.Cache(name)
        destinations={i:np.lib.format.open_memmap(ROOT/'cache'/f'{name}_prefix2_{i}.npy',mode='w+',
            dtype=np.float32,shape=(len(rows),2,512,20,20)) for i in (1,2)}
        error=0.
        for start in range(0,len(rows),16):
            ix=np.arange(start,min(start+16,len(rows)))
            data=old.get(ix)
            value=data['stage1'].flatten(0,1)
            for bi,block in enumerate(model.base.backbone.model.layer2):
                if bi in destinations:
                    destinations[bi][ix]=value.reshape(len(ix),2,512,20,20).cpu().numpy()
                value=block(value)
            error=max(error,float((value.reshape_as(data['raw2'])-data['raw2']).abs().max()))
        for dest in destinations.values():dest.flush()
        assert error<1e-4,(name,error)
        errors[name]=error
    write('prefix_replay.json',errors)
    del model
    gc.collect();torch.cuda.empty_cache()


def check(state,records,caches,bank):
    q=caches['source'].get(np.arange(4))
    legal=pair_mask(records['source'][:4],records['refs'],'cuda')
    labels=torch.tensor([r['label'] for r in records['source'][:4]],device='cuda')
    base=construct(state,None,17)
    with torch.no_grad():expected=base(q,bank,legal)
    del base
    result=dict(initial_identity={},nonzero_native_hook={},cached_live={},gradients={},parameters={},reused={})
    for arm,location in ARMS.items():
        model=construct(state,location,17)
        result['parameters'][arm]=sum(p.numel() for p in model.early.parameters())
        with torch.no_grad():error=float((model(q,bank,legal)-expected).abs().max())
        assert error<1e-4,(arm,'initial',error)
        result['initial_identity'][arm]=error
        if arm in ('block1','block2'):
            optimizer=torch.optim.AdamW(model.early.parameters(),lr=3e-4)
            for step in range(4):
                optimizer.zero_grad(set_to_none=True)
                torch.nn.functional.cross_entropy(model(q,bank,legal),labels).backward()
                for key,module in model.early.items():
                    output=sum(float(t[-1].weight.grad.abs().sum()) for t in module.transforms)
                    inner=sum(float(t[0].weight.grad.abs().sum()) for t in module.transforms)
                    assert output>0,(arm,key,step)
                    if step==3:
                        assert inner>0,(arm,key,'upstream')
                        result['gradients'][f'{arm}_{key}']=dict(output=output,inner=inner)
                optimizer.step()
            with torch.no_grad():
                direct=model.stage2(q,{})
                block=model.base.backbone.model.layer2[location]
                hook=block.bn3.register_forward_hook(lambda m,args,value:model.feedback(value,q,{}))
                try:native=model.base.backbone.model.layer2(q['stage1'].flatten(0,1)).reshape_as(direct)
                finally:hook.remove()
                error=float((direct-native).abs().max())
                assert error<1e-4,(arm,'native-full-stage',error)
                result['nonzero_native_hook'][arm]=error
                live={k:v for k,v in q.items() if not k.startswith('prefix2')}
                error=float((model(q,bank,legal)-model(live,bank,legal)).abs().max())
                assert error<1e-4,(arm,'cached-vs-live',error)
                result['cached_live'][arm]=error
            del optimizer
        del model
        gc.collect();torch.cuda.empty_cache()
    assert len({result['parameters'][a] for a in ('stage','block1','block2','block3')})==1
    saved=np.load(OLD/'predictions.npz')
    for arm,oldarm in [('baseline','baseline'),('stage','stage'),('block3','preadd')]:
        for seed in SEEDS:
            checkpoint=checkpoint_path(arm,seed)
            model=construct(state,ARMS[arm],seed)
            delta=torch.load(checkpoint,weights_only=True,map_location='cpu')['state']
            assert set(delta)=={n for n,p in model.named_parameters() if p.requires_grad}
            model.load_state_dict(delta,strict=False)
            actual=legacy.predict(model,caches['dev'],records['dev'][:16],bank,records['refs'],16)
            oldscore=saved['scores'][saved['arms'].tolist().index(oldarm),SEEDS.index(seed),:16]
            error=float(abs(actual-oldscore).max())
            assert error<1e-4,(arm,seed,'reuse',error)
            result['reused'][f'{arm}_{seed}']=error
            del model
            gc.collect();torch.cuda.empty_cache()
    write('diagnostics.json',result)
    print('PREPARE PASSED',flush=True)


def checkpoint_path(arm,seed):
    if arm in ('baseline','stage'):
        oldarm='baseline' if arm=='baseline' else 'layer2'
        return ROOT.parent/'shallow_m2_writeback_20261028/checkpoints'/f'{oldarm}_{seed}.pt'
    if arm=='block3':return OLD/'checkpoints'/f'preadd_{seed}.pt'
    return ROOT/'checkpoints'/f'{arm}_{seed}.pt'


def load_inputs():
    records={k:read_manifest(RELEASE/'manifests'/f'{v}.json') for k,v in
             [('source','source_adapt'),('dev','development'),('refs','references')]}
    oldcache=ROOT.parent/'shallow_m2_writeback_20261028/cache'
    for name,rows in records.items():
        ids=json.loads((oldcache/name/'complete.json').read_text())['ids']
        assert ids==[str(r['id']) for r in rows]
    state=torch.load(RELEASE/'weights/teacher.pt',weights_only=True,map_location='cpu')['state']
    return state,records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('prepare','train','cost'),default='train')
    args=parser.parse_args();setup('cuda')
    state,records=load_inputs()
    if args.mode=='prepare':build_prefix(state,records)
    caches={name:Cache(name) for name in records}
    bank=caches['refs'].get(np.arange(len(records['refs'])))
    if args.mode=='prepare':
        check(state,records,caches,bank)
        return
    if args.mode=='cost':
        benchmark(state,records,caches,bank)
        return
    if not (ROOT/'diagnostics.json').is_file():raise RuntimeError('Run prepare first')
    archived=json.loads((RELEASE/'manifests/selection_archived.json').read_text(encoding='utf-8'))
    lookup={str(r['id']):r for r in archived}
    selection=[lookup[str(r['id'])] for r in records['source']]
    legacy.ROOT,legacy.ARMS,legacy.construct=ROOT,ARMS,construct
    old=np.load(OLD/'predictions.npz')
    assert old['ids'].tolist()==[str(r['id']) for r in records['dev']]
    scores=np.zeros((len(ARMS),len(SEEDS),len(records['dev']),8),np.float32)
    bypasses=scores.copy();runs=[]
    oldruns=json.loads((OLD/'runs.json').read_text())
    for ai,arm in enumerate(ARMS):
        for si,seed in enumerate(SEEDS):
            if arm in ('baseline','stage','block3'):
                oldarm='preadd' if arm=='block3' else arm
                oi=old['arms'].tolist().index(oldarm)
                score,bypass=old['scores'][oi,si],old['bypasses'][oi,si]
                receipt=next(r.copy() for r in oldruns if r['arm']==oldarm and r['seed']==seed)
                receipt.update(arm=arm,reused_from=str(OLD/'runs.json'))
            elif (ROOT/f'completed_{arm}_{seed}.npz').is_file():
                saved=np.load(ROOT/f'completed_{arm}_{seed}.npz')
                score,bypass=saved['score'],saved['bypass']
                receipt=json.loads((ROOT/f'run_{arm}_{seed}.json').read_text())
            else:
                score,bypass,receipt=legacy.train_one(state,arm,seed,caches['source'],caches['dev'],
                    records['source'],records['dev'],bank,records['refs'],selection,16)
                receipt.update(block_index=ARMS[arm],position='preadd',new_m2_per_polarization=1,
                    adapter_channels=512,grid=[20,20])
                np.savez(ROOT/f'completed_{arm}_{seed}.npz',score=score,bypass=bypass)
                write(f'run_{arm}_{seed}.json',receipt)
            scores[ai,si]=score;bypasses[ai,si]=bypass;runs.append(receipt)
            write('runs.json',runs)
    np.savez(ROOT/'predictions.npz',ids=old['ids'],labels=old['labels'],initial=old['initial'],
        scores=scores,bypasses=bypasses,arms=np.array(list(ARMS)),seeds=np.array(SEEDS))
    write('complete.json',dict(new_fits=6,reused_fits=9,steps_per_fit=54,endpoint='fixed_epoch6',
        tuning=False,default_model_changed=False,scope='historical development comparison, not independent selection'))
    print('COMPLETE: six new fits, nine reused controls',flush=True)


@torch.inference_mode()
def benchmark(state,records,caches,bank):
    # Identical uncached stage1 entry for every arm; excludes shared chip encoding and IO.
    q={k:v for k,v in caches['dev'].get(np.arange(16)).items() if not k.startswith('prefix2')}
    b={k:v for k,v in bank.items() if not k.startswith('prefix2')}
    legal=pair_mask(records['dev'][:16],records['refs'],'cuda')
    output={}
    for arm,location in ARMS.items():
        model=construct(state,location,17)
        model.load_state_dict(torch.load(checkpoint_path(arm,17),weights_only=True,map_location='cpu')['state'],strict=False)
        def forward():
            if arm in ('baseline','stage'):
                # These arms normally reuse raw2; include their live stage2 cost too.
                live=[]
                for data in (q,b):
                    item=dict(data)
                    item['raw2']=model.base.backbone.model.layer2(data['stage1'].flatten(0,1)).reshape_as(data['raw2'])
                    live.append(item)
                return model(live[0],live[1],legal)
            return model(q,b,legal)
        for _ in range(4):forward()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        times=[]
        for _ in range(20):
            start=time.perf_counter();forward();torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
        output[arm]=dict(batch16_median_ms=float(np.median(times)),batch16_p90_ms=float(np.percentile(times,90)),
            extra_parameters=sum(p.numel() for p in model.early.parameters()),peak_cuda_mb=torch.cuda.max_memory_allocated()/1024**2)
        del model
        gc.collect();torch.cuda.empty_cache()
    write('cost.json',dict(results=output,scope='stage1/raw2/anchors ready, 16 queries and16 refs, cached prefix disabled; not chip-to-score latency'))


if __name__=='__main__':main()
