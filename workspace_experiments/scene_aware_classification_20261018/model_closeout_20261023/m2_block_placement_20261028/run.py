"""Fixed-budget M2 position experiment, reusing matched baseline/stage fits."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[2]
PREVIOUS = ROOT.parent/'shallow_m2_writeback_20261028'
RELEASE = WORK/'sar_m123_current_20261027'
sys.path.insert(0, str(WORK/'sar_m123_current_20261027'))
spec = importlib.util.spec_from_file_location('matched_shallow_runner', PREVIOUS/'run.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
from sar_m123.block_m2_placement import Layer2BlockM2
from sar_m123.current.data import read_manifest, pair_mask
from sar_m123.current.model import setup

ARMS = dict(baseline=(), stage='stage', pre3x3='pre3x3', post3x3='post3x3')
SEEDS = (17, 29, 43)


def write(name, value):
    (ROOT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def construct(state, position, seed):
    torch.manual_seed(903)
    return Layer2BlockM2(state, position or 'none').cuda().eval()


class Cache:
    def __init__(self, name):
        self.arrays = {p.stem:np.load(p,mmap_mode='r') for p in (PREVIOUS/'cache'/name).glob('*.npy')}
        self.arrays['prefix2'] = np.load(ROOT/'cache'/f'{name}_prefix2.npy',mmap_mode='r')

    def get(self, indices, device='cuda'):
        return {k:torch.as_tensor(np.array(v[indices]),device=device) for k,v in self.arrays.items()}


@torch.no_grad()
def build_prefix(state, records):
    model = construct(state, (), 17)
    (ROOT/'cache').mkdir(exist_ok=True)
    errors = {}
    for name, rows in records.items():
        cached_ids = json.loads((PREVIOUS/'cache'/name/'complete.json').read_text())['ids']
        assert cached_ids == [str(r['id']) for r in rows]
        x1 = np.load(PREVIOUS/'cache'/name/'stage1.npy',mmap_mode='r')
        raw2 = np.load(PREVIOUS/'cache'/name/'raw2.npy',mmap_mode='r')
        dest = np.lib.format.open_memmap(ROOT/'cache'/f'{name}_prefix2.npy',mode='w+',
                                       dtype=np.float32,shape=(len(rows),2,512,20,20))
        error = 0.
        for start in range(0,len(rows),16):
            ix = np.arange(start,min(start+16,len(rows)))
            z = torch.as_tensor(np.array(x1[ix]),device='cuda').flatten(0,1)
            for block in list(model.base.backbone.model.layer2.children())[:-1]:
                z = block(z)
            dest[ix] = z.reshape(len(ix),2,512,20,20).cpu().numpy()
            actual = model.base.backbone.model.layer2[-1](z).reshape(len(ix),2,512,20,20)
            error = max(error,float((actual-torch.as_tensor(np.array(raw2[ix]),device='cuda')).abs().max()))
        dest.flush()
        assert error<1e-4,(name,error)
        errors[name] = error
    del model
    gc.collect();torch.cuda.empty_cache()
    write('prefix_replay.json',errors)


def check(state, records, caches, bank):
    q = caches['source'].get(np.arange(4))
    legal = pair_mask(records['source'][:4], records['refs'], 'cuda')
    base = construct(state, (), 17)
    with torch.no_grad():
        expected = base(q,bank,legal)
    del base
    report = dict(initial_identity={}, gradient={}, native_block_hook_parity={},
                  cached_vs_live_prefix={}, reused_checkpoints={}, region_coverage={})
    for arm,position in ARMS.items():
        m = construct(state,position,17)
        with torch.no_grad():
            error = float((m(q,bank,legal)-expected).abs().max())
        assert error<1e-4,(arm,error)
        report['initial_identity'][arm]=error
        if arm in ('pre3x3','post3x3'):
            opt=torch.optim.AdamW([p for p in m.early.parameters()],lr=3e-4)
            labels=torch.tensor([r['label'] for r in records['source'][:4]],device='cuda')
            for step in range(4):
                opt.zero_grad(set_to_none=True)
                loss=torch.nn.functional.cross_entropy(m(q,bank,legal),labels)
                loss.backward()
                for key,module in m.early.items():
                    g=sum(float(t[-1].weight.grad.abs().sum()) for t in module.transforms)
                    inner=sum(float(t[0].weight.grad.abs().sum()) for t in module.transforms)
                    assert g>0,(arm,key,step,'output')
                    if step==3:
                        assert inner>0,(arm,key,'upstream')
                        report['gradient'][f'{arm}_{key}']=dict(output=g,inner=inner)
                opt.step()
            with torch.no_grad():
                direct=m.stage2(q,{})
                block=m.base.backbone.model.layer2[-1]
                target=block.conv2 if arm=='pre3x3' else block.conv3
                hook=target.register_forward_pre_hook(lambda module,args: (m.feedback(args[0],q,{}),))
                try:
                    native=block(q['prefix2'].flatten(0,1)).reshape_as(direct)
                finally:
                    hook.remove()
                error=float((direct-native).abs().max())
                assert error<1e-5,(arm,'native-block-order',error)
                report['native_block_hook_parity'][arm]=error
                live={k:v for k,v in q.items() if k!='prefix2'}
                error=float((m(q,bank,legal)-m(live,bank,legal)).abs().max())
                assert error<1e-4,(arm,'prefix',error)
                report['cached_vs_live_prefix'][arm]=error
            del opt
        del m
        gc.collect();torch.cuda.empty_cache()
    for arm,oldarm in [('baseline','baseline'),('stage','layer2')]:
        for seed in SEEDS:
            checkpoint=torch.load(PREVIOUS/'checkpoints'/f'{oldarm}_{seed}.pt',weights_only=True,map_location='cpu')
            m=construct(state,ARMS[arm],seed)
            keys={n for n,p in m.named_parameters() if p.requires_grad}
            assert keys==set(checkpoint['state']),(arm,'checkpoint keys')
            m.load_state_dict(checkpoint['state'],strict=False)
            score=legacy.predict(m,caches['dev'],records['dev'][:16],bank,records['refs'],16)
            saved=np.load(PREVIOUS/f'completed_{oldarm}_{seed}.npz')['score'][:16]
            error=float(abs(saved-score).max())
            assert error<1e-4,(arm,seed,error)
            report['reused_checkpoints'][f'{arm}_{seed}']=error
            del m
            gc.collect();torch.cuda.empty_cache()
    write('diagnostics.json',report)
    print('CHECKS PASSED',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('prepare','train'),default='train')
    args=parser.parse_args();setup('cuda')
    records={k:read_manifest(RELEASE/'manifests'/f'{v}.json') for k,v in
             [('source','source_adapt'),('dev','development'),('refs','references')]}
    selection=json.loads((RELEASE/'manifests/selection_archived.json').read_text(encoding='utf-8'))
    lookup={str(r['id']):r for r in selection}
    selection=[lookup[str(r['id'])] for r in records['source']]
    state=torch.load(RELEASE/'weights/teacher.pt',weights_only=True,map_location='cpu')['state']
    legacy.ROOT=ROOT
    legacy.ARMS=ARMS
    legacy.construct=construct
    if args.mode=='prepare':
        build_prefix(state,records)
    caches={name:Cache(name) for name in records}
    bank=caches['refs'].get(np.arange(len(records['refs'])))
    if args.mode=='prepare':
        check(state,records,caches,bank)
        return
    if not (ROOT/'diagnostics.json').exists():
        raise RuntimeError('Run prepare before training.')
    old=np.load(PREVIOUS/'predictions.npz')
    assert old['ids'].tolist()==[str(r['id']) for r in records['dev']]
    scores=np.zeros((4,3,len(records['dev']),8),np.float32)
    bypasses=scores.copy();runs=[]
    for ai,arm in enumerate(ARMS):
        for si,seed in enumerate(SEEDS):
            if arm in ('baseline','stage'):
                oldarm='layer2' if arm=='stage' else 'baseline'
                saved=np.load(PREVIOUS/f'completed_{oldarm}_{seed}.npz')
                score,bypass=saved['score'],saved['bypass']
                receipt=json.loads((PREVIOUS/f'run_{oldarm}_{seed}.json').read_text())
                receipt.update(arm=arm,reused_from=str(PREVIOUS/f'run_{oldarm}_{seed}.json'))
            elif (ROOT/f'completed_{arm}_{seed}.npz').exists():
                saved=np.load(ROOT/f'completed_{arm}_{seed}.npz')
                score,bypass=saved['score'],saved['bypass']
                receipt=json.loads((ROOT/f'run_{arm}_{seed}.json').read_text())
            else:
                score,bypass,receipt=legacy.train_one(state,arm,seed,caches['source'],caches['dev'],records['source'],
                    records['dev'],bank,records['refs'],selection,16)
                receipt.update(model_class='Layer2BlockM2',position=ARMS[arm],new_m2_per_polarization=1,
                               adapter_channels=128,grid=[20,20])
                np.savez(ROOT/f'completed_{arm}_{seed}.npz',score=score,bypass=bypass)
                write(f'run_{arm}_{seed}.json',receipt)
            scores[ai,si]=score;bypasses[ai,si]=bypass;runs.append(receipt)
            write('runs.json',runs)
    np.savez(ROOT/'predictions.npz',ids=old['ids'],labels=old['labels'],initial=old['initial'],
             scores=scores,bypasses=bypasses,arms=np.array(list(ARMS)),seeds=np.array(SEEDS))
    write('complete.json',dict(new_fits=6,reused_fits=6,endpoint='fixed_epoch6',tuning=False,
                              default_model_changed=False))
    print('COMPLETE: six new internal-position fits and six replayed matched controls',flush=True)


if __name__=='__main__':
    main()
