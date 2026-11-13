"""Portable entry point for the current chip-to-classification Full model."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

BUNDLE = Path(__file__).resolve().parent
WORK = BUNDLE/'workspace'
PAPER = WORK/'paper_draft_20261108'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def records(dataset):
    root = BUNDLE/'data'/dataset
    rows = read(root/'manifest.json')
    for row in rows:
        for pol in ('vv','vh'):
            if row.get(pol) is not None:
                row[pol]['image'] = str((root/row[pol]['image']).resolve())
    return rows, read(root/'groups.json')


def verify(dataset):
    rows, groups = records(dataset)
    missing = []
    for row in rows:
        for pol in ('vv','vh'):
            if row.get(pol) is not None and not Path(row[pol]['image']).is_file():
                missing.append(row[pol]['image'])
    if missing:
        raise FileNotFoundError(str(missing[:10]))
    for left,right in (('small','validation'),('small','evaluation'),('validation','evaluation')):
        for field in ('id','mmsi'):
            assert not {rows[i][field] for i in groups[left]} & {rows[i][field] for i in groups[right]}, (left,right,field)
        orbit = lambda i: '_'.join(rows[i]['pass_group'].split('_')[:2])
        assert not {orbit(i) for i in groups[left]} & {orbit(i) for i in groups[right]}, (left,right,'orbit')
    assert set(groups['references']) <= set(groups['small'])
    result = dict(dataset=dataset, records=len(rows),
        counts={k:len(groups[k]) for k in ('small','validation','evaluation','references')},
        all_chip_paths_exist=True, ship_and_pass_splits_disjoint=True)
    print(json.dumps(result),flush=True)
    return result


def runtime(device):
    sys.path.insert(0,str(PAPER/'full_finetune_20261111'))
    spec = importlib.util.spec_from_file_location('portable_ports',PAPER/'ports24_sync_20261112/train.py')
    ports = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ports)
    ft = ports.ft
    for module in (ft,ft.r50,ft.cost,ft.staged.task,ports.e.task):
        module.DEVICE = device
    if device=='cuda' and not ft.torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu for verify/smoke or install a CUDA-compatible PyTorch.')
    return ports,ft


def direct_batch(ft, rows):
    return ft.cost.prepare(rows)


def smoke(args,ports,ft):
    rows, groups = records(args.dataset)
    ids = groups['validation'][:4]
    refs = [rows[i] for i in groups['references']]
    model = ports.LiveFull(args.seed) if args.dataset=='ports24' else ft.LiveFull(args.seed,'full')
    path = BUNDLE/'checkpoints'/args.dataset/f'seed{args.seed}'
    model.load_state_dict(ft.torch.load(path/'best.pt',map_location='cpu',weights_only=True),strict=True)
    model.eval()
    with ft.torch.no_grad():
        ref = model.reference(direct_batch(ft,refs))
        z,_ = model(direct_batch(ft,[rows[i] for i in ids]),[rows[i] for i in ids],ref,refs)
    archived = ft.np.load(path/'validation.npz')
    lookup = {int(i):j for j,i in enumerate(archived['ids'])}
    expected = archived['score'][[lookup[i] for i in ids]]
    actual = z.detach().cpu().numpy()
    error = float(abs(actual-expected).max())
    if error>=1e-3 or not ft.np.array_equal(actual.argmax(1),expected.argmax(1)):
        raise AssertionError(dict(max_error=error,expected=expected.tolist(),actual=actual.tolist()))
    if args.backward:
        model.train()
        ref=model.reference(direct_batch(ft,refs))
        z,aux=model(direct_batch(ft,[rows[i] for i in ids]),[rows[i] for i in ids],ref,refs)
        y=ft.torch.tensor([rows[i]['label'] for i in ids],device=ft.DEVICE)
        ft.F.cross_entropy(z,y).backward()
        prefixes=('net.m1.','net.base.internal_m2.','net.matcher.','net.base.backbone.model.conv1.')
        grads={prefix:sum(float(p.grad.norm()) for n,p in model.named_parameters()
                         if n.startswith(prefix) and p.grad is not None) for prefix in prefixes}
        assert all(v>0 and ft.np.isfinite(v) for v in grads.values()),grads
    else:
        grads=None
    result=dict(dataset=args.dataset,seed=args.seed,device=ft.DEVICE,queries=len(ids),references=len(refs),
        validation_logit_max_error=error,top1_equal=True,gradient_norms=grads,training=False,
        torch=ft.torch.__version__,python=sys.version)
    save(BUNDLE/'checks'/f'{args.dataset}_s{args.seed}_{args.device}.json',result)
    print(json.dumps(result),flush=True)


def inputs_for(args,ports,ft):
    rows,groups=records(args.dataset)
    cache=BUNDLE/'cache'/args.dataset
    cache.mkdir(parents=True, exist_ok=True)
    if args.dataset=='ports24':
        class Inputs(ports.Inputs):
            def __init__(self):
                super().__init__()
                self.rows=rows
                self.reference_rows=[rows[i] for i in self.groups['references']]
                self.cache=cache
        result=Inputs()
        result.prepare()
        return result
    native=cache/'native'
    native.mkdir(parents=True,exist_ok=True)
    if not (native/'complete.json').exists():
        arrays={k:ft.np.lib.format.open_memmap(native/(k+'.npy'),mode='w+',dtype=ft.np.uint8,shape=shape)
            for k,shape in dict(pixels=(len(rows),2,160,160),footprint=(len(rows),2,160,160),
                                available=(len(rows),2)).items()}
        for i,row in enumerate(rows):
            for a in arrays.values():a[i]=0
            for p,v in enumerate(ft.staged.task.data.prepare(row)):
                if v is not None and v['valid_target']:
                    arrays['pixels'][i,p]=v['pixels']
                    arrays['footprint'][i,p]=v['footprint']
                    arrays['available'][i,p]=1
            if (i+1)%200==0:print('NATIVE',i+1,len(rows),flush=True)
        for a in arrays.values():a.flush()
        del arrays
        save(native/'complete.json',dict(records=len(rows)))
    class Inputs(ft.Inputs):
        def __init__(self):
            self.rows=rows
            self.groups={k:ft.np.asarray(v,dtype=ft.np.int64) for k,v in groups.items()}
            self.y=ft.np.asarray([r['label'] for r in rows])
            self.weights=ports.e.weights(rows,self.groups['small'])
            self.pixels=ft.np.load(native/'pixels.npy',mmap_mode='r')
            self.footprints=ft.np.load(native/'footprint.npy',mmap_mode='r')
            self.available=ft.np.load(native/'available.npy',mmap_mode='r')
            self.loaded=set();self.resident={};self.geometry={}
            self.cache=cache/'geometry';self.cache.mkdir(parents=True,exist_ok=True)
            self.reference_rows=[rows[i] for i in self.groups['references']]
            for split in ('small','validation'):
                self.load_split(split)
                self.prepare_geometry(split)
    return Inputs()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['verify','smoke','prepare','train','evaluate'])
    parser.add_argument('--dataset',choices=['oss2','ports24'],required=True)
    parser.add_argument('--seed',type=int,choices=[17,29,43],default=17)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--backward',action='store_true')
    parser.add_argument('--no-cuda-graph',action='store_true')
    parser.add_argument('--from-checkpoints',action='store_true',help='Evaluate included current weights, without retraining.')
    args=parser.parse_args()
    verify(args.dataset)
    if args.action=='verify':return
    if args.action=='train' and args.device!='cuda':
        raise ValueError('Full training requires CUDA; CPU is supported for smoke/prepare/evaluate.')
    ports,ft=runtime(args.device)
    if args.no_cuda_graph or args.device=='cpu':
        ft.LiveFull.accelerate=lambda self:None
    if args.action=='smoke':return smoke(args,ports,ft)
    dest=BUNDLE/'runs'/args.dataset
    dest.mkdir(parents=True,exist_ok=True)
    ports.ROOT=dest
    # ft.PAPER remains the read-only source/checkpoint tree.
    ft.ROOT=dest
    inputs=inputs_for(args,ports,ft)
    if args.action=='prepare':return
    if args.action=='train':
        if args.dataset=='ports24':ports.fit(inputs,args.seed)
        else:ft.fit(inputs,'full',1e-4,args.seed,dest/f'selected_s{args.seed}')
        return
    folder=(BUNDLE/'checkpoints'/args.dataset/f'seed{args.seed}' if args.from_checkpoints
            else dest/f'selected_s{args.seed}')
    model=ports.LiveFull(args.seed) if args.dataset=='ports24' else ft.LiveFull(args.seed,'full')
    model.load_state_dict(ft.torch.load(folder/'best.pt',map_location='cpu',weights_only=True),strict=True)
    model.eval()
    if args.device=='cuda':model.accelerate()
    if args.dataset=='oss2':
        inputs.load_split('evaluation');inputs.prepare_geometry('evaluation')
    ids=inputs.groups['evaluation']
    scores=ft.predict(model,inputs,ids,'evaluation')
    target=dest/('replay' if args.from_checkpoints else 'evaluation')/f'seed{args.seed}'
    target.mkdir(parents=True,exist_ok=True)
    ft.np.savez_compressed(target/'predictions.npz',ids=ids,score=scores,label=inputs.y[ids])
    metrics=(ports.metrics if args.dataset=='ports24' else ft.metrics)(inputs.y[ids],scores)
    if args.from_checkpoints:
        saved=ft.np.load(folder/'evaluation.npz')
        assert ft.np.array_equal(saved['ids'],ids)
        error=float(abs(scores-saved['score']).max())
        metrics['saved_prediction_max_error']=error
        metrics['saved_top1_equal']=bool(ft.np.array_equal(scores.argmax(1),saved['score'].argmax(1)))
    save(target/'metrics.json',metrics)
    print(json.dumps(metrics),flush=True)


if __name__=='__main__':main()
