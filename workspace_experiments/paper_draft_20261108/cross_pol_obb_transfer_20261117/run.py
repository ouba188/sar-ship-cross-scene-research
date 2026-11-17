"""Fixed-weight query-side cross-polar OBB transfer, with matched global fallback."""
from pathlib import Path
import argparse
import copy
import csv
import gc
import importlib.util
import json
import os
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
PREV = ROOT.parent/'pcea_conditional_modulation_20261117'
CACHE = ROOT.parent/'recent_baselines_20261111/native_cache'
POLS = ('vv', 'vh')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare():
    native = load('transfer_native_input', WORK/'sar_m123_standalone/sar_m123/input.py')
    rows = read(WORK/'m1_coverage_20261101/manifest.json')
    groups = read(WORK/'m1_coverage_20261101/groups.json')
    available = np.load(CACHE/'available.npy')
    pixels = np.load(CACHE/'pixels.npy', mmap_mode='r')
    footprints = np.load(CACHE/'footprint.npy', mmap_mode='r')
    records, skipped, arrays = [], [], dict(ids=[], pol=[], pixels=[], footprint=[], geometry=[])
    for split in ('validation', 'evaluation'):
        for idx in groups[split]:
            if available[idx].sum() != 1:
                continue
            row = rows[idx]
            target = int(np.flatnonzero(available[idx] == 0)[0]); donor = 1-target
            dp, tp = POLS[donor], POLS[target]
            spec = row[dp]
            source = row.get('source_paths', {})
            reason = None
            if not source.get(dp) or not source.get(tp):
                reason = 'no paired original source paths'
            elif Path(source[dp]).parent != Path(source[tp]).parent or Path(source[dp]).stem[:-3] != Path(source[tp]).stem[:-3]:
                reason = 'different source window'
            donor_path = Path(spec['image'])
            target_path = donor_path.parent.parent/tp/donor_path.name
            if not target_path.exists():
                reason = 'missing paired PNG'
            if reason:
                skipped.append(dict(index=idx,id=row['id'],split=split,reason=reason)); continue
            raw_d = np.array(Image.open(donor_path).convert('L'))
            raw_t = np.array(Image.open(target_path).convert('L'))
            src_d = np.array(Image.open(source[dp]).convert('L'))
            src_t = np.array(Image.open(source[tp]).convert('L'))
            if raw_t.shape != raw_d.shape or not np.array_equal(raw_t, src_t) or not np.array_equal(raw_d, src_d):
                skipped.append(dict(index=idx,id=row['id'],split=split,reason='PNG/source grid or pixel mismatch')); continue
            donor_view = native.prepare_view(spec)
            assert donor_view['valid_target']
            assert np.array_equal(donor_view['pixels'], pixels[idx,donor])
            assert np.array_equal(donor_view['footprint'], footprints[idx,donor])
            det = copy.deepcopy(spec['detections'][spec['target_index']])
            # Confidence is explicitly the donor's localization score, never target detection confidence.
            transferred = dict(image=str(target_path), detections=[det], target_index=0, alignment_valid=True)
            view = native.prepare_view(transferred)
            assert view['valid_target']
            assert view['placement']['crop_x'] == donor_view['placement']['crop_x']
            assert view['placement']['crop_y'] == donor_view['placement']['crop_y']
            assert np.allclose(view['geometry'], donor_view['geometry'])
            record = dict(index=idx, id=row['id'], split=split, label=row['label'],
                pass_group=row['pass_group'], donor_pol=dp, target_pol=tp,
                donor_confidence=det['confidence'], target_detection_confidence=None,
                native_shape=list(raw_t.shape), corners=det['corners'], placement=view['placement'],
                geometry=view['geometry'].tolist(), source_paths=source,
                donor_png=str(donor_path), target_png=str(target_path),
                png_equals_source=True, geometry_provenance='paired-polarization OBB, not target detection')
            records.append(record)
            for key, value in dict(ids=idx,pol=target,pixels=view['pixels'],footprint=view['footprint'],geometry=view['geometry']).items():
                arrays[key].append(value)
    np.savez_compressed(ROOT/'recovered_inputs.npz', **{k:np.asarray(v) for k,v in arrays.items()})
    write(ROOT/'records_local.json',records)
    counts = {s:dict(total=len(groups[s]),single=int(sum(available[i].sum()==1 for i in groups[s])),
                    recovered=sum(r['split']==s for r in records),
                    missing_vv=sum(r['split']==s and r['target_pol']=='vv' for r in records),
                    missing_vh=sum(r['split']==s and r['target_pol']=='vh' for r in records))
              for s in ('validation','evaluation')}
    write(ROOT/'input_audit.json',dict(counts=counts,skipped=skipped,policy='paired source window and exact PNG-source pixel equality'))
    print('PREPARED',counts,'skipped',len(skipped),flush=True)


def restore_score(path, ids):
    with np.load(path) as a:
        z = a['score'] if 'score' in a else a['scores']
        lookup = {int(i):j for j,i in enumerate(a['ids'])}
        return z[[lookup[int(i)] for i in ids]].copy()


def paired(y, z, original):
    b, a = original.argmax(1)==y,z.argmax(1)==y
    return dict(rescued=int((a&~b).sum()),harmed=int((~a&b).sum()),changed=int((z.argmax(1)!=original.argmax(1)).sum()))


def evaluate(seeds):
    model = load('transfer_conditional_model',PREV/'model.py')
    full, torch = model.full, model.torch
    inputs = full.Inputs()
    inputs.prepare_geometry('small')
    for split in ('validation','evaluation'):
        inputs.load_split(split); inputs.prepare_geometry(split)
    recovered = dict(np.load(ROOT/'recovered_inputs.npz'))
    rlookup = {int(i):j for j,i in enumerate(recovered['ids'])}
    selected = read(PREV/'selected_checkpoints.json')
    overrides = {}
    for idx,j in rlookup.items():
        native = full.cost.raw.tokens.slots(recovered['geometry'][j],recovered['footprint'][j])
        rd = full.cost.raw.tokens.cell_mass(native,8)/full._cell_area(20)
        wr = full.cost.raw.tokens.cell_mass(native,16)/full._cell_area(10)
        wr /= np.maximum(wr.sum(0,keepdims=True),1)
        values = dict(images=recovered['pixels'][j][None].astype(np.float32)/255,
                      valid=recovered['footprint'][j][None],read=rd,write=wr,mass=rd.sum((-2,-1)))
        for key,size in (('slots',10),('local_slots',20)):
            values[key] = full.cost.raw.grid_occupancy(torch.from_numpy(native).float()[None],(size,size))[0].numpy()
        overrides[idx]={k:torch.as_tensor(v,device='cuda') for k,v in values.items()}

    def data(ids,split,policy):
        original = inputs.data(ids,split)
        if policy=='closed': return original
        out={k:v.clone() for k,v in original.items()}
        for n,idx in enumerate(ids):
            if int(idx) not in rlookup: continue
            p=int(recovered['pol'][rlookup[int(idx)]])
            for key,value in overrides[int(idx)].items():
                out[key][n,p]=value if policy=='transfer' or key in ('images','valid') else 0
            out['available'][n,p]=1
            donor=1-p
            assert all(torch.equal(out[k][n,donor],original[k][n,donor]) for k in out)
        return out

    with torch.no_grad():
        for seed in seeds:
            net=model.Model(seed,'affine_frozen')
            net.load_state_dict(torch.load(selected[str(seed)]['affine_frozen']['checkpoint'],map_location='cpu',weights_only=True))
            net.eval().requires_grad_(False);net.accelerate()
            reference=net.reference(inputs.data(inputs.groups['references'],'small'))
            for split in ('validation','evaluation'):
                ids=inputs.groups[split];y=inputs.y[ids]
                affected=np.array([i for i in ids if int(i) in rlookup],dtype=int)
                controls=np.array([i for i in ids if int(i) not in rlookup and inputs.available[i].sum()==2][:3],dtype=int)
                check=np.concatenate((affected,controls))
                score={};replay={};activation={};images_equal=True
                for policy in ('closed','global','transfer'):
                    for active in (False,True):
                        name=policy+('_pcea' if active else '_full')
                        net.exchange.disabled=not active
                        chunks=[];traces=[]
                        for first in range(0,len(check),6):
                            part=check[first:first+6]
                            batch=data(part,split,policy)
                            if policy=='transfer':
                                g=data(part,split,'global')
                                assert all(torch.equal(batch[k],g[k]) for k in ('images','valid','available'))
                            z,_=net(batch,[inputs.rows[i] for i in part],reference,inputs.reference_rows)
                            chunks.append(z.cpu().numpy())
                            traces.extend(net.exchange.trace['active'].any(1).cpu().tolist())
                        measured=np.concatenate(chunks)
                        path=(Path(selected[str(seed)]['affine_frozen']['checkpoint']).parent/'validation.npz' if split=='validation' else
                              PREV/'evaluation'/f'affine_frozen_s{seed}'/'predictions.npz') if active else full.ROOT/f'selected_s{seed}'/f'{split}.npz'
                        base=restore_score(path,ids)
                        lookup={int(i):j for j,i in enumerate(ids)}
                        order=[lookup[int(i)] for i in check]
                        if policy=='closed':
                            error=float(np.abs(measured-base[order]).max())
                            assert error<1e-3 and np.array_equal(measured.argmax(1),base[order].argmax(1)),(seed,split,error)
                            replay[name]=error
                        else:
                            oldscore=score['closed'+('_pcea' if active else '_full')]
                            assert np.allclose(measured[-len(controls):],oldscore[[lookup[int(i)] for i in controls]],atol=1e-4)
                        base[order]=measured
                        score[name]=base
                        activation[name]=int(sum(traces[:len(affected)]))
                        print(seed,split,name,full.metrics(y,base)['oa'],'active',activation[name],flush=True)
                mask=np.isin(ids,affected)
                np.savez_compressed(ROOT/f'{split}_s{seed}.npz',ids=ids,label=y,affected=mask,**score)
                contrasts={}
                for suffix in ('full','pcea'):
                    for a,b in (('global','closed'),('transfer','closed'),('transfer','global')):
                        contrasts[a+'-'+b+'_'+suffix]=paired(y,score[a+'_'+suffix],score[b+'_'+suffix])
                for policy in ('closed','global','transfer'):
                    contrasts[policy+'_pcea-full']=paired(y,score[policy+'_pcea'],score[policy+'_full'])
                write(ROOT/f'{split}_s{seed}.json',dict(seed=seed,split=split,n=len(ids),affected=int(mask.sum()),
                    metrics={k:full.metrics(y,z) for k,z in score.items()},
                    affected_metrics={k:full.metrics(y[mask],z[mask]) for k,z in score.items()},
                    contrasts=contrasts,replay_error=replay,eligible_among_affected=activation,
                    pcea_enabled={k:k.endswith('_pcea') for k in score},
                    checkpoint=selected[str(seed)]['affine_frozen']['checkpoint'],
                    note='Fixed original source references and statistics; query-only repair. No training.'))
            del reference,net;gc.collect();torch.cuda.empty_cache()


def summarize():
    records=[]
    for path in sorted(ROOT.glob('*_s*.json')):
        if not path.stem.startswith(('validation_','evaluation_')): continue
        result=read(path)
        for arm,m in result['metrics'].items():
            records.append(dict(split=result['split'],seed=result['seed'],arm=arm,
                                oa=m['oa'],ba=m['ba'],nll=m['nll'],affected=result['affected']))
    with (ROOT/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    means=[]
    for split in ('validation','evaluation'):
        for arm in sorted(set(r['arm'] for r in records)):
            part=[r for r in records if r['split']==split and r['arm']==arm]
            means.append(dict(split=split,arm=arm,seeds=[r['seed'] for r in part],
                              **{k:float(np.mean([r[k] for r in part])) for k in ('oa','ba','nll')}))
    write(ROOT/'summary.json',means)
    print(json.dumps(means,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','evaluate','summarize'),required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[17,29,43])
    args=parser.parse_args()
    if args.phase=='prepare': prepare()
    elif args.phase=='evaluate': evaluate(args.seeds)
    else: summarize()
