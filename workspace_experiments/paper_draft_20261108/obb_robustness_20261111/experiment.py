"""Query-only OBB interventions on frozen OSS2 Full checkpoints."""
from pathlib import Path
import argparse
import csv
from collections import Counter
import gc
import json
import os
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
SOURCE = ROOT.parent / 'staged_ablation_20261109'
os.environ['MPLCONFIGDIR'] = str(ROOT / 'mplconfig')
sys.path.insert(0, str(SOURCE))
from obb_geometry import conditions, pipeline_conditions, perturb as geometry_perturb
SEEDS = (17, 29, 43)


def initialize():
    global e, np, pd, tokens, build_masks, pool_multiscale, torch
    import staged as e
    import numpy as np
    import pandas as pd
    from sar_m123 import tokens
    from geometry import build_masks, pool_multiscale
    torch = e.torch


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def perturb(geometry, condition):
    return np.asarray(geometry_perturb(geometry, condition), dtype=np.float64)


def data(limit=None, geometry_only=False):
    source = WORK / 'm1_coverage_20261101'
    rows = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    groups = json.loads((source / 'groups.json').read_text(encoding='utf-8'))
    ids = np.array(groups['evaluation'], dtype=int)
    if limit is not None:ids=ids[:limit]
    rid = np.array(groups['references'], dtype=int)
    selected = np.r_[ids, rid]
    a, c = e.prev.arrays()
    a['write'] = np.load(e.PREVIOUS / 'common_init/write.npy', mmap_mode='r')
    # Only the fixed 378 queries and 32 source references are materialized.
    akeys=('valid','available','slots') if geometry_only else ('valid','available','slots','desc','mass','write')
    ckeys=('local_slots',) if geometry_only else ('early_x','early_identity','local_slots')
    a = {k: np.asarray(a[k][selected]) for k in akeys}
    c = {k: np.asarray(c[k][selected]) for k in ckeys}
    return [rows[i] for i in selected], ids, np.arange(len(ids)), np.arange(len(ids), len(selected)), a, c


def region_array(slots):
    return np.stack([slots[..., ix.tolist(), :, :].sum(-3) for ix in
                     (tokens.REGIONS[0], tokens.REGIONS[2], tokens.REGIONS[1])], axis=-3)


def preparation_plan(root=ROOT):
    counts=Counter()
    audit=root/'mask_audit.csv'
    if audit.exists():
        with audit.open(newline='',encoding='utf-8') as stream:
            counts.update(row['condition'] for row in csv.DictReader(stream))
    sources=root/'native_sources.json'
    expected=3*len(json.loads(sources.read_text(encoding='utf-8'))) if sources.exists() else None
    reuse=[cfg['name'] for cfg in conditions() if (root/'geometry'/f'{cfg["name"]}.npz').exists()
           and expected is not None and counts[cfg['name']]==expected]
    return dict(reuse=reuse,pending=[cfg for cfg in conditions() if cfg['name'] not in reuse],
                expected_audit_rows_per_condition=expected)


def merge_audit_rows(existing, added):
    replaced={row['condition'] for row in added}
    return [row for row in existing if row['condition'] not in replaced]+added


def save_condition_audit(added,root=ROOT):
    path=root/'mask_audit.csv'
    existing=[]
    if path.exists():
        with path.open(newline='',encoding='utf-8') as stream:
            existing=list(csv.DictReader(stream))
    # Preserve all existing field strings; replace only an incomplete condition.
    merged=merge_audit_rows(existing,added)
    pending=root/'mask_audit.pending.csv'
    with pending.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(merged[0]))
        writer.writeheader()
        writer.writerows(merged)
    pending.replace(path)


def prepare():
    plan=preparation_plan()
    prior=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8')) if (ROOT/'protocol.json').exists() else {}
    if not plan['pending']:
        print('PREPARATION_ALREADY_COMPLETE',len(plan['reuse']),flush=True)
        return
    e.task.DEVICE = 'cpu'
    torch.set_num_threads(2)
    rows, original_ids, qid, rid, a, c = data(geometry_only=True)
    dest = ROOT / 'geometry'
    dest.mkdir(exist_ok=True)
    views = []
    for row in rows[:len(qid)]:
        pair=e.task.data.prepare(row)
        views.append([None if v is None else dict(geometry=v['geometry'],
            footprint=v['footprint'].astype(np.uint8),valid_target=v['valid_target'],
            placement=v['placement'],raw_shape=list(v['raw'].shape)) for v in pair])
    samples = []
    for j, pair in enumerate(views):
        for p, v in enumerate(pair):
            if v is None or not a['available'][j, p]:
                continue
            assert v['valid_target'] and np.array_equal(v['footprint'], a['valid'][j, p, 0])
            samples.append(dict(index=j, original_index=int(original_ids[j]), id=rows[j]['id'],
                label=rows[j]['label'], mmsi=rows[j]['mmsi'], polarization=('VV', 'VH')[p],
                geometry=v['geometry'].tolist(), placement=v['placement'],
                original_shape=v['raw_shape'], source=rows[j][('vv', 'vh')[p]]['image']))
    if not (ROOT/'native_sources.json').exists():write('native_sources.json', samples)
    baseline = prior.get('mask_replay')
    tick = time.perf_counter()
    print('REUSE_GEOMETRY',plan['reuse'],'APPEND',[v['name'] for v in plan['pending']],flush=True)
    for condition in plan['pending']:
        audits = []
        name = condition['name']
        slots = np.zeros_like(a['slots'][:len(qid)])
        local = np.zeros_like(c['local_slots'][:len(qid)])
        for j, pair in enumerate(views):
            for p, v in enumerate(pair):
                if v is None or not a['available'][j, p]:
                    continue
                original = v['geometry']
                changed = perturb(original, condition)
                native = tokens.slots(changed, v['footprint'])
                native_t = torch.from_numpy(native).float()[None]
                slots[j, p] = e.old.grid_occupancy(native_t, (10, 10))[0].numpy()
                local[j, p] = e.old.grid_occupancy(native_t, (20, 20))[0].numpy()
                before = tokens.soft_regions(original)[[0, 2, 1]] * v['footprint']
                after = tokens.soft_regions(changed)[[0, 2, 1]] * v['footprint']
                raw_after = tokens.soft_regions(changed)[[0, 2, 1]]
                for region, title in enumerate(('body', 'side', 'end')):
                    m0, m1 = before[region], after[region]
                    intersection, union = np.minimum(m0, m1).sum(), np.maximum(m0, m1).sum()
                    old20 = region_array(c['local_slots'][j, p])[region if region == 0 else 3-region]
                    new20 = region_array(local[j, p])[region if region == 0 else 3-region]
                    old10 = region_array(a['slots'][j, p])[region if region == 0 else 3-region]
                    new10 = region_array(slots[j, p])[region if region == 0 else 3-region]
                    audits.append(dict(condition=name,index=j,id=rows[j]['id'],label=rows[j]['label'],
                        polarization=('VV', 'VH')[p],region=title,length_px=float(original[2]),width_px=float(original[3]),
                        dx_px=float(changed[0]-original[0]),dy_px=float(changed[1]-original[1]),
                        displacement_px=float(np.linalg.norm(changed[:2]-original[:2])),
                        angle_delta_deg=float(np.rad2deg(changed[4]-original[4])),
                        clean_area_px=float(m0.sum()),changed_area_px=float(m1.sum()),
                        original_target_covered=float(intersection / max(m0.sum(), 1e-12)),
                        soft_iou=float(intersection/max(union,1e-12)),
                        outside_footprint_px=float((raw_after[region]*(1-v['footprint'])).sum()),
                        native_l1_px=float(np.abs(m1-m0).sum()),
                        grid20_l1=float(np.abs(new20-old20).sum()),grid10_l1=float(np.abs(new10-old10).sum()),
                        grid20_changed_cells=int((np.abs(new20-old20)>1e-7).sum()),
                        grid10_changed_cells=int((np.abs(new10-old10)>1e-7).sum())))
        if name == 'zero':
            baseline = dict(slots_max_error=float(abs(slots-a['slots'][:len(qid)]).max()),
                            local_max_error=float(abs(local-c['local_slots'][:len(qid)]).max()))
            assert max(baseline.values()) < 1e-6, baseline
        np.savez_compressed(dest / f'{name}.npz', slots=slots, local_slots=local)
        save_condition_audit(audits)
        print('PREPARED', name, round(time.perf_counter()-tick, 1), flush=True)
    write('protocol.json', dict(seeds=SEEDS,conditions=conditions(),pipeline_conditions=pipeline_conditions(),
        original_query_indices=original_ids.tolist(),query_ids=[r['id'] for r in rows[:len(qid)]],
        reference_ids=[rows[i]['id'] for i in rid],n_queries=len(qid),n_references=len(rid),
        m2_only_changed=['query local_slots','query slots'],
        pipeline_changed=['query local_slots','query slots','query desc','query mass','query write'],
        fixed=['raw chips','native160 placement','labels','checkpoint weights','M1 inputs','availability',
               'valid footprints','query order','reference bank','reference descriptors and geometry'],
        shift_units='along fraction of original length; across fraction of original width; native pixels',
        angle_units='degrees; positive rotates clockwise in x-right/y-down image coordinates',
        policy='both originally available query polarizations receive the same signed fractional error in their own OBB frame',
        scope='fixed-crop geometry pipeline, not redetection or recropping; no training or test selection',
        mask_replay=baseline,prepare_seconds=prior.get('prepare_seconds',0)+time.perf_counter()-tick,
        last_prepare_seconds=time.perf_counter()-tick,prepared_in_this_call=[v['name'] for v in plan['pending']],
        reused_in_this_call=plan['reuse'],pipeline_this_round=False))
    status=json.loads((ROOT/'status.json').read_text()) if (ROOT/'status.json').exists() else {}
    status.update(state='cpu_prepared_waiting_for_gpu_release',complete=False,gpu_used=False,
        geometry_conditions=len(conditions()),planned_geometry_conditions=len(conditions()),
        mask_audit_rows=len(conditions())*len(samples)*3,pending_geometry_conditions=[])
    write('status.json',status)


def load_net(seed, device):
    e.task.DEVICE = device
    e.task.current.setup(device)
    net = e.Network('full', seed, None)
    net.load_state_dict(torch.load(SOURCE/f'full_s{seed}_lr1/readout/final.pt',map_location='cpu',weights_only=True), strict=True)
    e.configure(net, 3)
    return net.eval().requires_grad_(False)


def fixed_delta(net, a, rows, qid, rid, batch):
    result = np.zeros((len(rows),2,1024,10,10),np.float32)
    rd=e.task.tensor(a['desc'],rid);rm=e.task.tensor(a['mass'],rid)
    ra=e.task.tensor(a['available'],rid).bool()
    label=torch.tensor([rows[i]['label'] for i in rid],device=e.task.DEVICE)
    for first in range(0,len(qid),batch):
        ix=qid[first:first+batch]
        pair=e.task.legal(rows,ix,rid)[...,None]&e.task.tensor(a['available'],ix)[:,None].bool()&ra[None]
        result[ix]=net.matcher(e.task.tensor(a['desc'],ix),rd,e.task.tensor(a['mass'],ix),rm,
            e.task.tensor(a['write'],ix),pair,label,arm='aggregate')[0].cpu().numpy()
    net.fixed_delta=result


def predict(net,a,c,rows,qid,rid,batch):
    scores=[]
    for first in range(0,len(qid),batch):
        scores.append(net(a,c,rows,qid[first:first+batch],rid)[0].cpu().numpy())
    return np.concatenate(scores)


def changed_inputs(a,c,condition,n):
    masks=np.load(ROOT/'geometry'/f'{condition}.npz')
    aa,cc=dict(a),dict(c)
    aa['slots']=np.concatenate((masks['slots'][:n],a['slots'][n:]))
    cc['local_slots']=np.concatenate((masks['local_slots'][:n],c['local_slots'][n:]))
    return aa,cc


def pipeline_descriptors(net,rows,a,qid,batch):
    """Reuse clean native images and the original frozen pre-M1 M3 feature maps."""
    folder=ROOT/'pipeline_geometry'
    folder.mkdir(exist_ok=True)
    configs=pipeline_conditions()
    outputs={v['name']:dict(desc=np.zeros_like(a['desc'][:len(qid)]),
                           mass=np.zeros_like(a['mass'][:len(qid)]),
                           write=np.zeros_like(a['write'][:len(qid)])) for v in configs}
    public=np.load(e.task.PUBLIC/'cache/prefix.npy',mmap_mode='r')
    for first in range(0,len(qid),batch):
        ix=qid[first:first+batch]
        views=[e.task.data.prepare(rows[i]) for i in ix]
        images=np.zeros((len(ix),2,1,160,160),np.float32)
        for j,pair in enumerate(views):
            for p,v in enumerate(pair):
                if a['available'][ix[j],p]:images[j,p,0]=v['pixels']/255.
        f1=net.base.prefix(torch.tensor(images,device=e.task.DEVICE),stop=1).flatten(0,1)
        ti=[rows[i]['indices']['table_index'] for i in ix]
        f2=torch.tensor(np.array(public[ti]),device=e.task.DEVICE).flatten(0,1)
        for cfg in configs:
            gs=[]
            for j,pair in enumerate(views):
                gs.append([build_masks(perturb(v['geometry'],cfg) if a['available'][ix[j],p] else None,
                    v['footprint'] if a['available'][ix[j],p] else np.zeros((160,160)), 'coarse_fixed')
                    for p,v in enumerate(pair)])
            read=torch.tensor(np.array([[g['read'] for g in pair] for pair in gs]),device=e.task.DEVICE)
            out=outputs[cfg['name']]
            out['desc'][ix]=pool_multiscale(f1,f2,read.flatten(0,1),'coarse_fixed').reshape(len(ix),2,20,768).cpu().numpy()
            for key in ('mass','write'):out[key][ix]=np.array([[g[key] for g in pair] for pair in gs])
        print('PIPELINE_PREPARED',first+len(ix),len(qid),flush=True)
    errors={k:float(np.max(np.abs(v-a[k][:len(qid)]))) for k,v in outputs['zero'].items()}
    write('pipeline_input_replay.json',errors)
    assert max(errors.values())<1e-4,errors
    for name,values in outputs.items():np.savez_compressed(folder/f'{name}.npz',**values)


def smoke():
    rows,oids,qid,rid,a,c=data(limit=4)
    net=load_net(17,'cpu')
    ix=qid[:4]
    tick=time.perf_counter()
    fixed_delta(net,a,rows,ix,rid,4)
    baseline=predict(net,a,c,rows,ix,rid,4)
    expected=np.load(SOURCE/'full_s17_lr1/evaluation.npz')['score'][:len(ix)]
    error=float(abs(baseline-expected).max())
    write('smoke_zero_replay.json',dict(n=len(ix),score_max_error=error,
        predictions_identical=bool(np.array_equal(baseline.argmax(1),expected.argmax(1)))))
    assert error<1e-4 and np.array_equal(baseline.argmax(1),expected.argmax(1)),error
    seen={}
    def hook(name):
        def record(module,args):seen[name]=args[1].detach().cpu().numpy().copy()
        return record
    handles=[]
    for p in range(2):
        for name,mod in [('low',net.base.m2[p]),('internal',net.base.internal_m2[p]),('deep',net.base.deep_pool[p])]:
            handles.append(mod.register_forward_pre_hook(hook(f'{name}_{p}')))
    predict(net,a,c,rows,ix,rid,4)
    clean=dict(seen)
    aa,cc=changed_inputs(a,c,'across_p20',len(qid))
    changed=predict(net,aa,cc,rows,ix,rid,4)
    changes={k:float(abs(v-clean[k]).max()) for k,v in seen.items()}
    assert all(v>0 for v in changes.values()),changes
    assert all(aa[k] is a[k] for k in ('desc','mass','write','valid','available'))
    assert cc['early_x'] is c['early_x'] and cc['early_identity'] is c['early_identity']
    assert np.array_equal(aa['slots'][rid],a['slots'][rid])
    assert np.array_equal(cc['local_slots'][rid],c['local_slots'][rid])
    for handle in handles:handle.remove()
    result=dict(seed=17,n=len(ix),baseline_score_max_error=error,
        hook_mask_max_changes=changes,score_max_change=float(abs(changed-baseline).max()),
        cpu_seconds=time.perf_counter()-tick,m2_only_m3_image_reference_unchanged=True)
    write('smoke.json',result)
    print('CPU_SMOKE',json.dumps(result),flush=True)


def run(args):
    if args.device.startswith('cuda') and not args.gpu_released:
        raise SystemExit('GPU not released by main agent: use --prepare/--smoke on CPU first.')
    if args.device=='cpu':
        raise SystemExit('Full CPU run is outside current CPU preparation/small-test authorization.')
    rows,oids,qid,rid,a,c=data()
    tick=time.perf_counter()
    complete=[]
    for seed in args.seeds:
        net=load_net(seed,args.device)
        dest=ROOT/f'seed_{seed}'
        dest.mkdir(exist_ok=True)
        fixed_delta(net,a,rows,qid,rid,args.batch)
        aa,cc=changed_inputs(a,c,'zero',len(qid))
        zero=predict(net,aa,cc,rows,qid,rid,args.batch)
        saved=np.load(SOURCE/f'full_s{seed}_lr1/evaluation.npz')
        assert np.array_equal(saved['ids'],oids)
        error=float(abs(zero-saved['score']).max())
        identical=bool(np.array_equal(zero.argmax(1),saved['score'].argmax(1)))
        write(f'seed_{seed}/zero_replay.json',dict(score_max_error=error,predictions_identical=identical,n=len(qid)))
        assert error<1e-4 and identical,error
        np.savez_compressed(dest/'m2_zero.npz',ids=oids,score=zero)
        for cfg in conditions()[1:]:
            aa,cc=changed_inputs(a,c,cfg['name'],len(qid))
            out=dest/f'm2_{cfg["name"]}.npz'
            if not out.exists():
                score=predict(net,aa,cc,rows,qid,rid,args.batch)
                np.savez_compressed(out,ids=oids,score=score)
            print('DONE',seed,'m2',cfg['name'],round(time.perf_counter()-tick,1),flush=True)
            write('status.json',dict(state='running',seed=seed,mode='m2',condition=cfg['name'],complete=False))
        if args.pipeline and not (ROOT/'pipeline_geometry/angle_p20.npz').exists():
            pipeline_descriptors(net,rows,a,qid,args.batch)
        for cfg in pipeline_conditions() if args.pipeline else []:
            aa,cc=changed_inputs(a,c,cfg['name'],len(qid))
            geom=np.load(ROOT/'pipeline_geometry'/f'{cfg["name"]}.npz')
            for key in ('desc','mass','write'):aa[key]=np.concatenate((geom[key],a[key][len(qid):]))
            fixed_delta(net,aa,rows,qid,rid,args.batch)
            score=predict(net,aa,cc,rows,qid,rid,args.batch)
            np.savez_compressed(dest/f'pipeline_{cfg["name"]}.npz',ids=oids,score=score)
            if cfg['name']=='zero':
                gap=float(abs(score-zero).max())
                assert gap<1e-4 and np.array_equal(score.argmax(1),zero.argmax(1)),gap
                write(f'seed_{seed}/pipeline_zero_replay.json',dict(score_max_error=gap,predictions_identical=True))
            print('DONE',seed,'pipeline',cfg['name'],round(time.perf_counter()-tick,1),flush=True)
        complete.append(seed)
        del net
        gc.collect()
        torch.cuda.empty_cache()
    write('status.json',dict(state='inference_complete',complete=False,inference_complete=True,
        seeds=complete,seconds=time.perf_counter()-tick,gpu_used=True,analysis_pending=True,
        pipeline_completed=args.pipeline))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--pipeline',action='store_true',help='Optional additional fixed-crop M2+M3 geometry sensitivity.')
    parser.add_argument('--resources-released',action='store_true',help='Main agent explicitly released CPU RAM for preparation/smoke.')
    parser.add_argument('--gpu-released',action='store_true')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch',type=int,default=12)
    parser.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS))
    args=parser.parse_args()
    if (args.prepare or args.smoke) and not args.resources_released:
        parser.error('RAM is reserved for the other job. Wait for explicit release before preparation/smoke.')
    if args.run and not args.gpu_released:
        parser.error('Wait for main agent GPU release; no automatic GPU execution.')
    if args.prepare or args.smoke or args.run:
        try:
            initialize()
            with torch.inference_mode():
                if args.prepare:prepare()
                if args.smoke:smoke()
                if args.run:run(args)
        except Exception:
            import traceback
            write('last_failure.json',dict(traceback=traceback.format_exc(),arguments=vars(args)))
            raise
