"""Missing dataset replications only, reusing splits and the frozen staged recipe."""
from pathlib import Path
import argparse
import gc
import importlib.util
import json
import sys
import time
import numpy as np
import pandas as pd
import engine as e
import intervention_inputs as interventions

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[1]
CLOSE=WORK/'scene_aware_classification_20261018/model_closeout_20261023'
STORE=Path('C:/Users/25791/AppData/Local/sar_paper_experiments_20261110')
ROLES=dict(fit_train='source_fit',utility_train='source_adapt',fit_validation='validation',
           selection='selection',evaluation='evaluation')
NAMES=['bulk_carrier','general_cargo','product_chemical_tanker','container_ship']


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):e.task.write(path,value)


def setup(name):
    dest=STORE/name;dest.mkdir(parents=True,exist_ok=True)
    report=ROOT/name;report.mkdir(parents=True,exist_ok=True)
    e.ROOT=dest;interventions.ROOT=dest/'intervention_inputs'
    e.prev.records=lambda: records(name)
    e.prev.arrays=lambda: arrays(name)
    return dest,report


def records(name):
    dest=STORE/name
    rows=read(dest/'manifest.json')
    groups={k:np.array(v,dtype=np.int64) for k,v in read(dest/'groups.json').items()}
    return rows,groups


def arrays(name):
    return tuple({p.stem:np.load(p,mmap_mode='r') for p in (STORE/name/'cache'/k).glob('*.npy')} for k in ('a','c'))


def manifest_oss1():
    dest,report=setup('oss1')
    if (dest/'manifest.json').exists():return
    old=CLOSE/'oss1_m123_replication_20261026'
    helper=ROOT.parent/'experiment_completion_20261109/oss1_transfer/prepare_transfer.py'
    spec=importlib.util.spec_from_file_location('oss1_manifest_helper',helper)
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    table=h.csv_read(old/'stack_table.csv')
    views={(int(v['obs_id']),v['pol']):v for v in h.csv_read(old/'four_class_views.csv')}
    geom={v['sample_id']:v for v in h.csv_read(WORK/'oss1_joint_geometry_20261017/geometry.csv')}
    rows=[]
    for row in table:
        i=int(row['obs_id']);label=int(row['class_id']);g=geom[row['sample_id']]
        assert int(g['class_id'])==(0,2,3,4)[label]
        r=dict(id=row['sample_id'],mmsi=row['mmsi'],pass_group=row['pass_group'],scene_id=row['scene_id'],
               label=label,original_class_id=(0,2,3,4)[label],class_name=NAMES[label],
               split=ROLES[row['role']],dataset='OpenSARShip1',input_scale=1.,canvas_size=160,
               indices=dict(obs_id=i,table_index=int(row['table_index'])))
        for pol in ('vv','vh'):
            v=views[i,pol];assert v['sample_id']==r['id']
            r[pol]=h.make_view(v,g)
            if r[pol]:assert Path(r[pol]['image']).is_file(),r[pol]['image']
        rows.append(r)
    groups={name:[i for i,r in enumerate(rows) if r['split']==name] for name in ROLES.values()}
    byobs={r['indices']['obs_id']:i for i,r in enumerate(rows)}
    groups['references']=[byobs[int(r['obs_id'])] for r in h.csv_read(old/'m3_results_grid01/reference_bank.csv')]
    groups['small']=groups['source_fit']+groups['source_adapt']
    assert len(rows)==866 and [len(groups[k]) for k in ('source_fit','source_adapt','validation','evaluation')]==[405,122,89,63]
    save(dest/'manifest.json',rows);save(dest/'groups.json',groups)
    initialize_protocol('oss1',4,str(old/'STACK_SPLIT.json'))


def manifest_ports24():
    dest,report=setup('ports24')
    if (dest/'protocol.json').exists():return
    path=CLOSE/'spatial_m12_full_20261029/cohorts.json'
    parts=read(path);roles={'fit':'source_fit','adapt':'source_adapt','development':'validation','expanded':'evaluation'}
    rows=[];groups={}
    for key,role in roles.items():
        groups[role]=list(range(len(rows),len(rows)+len(parts[key])))
        for row in parts[key]:
            r=dict(row);r['split']=role;r['dataset']='ports24';rows.append(r)
    lookup={r['id']:i for i,r in enumerate(rows)};assert len(lookup)==len(rows)
    groups['references']=[lookup[r['id']] for r in parts['references']]
    groups['small']=groups['source_fit']+groups['source_adapt'];groups['selection']=[]
    assert [len(groups[k]) for k in ('source_fit','source_adapt','validation','evaluation','references')]==[1205,136,160,3075,16]
    save(dest/'manifest.json',rows);save(dest/'groups.json',groups)
    initialize_protocol('ports24',8,str(path))


def initialize_protocol(name,classes,split_source):
    dest,report=setup(name);rows,groups=records(name)
    for left,right in [('small','validation'),('small','evaluation'),('validation','evaluation')]:
        for field in ('id','mmsi','orbit'):
            vals=lambda ids:{e.task.orbit(rows[i]) if field=='orbit' else rows[i][field] for i in ids}
            overlap=vals(groups[left])&vals(groups[right])
            if name=='ports24' and (left,right,field)==('small','validation','orbit'):
                assert not {rows[i]['pass_group'] for i in groups[left]}&{rows[i]['pass_group'] for i in groups[right]}
            else:assert not overlap,(left,right,field)
    assert set(groups['references'])<=set(groups['source_fit'])
    refcounts=np.bincount([rows[i]['label'] for i in groups['references']],minlength=classes)
    assert np.all(refcounts>0),refcounts
    protocol=read(ROOT.parent/'staged_ablation_20261109/protocol.json')
    protocol.update(dataset=name,classes=classes,arms=['full','plain','se'],seeds=[17,29,43],
        scales={'full':1.,'plain':1.,'se':.3},selection='Frozen OSS2 development-selected learning-rate scales; no new target tuning',
        split_source=split_source,completed_runs_reused=True,
        purpose='New dataset-source replication of surviving modules, not repetition of old versions')
    save(dest/'protocol.json',protocol);save(report/'protocol.json',protocol)
    source_orbits={e.task.orbit(rows[i]) for i in groups['small']}
    source_ships={rows[i]['mmsi'] for i in groups['small']}
    audit={role:dict(n=len(groups[role]),source_orbit_overlap=sum(e.task.orbit(rows[i]) in source_orbits for i in groups[role]),
                    source_ship_overlap=sum(rows[i]['mmsi'] in source_ships for i in groups[role])) for role in ('validation','evaluation')}
    save(report/'overlap_summary.json',audit)
    coverage=[]
    for role,ids in groups.items():
        coverage.append(dict(role=role,n=len(ids),ships=len({rows[i]['mmsi'] for i in ids}),
            orbits=len({e.task.orbit(rows[i]) for i in ids}),**{f'class{c}':sum(rows[i]['label']==c for i in ids) for c in range(classes)}))
    pd.DataFrame(coverage).to_csv(report/'coverage.csv',index=False)
    save(report/'storage.json',dict(store=str(dest),manifest=str(dest/'manifest.json'),checkpoints='model-specific stages; resume completed receipts'))


@e.torch.no_grad()
def extract_batch(net,rows):
    raw=e.old.raw;torch=e.torch
    n=len(rows);images=np.zeros((n,2,1,160,160),np.float32);valid=np.zeros_like(images)
    av=np.zeros((n,2),np.float32);slots=np.zeros((n,2,20,10,10),np.float32)
    local=np.zeros((n,2,20,20,20),np.float32);geom=[]
    for j,row in enumerate(rows):
        gs=[]
        for p,view in enumerate(e.task.data.prepare(row)):
            good=view is not None and view['valid_target']
            if good:
                images[j,p,0]=view['pixels'].astype(np.float32)/255
                valid[j,p,0]=view['footprint'];av[j,p]=1
                native=raw.tokens.slots(view['geometry'],view['footprint'])
                for target,size in ((slots,10),(local,20)):
                    target[j,p]=raw.grid_occupancy(torch.from_numpy(native).float()[None],(size,size))[0].numpy()
            gs.append(raw.build_masks(view['geometry'] if good else None,view['footprint'] if good else np.zeros((160,160)),'coarse_fixed'))
        geom.append(gs)
    bb=net.base.backbone.model;block=bb.layer2[3]
    f1=net.base.prefix(torch.tensor(images,device=e.task.DEVICE),stop=1)
    z=f1.flatten(0,1)
    for layer in list(bb.layer2.children())[:3]:z=layer(z)
    identity=z;x=block.relu(block.bn1(block.conv1(z)));x=block.relu(block.bn2(block.conv2(x)))
    f2=block.relu(block.bn3(block.conv3(x))+identity).reshape(n,2,512,20,20)
    read_masks=torch.tensor(np.stack([[g['read'] for g in gs] for gs in geom]),device=e.task.DEVICE)
    desc=raw.pool_multiscale(f1.flatten(0,1),f2.flatten(0,1),read_masks.flatten(0,1),'coarse_fixed').reshape(n,2,20,768)
    a=dict(valid=valid,available=av,slots=slots,desc=desc.cpu().numpy(),
           mass=np.stack([[g['mass'] for g in gs] for gs in geom]),write=np.stack([[g['write'] for g in gs] for gs in geom]))
    c=dict(early_x=x.reshape(n,2,128,20,20).cpu().numpy(),early_identity=identity.reshape(n,2,512,20,20).cpu().numpy(),local_slots=local)
    return a,c


@e.torch.no_grad()
def prepare(name):
    dest,report=setup(name)
    if name=='oss1':manifest_oss1()
    elif name=='ports24':manifest_ports24()
    rows,groups=records(name);protocol=read(dest/'protocol.json')
    e.task.current.setup(e.task.DEVICE)
    if not (dest/'cache/complete.json').exists():
        net=e.Network('plain',17,protocol['pretrain'],protocol['classes']).eval()
        stores={};tick=time.perf_counter()
        for first in range(0,len(rows),8):
            values=extract_batch(net,rows[first:first+8])
            for folder,table in zip(('a','c'),values):
                path=dest/'cache'/folder;path.mkdir(parents=True,exist_ok=True)
                for key,value in table.items():
                    if (folder,key) not in stores:
                        stores[folder,key]=np.lib.format.open_memmap(path/(key+'.npy'),mode='w+',dtype=np.float32,shape=(len(rows),)+value.shape[1:])
                    stores[folder,key][first:first+len(value)]=value
            if first%160==0:print('CACHE',name,first,len(rows),flush=True)
        for value in stores.values():value.flush()
        save(dest/'cache/complete.json',dict(n=len(rows),seconds=time.perf_counter()-tick,source='common ImageNet frozen prefix; geometry only; no old learned delta'))
        save(report/'input_availability.json',dict(both=int((stores['a','available'].sum(1)==2).sum()),
            vv_only=int(((stores['a','available'][:,0]==1)&(stores['a','available'][:,1]==0)).sum()),
            vh_only=int(((stores['a','available'][:,0]==0)&(stores['a','available'][:,1]==1)).sum())))
        del net,stores;gc.collect();e.torch.cuda.empty_cache()
    interventions.prepare()
    print('PREPARE_DONE',name,flush=True)


def smoke(name):
    dest,report=setup(name);rows,groups=records(name);a,c=arrays(name)
    protocol=read(dest/'protocol.json');e.task.current.setup(e.task.DEVICE)
    ids=groups['source_fit'][:3];result={}
    for arm in ('full','plain','se'):
        net=e.Network(arm,17,protocol['pretrain'],protocol['classes'])
        for phase in range(4):
            e.configure(net,phase);z,aux=net(a,c,rows,ids,groups['references'])
            loss=e.F.cross_entropy(z,e.torch.tensor([rows[i]['label'] for i in ids],device=e.task.DEVICE))
            if aux is not None:loss=loss+.02*aux.square().mean()
            loss.backward()
            grads={n:float(p.grad.norm()) for n,p in net.named_parameters() if p.requires_grad and p.grad is not None}
            assert z.shape==(len(ids),protocol['classes']) and e.torch.isfinite(z).all()
            assert all(np.isfinite(list(grads.values()))) and max(grads.values())>0
            result[f'{arm}_{phase}']=dict(shape=list(z.shape),gradients=grads)
            net.zero_grad(set_to_none=True)
        del net;gc.collect();e.torch.cuda.empty_cache()
    if protocol['classes']==4:
        sys.path.insert(0,str(ROOT.parent/'staged_ablation_20261109'))
        import staged as original
        for arm in ('full','plain','se'):
            new=e.Network(arm,17,protocol['pretrain']);old=original.Network(arm,17,protocol['pretrain'])
            for key,val in new.state_dict().items():assert e.torch.equal(val,old.state_dict()[key]),key
            for phase in range(4):
                e.configure(new,phase);original.configure(old,phase)
                with e.torch.no_grad():error=float((new(a,c,rows,ids,groups['references'])[0]-old(a,c,rows,ids,groups['references'])[0]).abs().max())
                assert error<1e-5,(arm,phase,error)
                result[f'parity_{arm}_{phase}']=error
            del new,old;gc.collect();e.torch.cuda.empty_cache()
    save(report/'smoke.json',result);print('SMOKE_OK',name,flush=True)


def train(name):
    dest,report=setup(name);rows,groups=records(name);a,c=arrays(name)
    protocol=read(dest/'protocol.json');e.task.current.setup(e.task.DEVICE)
    # Evaluation rows never need to occupy accelerator memory during fitting.
    last=int(max(np.r_[groups['small'],groups['validation']]))+1
    assert not np.any(groups['evaluation']<last), 'Manifest must place source/dev before evaluation'
    rows=rows[:last]
    a={k:v[:last] for k,v in a.items()};c={k:v[:last] for k,v in c.items()}
    e.resident(a,c)
    for seed in protocol['seeds']:
        for arm in protocol['arms']:
            e.fit(arm,seed,protocol['scales'][arm],protocol,rows,groups,a,c)
    print('TRAIN_DONE',name,flush=True)


@e.torch.no_grad()
def evaluate(name):
    dest,report=setup(name);rows,groups=records(name);a,c=arrays(name)
    protocol=read(dest/'protocol.json');e.task.current.setup(e.task.DEVICE)
    if len(rows)<1800:e.resident(a,c)
    metrics=[];y=np.array([r['label'] for r in rows])
    for seed in protocol['seeds']:
        for arm in protocol['arms']:
            scale=protocol['scales'][arm];folder=dest/f'{arm}_s{seed}_lr{scale:g}'
            receipt=read(folder/'receipt.json');out=folder/'evaluation.npz'
            net=None
            if not out.exists():
                net=e.Network(arm,seed,protocol['pretrain'],protocol['classes'])
                net.load_state_dict(e.torch.load(receipt['final'],map_location='cpu',weights_only=True));e.configure(net,3)
                if len(rows)<1800:e.freeze_delta(net,a,rows,groups['references'])
                val=e.predict(net,a,c,rows,groups['validation'],groups['references'])
                error=float(abs(val-np.load(folder/'readout/validation.npz')['score']).max());assert error<1e-4,error
                ids=groups['evaluation'];score=e.predict(net,a,c,rows,ids,groups['references'])
                off=score.copy() if not net.use3 else np.concatenate([
                    net(a,c,rows,ids[i:i+12],groups['references'],m3_on=False)[0].cpu().numpy() for i in range(0,len(ids),12)])
                np.savez_compressed(out,ids=ids,score=score,m3_off=off)
                save(folder/'evaluation_receipt.json',dict(n=len(ids),validation_replay_error=error,selection='none; all fixed-seed results retained'))
            d=np.load(out);ids=d['ids']
            metrics.append(dict(arm=arm,seed=seed,n=len(ids),**e.task.metrics(y[ids],d['score'])))
            del net;gc.collect();e.torch.cuda.empty_cache()
    pd.DataFrame(metrics).to_csv(report/'metrics.csv',index=False)
    pd.DataFrame(metrics).groupby('arm')[['oa','ba','nll','balanced_nll']].agg(['mean','std']).to_csv(report/'summary.csv')
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','train','evaluate']);p.add_argument('dataset',choices=['oss1','ports24','f0921']);args=p.parse_args()
    globals()[args.action](args.dataset)
