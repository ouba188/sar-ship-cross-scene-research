"""Bounded sequential GPU passes: GAP, source scale, then regional summaries."""
from common import *
import argparse
import time
from torchvision.models import resnet50


def support_info(g,fp,template):
    dyn=soft_regions(g); fixed=soft_regions(g,template)
    q=region_quality(dyn,fp,g)
    fq=region_quality(fixed,fp,g,template)
    dv=(q['min_neff']>=4)&(q['area']>0)
    fv=(fq['min_neff']>=4)&(fq['area']>0)
    return np.concatenate((fp[None],dyn*fp,fixed*fp)),q,dv,fv


def main(stage,mode):
    tick=time.perf_counter()
    assert (HERE/'frozen_config.json').exists()
    if stage=='test': assert (HERE/'source_selection_frozen.json').exists()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    table,valid,snapshots=load_data(); n=len(table)
    roles,required_mask=required(stage,n)
    ids=np.flatnonzero(required_mask.any(0))
    model=resnet50(weights=None)
    model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True))
    model=model.cuda().float().eval().requires_grad_(False)
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    template=pd.read_csv(HERE/'source_templates.csv').set_index(['reader','pol'])
    if mode!='gap':
        W=[]
        for r in roles:
            with np.load(HERE/'readers'/f"{r['name']}.npz") as z: W.append(z['W'].reshape(8,2,256).transpose(1,0,2))
        W=torch.tensor(np.stack(W),device='cuda',dtype=torch.float64)
    if mode=='regions':
        tau=np.array(json.loads((HERE/'scales.json').read_text())['tau'])
    feature=np.zeros((n,2,256),np.float32)
    seen=np.zeros((n,2),int)
    reader_ids=[np.flatnonzero(rr) for rr in required_mask]
    positions=[{int(i):j for j,i in enumerate(ii)} for ii in reader_ids]
    outputs=[]
    for ii in reader_ids:
        outputs.append(dict(indices=ii,summary=np.zeros((len(ii),2,7,8,3)),
            sigma=np.zeros((len(ii),2,8)),region_valid=np.zeros((len(ii),2,7),bool),
            quality=np.zeros((len(ii),2,3,4)),geometry=np.zeros((len(ii),2,4))))
    checks=[]; last=tick
    timing=dict(native_read_and_crop_cpu_io=0.,region_mask_cpu=0.,backbone_gpu_wall=0.,response_gpu_and_transfer_wall=0.)
    with torch.inference_mode():
        for first in range(0,len(ids),12):
            io_tick=time.perf_counter()
            addresses=[]; images=[]; fps=[]; geoms=[]
            for i in ids[first:first+12]:
                for p in range(2):
                    if not valid[i,p]: continue
                    pix,fp,g,place=native(table,snapshots,int(i),p)
                    images.append(pix[None].astype(np.float32)/255); fps.append(fp); geoms.append(g); addresses.append((int(i),p))
            timing['native_read_and_crop_cpu_io']+=time.perf_counter()-io_tick
            mask_tick=time.perf_counter()
            x=torch.tensor(np.stack(images),device='cuda')
            fpweights=np.stack([np.stack([cell_mass(np.rot90(fp,k)) for fp in fps]) for k in range(4)])
            gaps=[]
            masks={}; locations={}
            if mode=='regions':
                for ri,r in enumerate(roles):
                    take=[b for b,(i,p) in enumerate(addresses) if required_mask[ri,i]]
                    if not take: continue
                    ww=[]
                    for b in take:
                        i,p=addresses[b]; loc=positions[ri][i]; g=geoms[b]
                        lw=template.loc[(r['name'],POL[p]),['length','width']].to_numpy(float)
                        m,q,dv,fv=support_info(g,fps[b],lw)
                        ww.append(np.stack([cell_mass(np.rot90(m,k,(-2,-1))) for k in range(4)]))
                        outputs[ri]['region_valid'][loc,p]=np.r_[True,dv,fv]
                        outputs[ri]['quality'][loc,p]=np.stack([q['area'],q['coverage'],q['min_neff'],dv],-1)
                        outputs[ri]['geometry'][loc,p]=np.r_[g[2:4],g[5],1.] if g is not None else 0
                    masks[ri]=np.stack(ww); locations[ri]=take
            timing['region_mask_cpu']+=time.perf_counter()-mask_tick
            for rotation in range(4):
                torch.cuda.synchronize(); forward_tick=time.perf_counter()
                xx=x.rot90(rotation,(-2,-1)).expand(-1,3,-1,-1)
                z=model.layer1(model.maxpool(model.relu(model.bn1(model.conv1((xx-mean)/std)))))
                assert z.shape[1:]==(256,40,40)
                torch.cuda.synchronize(); timing['backbone_gpu_wall']+=time.perf_counter()-forward_tick
                response_tick=time.perf_counter()
                zz=z.flatten(2).double()
                fw=torch.tensor(fpweights[rotation].reshape(len(addresses),1,-1),device='cuda')
                gap=(zz*fw).sum(-1)/fw.sum(-1)
                gaps.append(gap.cpu().numpy())
                if mode=='gap':
                    timing['response_gpu_and_transfer_wall']+=time.perf_counter()-response_tick
                    continue
                for ri,r in enumerate(roles):
                    take=locations[ri] if ri in locations else [b for b,(i,p) in enumerate(addresses) if required_mask[ri,i]]
                    if not take: continue
                    pols=[addresses[b][1] for b in take]
                    response=torch.einsum('bck,bku->bcu',W[ri,pols],zz[take])
                    if mode=='scale':
                        norm=fw[take]/fw[take].sum(-1,keepdim=True)
                        mu=(response*norm).sum(-1)
                        rr=response-mu[...,None]; rr=rr-rr.mean(1,keepdim=True)
                        sigma=(rr.square()*norm).sum(-1).clamp_min(0).sqrt().cpu().numpy()
                        for b,batchpos in enumerate(take):
                            i,p=addresses[batchpos]; outputs[ri]['sigma'][positions[ri][i],p]+=sigma[b]/4
                    else:
                        weight=torch.tensor(masks[ri][:,rotation].reshape(len(take),7,-1),device='cuda')
                        norm=weight/weight.sum(-1,keepdim=True).clamp_min(1e-300)
                        mu=torch.einsum('bru,bcu->brc',norm,response)
                        rr=response[:,None]-mu[...,None]; rr-=rr.mean(2,keepdim=True)
                        logw=torch.where(norm>0,norm.log(),-torch.inf)[:,:,None]
                        # Missing geometric regions use a harmless temporary distribution and are zeroed below.
                        empty=weight.sum(-1)==0
                        logw=torch.where(empty[:,:,None,None],torch.zeros_like(logw),logw)
                        t=torch.tensor(tau[r['fold'],pols],device='cuda')[:,None,None,None]
                        qp=((logw+rr/t).softmax(-1)*rr).sum(-1)
                        qm=((logw-rr/t).softmax(-1)*rr).sum(-1)
                        packed=torch.stack([mu,(qp-qm)/2,(qp+qm)/2],-1).cpu().numpy()
                        for b,batchpos in enumerate(take):
                            i,p=addresses[batchpos]; loc=positions[ri][i]
                            packed[b,~outputs[ri]['region_valid'][loc,p]]=0
                            outputs[ri]['summary'][loc,p]+=packed[b]/4
                timing['response_gpu_and_transfer_wall']+=time.perf_counter()-response_tick
            values=np.mean(gaps,axis=0)
            for b,(i,p) in enumerate(addresses): feature[i,p]=values[b]; seen[i,p]+=1
            if not checks:
                # Repeat the same first sample alone, detecting batch-dependent numerical drift.
                one=[]
                for k in range(4):
                    xx=x[:1].rot90(k,(-2,-1)).expand(-1,3,-1,-1)
                    z=model.layer1(model.maxpool(model.relu(model.bn1(model.conv1((xx-mean)/std))))).flatten(2).double()
                    ww=torch.tensor(fpweights[k,:1].reshape(1,1,-1),device='cuda')
                    one.append(((z*ww).sum(-1)/ww.sum(-1)).cpu().numpy()[0])
                error=float(abs(np.mean(one,axis=0)-values[0]).max())
                assert error<1e-4
                checks.append(dict(single_batch_gap_max_error=error))
            if time.perf_counter()-last>25:
                print(stage,mode,first+len(ids[first:first+12]),'/',len(ids),round(time.perf_counter()-tick,1),flush=True); last=time.perf_counter()
    assert np.array_equal(seen[ids],valid[ids].astype(int))
    save_tick=time.perf_counter()
    if mode=='gap':
        np.savez_compressed(HERE/f'features_{stage}.npz',instance_ids=table.instance_id.to_numpy(dtype=str),layer1=feature.reshape(n,-1),valid=valid,computed=required_mask.any(0))
    else:
        if stage=='source':
            with np.load(HERE/'features_source.npz') as z: previous=z['layer1'].reshape(n,2,256)
            err=float(abs(previous[ids]-feature[ids]).max()); assert err<1e-4
            checks.append(dict(reextracted_gap_max_error=err))
        if mode=='scale':
            assert stage=='source'
            tau=np.zeros((3,2)); sigma_rows=[]
            for f in range(3):
                for p in range(2):
                    sig=np.concatenate([o['sigma'][valid[o['indices'],p],p] for o,r in zip(outputs,roles) if r['fold']==f and r['held'] is not None])
                    tau[f,p]=np.median(sig); assert tau[f,p]>0
                for o,r in zip(outputs,roles):
                    if r['fold']==f:
                        np.savez_compressed(HERE/f"scale_{r['name']}.npz",indices=o['indices'],sigma=o['sigma'])
            save_json(HERE/'scales.json',dict(tau=tau.tolist(),created_utc=now(),source_only=True,definition='median C4-mean global sigma over legal held readers, valid pol and classes'))
        else:
            directory=HERE/f'summaries_{stage}'; directory.mkdir(exist_ok=True)
            qrows=[]
            for o,r in zip(outputs,roles):
                if not len(o['indices']):continue
                assert np.isfinite(o['summary']).all()
                assert not o['summary'][~o['region_valid']].any()
                np.savez_compressed(directory/f"{r['name']}.npz",**o,instance_ids=table.iloc[o['indices']].instance_id.to_numpy(dtype=str),tau=tau[r['fold']],regions=np.array(['global','body','end','side','fixed_body','fixed_end','fixed_side']))
                for j,i in enumerate(o['indices']):
                    for p in range(2):
                        for k,name in enumerate(REGIONS):
                            area,cov,eff,flag=o['quality'][j,p,k]
                            qrows.append(dict(reader=r['name'],row_index=int(i),instance_id=table.iloc[i].instance_id,pol=POL[p],region=name,area=area,coverage=cov,min_neff=eff,region_valid=bool(flag)))
            pd.DataFrame(qrows).to_csv(HERE/f'info_region_quality_{stage}.csv',index=False)
    torch.cuda.synchronize()
    timing['cache_finalize_serialize_io']=time.perf_counter()-save_tick
    receipt=dict(stage=stage,mode=mode,completed_utc=now(),objects=len(ids),pol_images=int(seen.sum()),seconds=time.perf_counter()-tick,gpu_peak_bytes=torch.cuda.max_memory_allocated(),checks=checks,cpu_threads=2,gpu=torch.cuda.get_device_name(),timing=timing,timing_note='Instrumented wall segments; excludes setup and single-sample replay. Not full old+new pipeline end-to-end.')
    save_json(HERE/f'extraction_{stage}_{mode}.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['source','test']); parser.add_argument('mode',choices=['gap','scale','regions'])
    args=parser.parse_args(); main(args.stage,args.mode)
