"""Frozen R50 source-only passes; layer1/layer3 maps are never stored."""
from pipeline import *
from torchvision.models import resnet50


def geometry_module():
    spec = importlib.util.spec_from_file_location('upstream_obb_geometry', ROOT/'obb_dynamic_readout/common.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def centered_response(response, norm):
    mu = torch.einsum('bru,bcu->brc',norm,response)
    r = response[:,None]-mu[...,None]
    r = r-r.mean(2,keepdim=True)
    return mu,r


def attention(response, weights, tau, recenter=False):
    norm = weights/weights.sum(-1,keepdim=True).clamp_min(1e-300)
    mu,r = centered_response(response,norm)
    if recenter:
        r = r-(r*norm[:,:,None]).sum(-1,keepdim=True)
    logw = torch.where(norm>0,norm.log(),-torch.inf)[:,:,None]
    empty = weights.sum(-1)==0
    logw = torch.where(empty[:,:,None,None],torch.zeros_like(logw),logw)
    qp = ((logw+r/tau).softmax(-1)*r).sum(-1)
    qm = ((logw-r/tau).softmax(-1)*r).sum(-1)
    packed = torch.stack([mu,(qp-qm)/2,(qp+qm)/2],-1)
    assert torch.isfinite(packed).all()
    assert (packed[...,1]>=-1e-8).all()
    assert (packed[...,2].abs()<=packed[...,1]+1e-8).all()
    return packed


def main(mode):
    tick = time.perf_counter()
    table,tr,va = load()
    geo = geometry_module()
    n = len(table)
    valid = np.load(HERE/'valid.npz')['valid'].astype(bool)
    ids = tr if mode in ('scale','probe') else np.arange(n)
    if mode=='probe':
        ids=ids[:8]
    # Filter snapshots by obs_id; no historical target labels or scores are used.
    wanted = set(table.obs_id)
    snapshots = {s['obs_id']:s for s in json.loads((OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8')) if s['obs_id'] in wanted}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = resnet50(weights=None)
    model.load_state_dict(torch.load(geo.WEIGHTS,map_location='cpu',weights_only=True))
    model = model.cuda().float().eval().requires_grad_(False)
    W = {}
    for layer,dim in [('layer1',256),('layer3',1024)]:
        with np.load(HERE/'models'/f'reader_{layer}.npz') as z:
            W[layer] = torch.tensor(z['W'].reshape(8,2,dim).transpose(1,0,2),device='cuda',dtype=torch.float64)
    normmean = torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    normstd = torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    sig = {k:np.zeros((n,2,8)) for k in W}
    mean3,hb3 = np.zeros((n,2,8)),np.zeros((n,2,8,2))
    summary = np.zeros((n,2,3,8,3))
    quality,geometry = np.zeros((n,2,3,4)),np.zeros((n,2,4))
    flags = np.zeros((n,2,3),bool)
    gaps = {'layer1':np.zeros((n,2,256)), 'layer3':np.zeros((n,2,1024))}
    polmean = {'layer1':np.zeros((n,2,8)), 'layer3':mean3}
    if mode=='summaries':
        tau = {k:np.array(v) for k,v in json.loads((HERE/'scales.json').read_text())['tau'].items()}
    timing = dict(native_cpu_io=0.,mask_cpu=0.,backbone_gpu_wall=0.,response_gpu_transfer_wall=0.)
    seen = np.zeros((n,2),int)
    last = tick
    with torch.inference_mode():
        for first in range(0,len(ids),8):
            stamp = time.perf_counter()
            pixels,fps,geoms,addresses = [],[],[],[]
            for i in ids[first:first+8]:
                for p in range(2):
                    if not valid[i,p]:
                        continue
                    pix,fp,g,place = geo.native(table,snapshots,int(i),p)
                    pixels.append(pix[None].astype(np.float32)/255)
                    fps.append(fp)
                    geoms.append(g)
                    addresses.append((int(i),p))
            timing['native_cpu_io'] += time.perf_counter()-stamp
            if not addresses:
                continue
            stamp = time.perf_counter()
            x = torch.tensor(np.stack(pixels),device='cuda')
            fp = torch.tensor(np.stack(fps)[:,None],device='cuda',dtype=torch.float32)
            weights1 = np.stack([np.stack([geo.cell_mass(np.rot90(f,k)) for f in fps]) for k in range(4)])
            # Exact 160->10 area bins avoid the observed CUDA single-channel
            # rotated-tensor stride path; these weights match the original 3-channel mask path.
            weights3 = np.stack([np.stack([np.rot90(f,k).reshape(10,16,10,16).mean((1,3)) for f in fps]) for k in range(4)])
            region_weights = []
            for j,(i,p) in enumerate(addresses):
                if mode=='summaries':
                    masks = geo.soft_regions(geoms[j])
                    q = geo.region_quality(masks,fps[j],geoms[j])
                    flag = (q['area']>0)&(q['min_neff']>=4)
                    flags[i,p] = flag
                    quality[i,p] = np.stack([q['area'],q['coverage'],q['min_neff'],flag],-1)
                    g = geoms[j]
                    if g is not None:
                        geometry[i,p] = np.r_[g[2:4],g[5],1.]
                    region_weights.append(np.stack([geo.cell_mass(np.rot90(masks*fps[j],k,(-2,-1))) for k in range(4)]))
            region_weights = np.stack(region_weights) if region_weights else None
            pol = torch.tensor([p for i,p in addresses],device='cuda')
            timing['mask_cpu'] += time.perf_counter()-stamp
            for rotation in range(4):
                torch.cuda.synchronize()
                stamp = time.perf_counter()
                xx = x.rot90(rotation,(-2,-1)).expand(-1,3,-1,-1)
                z1 = model.layer1(model.maxpool(model.relu(model.bn1(model.conv1((xx-normmean)/normstd)))))
                z3 = model.layer3(model.layer2(z1))
                torch.cuda.synchronize()
                timing['backbone_gpu_wall'] += time.perf_counter()-stamp
                stamp = time.perf_counter()
                ww = {'layer1':torch.tensor(weights1[rotation].reshape(len(addresses),1,-1),device='cuda'),
                      'layer3':torch.tensor(weights3[rotation].reshape(len(addresses),1,-1),device='cuda',dtype=torch.float64)}
                if mode=='probe':
                    wcheck=F.interpolate(fp.rot90(rotation,(-2,-1)),size=z3.shape[-2:],mode='area')
                    check=(z3*wcheck).sum((-2,-1))/wcheck.sum((-2,-1))
                    direct=((z3.flatten(2).double()*ww['layer3']).sum(-1)/ww['layer3'].sum(-1))
                    print('rotation_pool',rotation,float(abs(check-direct).max()),flush=True)
                for layer,z in [('layer1',z1),('layer3',z3)]:
                    z = z.flatten(2).double()
                    weight = ww[layer]
                    norm = weight/weight.sum(-1,keepdim=True)
                    gap = (z*norm).sum(-1).cpu().numpy()
                    response = torch.einsum('bck,bku->bcu',W[layer][pol],z)
                    mu,r = centered_response(response,norm)
                    means = mu[:,0].cpu().numpy()
                    for j,(i,p) in enumerate(addresses):
                        gaps[layer][i,p] += gap[j]/4
                        polmean[layer][i,p] += means[j]/4
                    if mode in ('scale','probe'):
                        sigma = (r.square()*norm[:,:,None]).sum(-1).clamp_min(0).sqrt()[:,0].cpu().numpy()
                        for j,(i,p) in enumerate(addresses):
                            sig[layer][i,p] += sigma[j]/4
                    elif layer=='layer3':
                        t = torch.tensor(tau[layer],device='cuda')[pol,None,None,None]
                        packed = attention(response,weight,t,recenter=True)[:,0].cpu().numpy()
                        for j,(i,p) in enumerate(addresses):
                            hb3[i,p] += packed[j,:,1:]/4
                    else:
                        weight = torch.tensor(region_weights[:,rotation].reshape(len(addresses),3,-1),device='cuda')
                        t = torch.tensor(tau[layer],device='cuda')[pol,None,None,None]
                        packed = attention(response,weight,t).cpu().numpy()
                        for j,(i,p) in enumerate(addresses):
                            packed[j,~flags[i,p]] = 0
                            summary[i,p] += packed[j]/4
                timing['response_gpu_transfer_wall'] += time.perf_counter()-stamp
            for i,p in addresses:
                seen[i,p] += 1
            if time.perf_counter()-last>20:
                print(mode,min(first+8,len(ids)),'/',len(ids),'seconds',round(time.perf_counter()-tick,1),flush=True)
                last=time.perf_counter()
    assert np.array_equal(seen[ids],valid[ids].astype(int))
    checks = []
    rows = table.row_index.to_numpy()
    for layer in ['layer1','layer3']:
        file = ROOT/'features.npz' if layer=='layer3' else ROOT/'obb_dynamic_readout/features_source.npz'
        with np.load(file) as z:
            prior = z[layer][rows[ids]].reshape(len(ids),2,-1)
        error = float(abs(gaps[layer][ids]-prior).max())
        print('GAP CHECK',layer,error,flush=True)
        if mode=='probe':
            diff=abs(gaps[layer][ids]-prior)
            bad=np.unravel_index(diff.argmax(),diff.shape)
            print('worst',int(rows[ids[bad[0]]]),bad,'new',gaps[layer][ids][bad],'old',prior[bad],flush=True)
        else:
            assert error < 1e-4, (layer,error)
        with np.load(HERE/'models'/f'reader_{layer}.npz') as z:
            direct = gaps[layer][ids].reshape(len(ids),-1)@z['W'].T+z['intercept']
            replay = polmean[layer][ids].sum(1)+z['intercept']
        closure = float(abs(direct-replay).max())
        assert closure < 1e-8
        checks.append(dict(layer=layer,original_GAP_max_error=error,mean_WGAP_closure=closure))
    if mode=='probe':
        spec=importlib.util.spec_from_file_location('upstream_old_features',ROOT/'prepare_features.py')
        original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
        with torch.inference_mode():
            old=original.encode(model,x,fp.expand(-1,3,-1,-1))['layer3']
            alt=[]
            for rotation in range(4):
                xx=x.rot90(rotation,(-2,-1)).expand(-1,3,-1,-1)
                zz=model.relu(model.bn1(model.conv1((xx-normmean)/normstd)))
                before=zz.clone()
                reg=fp.expand(-1,3,-1,-1).rot90(rotation,(-2,-1))
                for j in range(3):
                    original.pool(zz,reg[:,j:j+1])
                print('stem_mutation',rotation,float(abs(zz-before).max()),flush=True)
                zz=model.layer3(model.layer2(model.layer1(model.maxpool(zz))))
                mask=fp.rot90(rotation,(-2,-1))
                wa=F.interpolate(mask,size=zz.shape[-2:],mode='area')
                wb=F.interpolate(reg[:,0:1],size=zz.shape[-2:],mode='area')
                print('weight_diff',rotation,float(abs(wa-wb).max()),mask.stride(),reg[:,0:1].stride(),flush=True)
                alt.append(original.pool(zz,mask,maximum=False).cpu().numpy())
            alt=np.mean(alt,0)
            print('encode_vs_direct',float(abs(old-alt).max()),flush=True)
        for j,(i,p) in enumerate(addresses):
            print('original',int(rows[i]),p,float(abs(old[j]-gaps['layer3'][i,p]).max()),'vs_direct',float(abs(alt[j]-gaps['layer3'][i,p]).max()),flush=True)
        return
    if mode=='scale':
        tau = {layer:[float(np.median(sig[layer][tr[valid[tr,p]],p])) for p in range(2)] for layer in W}
        assert all(np.isfinite(v) and v>0 for values in tau.values() for v in values)
        save_json('scales.json',dict(tau=tau,fit_indices=table.iloc[tr].row_index.tolist(),created_utc=utc(),
            definition='Median over C4-averaged global sigma of fit_train samples and eight class coordinates; valid pol only; fixed for all seeds.'))
        np.savez_compressed(HERE/'scale_sigma.npz',indices=table.iloc[tr].row_index,layer1=sig['layer1'][tr],layer3=sig['layer3'][tr])
    else:
        assert not hb3[~valid].any() and not summary[~flags].any()
        np.savez_compressed(HERE/'summaries.npz',indices=rows,mean3=mean3,hb3=hb3,summary1=summary,
            quality=quality,geometry=geometry,region_valid=flags,valid=valid)
        # Current geometry/quality are label-free and must agree with the historical implementation.
        old = np.load(ROOT/'obb_dynamic_readout/summaries_source/fold0_outer.npz')
        locations = {int(v):j for j,v in enumerate(rows)}
        pairs = [(j,locations[int(v)]) for j,v in enumerate(old['indices']) if int(v) in locations]
        if pairs:
            a,b = np.array(pairs).T
            qerror = float(abs(quality[b]-old['quality'][a]).max())
            gerror = float(abs(geometry[b]-old['geometry'][a]).max())
            assert qerror<1e-8 and gerror<1e-8
            checks.append(dict(historical_geometry_quality_overlap=len(pairs),quality_max_error=qerror,geometry_max_error=gerror))
    receipt = dict(mode=mode,completed_utc=utc(),n=len(ids),pol_images=int(seen.sum()),seconds=time.perf_counter()-tick,
        gpu_peak_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(),batch_objects=8,
        checks=checks,timing=timing,no_spatial_F_cache=True,no_backbone_fit=True,external_labels_read=False)
    save_json(f'extraction_{mode}.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['scale','summaries','probe'])
    main(parser.parse_args().mode)
