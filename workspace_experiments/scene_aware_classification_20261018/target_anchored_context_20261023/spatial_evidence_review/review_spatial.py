"""Fixed-model, native C4 layer3 spatial accounting. No fitting or training."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.models import resnet50
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from shapely.geometry import Polygon, box

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
BASE = PARENT.parent
EXP = BASE/'context_expansion_20261022'
STRONG = BASE/'context_strong_reader_20261022'
NEW = PARENT/'partition_replication'
SCORE = PARENT/'score_scale_decoupling'
sys.path.insert(0, str(SCORE))
from run_score import ScoreDecoupled

CHECKPOINT = Path('C:/Users/25791/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth')
CASES = [('pair_0032624', 'rescued'), ('pair_0053959', 'harmed'),
         ('pair_0030906', 'CC'), ('pair_0030918', 'WW')]
ATOL = 5e-5


def receptive_field(model):
    r, jump, start = 1., 1., .5
    rows = []
    layers = [('conv1', model.conv1), ('maxpool', model.maxpool)]
    for stage in ['layer1', 'layer2', 'layer3']:
        for i, block in enumerate(getattr(model, stage)):
            layers.extend((f'{stage}.{i}.{name}', getattr(block, name))
                          for name in ['conv1', 'conv2', 'conv3'])
    for name, mod in layers:
        def one(v):
            return v[0] if isinstance(v, tuple) else v
        k, s, p = map(one, [mod.kernel_size, mod.stride, mod.padding])
        dilation = one(mod.dilation)
        effective = (k-1)*dilation+1
        start += ((effective-1)/2-p)*jump
        r += (effective-1)*jump
        jump *= s
        rows.append(dict(layer=name, kernel=k, stride=s, padding=p,
                         maximum_rf=r, jump=jump, start=start))
    pd.DataFrame(rows).to_csv(HERE/'receptive_field.csv', index=False)
    return r, jump, start


@torch.inference_mode()
def extract_maps(model, pixels):
    x = torch.tensor(pixels.copy(), dtype=torch.float32, device='cuda')[None,None]/255
    x = x.expand(-1,3,-1,-1)
    mean = x.new_tensor([.485,.456,.406])[None,:,None,None]
    std = x.new_tensor([.229,.224,.225])[None,:,None,None]
    x = (x-mean)/std
    maps, pooled = [], []
    for k in range(4):
        z = x.rot90(k,(-2,-1)).contiguous()
        z = model.maxpool(model.relu(model.bn1(model.conv1(z))))
        z = model.layer3(model.layer2(model.layer1(z)))
        maps.append(z[0].cpu().numpy())
        pooled.append(z.mean((-2,-1)))
    return np.stack(maps), torch.stack(pooled).mean(0)[0].cpu().numpy()


def sampling_centers(shape, image_shape, rotation, jump, start):
    gh, gw = shape
    height, width = image_shape
    yy, xx = np.meshgrid(start+jump*np.arange(gh), start+jump*np.arange(gw), indexing='ij')
    if rotation == 0:
        x, y = xx, yy
    elif rotation == 1:
        x, y = width-yy, xx
    elif rotation == 2:
        x, y = width-xx, height-yy
    else:
        x, y = yy, height-xx
    return np.stack([np.rot90(x,-rotation), np.rot90(y,-rotation)], axis=-1)


def grid_accounting(evidence, response, reference, polygon, width, height):
    gh, gw = evidence.shape
    xe, ye = np.linspace(0,width,gw+1), np.linspace(0,height,gh+1)
    fractions = np.empty((gh,gw))
    for iy in range(gh):
        for ix in range(gw):
            cell = box(xe[ix],ye[iy],xe[ix+1],ye[iy+1])
            fractions[iy,ix] = cell.intersection(polygon).area/cell.area
    cell_area = width*height/(gh*gw)
    clipped_area = polygon.intersection(box(0,0,width,height)).area
    assert abs(fractions.sum()*cell_area-clipped_area) < 1e-7
    out = dict(obb_area=polygon.area, obb_clipped_area=clipped_area,
               obb_area_fraction=clipped_area/(width*height),
               partial_cells=int(((fractions>0)&(fractions<1)).sum()),
               full_inside_cells=int((fractions>1-1e-9).sum()))
    for name, values in [('centered',evidence), ('response',response), ('reference',reference)]:
        out[name+'_total'] = float(values.sum())
        out[name+'_inside'] = float((values*fractions).sum())
        out[name+'_outside'] = float((values*(1-fractions)).sum())
    for region, fraction in [('inside',fractions), ('outside',1-fractions)]:
        out[region+'_positive'] = float((np.maximum(evidence,0)*fraction).sum())
        out[region+'_negative'] = float((np.minimum(evidence,0)*fraction).sum())
        out[region+'_absolute'] = float((abs(evidence)*fraction).sum())
    return fractions, xe, ye, out


def draw_case(case, images, grids):
    fig, axes = plt.subplots(2,3,figsize=(13,8), layout='constrained')
    limit = max(np.max(abs(g['evidence'])) for g in grids)
    norm = TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
    for j,(pixels,g) in enumerate(zip(images,grids)):
        height,width = pixels.shape
        points = np.r_[g['points'],g['points'][:1]]
        ax = axes[j,0]
        ax.imshow(pixels,cmap='gray',vmin=0,vmax=255,interpolation='nearest',extent=[0,width,height,0])
        ax.set_title(f'{g["pol"].upper()} native {width}x{height} px; recorded OBB')
        for col in [1,2]:
            ax = axes[j,col]
            values = g['evidence'] if col==1 else g['response']
            # Flat cells preserve the actual 4x4/5x5 accounting, no smoothed heatmap.
            scale = norm if col==1 else TwoSlopeNorm(vmin=-max(abs(values).max(),1e-12),vcenter=0,vmax=max(abs(values).max(),1e-12))
            mesh = ax.pcolormesh(g['xe'],g['ye'],values,cmap='RdBu_r',norm=scale,
                                 shading='flat',edgecolors='#666666',linewidth=.6)
            for iy in range(values.shape[0]):
                for ix in range(values.shape[1]):
                    ax.text((g['xe'][ix]+g['xe'][ix+1])/2,(g['ye'][iy]+g['ye'][iy+1])/2,
                            f'{values[iy,ix]:+.3f}',ha='center',va='center',fontsize=8,
                            color='white' if abs(values[iy,ix])>.6*scale.vmax else 'black')
            label = 'centered delta' if col==1 else 'uncentered response'
            ax.set_title(f'{label}: {values.shape[1]}x{values.shape[0]} cells; sum={values.sum():+.5f}')
            fig.colorbar(mesh,ax=ax,shrink=.75,label='margin contribution per cell')
        for ax in axes[j]:
            ax.plot(points[:,0],points[:,1],color='#16b845',lw=1.8)
            ax.set(xlim=(0,width),ylim=(height,0),aspect='equal',xlabel='native x (px)',ylabel='native y (px)')
    fig.suptitle(f'{case["instance_id"]} | new seed29 {case["outcome"]} | true {case["true_class"]} vs fixed wrong {case["wrong_class"]}\n'
                 f'A margin {case["base_margin"]:+.5f}; core delta {case["cached_delta"]:+.5f}; m={case["m"]:+.4f}\n'
                 'Coarse readout accounting, NOT pixel attribution. Green: recorded detection OBB. Reference subtraction is spatially uniform.',fontsize=11)
    path = HERE/f'{case["instance_id"]}_spatial.png'
    fig.savefig(path,dpi=150)
    plt.close(fig)
    return str(path)


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    targets = pd.read_csv(EXP/'readout/targets.csv',dtype={'mmsi':str})
    predictions = pd.read_csv(SCORE/'predictions.csv')
    predictions = predictions[(predictions.cohort=='new')&(predictions.seed==29)].set_index('instance_id')
    deep = np.load(STRONG/'deep_features.npz')
    h = deep['layer3']
    assert np.array_equal(deep['instance_ids'],targets.instance_id)
    model = resnet50(weights=None)
    model.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True))
    model = model.cuda().float().eval().requires_grad_(False)
    rf,jump,start = receptive_field(model)
    cases,pol_rows,cell_rows,rotation_rows,group_rows,source_rows = [],[],[],[],[],[]
    for ident,outcome in CASES:
        i = targets.index[targets.instance_id==ident][0]
        target = targets.iloc[i]
        saved = predictions.loc[ident]
        fold = int(saved.fold)
        inp_path = NEW/f'fold{fold}_inputs.npz'
        origin_path = NEW/'origins'/f'fold{fold}_outer.npz'
        with np.load(inp_path) as z:
            inp = {k:z[k] for k in z.files}
        with np.load(origin_path) as z:
            origin = {k:z[k] for k in z.files}
        assert i in inp['test'] and np.array_equal(inp['train'],origin['fit'])
        assert np.array_equal(inp['instance_ids'],targets.instance_id)
        w,h0 = origin['W'],origin['h0']
        np.testing.assert_allclose(h0,h[inp['train']].mean(0,dtype=np.float64),atol=1e-12,rtol=0)
        q_rebuilt = np.zeros((32,3))
        for j,cols in enumerate(np.array_split(np.arange(2048),16)):
            q_rebuilt[16+j] = (h[i,cols]-h0[cols])@w[:,cols].T
        q_error = float(abs(q_rebuilt-inp['q'][i]).max())
        assert q_error < 1e-10
        baseline_error = float(abs(h[i].astype(float)@w.T+origin['raw_intercept']-inp['base'][i]).max())
        assert baseline_error < ATOL
        gate_path = SCORE/'models'/f'new_fold{fold}_29.pt'
        gate = ScoreDecoupled('layer3').cuda().eval().requires_grad_(False)
        gate.load_state_dict(torch.load(gate_path,map_location='cpu',weights_only=True))
        with torch.inference_mode():
            env = torch.tensor(inp['env'],dtype=torch.float32,device='cuda')
            aa,mm,dd = gate.parts(env)
            # Full batch reproduces the original matrix multiplication layout.
            q_tensor = torch.tensor(inp['q'],dtype=torch.float32,device='cuda')
            deltas = (dd[:,:,None]*q_tensor).sum(1)
            d = dd[i].cpu().numpy()
            a = aa[i].cpu().numpy()
            m = float(mm[i,0])
            delta = deltas[i].cpu().numpy()
        base = inp['base'][i].astype(np.float64)
        core = base+delta
        scores = float(np.float32(1)+np.float32(m))*core
        truth = int(target.class_id)
        wrong = max((c for c in range(3) if c!=truth),key=lambda c:base[c])
        baseline_pred, pred = int(base.argmax()), int(core.argmax())
        observed = 'rescued' if baseline_pred!=truth and pred==truth else 'harmed' if baseline_pred==truth and pred!=truth else 'CC' if pred==truth else 'WW'
        assert observed==outcome
        np.testing.assert_allclose(core,saved[['u0','u1','u2']].to_numpy(float),atol=ATOL,rtol=0)
        np.testing.assert_allclose(scores,saved[['s0','s1','s2']].to_numpy(float),atol=ATOL,rtol=0)
        margin_w = w[truth]-w[wrong]
        gate_channels = np.repeat(d[16:],128)
        coeff = margin_w*gate_channels
        cached_delta = float(delta[truth])-float(delta[wrong])
        q_delta64 = float(np.sum(d.astype(float)*(inp['q'][i,:,truth]-inp['q'][i,:,wrong])))
        case = dict(instance_id=ident,outcome=outcome,fold=fold,seed=29,true_class=truth,wrong_class=wrong,
                    base_prediction=baseline_pred,prediction=pred,port=target.port,mmsi=target.mmsi,pass_group=target.pass_group,
                    base_margin=float(base[truth]-base[wrong]),core_margin=float(core[truth]-core[wrong]),
                    score_margin=float(scores[truth]-scores[wrong]),m=m,cached_delta=cached_delta,
                    q_delta64=q_delta64,q_rebuild_error=q_error,baseline_affine_error=baseline_error)
        source_rows.append(dict(instance_id=ident,fold=fold,alpha=float(inp['alpha']),source_n=len(inp['train']),
                    input_cache=str(inp_path),outer_origin=str(origin_path),gate_checkpoint=str(gate_path),
                    W_shape=str(w.shape),W_dtype=str(w.dtype),h0_source='unweighted outer source mean',
                    W_source='saved exact PCA/scaler/ridge backprojection with factor 3; no refit'))
        arrays = dict(W=w,h0=h0,a=a,d=d,m=np.array(m),base=base,core=core,score=scores,q=inp['q'][i],q_rebuilt=q_rebuilt,
                      true_class=np.array(truth),wrong_class=np.array(wrong),channel_coeff=coeff)
        images,grids = [],[]
        for pol_index,pol in enumerate(['vv','vh']):
            path = EXP/'readout/native_targets'/f'{ident}_{pol}.png'
            with Image.open(path) as image:
                assert image.mode=='L'
                pixels = np.array(image)
            window = np.array(json.loads(target[f'target_window_{pol}']))
            height,width = pixels.shape
            assert [width,height]==window[2:].tolist()
            points = np.array(json.loads(target.target_points))-window[:2]
            polygon = Polygon(points)
            assert polygon.is_valid
            maps,pooled = extract_maps(model,pixels)
            channels = slice(pol_index*1024,(pol_index+1)*1024)
            pooled_error = float(abs(pooled-h[i,channels]).max())
            np.testing.assert_allclose(pooled,h[i,channels],atol=ATOL,rtol=ATOL)
            gh,gw = maps.shape[-2:]
            cp,hp = coeff[channels],h0[channels]
            response_rot = np.einsum('c,kcyx->kyx',cp,maps.astype(float))/(gh*gw)
            reference_value = float(cp@hp)/(gh*gw)
            reference_rot = np.full_like(response_rot,reference_value)
            centered_rot = response_rot-reference_rot
            response_back = np.stack([np.rot90(v,-k) for k,v in enumerate(response_rot)])
            centered_back = np.stack([np.rot90(v,-k) for k,v in enumerate(centered_rot)])
            response = response_back.mean(0)
            evidence = centered_back.mean(0)
            reference = np.full_like(evidence,reference_value)
            fractions,xe,ye,account = grid_accounting(evidence,response,reference,polygon,width,height)
            centers = np.stack([sampling_centers(maps[k].shape[-2:],pixels.shape,k,jump,start) for k in range(4)])
            phase_span = float(np.ptp(centers,axis=0).max())
            expected_pol = float(cp@(h[i,channels]-hp))
            map_error = float(evidence.sum()-expected_pol)
            assert abs(map_error)<ATOL
            # This test detects sign, /HW, polarization or C4 inverse bookkeeping mistakes.
            algebra_error = float(abs(evidence.sum()-cp@(maps.mean((0,2,3),dtype=float)-hp)))
            assert algebra_error<1e-10
            for k in range(4):
                rotation_rows.append(dict(instance_id=ident,polarization=pol,rotation_deg=k*90,
                    response=float(response_rot[k].sum()),reference=float(reference_rot[k].sum()),
                    centered=float(centered_rot[k].sum()),inverse_sum_error=float(abs(centered_back[k].sum()-centered_rot[k].sum()))))
            for j in range(8):
                global_group=16+pol_index*8+j
                cc=slice(j*128,(j+1)*128)
                group_rows.append(dict(instance_id=ident,polarization=pol,group=global_group,d=float(d[global_group]),
                    q_margin=float(inp['q'][i,global_group,truth]-inp['q'][i,global_group,wrong]),
                    cached_contribution=float(d[global_group]*(inp['q'][i,global_group,truth]-inp['q'][i,global_group,wrong])),
                    map_contribution=float(cp[cc]@(maps[:,cc].mean((0,2,3),dtype=float)-hp[cc]))))
            for iy in range(gh):
                for ix in range(gw):
                    cell_rows.append(dict(instance_id=ident,polarization=pol,row=iy,col=ix,x0=xe[ix],x1=xe[ix+1],
                        y0=ye[iy],y1=ye[iy+1],obb_fraction=fractions[iy,ix],
                        centered=evidence[iy,ix],response=response[iy,ix],reference=reference[iy,ix],
                        inside=evidence[iy,ix]*fractions[iy,ix],outside=evidence[iy,ix]*(1-fractions[iy,ix])))
            pol_rows.append(dict(instance_id=ident,polarization=pol,native_width=width,native_height=height,
                grid_width=gw,grid_height=gh,png=str(path),window=json.dumps(window.tolist()),obb_local=json.dumps(points.tolist()),
                feature_max_error=pooled_error,map_cached_integral_error=map_error,algebra_error=algebra_error,
                cached_pol_delta=expected_pol,rf_max=rf,jump=jump,c4_rf_center_phase_span=phase_span,**account))
            arrays.update({f'{pol}_maps_c4':maps,f'{pol}_pooled':pooled,f'{pol}_response_c4':response_rot,
                f'{pol}_centered_c4':centered_rot,f'{pol}_centered_inverse_c4':centered_back,
                f'{pol}_evidence':evidence,f'{pol}_response':response,f'{pol}_reference':reference,
                f'{pol}_obb_fraction':fractions,f'{pol}_obb_local':points,f'{pol}_sampling_centers_inverse_c4':centers})
            images.append(pixels)
            grids.append(dict(pol=pol,evidence=evidence,response=response,reference=reference,points=points,xe=xe,ye=ye))
        case['spatial_integral'] = float(sum(g['evidence'].sum() for g in grids))
        case['integral_error_vs_cached_float32'] = case['spatial_integral']-cached_delta
        case['integral_error_vs_cached_float64'] = case['spatial_integral']-q_delta64
        case['uncentered_response'] = float(sum(g['response'].sum() for g in grids))
        case['source_reference_subtracted'] = float(sum(g['reference'].sum() for g in grids))
        assert abs(case['integral_error_vs_cached_float32'])<ATOL
        case['panel'] = draw_case(case,images,grids)
        np.savez_compressed(HERE/f'{ident}_decomposition.npz',**arrays)
        cases.append(case)
        print(json.dumps(case,ensure_ascii=False),flush=True)
    for name,rows in [('cases',cases),('polarization',pol_rows),('cells',cell_rows),('rotations',rotation_rows),
                      ('groups',group_rows),('sources',source_rows)]:
        pd.DataFrame(rows).to_csv(HERE/f'{name}.csv',index=False)
    summary = dict(cases=4,trained_models=0,reader_refits=0,checkpoint=str(CHECKPOINT),
        model_eval=True,fp32_extraction=True,tf32=False,no_input_resize_interpolation_or_padding=True,
        original_architecture_padding_unchanged=True,rotation_degrees=[0,90,180,270],
        max_integral_error_vs_cached_float32=max(abs(c['integral_error_vs_cached_float32']) for c in cases),
        max_feature_error=max(r['feature_max_error'] for r in pol_rows),absolute_integral_tolerance=ATOL,
        max_q_rebuild_error=max(c['q_rebuild_error'] for c in cases),
        max_affine_baseline_precision_error=max(c['baseline_affine_error'] for c in cases),
        max_spatial_algebra_error=max(r['algebra_error'] for r in pol_rows),
        theoretical_max_rf=rf,feature_stride=jump,first_rf_center=start,
        grid_convention='inverse-rotated index grid; equal-extent display cells, not exact RF support; phases recorded',
        inside_outside='intersection area fraction of recorded OBB with coarse cell, not pure causal support',
        reference_convention='uniformly allocate per-channel source h0 over each spatial grid; not image response',
        class_contrast='true versus highest scoring wrong class in frozen baseline, fixed for every map and rotation')
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
