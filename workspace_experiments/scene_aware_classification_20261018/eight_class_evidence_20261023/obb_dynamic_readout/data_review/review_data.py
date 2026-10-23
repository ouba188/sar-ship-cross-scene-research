"""Source-only OBB coverage observation; no feature extraction or training."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')
import sys
sys.dont_write_bytecode = True
import json
import time
import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
from torchvision.models import resnet50
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = Path('E:/临时会话/same_vessel_gallery_20261019/local_condition_module')
sys.path.insert(0, str(OLD))
from native_input import crop_pad
from prepare import masks_for
RHO_L, RHO_W, EDGE = .25, .5, 1.
REGIONS = ['body', 'end', 'side']
YY, XX = np.mgrid[:160, :160]


def geometry(view):
    p = view['priors']
    dets, j = p.get('detections', []), p.get('target_index', -1)
    if not (isinstance(j, int) and 0 <= j < len(dets)):
        return None, 'no_target_index'
    d = dets[j]
    if not d.get('shape_usable') or d['confidence'] < .25:
        return None, 'unusable_target'
    corners = np.asarray(d['corners'], float)
    edges = np.roll(corners, -1, axis=0)-corners
    lens = np.linalg.norm(edges, axis=1)
    a = int(np.argmax(lens))
    e = edges[a]/lens[a]
    return dict(center=corners.mean(0), e=e, side=np.array([-e[1], e[0]]),
        L=float(lens[a]), W=float(lens[(a+1)%4]), confidence=d['confidence'],
        corners=corners, target_index=j, touches_edge=d.get('touches_edge', False)), ''


def box(g, length_scale=1., width_scale=1.):
    u, v = g['e']*g['L']*length_scale/2, g['side']*g['W']*width_scale/2
    return np.array([g['center']-u-v, g['center']+u-v, g['center']+u+v, g['center']-u+v])


def clip_areas(g, rect):
    areas = []
    for ls, ws in [(1, 1), (1+2*RHO_L, 1), (1, 1+2*RHO_W)]:
        value, _ = cv2.intersectConvexConvex(box(g, ls, ws).astype(np.float32), rect.astype(np.float32))
        areas.append(max(0., float(value)))
    return np.array([areas[0], max(0, areas[1]-areas[0]), max(0, areas[2]-areas[0])])


def soft_masks(g, placement):
    x = XX+placement['crop_x']-placement['pad_left']
    y = YY+placement['crop_y']-placement['pad_top']
    u = (x-g['center'][0])*g['e'][0]+(y-g['center'][1])*g['e'][1]
    v = (x-g['center'][0])*g['side'][0]+(y-g['center'][1])*g['side'][1]
    def soft(length, width):
        return np.clip(.5+(length/2-abs(u))/EDGE, 0, 1)*np.clip(.5+(width/2-abs(v))/EDGE, 0, 1)
    body = soft(g['L'], g['W'])
    return np.stack([body, np.maximum(soft(g['L']*(1+2*RHO_L), g['W'])-body, 0),
        np.maximum(soft(g['L'], g['W']*(1+2*RHO_W))-body, 0)])


def voronoi_mass(masks):
    # Integrate piecewise-constant pixel cells over actual feature-center Voronoi bins.
    edge = np.r_[-.5, np.arange(2., 156., 4.), 159.5]+.5
    lo = np.floor(edge).astype(int); hi = np.minimum(lo+1, 160); frac = edge-lo
    prefix = np.pad(masks.cumsum(-1).cumsum(-2), ((0, 0), (1, 0), (1, 0)))
    corners = np.zeros((len(masks), 41, 41))
    for yi, wy in [(lo, 1-frac), (hi, frac)]:
        for xi, wx in [(lo, 1-frac), (hi, frac)]:
            corners += prefix[:, yi[:, None], xi[None, :]]*wy[None, :, None]*wx[None, None, :]
    return np.maximum(np.diff(np.diff(corners, axis=-1), axis=-2), 0)


def network():
    torch.set_num_threads(2)
    m = resnet50(weights=None).eval()
    rf, step, origin = 1, 1, 0.
    layers = [m.conv1, m.maxpool]
    for block in m.layer1:
        layers.extend([block.conv1, block.conv2, block.conv3])
    for layer in layers:
        def first(x):
            return x if isinstance(x, int) else x[0]
        k, s, p, d = [first(getattr(layer, name)) for name in ['kernel_size', 'stride', 'padding', 'dilation']]
        origin += ((k-1)*d/2-p)*step
        rf += (k-1)*d*step
        step *= s
    with torch.inference_mode():
        z = m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(torch.zeros(1, 3, 160, 160))))))
    return dict(shape=list(z.shape), receptive_field=rf, stride=step, center_origin_pixel_index=origin,
        implementation=inspect.getfile(type(m)), pretrained_forward=False)


def make_panel(row, snapshots, coverage, reason, serial):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), layout='constrained')
    out = []
    colors = ['#39dd67', '#ffb000', '#2dbeff']
    for p, pol in enumerate(['vv', 'vh']):
        view = snapshots[int(row.obs_id)]['images'][pol]
        with Image.open(row[f'image_{pol}']) as im:
            raw = np.asarray(im.convert('L'))
        assert raw.shape == (view['h'], view['w'])
        canvas, regions, place = crop_pad(raw, masks_for(view), 160)
        saved = coverage.loc[(row.instance_id, pol)]
        for key, value in place.items():
            assert value == saved[key], (row.instance_id, pol, key)
        g, missing = geometry(view)
        axes[p, 0].imshow(raw, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        axes[p, 0].set_title(f'{pol.upper()} original {view["w"]}x{view["h"]}; valid={saved.valid}')
        if g is None:
            for ax in axes[p, 1:]:
                ax.text(.5, .5, f'No usable target OBB: {missing}', ha='center')
            continue
        support = regions.sum(0)
        soft = soft_masks(g, place)*support
        gm = voronoi_mass(soft)
        for a, b, color in [(1, 1, colors[0]), (1.5, 1, colors[1]), (1, 2, colors[2])]:
            axes[p, 0].add_patch(Polygon(box(g, a, b), fill=False, edgecolor=color, lw=1.2))
        for j, name in enumerate(REGIONS):
            ax = axes[p, j+1]
            ax.imshow(canvas, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            rgba = np.zeros((160, 160, 4))
            rgba[..., :3] = matplotlib.colors.to_rgb(colors[j]); rgba[..., 3] = soft[j]*.55
            ax.imshow(rgba, interpolation='nearest')
            neff = gm[j].sum()**2/(gm[j]**2).sum() if gm[j].sum() else 0
            ax.set_title(f'{pol.upper()} {name}: mass={soft[j].sum():.1f}px, n_eff={neff:.1f}\n'
                f'OBB {g["L"]:.1f}x{g["W"]:.1f}; crop=({place["crop_x"]},{place["crop_y"]}) '
                f'pad=({place["pad_left"]},{place["pad_top"]})')
            ax.set_xlim(-.5, 159.5); ax.set_ylim(159.5, -.5)
        out.append(dict(instance_id=row.instance_id, pol=pol, reason=reason, image_path=row[f'image_{pol}'],
            valid=bool(saved.valid), length=g['L'], width=g['W'], **place))
    fig.suptitle(f'SOURCE geometry only / {row.instance_id} / {row.class_name} / {row.port} / {reason}\n'
        'rhoL=.25 rhoW=.5; 1px linear soft transition; native160 no resize; green=body orange=end blue=side; RF35', fontsize=11)
    path = HERE/f'case{serial}.png'
    fig.savefig(path, dpi=110); plt.close(fig)
    for item in out:
        item['panel'] = str(path)
    return out


def main():
    tick = time.perf_counter()
    table = pd.read_csv(ROOT/'cohort.csv', dtype={'mmsi': str})
    snapshots = {s['obs_id']: s for s in json.loads((OLD/'view_snapshot.json').read_text(encoding='utf-8'))}
    coverage = pd.read_csv(ROOT/'coverage.csv').set_index(['instance_id', 'pol'])
    aligned = pd.read_csv(OLD/'crossview_alignment.csv').set_index('obs_id')
    roles = {r: set() for r in ['train', 'validation', 'test']}
    role_rows = []
    for fold in range(3):
        with np.load(ROOT/'baseline'/f'fold{fold}_inputs.npz') as z:
            np.testing.assert_array_equal(z['instance_ids'], table.instance_id)
            for role in roles:
                roles[role].update(z[role].tolist())
                role_rows.append(dict(fold=fold, role=role, n=len(z[role])))
    source = roles['train'] | roles['validation']
    assert len(source) == 8750 and len(roles['test']) == 4744 and not source & roles['test']
    cohort = table.iloc[sorted(source)].copy()
    rows, region_rows = [], []
    for num, row in enumerate(cohort.itertuples()):
        for pol in ['vv', 'vh']:
            view = snapshots[int(row.obs_id)]['images'][pol]
            saved = coverage.loc[(row.instance_id, pol)]
            g, reason = geometry(view)
            rec = dict(instance_id=row.instance_id, obs_id=row.obs_id, class_id=row.class_id,
                class_name=row.class_name, mmsi=row.mmsi, port=row.port, product=row.product,
                source_role='train' if row.Index in roles['train'] else 'validation', pol=pol,
                w=view['w'], h=view['h'], native_valid=bool(saved.valid),
                disabled_alignment=bool(saved.disabled_alignment), target_usable=g is not None,
                missing_reason=reason, match_status=view['priors'].get('match_status'),
                crossview_status=aligned.loc[row.obs_id, 'status'],
                crossview_reason=aligned.loc[row.obs_id, 'reason'],
                **{k:saved[k] for k in ['crop_x','crop_y','crop_width','crop_height','pad_left','pad_top']})
            if g is None:
                rows.append(rec); continue
            rec.update(length=g['L'], width=g['W'], aspect=g['L']/g['W'], confidence=g['confidence'],
                center_x=g['center'][0], center_y=g['center'][1], axis_x=g['e'][0], axis_y=g['e'][1],
                corners=json.dumps(g['corners'].tolist()), target_index=g['target_index'],
                touches_edge=g['touches_edge'], center160_x=g['center'][0]-saved.crop_x+saved.pad_left,
                center160_y=g['center'][1]-saved.crop_y+saved.pad_top)
            native = np.array([[-.5,-.5], [view['w']-.5,-.5], [view['w']-.5,view['h']-.5],[-.5,view['h']-.5]])
            cx, cy, cw, ch = saved.crop_x, saved.crop_y, saved.crop_width, saved.crop_height
            retained = np.array([[cx-.5,cy-.5],[cx+cw-.5,cy-.5],[cx+cw-.5,cy+ch-.5],[cx-.5,cy+ch-.5]])
            ideal = np.array([g['L']*g['W'], 2*RHO_L*g['L']*g['W'], 2*RHO_W*g['L']*g['W']])
            na, ca = clip_areas(g, native), clip_areas(g, retained)
            support = np.zeros((160,160))
            px, py = int(saved.pad_left), int(saved.pad_top)
            support[py:py+int(ch),px:px+int(cw)] = 1
            soft = soft_masks(g, saved)*support
            grid = voronoi_mass(soft)
            rec['axis_corner_roundtrip_error'] = float(np.max(np.min(np.linalg.norm(box(g)[:,None]-g['corners'][None],axis=-1),axis=1)))
            for j, name in enumerate(REGIONS):
                mass, mass2 = grid[j].sum(), np.square(grid[j]).sum()
                region_rows.append(dict(instance_id=row.instance_id, pol=pol, region=name,
                    native_valid=bool(saved.valid), class_id=row.class_id, source_role=rec['source_role'],
                    hard_planned_area=ideal[j], hard_native_area=na[j], hard_retained_area=ca[j],
                    hard_native_fraction=float(np.clip(na[j]/ideal[j],0,1)),
                    hard_retained_fraction=float(np.clip(ca[j]/ideal[j],0,1)),
                    additional_crop_loss=max(0,float(na[j]-ca[j])), soft_pixel_mass=soft[j].sum(),
                    voronoi_mass=mass, positive_grid_cells=int((grid[j]>1e-8).sum()),
                    n_eff=mass**2/mass2 if mass2 else 0,
                    mass_conservation_error=abs(mass-soft[j].sum())))
            rows.append(rec)
        if num and num%2000 == 0:
            print('source coverage',num,'/',len(cohort),'seconds',round(time.perf_counter()-tick,1),flush=True)
    frame, regions = pd.DataFrame(rows), pd.DataFrame(region_rows)
    frame.to_csv(HERE/'source_geometry.csv',index=False)
    regions.to_csv(HERE/'source_region_coverage.csv',index=False)
    pd.DataFrame(role_rows).to_csv(HERE/'fold_membership_counts.csv',index=False)
    summary_rows = []
    for (pol, name), group in regions[regions.native_valid].groupby(['pol','region']):
        summary_rows.append(dict(pol=pol,region=name,n=len(group),
            coverage_below_half=int((group.hard_retained_fraction<.5).sum()),
            coverage_below_90pct=int((group.hard_retained_fraction<.9).sum()),
            coverage_zero=int((group.hard_retained_area<1e-8).sum()),
            soft_mass_below4=int((group.soft_pixel_mass<4).sum()),
            neff_below2=int((group.n_eff<2).sum()), neff_below4=int((group.n_eff<4).sum()),
            **{f'{key}_{q}':float(group[key].quantile(prob)) for key in ['hard_retained_fraction','soft_pixel_mass','n_eff']
                for q,prob in [('p05',.05),('median',.5),('p95',.95)]}))
    pd.DataFrame(summary_rows).to_csv(HERE/'coverage_summary.csv',index=False)
    vv=frame[(frame.pol=='vv')&frame.native_valid].copy()
    vv['area']=vv.length*vv.width
    ends=regions[(regions.pol=='vv')&(regions.region=='end')].set_index('instance_id')
    vv['end_fraction']=vv.instance_id.map(ends.hard_retained_fraction)
    used, chosen = set(), []
    for name, metric, quantile in [('small','area',.05),('large','area',.95),('narrow','width',.01),
            ('elongated','aspect',.99),('tight_crop','end_fraction',0),('near_square','aspect',.01)]:
        value=vv[metric].quantile(quantile)
        candidates=vv[~vv.instance_id.isin(used)].copy()
        candidates['distance']=abs(candidates[metric]-value)
        pick=candidates.sort_values(['distance','instance_id']).iloc[0]
        chosen.append(dict(instance_id=pick.instance_id, selection=name, metric=metric,
            source_quantile=quantile, target=value, actual=pick[metric]))
        used.add(pick.instance_id)
    pd.DataFrame(chosen).to_csv(HERE/'selected_cases.csv',index=False)
    cases=[]
    indexed=cohort.set_index('instance_id',drop=False)
    for j,c in enumerate(chosen,1):
        cases.extend(make_panel(indexed.loc[c['instance_id']],snapshots,coverage,c['selection'],j))
    pd.DataFrame(cases).to_csv(HERE/'review_cases.csv',index=False)
    info=dict(source_unique=len(source),source_train_unique=len(roles['train']),source_validation_unique=len(roles['validation']),
        test_unique=len(roles['test']), test_results_read=False, source_test_overlap=0,
        source_pol_counts=frame.groupby('pol')[['native_valid','target_usable','disabled_alignment']].sum().to_dict(),
        match_status=frame.groupby(['pol','match_status']).size().to_dict().__str__(),
        network=network(),rhoL=RHO_L,rhoW=RHO_W,soft_transition_width=EDGE,
        geometry_max_corner_error=float(frame.axis_corner_roundtrip_error.max()),
        max_mass_conservation_error=float(regions.mass_conservation_error.max()),
        seconds=time.perf_counter()-tick,new_fits=0)
    (HERE/'summary.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(info,ensure_ascii=False),flush=True)
    print(pd.DataFrame(summary_rows).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
