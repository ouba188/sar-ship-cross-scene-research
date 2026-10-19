"""Select and render traceable same-vessel diagnostic pairs, without training."""
import csv
import json
import math
from pathlib import Path
from urllib.parse import unquote, urlparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
GALLERY = ROOT.parent


def read_gallery():
    text = (GALLERY / 'gallery_data.js').read_text(encoding='utf-8')
    return json.loads(text.removeprefix('window.GALLERY_DATA=').rstrip(';\n'))


def angle_gap(a, b):
    return abs((a-b+math.pi/2) % math.pi-math.pi/2)*180/math.pi


def valid_axis(o, pol='vv'):
    p = o['images'][pol]['priors']
    a = p.get('ship_axis')
    return a if a and a['length_px']/a['width_px'] >= 1.5 else None


def path_of(view):
    return Path(unquote(urlparse(view['url']).path).lstrip('/'))


def row_of(pair, obs):
    a, b = (obs[str(pair[k])] for k in ('a', 'b'))
    row = dict(pair)
    row.update(mmsi=a['mmsi'], category=a['category'], same_port=a['port']==b['port'],
               correct_count=sum(o['predictions']['dual']['correct'] for o in (a,b)),
               margin_gap=abs(a['predictions']['dual']['margin']-b['predictions']['dual']['margin']))
    axes = [valid_axis(o) for o in (a,b)]
    row['axis_gap'] = angle_gap(*(x['angle'] for x in axes)) if all(axes) else None
    row['axis_usable'] = all(axes)
    row['radar_usable'] = all(o['images']['vv']['priors']['radar_axis'] is not None for o in (a,b))
    row['relative_gap'] = angle_gap(*[x['angle']-o['images']['vv']['priors']['radar_axis'] for x,o in zip(axes,(a,b))]) if all(axes) and row['radar_usable'] else None
    row['size_ratio'] = max(o['images']['vv']['w'] for o in (a,b))/min(o['images']['vv']['w'] for o in (a,b))
    row['occupancy_gap'] = abs(axes[0]['length_px']/a['images']['vv']['w']-axes[1]['length_px']/b['images']['vv']['w']) if all(axes) else None
    ns = [sum(d['confidence']>=.25 and d['shape_usable'] for d in o['images']['vv']['priors']['detections']) for o in (a,b)]
    row['candidate_count_gap'] = abs(ns[0]-ns[1])
    row['max_candidates'] = max(ns)
    row['pol_axis_gap_max'] = max([angle_gap(valid_axis(o)['angle'],valid_axis(o,'vh')['angle']) for o in (a,b) if valid_axis(o) and valid_axis(o,'vh')] or [0])
    return row


def write_csv(path, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def pixel_summary(o, pol):
    v=o['images'][pol]; p=v['priors']; im=np.asarray(Image.open(path_of(v)).convert('L'))
    def mask_of(d):
        mask=Image.new('L',(v['w'],v['h']))
        ImageDraw.Draw(mask).polygon([tuple(x) for x in d['corners']],fill=1)
        return np.asarray(mask,dtype=bool)
    target=mask_of(p['detections'][p['target_index']]) if p['target_index']>=0 else np.zeros(im.shape,dtype=bool)
    others=np.zeros(im.shape,dtype=bool)
    for i,d in enumerate(p['detections']):
        if i!=p['target_index'] and d['confidence']>=.25 and d['shape_usable']: others |= mask_of(d)
    others &= ~target
    row={'obs_id':o['id'],'pol':pol,'correct':o['predictions']['dual']['correct']}
    for name,mask in (('target',target),('other',others),('rest',~(target|others))):
        values=im[mask]
        row[name+'_pixels']=len(values)
        row[name+'_mean_u8']=float(values.mean()) if len(values) else None
        row[name+'_fraction_255']=float((values==255).mean()) if len(values) else None
    return row


def render(pair, obs, classes, reason):
    fig, axs=plt.subplots(2,4,figsize=(16,8))
    details=[]
    for r,k in enumerate(('a','b')):
        o=obs[str(pair[k])]
        for j,pol in enumerate(('vv','vh')):
            v=o['images'][pol]; p=v['priors']; im=np.asarray(Image.open(path_of(v)))
            for overlay in (0,1):
                ax=axs[r,2*j+overlay]; ax.imshow(im,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                ax.set_xlim(-.5,v['w']-.5); ax.set_ylim(v['h']-.5,-.5); ax.set_aspect('equal')
                if overlay:
                    for i,d in enumerate(p['detections']):
                        if d['confidence']<.25 or not d['shape_usable']: continue
                        pts=np.array(d['corners']+[d['corners'][0]])
                        ax.plot(pts[:,0],pts[:,1],color='lime' if i==p['target_index'] else 'orange',lw=1)
                    a=p.get('ship_axis')
                    if a:
                        center=np.array(a['center']); delta=np.array([math.cos(a['angle']),math.sin(a['angle'])])*a['length_px']/2
                        ax.plot([center[0]-delta[0],center[0]+delta[0]],[center[1]-delta[1],center[1]+delta[1]],color='yellow',lw=.8)
                    if p['radar_axis'] is not None:
                        center=np.array([v['w']/2,v['h']/2]); delta=np.array([math.cos(p['radar_axis']),math.sin(p['radar_axis'])])*min(v['w'],v['h'])*.4
                        ax.plot([center[0]-delta[0],center[0]+delta[0]],[center[1]-delta[1],center[1]+delta[1]],'--',color='cyan',lw=.8)
                pred=o['predictions']['dual']; name=classes[str(pred['class_id'])]
                ax.set_title(f"obs {o['id']} {pol.upper()} {v['w']}x{v['h']} {'prior overlay' if overlay else 'unaltered PNG'}\n{name} ({'correct' if pred['correct'] else 'wrong'}), margin {pred['margin']:.3f}",fontsize=9)
                ax.tick_params(labelsize=7)
            a=p.get('ship_axis',{})
            details.append(dict(pair_id=pair['id'],reason=reason,obs_id=o['id'],mmsi=o['mmsi'],category=o['category'],port=o['port'],time=o['time'],product=o['product'],pol=pol,path=str(path_of(v)),width=v['w'],height=v['h'],prediction=classes[str(o['predictions']['dual']['class_id'])],correct=o['predictions']['dual']['correct'],margin=o['predictions']['dual']['margin'],incidence=o['incidence'],target_status=p['match_status'],target_index=p['target_index'],n_candidates=len(p['detections']),axis_deg=a.get('angle',float('nan'))*180/math.pi,length_px=a.get('length_px'),width_px=a.get('width_px'),length_m=a.get('length_m'),radar_axis_deg=p['radar_axis']*180/math.pi if p['radar_axis'] is not None else None,scale_x=p['scale']['x'] if p['scale'] else None,scale_y=p['scale']['y'] if p['scale'] else None))
    a,b=(obs[str(pair[k])] for k in ('a','b'))
    fig.suptitle(f"Pair {pair['id']} | MMSI {a['mmsi']} | {a['category']} | {reason}\n{a['port']} {a['time']} (top) / {b['port']} {b['time']} (bottom)\nDiagnostic classifier from Oct 10; PNG values unchanged; OBB is candidate, cyan = unsigned ground-range axis",fontsize=11)
    fig.tight_layout(rect=(0,0,1,.91))
    fig.savefig(ROOT/f"pair_{pair['id']}.png",dpi=130); plt.close(fig)
    return details


def main():
    data=read_gallery(); obs=data['observations']
    rows=[row_of(p,obs) for p in data['pairs']]
    write_csv(ROOT/'pair_diagnostics.csv',rows)
    # Purposeful examples, not prevalence estimates or a model-selection set.
    mixed=[r for r in rows if r['correct_count']==1 and r['axis_usable'] and r['margin_gap']>.25]
    rules=[
        ('relative_angle_change',lambda r:r['radar_usable'] and r['same_port'],lambda r:r['relative_gap']),
        ('angle_change_nonboundary_small_incidence',lambda r:r['radar_usable'] and r['same_port'] and r['delta_incidence']!='' and abs(float(r['delta_incidence']))<.05 and min(abs(obs[str(r[k])]['predictions']['dual']['margin']) for k in ('a','b'))>.2,lambda r:r['relative_gap']),
        ('near_axis_correct_wrong',lambda r:r['axis_gap']<=10 and r['same_port'] and r['size_ratio']<=1.25,lambda r:r['margin_gap']),
        ('crop_extent_change',lambda r:r['size_ratio']>=1.5,lambda r:r['size_ratio']),
        ('multiple_candidate_change',lambda r:r['candidate_count_gap']>=2,lambda r:r['candidate_count_gap']),
        ('polarization_axis_disagreement',lambda r:r['pol_axis_gap_max']>=30,lambda r:r['pol_axis_gap_max']),
        ('cross_port_correct_wrong',lambda r:not r['same_port'] and r['radar_usable'],lambda r:r['margin_gap']),
    ]
    selected=[]; ships=set()
    for reason,condition,score in rules:
        pool=sorted((r for r in mixed if condition(r) and r['mmsi'] not in ships),key=lambda r:(-score(r),r['id']))
        if pool:
            r=dict(pool[0],selection_reason=reason); selected.append(r); ships.add(r['mmsi'])
    for count,reason in ((2,'persistent_correct_control'),(0,'persistent_wrong_control')):
        pool=sorted((r for r in rows if r['correct_count']==count and r['radar_usable'] and r['axis_usable'] and r['mmsi'] not in ships),key=lambda r:(-r['relative_gap'],r['id']))
        if pool:
            r=dict(pool[0],selection_reason=reason); selected.append(r); ships.add(r['mmsi'])
    write_csv(ROOT/'selected_pairs.csv',selected)
    details=[]
    for r in selected: details+=render(r,obs,data['classes'],r['selection_reason'])
    write_csv(ROOT/'case_observations.csv',details)
    write_csv(ROOT/'pixel_statistics.csv',[pixel_summary(obs[str(r[k])],pol) for r in selected for k in ('a','b') for pol in ('vv','vh')])
    summary={'pairs':len(rows),'paired_observations':len(obs),'mixed_pairs':sum(r['correct_count']==1 for r in rows),'mixed_unique_ships':len({r['mmsi'] for r in rows if r['correct_count']==1}),'mixed_both_vv_axes_elongated_margin_gt025':len(mixed),'mixed_both_radar_and_elongated_margin_gt025':sum(r['radar_usable'] for r in mixed),'selection':'Purposeful contrast selection; includes outcome-based filters; NOT unbiased prevalence, causal inference, or independent validation. Pair rows share observations and vessels.','selected':selected}
    (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
