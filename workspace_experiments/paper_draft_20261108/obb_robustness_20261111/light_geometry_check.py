"""Small native-geometry check: standard library only, no torch/numpy/cache loading."""
from pathlib import Path
import array
import ast
import csv
import json
import math
import statistics
import struct
import sys
import zipfile

sys.dont_write_bytecode = True
from obb_geometry import conditions, perturb

ROOT = Path(__file__).resolve().parent


def soft_regions_at(g, x, y):
    cx, cy, length, width, angle = g[:5]
    u = (x-cx)*math.cos(angle)+(y-cy)*math.sin(angle)
    v = -(x-cx)*math.sin(angle)+(y-cy)*math.cos(angle)
    def rectangle(l, w):
        return max(0., min(1., l/2+.5-abs(u))) * max(0., min(1., w/2+.5-abs(v)))
    body = rectangle(length, width)
    return body, max(0., rectangle(length, 2*width)-body), max(0., rectangle(1.5*length,width)-body)


def geometry_check(record, cfg):
    g, p = record['geometry'], record['placement']
    changed = perturb(g, cfg)
    before, after, l1, intersection = [[0.]*3 for _ in range(4)]
    for y in range(p['pad_top'], p['pad_top']+p['crop_height']):
        for x in range(p['pad_left'],p['pad_left']+p['crop_width']):
            b, a = soft_regions_at(g,x,y), soft_regions_at(changed,x,y)
            for k in range(3):
                before[k] += b[k]
                after[k] += a[k]
                l1[k] += abs(a[k]-b[k])
                intersection[k] += min(a[k],b[k])
    return [dict(id=record['id'],polarization=record['polarization'],condition=cfg['name'],
        region=name,length_px=g[2],width_px=g[3],dx_px=changed[0]-g[0],dy_px=changed[1]-g[1],
        displacement_px=math.hypot(changed[0]-g[0],changed[1]-g[1]),
        before_area_px=before[k],after_area_px=after[k],native_l1_px=l1[k],
        covered_fraction=intersection[k]/max(before[k],1e-12),zero_native_change=l1[k]<1e-10)
        for k,name in enumerate(('body','side','end'))]


def read_one_npy(path, member, row):
    # Stream a single already-produced row; never load the full model/cache arrays.
    with zipfile.ZipFile(path) as archive, archive.open(member+'.npy') as stream:
        assert stream.read(6)==b'\x93NUMPY'
        version=tuple(stream.read(2))
        length=struct.unpack('<H' if version==(1,0) else '<I',stream.read(2 if version==(1,0) else 4))[0]
        header=ast.literal_eval(stream.read(length).decode('latin1'))
        assert header['descr']=='<f4' and not header['fortran_order']
        count=math.prod(header['shape'][1:])
        skip=row*count*4
        while skip:
            part=stream.read(min(skip,65536))
            if not part:raise EOFError(path)
            skip-=len(part)
        result=array.array('f')
        result.frombytes(stream.read(count*4))
        return result,header['shape'][1:]


def main():
    records=json.loads((ROOT/'native_sources.json').read_text(encoding='utf-8'))
    ordered=sorted(records,key=lambda r:r['geometry'][3])
    chosen=[ordered[0],ordered[len(ordered)//2],max(records,key=lambda r:r['geometry'][2])]
    configs={v['name']:v for v in conditions()}
    # Cardinal axes catch swapped axes/units; transforms must preserve input and confidence.
    g=[80.,80.,22.,8.,0.,.7]
    assert perturb(g,configs['along_p10'])==[82.2,80.,22.,8.,0.,.7]
    assert perturb(g,configs['across_p10'])==[80.,80.8,22.,8.,0.,.7]
    turned=perturb([80.,80.,22.,8.,math.pi/2,.7],configs['along_p10'])
    assert abs(turned[0]-80)<1e-12 and abs(turned[1]-82.2)<1e-12
    assert abs(perturb(g,configs['angle_p05'])[4]-math.pi/36)<1e-12
    assert g==[80.,80.,22.,8.,0.,.7]
    audits=[]
    for record in chosen:
        for cfg in conditions():audits.extend(geometry_check(record,cfg))
    with (ROOT/'light_native_audit.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(audits[0]));writer.writeheader();writer.writerows(audits)
    grid=[]
    for record in chosen:
        for member in ('slots','local_slots'):
            base,shape=read_one_npy(ROOT/'geometry/zero.npz',member,record['index'])
            p=0 if record['polarization']=='VV' else 1
            cells=shape[-1]*shape[-2]
            for name in ('along_p05','across_p05'):
                path=ROOT/'geometry'/f'{name}.npz'
                if not path.exists():continue
                other,_=read_one_npy(path,member,record['index'])
                for region,ix in [('body',range(8)),('side',range(8,16)),('end',range(16,20))]:
                    difference=[abs(sum(other[(p*20+s)*cells+j]-base[(p*20+s)*cells+j] for s in ix)) for j in range(cells)]
                    grid.append(dict(id=record['id'],polarization=record['polarization'],condition=name,
                        grid=shape[-1],region=region,l1=sum(difference),changed_cells=sum(v>1e-7 for v in difference)))
    with (ROOT/'light_grid_audit.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(grid[0]));writer.writeheader();writer.writerows(grid)
    result=dict(status='passed_small_geometry_checks_only',torch_imported='torch' in sys.modules,
        n_present_views=len(records),n_selected_views=len(chosen),n_conditions=len(configs),
        median_length_px=statistics.median(r['geometry'][2] for r in records),
        median_width_px=statistics.median(r['geometry'][3] for r in records),
        selected=[dict(id=r['id'],polarization=r['polarization'],geometry=r['geometry']) for r in chosen],
        selected_nonzero_conditions_zero_native_body_change=sum(r['condition']!='zero' and r['region']=='body' and r['zero_native_change'] for r in audits),
        limitation='Small real-geometry checks and partial saved grid masks; no model forward or score reproduction.')
    (ROOT/'light_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
