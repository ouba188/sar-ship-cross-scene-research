"""Frozen Full-Tuned query-only polarization recovery; no label-based selection."""
from pathlib import Path
import argparse
import copy
import gc
import importlib.util
import json
import re
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent.parent
OLD = ROOT.parent / 'full_finetune_20261111'
CHIPS = Path('D:/Documents/project_root/project_data/public_opensarship2/opensarship2/chips')
SEEDS = (17, 29, 43)


def save(name, obj):
    (ROOT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def records():
    rows = json.loads((WORK/'m1_coverage_20261101/manifest.json').read_text(encoding='utf-8'))
    groups = json.loads((WORK/'m1_coverage_20261101/groups.json').read_text(encoding='utf-8'))
    return rows, groups


def recover(row):
    out = copy.deepcopy(row)
    missing = [p for p in ('vv','vh') if row.get(p) is None]
    if not missing:
        return out, None
    if len(missing) != 1:
        raise ValueError('Expected exactly one missing view')
    dst = missing[0]
    src = 'vh' if dst == 'vv' else 'vv'
    donor = row[src]
    target = donor['detections'][donor['target_index']]
    a, b = (Path(row['source_paths'][p]) for p in (src,dst))
    # Original products and crop centers must agree, not only raster sizes.
    canonical = lambda p: re.sub(r'_(vv|vh)$', '', p.stem)
    if a.parent != b.parent or canonical(a) != canonical(b):
        raise ValueError(f'Unmatched source crop: {row["id"]}')
    png = CHIPS/dst/(row['id']+'.png')
    with Image.open(donor['image']) as im: source = np.asarray(im.convert('L'))
    with Image.open(png) as im: dest = np.asarray(im.convert('L'))
    with Image.open(a) as im: source_raw = np.asarray(im.convert('L'))
    with Image.open(b) as im: dest_raw = np.asarray(im.convert('L'))
    if source.shape != dest.shape or not np.array_equal(source,source_raw) or not np.array_equal(dest,dest_raw):
        raise ValueError(f'Pixel provenance/shape mismatch: {row["id"]}')
    corners = np.asarray(target['corners'])
    if not np.isfinite(corners).all() or target['confidence'] < .25:
        raise ValueError('Invalid donor box')
    if dest.max() == dest.min():
        raise ValueError('No image variation in missing view')
    out[dst] = dict(image=str(png), detections=[copy.deepcopy(target)], target_index=0,
                    alignment_valid=True, box_origin=f'{src}_transferred',
                    ship_axis=copy.deepcopy(donor.get('ship_axis')),
                    radar_axis_rad=donor.get('radar_axis_rad'))
    # Confidence remains donor detection confidence, not a destination detection.
    out[dst]['detections'][0]['confidence_origin'] = src
    audit = dict(id=row['id'], restored=dst, donor=src, shape=list(dest.shape),
                 source_pixel_identity=True, destination_pixel_identity=True,
                 shared_product_crop_key=True, donor_confidence=target['confidence'],
                 source_image=donor['image'], restored_image=str(png),
                 source_tif=str(a), restored_tif=str(b), corners=target['corners'])
    return out, audit


def panel(row, audit, path, footer=''):
    canvas = Image.new('RGB',(760,300),'white')
    d = ImageDraw.Draw(canvas)
    d.text((8,6),f"{row['id']} | restore {audit['restored'].upper()} | native pixels",fill='black')
    for n,pol in enumerate(('vv','vh')):
        image_path = row[pol]['image'] if row.get(pol) else audit['restored_image']
        with Image.open(image_path) as im: im = im.convert('RGB')
        x,y = 10+380*n,45
        canvas.paste(im,(x,y))
        pts = [(x+float(a),y+float(b)) for a,b in audit['corners']]
        d.line(pts+[pts[0]],fill='red' if pol == audit['restored'] else 'green',width=1)
        d.text((x,27),f"{pol.upper()} | {im.width}x{im.height} | {'borrowed' if pol==audit['restored'] else 'original'} box",fill='black')
    d.text((8,260),footer,fill='black')
    canvas.save(path)


def prepare():
    rows,groups = records()
    audit=[]
    for split in ('small','validation','evaluation'):
        for idx in groups[split]:
            _,record = recover(rows[idx])
            if record:
                audit.append(dict(split=split,index=idx,**record))
    save('pixel_audit_local.json',audit)
    save('protocol.json',dict(seeds=SEEDS,weights='Full-Tuned selected checkpoints',
        train=False, query_only=True, references='unchanged original 32',
        target_rule='borrow only when exactly one view absent; matching original crop/product and pixels',
        alignment_requirement='source provenance audit must confirm common VV/VH grid before inference',
        tuning=False, held_out_labels_used_for_recovery=False,
        splits={s:len(groups[s]) for s in ('small','validation','evaluation')},
        recovery_counts={s:sum(r['split']==s for r in audit) for s in ('small','validation','evaluation')}))
    (ROOT/'panels').mkdir(exist_ok=True)
    candidates=[r for r in audit if r['split']=='evaluation']
    for r in candidates[:8]:
        panel(rows[r['index']],r,ROOT/'panels'/f"before_{r['index']}.png")
    print('PREPARED',json.dumps({s:sum(r['split']==s for r in audit) for s in ('small','validation','evaluation')}),flush=True)


def run():
    if not (ROOT/'ALIGNMENT_CONFIRMED.md').exists():
        raise RuntimeError('Read source provenance and confirm alignment before inference')
    spec=importlib.util.spec_from_file_location('full_recovery_parent',OLD/'train.py')
    ft=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=ft
    spec.loader.exec_module(ft)
    torch=ft.torch
    torch.set_num_threads(2)
    rows,groups=records()
    from geometry import _cell_area

    def inputs(batch):
        d=ft.cost.prepare(batch)
        # Reproduce the exact native masks and integration used during fine-tuning.
        for j,row in enumerate(batch):
            for p,view in enumerate(ft.staged.task.data.prepare(row)):
                if view is None or not view['valid_target']: continue
                mask=ft.cost.raw.tokens.slots(view['geometry'],view['footprint'])
                read=ft.cost.raw.tokens.cell_mass(mask,8)/_cell_area(20)
                writer=ft.cost.raw.tokens.cell_mass(mask,16)/_cell_area(10)
                writer/=np.maximum(writer.sum(0,keepdims=True),1)
                for key,value in (('read',read),('write',writer),('mass',read.sum((-2,-1)))):
                    d[key][j,p]=torch.as_tensor(value,device='cuda')
        return d

    reference_rows=[rows[i] for i in groups['references']]
    refdata=inputs(reference_rows)
    records_out=[]
    start=time.perf_counter()
    with torch.inference_mode():
        for seed in SEEDS:
            net=ft.LiveFull(seed,'full')
            net.load_state_dict(torch.load(OLD/f'selected_s{seed}/best.pt',map_location='cpu',weights_only=True),strict=True)
            net.eval()
            ref=net.reference(refdata)
            for split in ('validation','evaluation'):
                ids=groups[split]
                old=np.load(OLD/f'selected_s{seed}/{split}.npz')
                assert np.array_equal(old['ids'],ids)
                base=[];new=[]
                for k in range(0,len(ids),12):
                    batch=[rows[i] for i in ids[k:k+12]]
                    z,_=net(inputs(batch),batch,ref,reference_rows)
                    base.append(z.cpu().numpy())
                    restored=[recover(r)[0] for r in batch]
                    z,_=net(inputs(restored),restored,ref,reference_rows)
                    new.append(z.cpu().numpy())
                base,new=np.concatenate(base),np.concatenate(new)
                err=float(abs(base-old['score']).max())
                assert err<1e-3 and np.array_equal(base.argmax(1),old['score'].argmax(1)),err
                available=np.array([r.get('vv') is not None and r.get('vh') is not None for r in (rows[i] for i in ids)])
                unchanged=float(abs(base[available]-new[available]).max())
                assert unchanged<1e-5,unchanged
                labels=np.array([rows[i]['label'] for i in ids])
                np.savez_compressed(ROOT/f'{split}_s{seed}.npz',ids=ids,label=labels,base=base,recovered=new)
                for group,mask in (('all',np.ones(len(ids),bool)),('recovered',~available),('both',available)):
                    before=ft.metrics(labels[mask],base[mask]);after=ft.metrics(labels[mask],new[mask])
                    p0=base[mask].argmax(1);p1=new[mask].argmax(1);y=labels[mask]
                    records_out.append(dict(seed=seed,split=split,group=group,n=int(mask.sum()),
                        before=before,after=after,rescued=int(((p0!=y)&(p1==y)).sum()),
                        harmed=int(((p0==y)&(p1!=y)).sum()),replay_error=err,unchanged_error=unchanged))
                print('DONE',seed,split,'replay',err,records_out[-2],flush=True)
            del net,ref
            gc.collect();torch.cuda.empty_cache()
    save('results.json',records_out)
    save('status.json',dict(inference_complete=True,seconds=time.perf_counter()-start,
        trained=False,formal_model_changed=False,analysis_pending=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','run'))
    args=parser.parse_args()
    prepare() if args.mode=='prepare' else run()
