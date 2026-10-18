"""Windowed WorldCover context for frozen detections; never edits SAR pixels."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from osgeo import gdal
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
WC = Path('E:/临时会话/safe841_redownload_plan_20260918/annotation_bundle/assets/worldcover')
LAND = [10, 20, 30, 40, 50, 60, 100]
BUFFERS = [50, 100, 200]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interior_distance(classes, pixel_m):
    # Missing coverage, wetland and mangrove are not safe land exclusions.
    land = np.isin(classes, LAND)
    padded = np.pad(land, 1, constant_values=False)
    return distance_transform_edt(padded, sampling=pixel_m)[1:-1, 1:-1]


def region_stats(classes, distance, region):
    n = int(region.sum())
    if n == 0:
        return dict(water_fraction=0., land_fraction=0., uncertain_fraction=1., land_clearance_lower_bound_m=0., action='keep_uncertain')
    vals = classes[region]
    water = float(np.mean(vals == 80))
    land = float(np.mean(np.isin(vals, LAND)))
    minimum = float(distance[region].min())
    action = 'exclude_land_interior' if minimum > 100 else ('keep_water' if water == 1 else 'keep_coastal_or_uncertain')
    return dict(water_fraction=water, land_fraction=land, uncertain_fraction=float(np.mean(~np.isin(vals, LAND + [80]))), land_clearance_lower_bound_m=minimum, action=action)


class WaterMap:
    def __init__(self, paths):
        self.vrt = gdal.BuildVRT('', [str(p) for p in paths], srcNodata=0, VRTNodata=0)

    def window(self, ds, window):
        x, y, w, h = map(int, window)
        gt = ds.GetGeoTransform()
        if gt[2] != 0 or gt[4] != 0 or gt[1] <= 0 or gt[5] >= 0:
            raise ValueError('Expected north-up metric SAR raster')
        west, north = gdal.ApplyGeoTransform(gt, x, y)
        east, south = gdal.ApplyGeoTransform(gt, x+w, y+h)
        warped = gdal.Warp('', self.vrt, format='MEM', outputBounds=[west, south, east, north], width=w, height=h,
                           dstSRS=ds.GetProjection(), resampleAlg='near', dstNodata=0, outputType=gdal.GDT_Byte)
        classes = warped.ReadAsArray()
        return classes, interior_distance(classes, (abs(gt[5]), gt[1]))


def object_window(obj, ds, pad=24):
    points = np.concatenate([np.asarray(m['source_corners']) for m in obj['members']])
    lo = np.maximum(0, np.floor(points.min(0)).astype(int)-pad)
    hi = np.minimum([ds.RasterXSize, ds.RasterYSize], np.ceil(points.max(0)).astype(int)+pad+1)
    x,y = lo
    w,h = hi-lo
    mask = Image.new('1', (int(w),int(h)))
    draw = ImageDraw.Draw(mask)
    for member in obj['members']:
        draw.polygon([tuple(p) for p in np.asarray(member['source_corners'])-lo], fill=1)
    return (int(x),int(y),int(w),int(h)), np.asarray(mask,dtype=bool)


def main():
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
    gdal.SetCacheMax(128*1024**2)
    HERE.mkdir(exist_ok=True)
    paths = sorted(WC.glob('*_Map.tif'))
    wm = WaterMap(paths)
    batches = [PARENT/x for x in ['object_bridge','object_bridge_b529','object_bridge_31cc_2ec7']]
    datasets, paths_by_product, objects, links, tiles = {}, {}, [], [], []
    for batch in batches:
        data = read_json(batch/'INPUTS.json')
        for source in data['sources']:
            paths_by_product[source['product']] = {p:source['raster_'+p.lower()] for p in ['VV','VH']}
        objects.extend(read_json(batch/'objects.json'))
        links.extend(read_json(batch/'source_links.json'))
        tiles.extend(data['tiles'])
    for product, pair in paths_by_product.items():
        datasets[product] = {p:gdal.Open(path) for p,path in pair.items()}
    provisional = {r['candidate_object_ids'][0] for r in links if r['status']=='provisional_unique'}
    rows, object_by_id = [], {}
    for i,obj in enumerate(objects):
        product = obj['product']
        ds = datasets[product]['VV']
        window, region = object_window(obj, ds)
        classes, distance = wm.window(ds,window)
        stats = region_stats(classes,distance,region)
        x,y,w,h = window
        cx,cy = np.round(np.asarray(obj['source_center'])-[x,y]).astype(int)
        row = dict(object_id=obj['object_id'], product=product, pair_status=obj['pair_status'], provisional_unique=obj['object_id'] in provisional,
                   **stats, centroid_land=bool(np.isin(classes[np.clip(cy,0,h-1),np.clip(cx,0,w-1)],LAND)))
        for b in BUFFERS:
            row[f'exclude_{b}m'] = stats['land_clearance_lower_bound_m'] > b
        rows.append(row)
        object_by_id[obj['object_id']] = obj
        if (i+1)%200==0:
            print(f'objects {i+1}/{len(objects)}',flush=True)
    csv_write(HERE/'object_policy.csv',rows)
    row_by_id = {r['object_id']:r for r in rows}
    source_rows = []
    for link in links:
        row = dict(source_key_json=link['source_key_json'], product=link['product'], old_status=link['status'], original_candidates=len(link['candidate_object_ids']))
        for b in BUFFERS:
            retained = [oid for oid in link['candidate_object_ids'] if not row_by_id[oid][f'exclude_{b}m']]
            row[f'retained_{b}m'] = len(retained)
        source_rows.append(row)
    csv_write(HERE/'source_retention.csv',source_rows)
    tile_rows, cache = [], {}
    for tile in tiles:
        x,y = tile['origin']; w,h = tile['width'],tile['height']
        ds = datasets[tile['product']]['VV']
        key = (tile['product'],x,y,w,h)
        if key not in cache:
            pad = 24
            left,top = max(0,x-pad),max(0,y-pad)
            right,bottom = min(ds.RasterXSize,x+w+pad),min(ds.RasterYSize,y+h+pad)
            classes,distance = wm.window(ds,(left,top,right-left,bottom-top))
            core = distance[y-top:y-top+h,x-left:x-left+w]
            cache[key] = float(core.min())
        tile_rows.append(dict(tile_id=tile['tile_id'],product=tile['product'],skip_land_interior=cache[key]>100,land_clearance_lower_bound_m=cache[key]))
    csv_write(HERE/'tile_policy.csv',tile_rows)
    for name, keep in [('objects_retained.json',lambda r:r['action']!='exclude_land_interior'),('objects_land_review.json',lambda r:r['action']=='exclude_land_interior')]:
        selected = [object_by_id[r['object_id']] for r in rows if keep(r)]
        (HERE/name).write_text(json.dumps(selected,indent=2),encoding='utf-8')
    retained_ids={r['object_id'] for r in rows if r['action']!='exclude_land_interior'}
    manifests,edges=[],[]
    for batch in batches:
        with (batch/'crop_manifest.csv').open(encoding='utf-8-sig',newline='') as f:
            manifests.extend(dict(r,source_bridge=str(batch)) for r in csv.DictReader(f) if r['object_id'] in retained_ids)
        edges.extend(r for r in read_json(batch/'neighbor_edges.json') if r['object_id'] in retained_ids and r['neighbor_id'] in retained_ids)
    csv_write(HERE/'crop_manifest_retained.csv',manifests)
    (HERE/'neighbor_edges_retained.json').write_text(json.dumps(edges,indent=2),encoding='utf-8')
    summary = dict(objects=len(rows),sources=len(links),tiles=len(tiles),unique_spatial_tiles=len(cache),
                   actions=dict(Counter(r['action'] for r in rows)), tiles_skippable=sum(r['skip_land_interior'] for r in tile_rows),
                   centroid_land=sum(r['centroid_land'] for r in rows),
                   uncertain_objects=sum(r['uncertain_fraction']>0 for r in rows),retained_manifest_rows=len(manifests),retained_neighbor_edges=len(edges),
                   by_product={p:dict(Counter(r['action'] for r in rows if r['product']==p)) for p in datasets},
                   sensitivity={str(b):dict(excluded=sum(r[f'exclude_{b}m'] for r in rows),provisional_excluded=sum(r[f'exclude_{b}m'] and r['provisional_unique'] for r in rows),sources_losing_all_candidates=sum(r['original_candidates']>0 and r[f'retained_{b}m']==0 for r in source_rows)) for b in BUFFERS},
                   source='Local ESA WorldCover 2021 v200; classes80 water,10/20/30/40/50/60/100 land; all others uncertain',
                   watermap_files=[str(p) for p in paths], raster_paths=paths_by_product,
                   scope='Five source-selected acquisition subsets, not full-scene sweep. No verified detection truth or new classification scores. 100m provisional operational margin,50/200m sensitivity,not calibrated geolocation uncertainty. Distance is a lower bound to non-confident-land (water/unknown/local-window edge),NOT exact coastline distance. 2021 map versus2025 SAR.')
    (HERE/'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # Review extremes and near-coast cases, including source-associated candidates.
    review = []
    for product in datasets:
        pool = [r for r in rows if r['product']==product]
        selectors = [lambda r:r['land_clearance_lower_bound_m'],lambda r:float(r['provisional_unique'])+float(r['centroid_land']),lambda r:float(r['action']=='keep_coastal_or_uncertain')*(1-abs(r['water_fraction']-.5))]
        for selector in selectors:
            ranked = sorted(pool,key=selector,reverse=True)
            pick = next((r for r in ranked if r not in review),None)
            if pick: review.append(pick)
    csv_write(HERE/'review_cases.csv',review)
    for start in range(0,len(review),5):
        subset = review[start:start+5]
        panel = Image.new('RGB',(1000,len(subset)*300),'white')
        draw = ImageDraw.Draw(panel)
        for j,row in enumerate(subset):
            obj=object_by_id[row['object_id']]; ds=datasets[obj['product']]['VV']
            cx,cy=np.round(obj['source_center']).astype(int)
            x,y=max(0,min(cx-128,ds.RasterXSize-256)),max(0,min(cy-128,ds.RasterYSize-256))
            classes,_=wm.window(ds,(x,y,256,256))
            draw.text((8,j*300+3),f"{start+j+1} {obj['product'][-4:]} {row['action']} water={row['water_fraction']:.2f} clearance>={row['land_clearance_lower_bound_m']:.0f}m provisional={row['provisional_unique']}",fill='black')
            for k,pol in enumerate(['VV','VH','MAP']):
                if pol=='MAP':
                    rgb=np.full((256,256,3),150,dtype=np.uint8)
                    rgb[classes==80]=[55,145,215]; rgb[np.isin(classes,LAND)]=[175,185,100]
                    im=Image.fromarray(rgb)
                else:
                    im=Image.fromarray(datasets[obj['product']][pol].ReadAsArray(int(x),int(y),256,256)).convert('RGB')
                ink=ImageDraw.Draw(im)
                for member in obj['members']:
                    points=[tuple(p) for p in np.asarray(member['source_corners'])-[x,y]]
                    ink.line(points+[points[0]], fill=(255,80,20),width=1)
                draw.text((10+k*330,j*300+20),pol,fill='black')
                panel.paste(im,(10+k*330,j*300+38))
        panel.save(HERE/f'review_{start//5+1}.png')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['watermap_files','raster_paths']},indent=2),flush=True)


if __name__=='__main__':
    main()
