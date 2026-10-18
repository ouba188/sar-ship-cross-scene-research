import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
DATA = Path('E:/Hermes/paper_tracking/f0921_chips_dyn')
SOURCE = ROOT.parent / 'recovery_31cc_2ec7/five_scene_sources.csv'


def main():
    d = pd.read_csv(DATA / 'index.csv')
    d['date'] = d.scene.str.extract(r'Chip_(\d{8})_')[0]
    arrays = {p: np.load(DATA / f'F0921_{p}.npy', mmap_mode='r') for p in ['VV', 'VH']}
    coverage = d.groupby('class_raw').agg(rows=('row', 'size'), mmsi=('mmsi', 'nunique'), date_utm_groups=('scene', 'nunique'))
    coverage.sort_values('rows', ascending=False).to_csv(ROOT / 'raw_class_coverage.csv')
    sources = pd.read_csv(SOURCE)
    # Date and rounded coordinates provide candidate linkage, not product/identity proof.
    groups = d.groupby(['date', 'mmsi'])
    matches = []
    for s in sources.to_dict('records'):
        key = (s['observation_time_token'][:8], s['mmsi'])
        candidates = groups.get_group(key) if key in groups.groups else d.iloc[:0]
        near = candidates[(candidates.lon - s['lon']).abs().le(0.00003) & (candidates.lat - s['lat']).abs().le(0.00003)]
        matches.append(dict(product=s['product'], source_filename=s['source_filename'], mmsi=s['mmsi'], class_raw=s['class_raw'], date_mmsi_candidates=len(candidates), coordinate_candidates=len(near), chip_rows=json.dumps(near.row.tolist()), raw_labels=json.dumps(near.class_raw.tolist())))
    links = pd.DataFrame(matches)
    links.to_csv(ROOT / 'five_scene_candidate_links.csv', index=False)
    summary = dict(rows=len(d), unique_mmsi=d.mmsi.nunique(), date_utm_groups=d.scene.nunique(), raw_labels=d.class_raw.nunique(), class8_counts=d.class8.value_counts().to_dict(), shape={p:list(a.shape) for p,a in arrays.items()}, dtype={p:str(a.dtype) for p,a in arrays.items()}, sequential_rows=bool(np.array_equal(d.row,np.arange(len(d)))), crop_source_counts=d.len_src_from.value_counts().to_dict(), five_scene_sources=len(links), coordinate_link_sources=int(links.coordinate_candidates.gt(0).sum()), unique_coordinate_link_sources=int(links.coordinate_candidates.eq(1).sum()))
    (ROOT / 'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    focus = d[(d.date.eq('20250202') & d.mmsi.isin([564902000,431651000])) | (d.date.eq('20250315') & d.mmsi.isin([419001909,636022833]))]
    selected = focus.copy()
    for cls in ['oil_products_tanker','lpg_tanker','chemical_oil_products_tanker','container_ship']:
        pool = d[d.class_raw.eq(cls) & d.date.ge('20250101')]
        if len(pool):
            selected = pd.concat([selected,pool.iloc[[0]]])
    selected = selected.drop_duplicates('row')
    selected.to_csv(ROOT / 'viewed_chip_rows.csv',index=False)
    canvas = Image.new('RGB',(760,280*len(selected)), 'white')
    draw = ImageDraw.Draw(canvas)
    for j,r in enumerate(selected.to_dict('records')):
        y=j*280
        draw.text((8,y+4),f"row={r['row']} MMSI={r['mmsi']} {r['class_raw']}",fill='black')
        draw.text((8,y+20),f"{r['scene']} source_side={r['crop_side_src']} length_from={r['len_src_from']}",fill='black')
        for k,p in enumerate(['VV','VH']):
            draw.text((20+k*370,y+37),p,fill='black')
            canvas.paste(Image.fromarray(arrays[p][int(r['row'])]).convert('RGB'),(20+k*370,y+53))
    canvas.save(ROOT / 'paired_chips.png')
    print(json.dumps(summary,indent=2))
    print(links.groupby('product').coordinate_candidates.agg(['size',lambda x:int(x.gt(0).sum())]).to_string())
    print('focus rows:',focus[['row','scene','mmsi','class_raw']].to_string(index=False))


if __name__ == '__main__':
    main()
