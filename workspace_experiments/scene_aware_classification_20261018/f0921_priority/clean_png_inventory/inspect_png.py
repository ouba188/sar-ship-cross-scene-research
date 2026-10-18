"""Inspect existing PNG pairs and ZIP metadata without extracting large arrays."""
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE=Path(__file__).resolve().parent
PNG=Path('E:/Hermes/paper_tracking/f0921_chips_clean_png')
ZIP=Path('E:/Hermes/paper_tracking/f0921_chips_clean_npy.zip')


def main():
    with zipfile.ZipFile(ZIP) as z:
        with z.open('index.csv') as f:
            index=pd.read_csv(f)
        rows=[]
        samples=[]
        for folder in sorted(PNG.iterdir()):
            if not folder.is_dir() or folder.name=='sheets':
                continue
            names={e.name for e in os.scandir(folder) if e.name.endswith('.png')}
            vv={n[:-7] for n in names if n.endswith('_VV.png')}
            vh={n[:-7] for n in names if n.endswith('_VH.png')}
            both=vv&vh
            ids=[int(n.split('__row')[1].split('__')[0]) for n in both]
            expected=set(index[index.class8.eq(folder.name)].row.astype(int))
            actual=set(ids)
            rows.append(dict(class8=folder.name,png=len(names),pairs=len(both),unpaired=len(vv^vh),missing_index_rows=len(expected-actual),unexpected_rows=len(actual-expected)))
            first=sorted(both)[0]
            row=int(first.split('__row')[1].split('__')[0])
            meta=index.iloc[row].to_dict()
            for pol in ['VV','VH']:
                path=folder/(first+'_'+pol+'.png')
                im=np.asarray(Image.open(path))
                samples.append(dict(row=row,row_src=int(meta['row_src']),class_raw=meta['class_raw'],mmsi=int(meta['mmsi']),pol=pol,path=str(path),shape=str(im.shape),std=float(im.std())))
        # One paired pixel replay checks lossless export using the first indexed row.
        meta=index.iloc[0]
        tag='%s__row%06d__%s__d%05.1fkm' % (meta.class8,0,meta.scene[-30:],meta.d_land_km)
        exact={}
        for pol in ['VV','VH']:
            with z.open(f'F0921_{pol}.npy') as f:
                version=np.lib.format.read_magic(f)
                shape,fortran,dtype=np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f)
                if fortran:
                    raise ValueError('Expected row-major stored chips')
                first=np.frombuffer(f.read(224*224*dtype.itemsize),dtype=dtype).reshape(224,224)
                actual=np.asarray(Image.open(PNG/meta.class8/(tag+'_'+pol+'.png')))
                exact[pol]=dict(shape=list(shape),dtype=str(dtype),first_row_exact=bool(np.array_equal(first,actual)))
    pd.DataFrame(rows).to_csv(HERE/'class_counts.csv',index=False)
    pd.DataFrame(samples).to_csv(HERE/'sample_paths.csv',index=False)
    index.groupby('class_raw').agg(rows=('row','size'),mmsi=('mmsi','nunique')).sort_values('rows',ascending=False).to_csv(HERE/'raw_class_counts.csv')
    summary=dict(png_root=str(PNG),metadata_zip=str(ZIP),index_rows=len(index),unique_mmsi=index.mmsi.nunique(),date_utm_groups=index.scene.nunique(),raw_labels=index.class_raw.nunique(),png_files=sum(r['png'] for r in rows),paired_chips=sum(r['pairs'] for r in rows),unpaired=sum(r['unpaired'] for r in rows),missing_index_rows=sum(r['missing_index_rows'] for r in rows),unexpected_rows=sum(r['unexpected_rows'] for r in rows),sequential_index=bool(np.array_equal(index.row,np.arange(len(index)))),pixel_replay=exact,dynamic_directory_currently_exists=Path('E:/Hermes/paper_tracking/f0921_chips_dyn').exists())
    (HERE/'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(samples).head(4).to_string(index=False))


if __name__=='__main__':
    main()
