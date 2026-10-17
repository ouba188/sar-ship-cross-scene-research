"""Check the plotted masks against original crop records, not guessed boxes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'geometry_recovery_effect_20261016/visual_diagnosis'
m=pd.read_csv(ROOT/'geometry_source_recovery_20261016/geometry_input_manifest.csv',low_memory=False)
r=pd.read_csv(ROOT/'region_cue_replication_20261013/ports/frame.csv').set_index('sample_id')
s=pd.read_csv('E:/SAR_AIS_T_shipchip_dataset/v5_context_dynamic_64_128_return_20260915/metadata/server2/manifest.csv',low_memory=False)
checks=[]
for grow in (24846,35475):
    row=m.iloc[grow]; frame=r.loc[row.instance_id]
    for pol in ('vv','vh'):
        source=s.loc[(s.obj_uid==row['obj_uid_'+pol])&(s.polarization.str.lower()==pol)]
        assert len(source)==1
        source=source.iloc[0]
        actual=np.array([source.left-source.crop_left,source.top-source.crop_top,
                         source.right-source.crop_left,source.bottom-source.crop_top],float)
        expected=frame[[f'bbox_{c}_{pol}' for c in ('x0','y0','x1','y1')]].to_numpy(float)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-8)
        with Image.open(row['image_relpath_'+pol]) as im:
            image=np.asarray(im.convert('L'))
            assert im.size==(int(frame['native_w_'+pol]),int(frame['native_h_'+pol]))
        yy,xx=np.indices(image.shape); x0,y0,x1,y1=actual
        mask=(xx+.5>=x0)&(xx+.5<x1)&(yy+.5>=y0)&(yy+.5<y1)
        bright=image>=200
        checks.append(dict(global_row=grow,pol=pol,box=actual.tolist(),native_shape=list(image.shape),
            source_crop_record_matches=True,display_threshold=200,
            bright_pixels=int(bright.sum()),bright_outside_bbox=int((bright&~mask).sum())))
(OUT/'region_source_check.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
