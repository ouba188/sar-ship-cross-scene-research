"""Show actual polygon memberships next to unmodified native SAR displays."""
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from PIL import Image

from analyze_conditions import OUT, OLD, ROOT, polygon_mask


def main():
    selected=pd.concat([pd.read_csv(OUT/'selected_cases.csv'),pd.read_csv(OUT/'crude_selected_cases.csv')]).drop_duplicates('pair_id')
    views=pd.read_csv(OLD/'data_review/main_pair_views.csv').set_index(['obs_id','pol'])
    ids=set(selected.obs_id_a)|set(selected.obs_id_b)
    snap={x['obs_id']:x for x in json.loads((ROOT/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8')) if x['obs_id'] in ids}
    stats=pd.read_csv(OUT/'image_statistics.csv').set_index(['obs_id','pol'])
    for p in selected.itertuples():
        fig,axs=plt.subplots(2,4,figsize=(14,7))
        fig.suptitle(f'{p.pair_id}: native pixels and exact statistical regions\n'
            'Green=target OBB; orange=other usable OBB; gray=remaining pixels (not verified sea only)')
        for row,oid in enumerate((p.obs_id_a,p.obs_id_b)):
            for col,pol in enumerate(('vv','vh')):
                v=views.loc[(oid,pol)];pr=snap[oid]['images'][pol]['priors']
                with Image.open(v.image_path) as im:raw=np.asarray(im.convert('L'))
                target=polygon_mask((v.width,v.height),pr['detections'][pr['target_index']]['corners'])
                other=np.zeros_like(target)
                for j,d in enumerate(pr['detections']):
                    if j!=pr['target_index'] and d.get('shape_usable') and d['confidence']>=.25:
                        other|=polygon_mask((v.width,v.height),d['corners'])
                region=np.zeros_like(raw);region[other]=2;region[target]=1
                assert target.sum()==stats.loc[(oid,pol),'target_pixels']
                assert (~(target|other)).sum()==stats.loc[(oid,pol),'rest_pixels']
                ax=axs[row,2*col];ax.imshow(raw,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                ax.set_title(f'{oid} {pol.upper()} original {v.width}x{v.height}',fontsize=9)
                ax=axs[row,2*col+1];ax.imshow(region,cmap=ListedColormap(['#777777','#20a66b','#dc9030']),vmin=0,vmax=2,interpolation='nearest')
                ax.set_title(f'T={target.sum()}px, R={(~(target|other)).sum()}px',fontsize=9)
        fig.tight_layout(rect=(0,0,1,.92));fig.savefig(OUT/f'regions_{p.pair_id}.png',dpi=120);plt.close(fig)
    print(f'{len(selected)} paired region panels, pixel counts match stored statistics')


if __name__=='__main__':
    main()
