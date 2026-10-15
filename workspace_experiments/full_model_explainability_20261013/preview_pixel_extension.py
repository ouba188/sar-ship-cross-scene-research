"""Show selected regions and check real-pixel intervention controls."""

from pixel_extension import DEST, FOLDER_F, read_images, transform_images, f_region, apply, box_mask
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from run_paths import write


def main():
    f=pd.read_csv(DEST/'f0921/f0921/selected.csv')
    port=pd.read_csv(DEST/'self24ports/Antwerp-Bruges/selected.csv')
    pixel=np.load(FOLDER_F/'inputs128.npy',mmap_mode='r')
    cases=[('F_0921',r) for r in f.groupby('class_id',sort=True).head(1).itertuples()]
    cases += [('Antwerp-Bruges',r) for r in port.iloc[:4].itertuples()]
    fig,axes=plt.subplots(3,4,figsize=(10,8),layout='constrained');checks=0
    for ax,(ds,row) in zip(axes.flat,cases):
        if ds=='F_0921':
            pair=np.asarray(pixel[row.query_index]);a=pair[1]
            region,sham,valid,sv=f_region(row)
            mask=box_mask(a.shape,region)
            edited=apply(pair,mask,'shuffle',20261013+row.query_index)
            np.testing.assert_array_equal(np.sort(pair[:,mask],axis=1),np.sort(edited[:,mask],axis=1))
            np.testing.assert_array_equal(pair[:,~mask],edited[:,~mask]);checks+=2
        else:
            native=read_images(row);a=native[1]
            edited,info=transform_images(native,row,'target_shuffle',row.query_index)
            region=tuple(info[k] for k in ('x0','y0','x1','y1'))
            mask=box_mask(a.shape,region)
            np.testing.assert_array_equal(np.sort(a[mask]),np.sort(edited[1][mask]))
            np.testing.assert_array_equal(a[~mask],edited[1][~mask]);checks+=2
        ax.imshow(a,cmap='gray',vmin=0,vmax=255)
        l,t,r,b=region;ax.add_patch(Rectangle((l-.5,t-.5),r-l,b-t,fill=False,color='#ee4433',lw=1))
        ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(f'{ds}: class {row.class_id}\n{row.sample_id}',fontsize=9)
    fig.suptitle('F: detector-center/length neighborhood; port: actual detection-box neighborhood',fontsize=11)
    fig.savefig(DEST/'REGION_PREVIEW.png',dpi=160);plt.close(fig)
    write(DEST/'PIXEL_CONTROL_CHECK.json',dict(real_images=len(cases),checks=checks,
          preserved='Unedited pixels and within-region gray-value multiset',passed=True))


if __name__=='__main__':main()
