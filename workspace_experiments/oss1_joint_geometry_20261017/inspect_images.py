"""Outcome-blind native images and detected directions, before fitting readers."""
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from common import HERE,MANIFEST


def main():
    m=pd.read_csv(MANIFEST);d=pd.read_csv(HERE/'detections.csv').set_index('global_row')
    rng=np.random.default_rng(20261017)
    rows=[int(rng.choice(g.index)) for _,g in m.groupby('class_id')]
    pd.DataFrame(dict(global_row=rows)).to_csv(HERE/'visual_selection.csv',index=False)
    for start in (0,4):
        fig,axes=plt.subplots(4,2,figsize=(8,13),layout='constrained')
        for i,row in enumerate(rows[start:start+4]):
            t=d.loc[row];theta=t.ship_axis
            co,si=np.cos(theta),np.sin(theta)
            corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*[t.length/2,t.width/2]
            corners=corners@np.array([[co,si],[-si,co]])+np.array([t.cx,t.cy])
            for j,pol in enumerate(('vv','vh')):
                with Image.open(m.loc[row,'chip_path_'+pol]) as im:
                    x=np.asarray(im.convert('L'))
                axes[i,j].imshow(x,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                axes[i,j].add_patch(Polygon(corners,fill=False,color='lime',lw=1))
                axes[i,j].plot(t.cx+np.array([-1,1])*t.length*co,
                               t.cy+np.array([-1,1])*t.length*si,color='cyan',lw=.8)
                axes[i,j].set_title(f'{row} | {m.loc[row,"class_name"]} | {pol.upper()}\n{x.shape} | axis valid={t.axis_valid}',fontsize=9)
                axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
        fig.suptitle('Native uint8; image-estimated axis, not AIS heading; no outcome selection')
        fig.savefig(HERE/f'native_axes_{start//4+1}.png',dpi=130);plt.close(fig)


if __name__=='__main__':
    main()
