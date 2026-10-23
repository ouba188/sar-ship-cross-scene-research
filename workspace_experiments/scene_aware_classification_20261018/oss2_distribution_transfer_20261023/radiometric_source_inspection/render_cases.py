"""Native pixels, shared float display, and fixed normalization reconstruction."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
from osgeo import gdal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    gdal.UseExceptions()
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    df = pd.read_csv(HERE / 'source_correspondence.csv')
    cohort = pd.read_csv(ROOT / 'cohort.csv').set_index('sample_id')
    with np.load(ROOT / 'calibrated/fold0_seed17.npz') as z:
        score = pd.DataFrame(z['base'], index=z['sample_ids'])
        train_ids = df.loc[df.role.eq('train'), 'sample_id'].unique()
        training = pd.DataFrame(z['train_base'], index=z['sample_ids'])
        score.loc[train_ids] = training.loc[train_ids]
    limits = {}
    for pol, g in df[df.role.eq('train')].groupby('pol'):
        low = g['float_q0.01'].to_numpy()
        high = g['float_q0.99'].to_numpy()
        limits[pol] = [float(10*np.log10(np.quantile(low[low>0], .01))),
                       float(10*np.log10(np.quantile(high, .99)))]
    # Four previously inspected source-validation examples plus the worst reconstruction.
    ids = ['pub2_opensarship2_'+s for s in ['000556','000557','000560','000569']]
    ids.append(df.sort_values('amplitude_maxnorm_mae', ascending=False).iloc[0].sample_id)
    selected = []
    (HERE / 'panels').mkdir(exist_ok=True)
    for sid in ids:
        records = df[df.sample_id.eq(sid)].set_index('pol')
        a = gdal.Open(records.iloc[0].calibrated_path).ReadAsArray()
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
        for p, pol in enumerate(['VV','VH']):
            r = records.loc[pol]
            v = a[int(r.band)-1].astype(float)
            with Image.open(r.png_path) as im:
                png = np.array(im)
            log = np.full(v.shape, np.nan)
            log[v>0] = 10*np.log10(v[v>0])
            rec = 255*np.sqrt(v/v.max())
            for col, val, title in [(0,png,'Current display PNG'),
                                    (1,log,'Stored float: shared log scale'),
                                    (2,rec,'255 sqrt(float / max)'),
                                    (3,np.minimum(255, r.amplitude_to_display_slope*np.sqrt(v)),
                                     'Fitted amplitude scale + clipping')]:
                lo, hi = limits[pol] if col==1 else (0,255)
                im = axes[p,col].imshow(val, cmap='gray', vmin=lo, vmax=hi,
                                       interpolation='nearest')
                axes[p,col].set_title(f'{pol}: {title}', fontsize=10)
                axes[p,col].set_xticks([]); axes[p,col].set_yticks([])
                fig.colorbar(im, ax=axes[p,col], fraction=.04)
            axes[p,0].set_xlabel(f'Native {v.shape[1]}x{v.shape[0]}; no resizing')
            axes[p,1].set_xlabel(f'max={v.max():.5g}; median={np.median(v):.5g}; unit unverified')
            axes[p,2].set_xlabel(f'MAE={r.amplitude_maxnorm_mae:.3f}; rounded match={r.maxnorm_round_equal:.3%}')
            axes[p,3].set_xlabel(f'MAE={r.fitted_display_mae:.3f}; rounded match={r.fitted_round_equal:.3%}')
            selected.append(dict(sample_id=sid, pol=pol, role=r.role, class_id=int(r.class_id),
                                 base17_prediction=int(score.loc[sid].to_numpy().argmax()),
                                 float_max=float(v.max()), float_median=float(np.median(v)),
                                 png_median=float(np.median(png)), native_height=v.shape[0],
                                 native_width=v.shape[1], png_path=r.png_path,
                                 calibrated_path=r.calibrated_path))
        y = int(cohort.loc[sid].class_id)
        pred = int(score.loc[sid].to_numpy().argmax())
        fig.suptitle(f'{sid}: class={y}, existing B17={pred}; {records.iloc[0].role}\n'
                     'Float limits from source training; this is input inspection, NOT a new prediction', fontsize=11)
        fig.savefig(HERE / 'panels' / f'{sid}.png', dpi=120)
        plt.close(fig)
    pd.DataFrame(selected).to_csv(HERE / 'review_cases.csv', index=False)
    (HERE / 'display_limits.json').write_text(json.dumps(limits, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
