"""Inspect existing source float patches and their display PNG correspondence."""
import sys
sys.dont_write_bytecode = True
import json
import time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from osgeo import gdal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    tick = time.perf_counter()
    gdal.UseExceptions()
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    with np.load(ROOT / 'pair_local_evidence/source_moments.npz') as z:
        ids, tr, va = [z[k] for k in ['sample_ids', 'train', 'validation']]
    cohort = pd.read_csv(ROOT / 'cohort.csv', dtype={'mmsi': str}).set_index('sample_id').loc[ids]
    metadata, rows, products = {}, [], []
    for i, (sid, row) in enumerate(cohort.iterrows()):
        src = Path(row.source_path_vv)
        product = src.parent.parent
        if product not in metadata:
            meta_path = product / 'Metadata.xml'
            if not meta_path.exists():
                candidates = [p for p in product.glob('*.xml') if p.name.lower() != 'ship.xml']
                assert len(candidates) == 1, (product, candidates)
                meta_path = candidates[0]
            attrs = {x.attrib['name']: x.attrib.get('value')
                     for x in ET.parse(meta_path).iter('attrib')}
            order = [attrs.get(f'mds{k}_tx_rx_polar') for k in [1, 2]]
            metadata[product] = order
            products.append(dict(product=str(product), metadata_path=str(meta_path), band_order=','.join(order),
                                 source_abs_calibration_flag=attrs.get('abs_calibration_flag'),
                                 note='Product metadata is not a band-unit declaration for Patch_Cal'))
        order = metadata[product]
        name = src.stem.removeprefix('Visual_').removesuffix('_vv') + '.tif'
        path = product / 'Patch_Cal' / name
        common = dict(sample_id=sid, source_row=i, role='train' if i in tr else 'validation',
                      class_id=int(row.class_id), mmsi=row.mmsi, pass_group=row.pass_group,
                      calibrated_path=str(path), exists=path.exists())
        if not path.exists():
            rows.append(common)
            continue
        ds = gdal.Open(str(path))
        a = ds.ReadAsArray()
        assert a.ndim == 3 and a.shape[0] == 2
        assert set(order) == {'VV', 'VH'}
        for pol in ['VV', 'VH']:
            band = order.index(pol)
            v = a[band].astype(np.float64)
            pngpath = Path(row[f'chip_path_{pol.lower()}'])
            with Image.open(pngpath) as im:
                p = np.array(im)
            assert v.shape == p.shape
            finite = np.isfinite(v)
            record = dict(common, pol=pol, band=band+1, dtype=str(a.dtype),
                          height=v.shape[0], width=v.shape[1], png_path=str(pngpath),
                          unit=ds.GetRasterBand(band+1).GetUnitType(),
                          description=ds.GetRasterBand(band+1).GetDescription(),
                          finite=bool(finite.all()), negative=int((v[finite] < 0).sum()))
            if not finite.all() or np.min(v) < 0 or np.max(v) <= 0:
                rows.append(record)
                continue
            # Both formulas are fixed diagnostics, never fitted display reconstructions.
            amplitude_display = 255 * np.sqrt(v / v.max())
            intensity_display = 255 * v / v.max()
            amp = np.sqrt(v)
            amplitude_minmax = 255 * (amp - amp.min()) / (amp.max() - amp.min())
            usable = (p > 0) & (p < 255) & (amp > 0)
            slope = float(np.sum(amp[usable]*p[usable]) / np.sum(amp[usable]**2))
            fitted = np.minimum(255, slope*amp)
            other = a[1-band]
            record.update(spearman_correct=float(spearmanr(v.ravel(), p.ravel()).statistic),
                          spearman_other=float(spearmanr(other.ravel(), p.ravel()).statistic),
                          amplitude_maxnorm_mae=float(np.mean(abs(amplitude_display-p))),
                          amplitude_minmax_mae=float(np.mean(abs(amplitude_minmax-p))),
                          minmax_round_equal=float(np.mean(np.rint(amplitude_minmax)==p)),
                          float_min=float(v.min()),
                          amplitude_to_display_slope=slope,
                          amplitude_anchor_255=255/slope,
                          fitted_display_mae=float(np.mean(abs(fitted-p))),
                          fitted_round_equal=float(np.mean(np.rint(fitted)==p)),
                          intensity_maxnorm_mae=float(np.mean(abs(intensity_display-p))),
                          maxnorm_round_equal=float(np.mean(np.rint(amplitude_display)==p)),
                          png_saturated_fraction=float(np.mean(p==255)),
                          float_mean=float(v.mean()), float_sum=float(v.sum()))
            for quantile in [.01, .5, .9, .99, 1.]:
                record[f'float_q{quantile:g}'] = float(np.quantile(v, quantile))
                record[f'png_q{quantile:g}'] = float(np.quantile(p, quantile))
            rows.append(record)
        ds = None
        if (i+1) % 300 == 0:
            print('source float inspection', i+1, '/', len(ids), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / 'source_correspondence.csv', index=False)
    pd.DataFrame(products).to_csv(HERE / 'product_metadata.csv', index=False)
    good = df[df.exists & df.finite.fillna(False)].copy()
    group = good.groupby(['role', 'pol']).agg(
        n=('sample_id', 'size'), rank_median=('spearman_correct', 'median'),
        cross_rank_median=('spearman_other', 'median'),
        amplitude_mae_median=('amplitude_maxnorm_mae', 'median'),
        intensity_mae_median=('intensity_maxnorm_mae', 'median'),
        float_max_min=('float_q1', 'min'), float_max_median=('float_q1', 'median'),
        float_max_max=('float_q1', 'max'))
    group.to_csv(HERE / 'summary_by_role.csv')
    info = dict(source_rows=len(ids), train=len(tr), validation=len(va),
                existing_objects=int(df[df.exists].sample_id.nunique()),
                finite_band_rows=len(good), negative_band_rows=int((good.negative>0).sum()),
                products=len(products), unit_tags=sorted(good.unit.unique().tolist()),
                descriptions=sorted(good.description.unique().tolist()),
                test_used=False, trained=False, seconds=time.perf_counter()-tick)
    (HERE / 'summary.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info), flush=True)
    print(group.to_string(), flush=True)


if __name__ == '__main__':
    main()
