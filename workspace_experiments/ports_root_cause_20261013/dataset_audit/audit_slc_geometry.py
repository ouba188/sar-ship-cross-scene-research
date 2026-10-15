"""Check native SLC grid provenance without encoding or fitting models."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import tifffile
from lxml import etree as ET
from scipy.stats import spearmanr
from audit_local import OUT, NATIVE_ROOTS, read, save, write


def main():
    allpublic = read(OUT / 'public_metadata.csv')
    frame = allpublic.loc[allpublic.dataset.eq('oss2') & allpublic.product_type.eq('SLC')].copy()
    metadata = read(OUT / 'product_spacing_metadata.csv')
    metadata = metadata.loc[metadata.dataset.eq('oss2')].set_index('product')
    records, probes = [], []
    for product, group in frame.groupby('product'):
        root = NATIVE_ROOTS['oss2'] / product
        content = (root / 'Ship.xml').read_bytes().decode('utf-8', errors='replace').encode('utf-8')
        xml = ET.fromstring(content, ET.XMLParser(recover=True))
        ships = {(s.findtext('SARShipInformation/Center_x'), s.findtext('SARShipInformation/Center_y')): s
                 for s in xml.findall('.//ship')}
        meta = metadata.loc[product]
        for ordinal, row in enumerate(group.itertuples()):
            path = Path(row.source_path_vv)
            native_name = path.name.removeprefix('Visual_').removesuffix('_vv.tif') + '.tif'
            raw_path, cal_path = root / 'Patch' / native_name, root / 'Patch_Cal' / native_name
            rec = dict(sample_id=row.sample_id, product=product, class_name=row.class_name,
                       source_path_vv=row.source_path_vv, source_path_vh=row.source_path_vh,
                       chip_path_vv=row.chip_path_vv, chip_path_vh=row.chip_path_vh,
                       raw_path=str(raw_path), cal_path=str(cal_path),
                       range_spacing=float(meta.range_spacing), azimuth_spacing=float(meta.azimuth_spacing),
                       incidence_deg=float(row.incidence_xml), srgr_flag=meta.srgr_flag,
                       map_projection=meta.map_projection, terrain_corrected=meta.is_terrain_corrected)
            for prefix, location in [('uint8_vv', path), ('uint8_vh', Path(row.source_path_vh)),
                                     ('raw', raw_path), ('cal', cal_path)]:
                rec[prefix + '_exists'] = location.is_file()
                if location.is_file():
                    with tifffile.TiffFile(location) as tif:
                        page = tif.pages[0]
                        rec.update({prefix + '_width': page.imagewidth, prefix + '_height': page.imagelength,
                                    prefix + '_dtype': str(page.dtype), prefix + '_shape': str(page.shape),
                                    prefix + '_orientation': str(page.tags.get('Orientation', 'absent'))})
            ship = ships[(row.center_x, row.center_y)]
            get = lambda name: float(ship.findtext('SARShipInformation/' + name))
            expected_w = abs(get('UpperLeft_x') - get('LowerRight_x')) + 1
            expected_h = abs(get('UpperLeft_y') - get('LowerRight_y')) + 1
            rec.update(xml_width=expected_w, xml_height=expected_h,
                       uint8_matches_xml=rec['uint8_vv_width'] == expected_w and rec['uint8_vv_height'] == expected_h,
                       raw_uint8_same_shape=rec.get('raw_width') == rec['uint8_vv_width'] and rec.get('raw_height') == rec['uint8_vv_height'],
                       cal_uint8_same_shape=rec.get('cal_width') == rec['uint8_vv_width'] and rec.get('cal_height') == rec['uint8_vv_height'],
                       vv_vh_same_shape=rec['uint8_vv_width'] == rec['uint8_vh_width'] and rec['uint8_vv_height'] == rec['uint8_vh_height'])
            rec['ground_range_spacing_m'] = rec['range_spacing'] / np.sin(np.radians(rec['incidence_deg']))
            rec['range_compress_ratio'] = rec['ground_range_spacing_m'] / rec['azimuth_spacing']
            rec['native_pixel_aspect_ratio'] = rec['uint8_vv_width'] / rec['uint8_vv_height']
            rec['actual_aspect_ratio'] = rec['native_pixel_aspect_ratio'] * rec['range_compress_ratio']
            rec['actual_aspect_ratio_basis'] = 'local_ground_width_over_azimuth_height_from_metadata'
            records.append(rec)
            # One deterministic chip per product: check spatial alignment, not only header dimensions.
            if ordinal == 0 and raw_path.is_file():
                raw = tifffile.imread(raw_path)
                for pol, i, j in [('vh', 0, 1), ('vv', 2, 3)]:
                    image = tifffile.imread(Path(getattr(row, 'source_path_' + pol)))
                    amplitude = np.hypot(raw[..., i], raw[..., j])
                    corr = float(spearmanr(amplitude.ravel(), image.ravel()).statistic)
                    order = np.argsort(amplitude.ravel(), kind='stable')
                    delta = np.diff(image.ravel()[order].astype(int))
                    probes.append(dict(sample_id=row.sample_id, product=product, pol=pol,
                                       raw_shape=str(raw.shape), uint8_shape=str(image.shape),
                                       same_grid_amplitude_uint8_spearman=corr,
                                       amplitude_sorted_uint8_decreases=int((delta < 0).sum()),
                                       compared_adjacencies=len(delta),
                                       transposed_amplitude_spearman=float(spearmanr(amplitude.T.ravel(), image.ravel()).statistic)))
    save('oss2_slc_native_grid', records)
    save('oss2_slc_pixel_grid_probes', probes)
    f = pd.DataFrame(records)
    receipt = dict(rows=len(f), products=f['product'].nunique(),
                   raw_present=int(f.raw_exists.sum()), cal_present=int(f.cal_exists.sum()),
                   xml_shape_matches=int(f.uint8_matches_xml.sum()), raw_shape_matches=int(f.raw_uint8_same_shape.sum()),
                   cal_shape_matches=int(f.cal_uint8_same_shape.sum()), vv_vh_matches=int(f.vv_vh_same_shape.sum()),
                   srgr_flags=sorted(f.srgr_flag.unique()), terrain_corrected=sorted(f.terrain_corrected.unique()),
                   range_spacing_min=float(f.range_spacing.min()), range_spacing_max=float(f.range_spacing.max()),
                   azimuth_spacing_min=float(f.azimuth_spacing.min()), azimuth_spacing_max=float(f.azimuth_spacing.max()),
                   compression_min=float(f.range_compress_ratio.min()), compression_median=float(f.range_compress_ratio.median()),
                   compression_max=float(f.range_compress_ratio.max()), pixel_probes=len(probes),
                   pixel_probe_rank_min=float(pd.DataFrame(probes).same_grid_amplitude_uint8_spearman.min()),
                   pixel_probe_sorted_decreases=int(pd.DataFrame(probes).amplitude_sorted_uint8_decreases.sum()),
                   no_model_fits=True, no_encoding=True)
    write('SLC_GEOMETRY_COMPLETE.json', receipt)
    print(receipt)


if __name__ == '__main__':
    main()
