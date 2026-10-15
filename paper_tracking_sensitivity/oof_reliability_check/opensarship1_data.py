"""Join native OSS1 XML annotations to paired chips by scene and pixel center."""

from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from opensarship2_labels import map_opensarship2_labels, summarize_mapping
from run_opensarship2_transfer import write_json

ROOT = Path('D:/Documents/project_root/OpenSARShip_1')


def read_native(root=ROOT):
    records = []
    for xml in sorted(root.glob('*/Ship.xml')):
        scene = xml.parent.name
        images = {}
        for path in (xml.parent/'Patch_Uint8').glob('*_vv.tif'):
            match = re.search(r'_x(\d+)_y(\d+)_vv\.tif$', path.name)
            key = (int(match[1]), int(match[2]))
            if key in images:
                raise ValueError(f'nonunique chip center: {scene}/{key}')
            images[key] = path
        for ordinal, ship in enumerate(ET.parse(xml).getroot().findall('ship')):
            x = int(ship.findtext('SARShipInformation/Center_x'))
            y = int(ship.findtext('SARShipInformation/Center_y'))
            vv = images.get((x, y))
            vh = vv.with_name(vv.name.removesuffix('_vv.tif')+'_vh.tif') if vv else None
            available = vv is not None and vh.exists()
            records.append(dict(sample_id=f'{scene}:x{x}:y{y}', scene_id=scene,
                xml_ship_index=ordinal, product_type=scene.split('_')[2], center_x=x, center_y=y,
                mmsi_or_ship_id=ship.findtext('AISShipInformation/MMSI', '').strip(),
                label_elaborated=ship.findtext('MarineTrafficInformation/Elaborated_type', '').strip(),
                ais_type=ship.findtext('AISShipInformation/Ship_Type', '').strip(),
                chip_path_vv=str(vv) if vv else '', chip_path_vh=str(vh) if vh else '',
                image_available=available))
    result = pd.DataFrame(records)
    if result.sample_id.duplicated().any():
        raise ValueError('duplicate annotation center; explicit identity resolution required')
    return result


def prepare(out):
    native = read_native()
    mapped = map_opensarship2_labels(native)
    mapped.insert(0, 'target_manifest_row', np.arange(len(mapped)))
    mapped.to_csv(out/'native_label_mapping.csv', index=False)
    grd = mapped.loc[mapped.product_type == 'GRDH']
    selected = grd.loc[grd.label_included]
    if not selected.image_available.all():
        raise ValueError('compatible GRD sample is missing a polarization; do not silently remove')
    selected.to_csv(out/'target_manifest.csv', index=False)
    write_json(out/'label_mapping_summary.json', dict(native_rows=len(native),
        native_products=native.scene_id.nunique(), products_by_type=native.groupby('product_type').scene_id.nunique().to_dict(),
        grd_mapping=summarize_mapping(grd), target_rows=len(selected),
        protocol='GRDH primary cohort; SLC companion products excluded from evaluation, included in source identity purge',
        join='XML Center_x/Center_y + scene to paired Patch_Uint8; never file list position',
        no_quality_filter=True, no_coarse_to_fine_inference=True))
    return native
