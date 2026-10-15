"""Augment existing audit outputs only; no new input or image exploration."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent


def main():
    path = OUT / 'oss2_slc_native_grid.csv'
    grid = pd.read_csv(path, keep_default_na=False)
    public = pd.read_csv(OUT / 'public_metadata.csv', keep_default_na=False, dtype=str)
    public = public.loc[public.dataset.eq('oss2')].set_index('sample_id')
    paths = ['source_path_vv', 'source_path_vh', 'chip_path_vv', 'chip_path_vh']
    for column in paths:
        grid[column] = public.loc[grid.sample_id, column].to_numpy()
    for kind, folder in [('raw_path', 'Patch'), ('cal_path', 'Patch_Cal')]:
        grid[kind] = grid.source_path_vv.map(lambda s: str(
            Path(s).parent.parent / folder /
            (Path(s).name.removeprefix('Visual_').removesuffix('_vv.tif') + '.tif')))
    grid['native_pixel_aspect_ratio'] = grid.uint8_vv_width / grid.uint8_vv_height
    grid['actual_aspect_ratio'] = grid.native_pixel_aspect_ratio * grid.range_compress_ratio
    grid['actual_aspect_ratio_basis'] = 'local_ground_width_over_azimuth_height_from_metadata'
    assert len(grid) == 999 and grid.sample_id.is_unique
    assert np.isfinite(grid.actual_aspect_ratio).all()
    assert grid[paths].ne('').all().all()
    grid.to_csv(path, index=False, encoding='utf-8-sig')
    receipt_path = OUT / 'FINAL_RECEIPT.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    receipt['main_branch_status'] = 'Bounded two-arm SLC geometry vs equal-area axis-sham authorized; supersedes earlier temporary closing; outcomes not audited here'
    receipt['primary_R_T_DD_requires_mapper'] = False
    receipt['native_grid_export'] = {
        'file': path.name, 'rows': len(grid),
        'path_columns': paths + ['raw_path', 'cal_path'],
        'ratio_columns': ['range_compress_ratio', 'native_pixel_aspect_ratio', 'actual_aspect_ratio'],
        'actual_aspect_ratio_definition': 'native_width*ground_range_spacing/(native_height*azimuth_spacing)',
        'basis': 'local planar metadata approximation, not measured ship-body aspect ratio'
    }
    receipt['execution']['owned_processes_running'] = 0
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Final report receipt and 999-row native-grid path/aspect export complete.')


if __name__ == '__main__':
    main()
