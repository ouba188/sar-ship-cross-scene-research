"""Additional metadata-only summaries and bounded public export inventory."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from audit_local import OUT, WORK, ROOT, SOURCE, NATIVE_ROOTS, read, save, write


def main():
    public = read(OUT / 'public_metadata.csv')
    source = read(SOURCE / 'manifest.csv')
    inventory = read(OUT / 'xml_inventory.csv')
    exports = []
    for row in inventory.itertuples():
        root = NATIVE_ROOTS[row.dataset] / row.product
        names = [p.name for p in root.iterdir()]
        patches = list((root / 'Patch_Uint8').glob('*.tif'))
        counts = {pol: sum(p.stem.lower().endswith('_' + pol) for p in patches)
                  for pol in ['vv', 'vh', 'hh', 'hv']}
        exports.append(dict(dataset=row.dataset, product=row.product, xml_rows=int(row.xml_rows),
                            immediate_children='|'.join(sorted(names)),
                            uint8_files=len(patches), **{pol + '_files': n for pol, n in counts.items()}))
    save('export_inventory', exports)
    save('export_counts', pd.DataFrame(exports).groupby('dataset')[[
        'xml_rows', 'uint8_files', 'vv_files', 'vh_files', 'hh_files', 'hv_files']].sum().reset_index())
    repeated = read(OUT / 'oss2_within_datatake_repeated_rows.csv')
    dup = repeated.loc[repeated.duplicated(['identity', 'product'], keep=False)]
    save('oss2_same_product_repeated_identity', dup)
    source_paths = []
    for pol in ['vv', 'vh']:
        col = 'image_relpath_' + pol
        source_paths.append(dict(pol=pol, rows=len(source), unique_paths=source[col].nunique(),
                                 reused_path_rows=int(source[col].duplicated(keep=False).sum())))
    save('source_path_reuse', source_paths)
    source_meta_root = Path('E:/SAR_AIS_T_shipchip_dataset/v5_context_dynamic_64_128_return_20260915/metadata')
    returned = pd.concat([read(source_meta_root / server / 'manifest.csv') for server in ['server1', 'server2']], ignore_index=True)
    returned['basename'] = returned.context_output_png.map(lambda x: x.replace('\\', '/').rsplit('/', 1)[-1])
    selected = source.loc[source.data_origin.eq('returned_20260915')].copy()
    selected['basename'] = selected.image_relpath_vv.map(lambda x: x.replace('\\', '/').rsplit('/', 1)[-1])
    vv = returned.loc[returned.polarization.eq('VV')]
    selected = selected.merge(vv[['basename', 'position_method', 'abs_time_delta_seconds', 'uncertainty_m',
                                  'geometry_confidence', 'human_review_required', 'redetect_status']],
                              on='basename', how='left', validate='one_to_one', indicator=True)
    if not selected['_merge'].eq('both').all():
        raise ValueError('Returned source metadata join incomplete')
    save('source_returned_metadata', selected[['instance_id', 'class_name', 'mmsi', 'product', 'position_method',
                                             'abs_time_delta_seconds', 'uncertainty_m', 'geometry_confidence',
                                             'human_review_required', 'redetect_status']])
    details = []
    for field in ['position_method', 'abs_time_delta_seconds', 'uncertainty_m', 'geometry_confidence',
                  'human_review_required', 'redetect_status']:
        s = selected[field].fillna('')
        num = pd.to_numeric(s, errors='coerce').dropna()
        details.append(dict(field=field, total=len(s), nonempty=int(s.ne('').sum()),
                            numeric_valid=len(num), median=num.median(), p95=num.quantile(.95), max=num.max()))
    save('source_returned_metadata_summary', details)
    metadata = read(WORK / 'ports_public_data_study_20261012/audit/metadata.csv')
    desc1 = read(WORK / 'ports_public_data_study_20261012/image_diagnostics/oss1.csv')
    sizes = []
    for name, f in [('source24', metadata.loc[metadata.dataset.eq('self24ports')]),
                    ('oss2', metadata.loc[metadata.dataset.eq('oss2')]), ('oss1', desc1)]:
        if name == 'oss1' and 'class_name' not in f:
            f = f.merge(public.loc[public.dataset.eq('oss1'), ['sample_id', 'class_name']], on='sample_id', validate='one_to_one')
        for cls, part in f.groupby('class_name'):
            for col in ['native_width_px_vv', 'native_height_px_vv', 'native_width_px_vh', 'native_height_px_vh']:
                v = pd.to_numeric(part[col], errors='coerce')
                sizes.append(dict(dataset=name, class_name=cls, field=col, n=int(v.notna().sum()),
                                  min=v.min(), median=v.median(), max=v.max()))
    save('native_size_by_class', sizes)
    sensitivity = []
    classes = []
    for name in ['oss1', 'oss2', 'oss2_grdh']:
        origin = 'oss2' if name == 'oss2_grdh' else name
        frame = public.loc[public.dataset.eq(origin)].set_index('sample_id', drop=False)
        other = 'oss1' if origin == 'oss2' else 'oss2'
        other_ids = set(public.loc[public.dataset.eq(other), 'identity'])
        with np.load(WORK / f'public_oss_comparison_20261012/targets/shared_frozen/{name}/predictions.npz') as z:
            frame = frame.loc[z['ids']].reset_index(drop=True)
            for draw, support in enumerate(z['support']):
                query = z['eval_masks'][draw]
                no_other = ~frame.identity.isin(other_ids).to_numpy()
                unseen = ~frame.identity.isin(set(frame.iloc[support].identity)).to_numpy()
                for scope, mask in [('primary', query), ('public_no_other_release_vessel', query & no_other),
                                    ('query_unseen_support_vessel', query & unseen),
                                    ('both_identity_exclusions', query & unseen & no_other)]:
                    part = frame.loc[mask]
                    sensitivity.append(dict(dataset=name, draw=draw, scope=scope, query_rows=len(part),
                                             classes=part.class_name.nunique(), identities=part.identity.nunique()))
                    for cls, n in part.class_name.value_counts().items():
                        classes.append(dict(dataset=name, draw=draw, scope=scope, class_name=cls, rows=n))
    save('identity_sensitivity_cohorts', sensitivity)
    save('identity_sensitivity_class_counts', classes)
    probe = read(OUT / 'image_probes.csv')
    record = dict(public_xml_files=len(inventory),
                  xml_utf8_replacement_tags=sorted(set(inventory.replacement_tags) - {''}),
                  xml_replacement_files=int(pd.to_numeric(inventory.utf8_replacement_characters).gt(0).sum()),
                  tiff_png_probes=len(probe.loc[probe.dataset.eq('oss2')]),
                  all_probed_tiff_png_equal=bool(probe.loc[probe.dataset.eq('oss2'), 'tiff_png_equal'].eq('True').all()),
                  returned_rows_joined=len(selected), source_paths_unique=all(x['reused_path_rows'] == 0 for x in source_paths),
                  note='No new pixel hashes, model fits, GPU calls, or changed evaluation cohorts')
    write('DETAILS_COMPLETE.json', record)
    print(json.dumps(record, indent=2))
    print(pd.DataFrame(exports).groupby('dataset')[['xml_rows', 'uint8_files', 'vv_files', 'vh_files']].sum().to_string())


if __name__ == '__main__':
    main()
