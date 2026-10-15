"""Unambiguous paired-product metadata; no predictions or model fits."""
import numpy as np
import pandas as pd
from lxml import etree as ET
from audit_local import OUT, WORK, NATIVE_ROOTS, read, save, write


def main():
    public = read(OUT / 'public_metadata.csv')
    f = public.loc[public.dataset.eq('oss2')].copy()
    pairs, ambiguous = [], []
    fields = ['sample_id', 'row_id', 'target_manifest_row', 'product', 'class_id', 'class_name',
              'sar_datetime', 'ais_epoch', 'center_x', 'center_y', 'sar_lon', 'sar_lat',
              'manual_length', 'ais_length', 'ais_width', 'noise_xml', 'multi_ship_xml', 'port_noise_xml']
    for (identity, datatake), group in f.groupby(['identity', 'datatake_key']):
        counts = group.product_type.value_counts()
        ng, ns = int(counts.get('GRDH', 0)), int(counts.get('SLC', 0))
        if ng == 1 and ns == 1:
            g, s = group.loc[group.product_type.eq('GRDH')].iloc[0], group.loc[group.product_type.eq('SLC')].iloc[0]
            pair = dict(identity=identity, datatake_key=datatake,
                        same_sar_datetime=g.sar_datetime == s.sar_datetime,
                        same_label=g.class_id == s.class_id, same_ais_epoch=g.ais_epoch == s.ais_epoch)
            pair.update({key + '_grdh': g[key] for key in fields})
            pair.update({key + '_slc': s[key] for key in fields})
            pairs.append(pair)
        elif ng > 1 or ns > 1:
            ambiguous.append(dict(identity=identity, datatake_key=datatake, grdh_rows=ng, slc_rows=ns,
                                  sample_ids='|'.join(group.sample_id), reason='nonunique_product_type_or_product'))
    allpairs = pd.DataFrame(pairs)
    exact = allpairs.loc[allpairs.same_sar_datetime].copy()
    save('oss2_grdh_slc_unique_datatake_pairs', allpairs)
    save('oss2_grdh_slc_pairs', exact)
    save('oss2_grdh_slc_ambiguous_groups', ambiguous)
    eligibility = []
    with np.load(WORK / 'public_oss_comparison_20261012/targets/shared_frozen/oss2/predictions.npz') as z:
        lookup = {sample: i for i, sample in enumerate(z['ids'])}
        for draw, query in enumerate(z['eval_masks']):
            for row in exact.itertuples():
                eligibility.append(dict(draw=draw, sample_id_grdh=row.sample_id_grdh, sample_id_slc=row.sample_id_slc,
                                        both_query=bool(query[lookup[row.sample_id_grdh]] and query[lookup[row.sample_id_slc]]),
                                        same_label=row.same_label))
    save('oss2_grdh_slc_pair_query_eligibility', eligibility)
    metadata = []
    inv = read(OUT / 'xml_inventory.csv')
    for row in inv.itertuples():
        root = NATIVE_ROOTS[row.dataset] / row.product
        paths = [root / name for name in ['Metedata.xml', 'Metadata.xml', row.product + '.xml'] if (root / name).exists()]
        record = dict(dataset=row.dataset, product=row.product, metadata_path='', metadata_present=bool(paths))
        if paths:
            record['metadata_path'] = str(paths[0])
            tree = ET.parse(str(paths[0]), ET.XMLParser(recover=True))
            for key in ['range_spacing', 'azimuth_spacing', 'SAMPLE_TYPE', 'srgr_flag', 'MISSION', 'ACQUISITION_MODE',
                        'map_projection', 'is_terrain_corrected']:
                nodes = tree.findall(f'.//attrib[@name="{key}"]')
                record[key] = '|'.join(sorted({e.get('value', '') for e in nodes}))
        metadata.append(record)
    save('product_spacing_metadata', metadata)
    meta = pd.DataFrame(metadata).fillna('')
    meta['product_type'] = meta['product'].str.extract(r'_(GRDH|SLC)_', expand=False)
    stats = []
    for (dataset, kind), group in meta.groupby(['dataset', 'product_type']):
        result = dict(dataset=dataset, product_type=kind, products=len(group),
                      metadata_products=int(group.metadata_present.sum()))
        for key in ['range_spacing', 'azimuth_spacing']:
            v = pd.to_numeric(group[key], errors='coerce').dropna()
            result.update({key + '_valid': len(v), key + '_min': v.min(), key + '_median': v.median(), key + '_max': v.max()})
        stats.append(result)
    save('product_spacing_summary', stats)
    receipt = dict(unique_datatake_pairs=len(allpairs), exact_sar_time_pairs=len(exact),
                   exact_sar_time_same_label=int(exact.same_label.sum()),
                   unique_datatake_same_label=int(allpairs.same_label.sum()),
                   ambiguous_groups=len(ambiguous),
                   both_query_by_draw=pd.DataFrame(eligibility).groupby('draw').both_query.sum().to_dict(),
                   primary_pair_file='oss2_grdh_slc_pairs.csv',
                   rule='One GRDH and one SLC row per identity+datatake; exact SAR datetime for primary; retain label mismatch flag',
                   use='Use identical full-OSS2 task support/model; select both-query pairs by draw. No independent product-causal claim.')
    write('PAIRS_COMPLETE.json', receipt)
    print(receipt)
    print(pd.DataFrame(stats).to_string(index=False))


if __name__ == '__main__':
    main()
