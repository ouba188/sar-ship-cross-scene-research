"""Record source-label/AIS-description disagreements without relabeling anything."""

import json
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE/'knowledge_train_only_20261011'
AUDIT = BASE/'knowledge_train_only_design_20261011'


def main():
    mask = pd.read_csv(OUT/'attribute_mask.csv')
    source = mask[mask.split.eq('fit') & mask.attribute_valid].drop_duplicates('mmsi')
    ais = pd.read_csv('E:/Hermes/paper_tracking/port_ais_2024_2025_from_zip/dedup_tables/vessels_to_lookup_dedup.csv',
        usecols=['mmsi', 'name_mode', 'ais_type_code_mode', 'ais_type_description'])
    joined = source[['obs_id', 'mmsi', 'class_name', 'raw_exact_length', 'raw_exact_beam']].merge(
        ais, on='mmsi', how='left', validate='one_to_one')
    joined.to_csv(OUT/'source_label_ais_descriptions.csv', index=False)
    cross = pd.crosstab(joined.class_name, joined.ais_type_description)
    cross.to_csv(OUT/'source_label_ais_crosstab.csv')
    examples = []
    objects = pd.read_csv(AUDIT/'object_dimension_audit.csv')
    rows = objects[objects.mmsi.isin([538005751, 538005871, 341152000]) & objects.pol.eq('vv')]
    for row in rows.itertuples():
        payload = json.loads(Path(row.local_json).read_text(encoding='utf-8'))
        obj = [o for o in payload['objects'] if str(o.get('obj_uid') or o.get('source_obj_uid') or '') == row.uid]
        assert len(obj) == 1
        m = obj[0]['meta']
        assert str(m['mmsi']) == str(row.mmsi)
        sample = joined[joined.mmsi.eq(row.mmsi)].iloc[0]
        assert float(m['length_m']) == sample.raw_exact_length
        assert float(m['beam_m']) == sample.raw_exact_beam
        examples.append(dict(obs_id=int(row.obs_id), uid=row.uid, local_json=row.local_json,
            manifest_class=sample.class_name, json_meta=m, ais_description=sample.ais_type_description,
            ais_type_code=int(sample.ais_type_code_mode)))
    (OUT/'source_semantic_examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding='utf-8')
    print(cross.to_string())
    print('Examples match UID, identity and dimensions; no labels or training masks changed.')


if __name__ == '__main__':
    main()
