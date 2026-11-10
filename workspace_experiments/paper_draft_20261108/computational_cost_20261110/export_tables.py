"""Export the main table with isolated online memory and verify receipt counts."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
summary = pd.read_csv(ROOT / 'summary.csv')
memory = pd.read_csv(ROOT / 'online_memory.csv')
selected = summary[summary.arm.isin(['plain', 'se', 'full'])].copy()
selected = selected.drop(columns=['peak_allocated_mib', 'incremental_peak_mib'])
selected = selected.merge(memory[memory.batch.eq(1)][['classes', 'arm', 'resident_allocated_mib',
                          'peak_allocated_mib', 'incremental_peak_mib']], on=['classes', 'arm'], validate='one_to_one')
selected.to_csv(ROOT / 'paper_cost_table.csv', index=False)
phases = []
max_error = 0
for row in summary.itertuples():
    record = json.loads((ROOT / f'{row.classes}class_{row.arm}.json').read_text())
    if 'frozen_descriptor_parameters' in record:
        record['frozen_descriptor_conv_parameters'] = record.pop('frozen_descriptor_parameters')
        record['frozen_descriptor_bn_parameters'] = 1024 if record['references'] else 0
        (ROOT / f'{row.classes}class_{row.arm}.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    for name, count in record['training'].items():
        phases.append(dict(classes=row.classes, arm=row.arm, stage=name, parameters=count))
        if row.classes == 4 and name != 'unique_updated_parameters':
            receipt = Path(record['checkpoint']).parents[1] / name / 'receipt.json'
            assert json.loads(receipt.read_text())['trainable'] == count, receipt
    if row.classes == 4:
        error = max(record['raw_cache_max_logit_error'], record['saved_prediction_max_logit_error'])
        assert error < 1e-4, (row.arm, error)
        max_error = max(max_error, error)
pd.DataFrame(phases).to_csv(ROOT / 'trainable_parameters.csv', index=False)
print('Verified all six trained arms against original receipts and predictions.')
print('Maximum replay logit error:', max_error)
print(selected.to_string(index=False))
