"""Export the frozen full-model comparison without rerunning experiments."""
from pathlib import Path
import json

import pandas as pd


ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
OUT = ROOT / 'frequency_closeout_20261010'
PROVENANCE = {
    'learned': ('primary', 'learned_frequency_residual_20261009', 'selective'),
    'standalone_local': ('historical_candidate', 'frequency_preserve_20261010', 'standalone_support_local'),
    'preserved_hard': ('historical_candidate', 'frequency_preserve_20261010', 'preserved_support_local'),
    'margin_selected': ('historical_candidate', 'frequency_margin_selection_20261010', 'source_selected'),
    'pair_local': ('historical_candidate', 'frequency_pair_selection_20261010', 'pair_local'),
    'calibrated': ('forced_final_control', 'frequency_closeout_20261010', 'calibrated'),
    'source_selected': ('optional_ba_extension', 'frequency_closeout_20261010', 'source_selected'),
}


def main():
    summary = pd.read_csv(OUT / 'summary.csv')
    columns = ['dataset', 'mode', 'oa', 'ba', 'delta_oa_vs_learned',
               'delta_ba_vs_learned', 'positive_draws_ba_vs_learned']
    table = summary.loc[summary.stage.eq('full'), columns].copy()
    if len(table) != 21 or table.duplicated(['dataset', 'mode']).any():
        raise ValueError('Expected three datasets and seven unique modes each')
    table['role'] = table['mode'].map(lambda mode: PROVENANCE[mode][0])
    table['artifact_root'] = table['mode'].map(lambda mode: (ROOT / PROVENANCE[mode][1]).as_posix())
    table['original_mode'] = table['mode'].map(lambda mode: PROVENANCE[mode][2])
    for folder in table.artifact_root.unique():
        if not Path(folder).is_dir():
            raise FileNotFoundError(folder)
    freeze = json.loads((OUT / 'FREEZE.json').read_text(encoding='utf-8'))
    primary = Path(freeze['primary_model']['artifact_root'])
    for key in ('checkpoints', 'predictions'):
        files = list(primary.glob(freeze['primary_model'][key]))
        if len(files) != 25:
            raise ValueError(f'{key}: expected 25 task artifacts, found {len(files)}')
    completion = json.loads((OUT / 'REPORT_COMPLETE.json').read_text(encoding='utf-8'))
    if not completion['round_closed'] or completion['diagnosis_pending']:
        raise ValueError('Round is not closed')
    table.to_csv(OUT / 'FINAL_MODEL_TABLE.csv', index=False)
    print(table.to_string(index=False))
    print('Exported 21 rows; primary checkpoint/policy paths resolve for all 25 tasks.')


if __name__ == '__main__':
    main()
