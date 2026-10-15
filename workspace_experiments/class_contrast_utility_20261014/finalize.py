"""Retain seed variability and paired sample-level changes for the learned core."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    for folder in (ROOT, ROOT / 'balanced_objective'):
        cases = pd.read_csv(folder / 'case_paths.csv', dtype={'ship': str})
        keys = ['dataset', 'episode', 'budget', 'draw', 'cell', 'row']
        base = cases[cases.arm == 'class_mean'][keys + ['new_pred']].rename(columns={'new_pred': 'reference_pred'})
        candidates = cases[cases.arm.str.startswith('utility_')].merge(base, on=keys, validate='many_to_one')
        candidates['rescued'] = (candidates.reference_pred != candidates.class_id) & (candidates.new_pred == candidates.class_id)
        candidates['harmed'] = (candidates.reference_pred == candidates.class_id) & (candidates.new_pred != candidates.class_id)
        changed = candidates[candidates.rescued | candidates.harmed]
        changed.to_csv(folder / 'reference_change_events.csv', index=False)
        groups = []
        for keys0, g in changed.groupby(['dataset', 'budget', 'cell', 'arm', 'class_id']):
            groups.append(dict(zip(['dataset', 'budget', 'cell', 'arm', 'class_id'], keys0),
                               rescues=int(g.rescued.sum()), harms=int(g.harmed.sum()),
                               rescued_ships=g.loc[g.rescued, 'ship'].nunique(),
                               harmed_ships=g.loc[g.harmed, 'ship'].nunique(),
                               rescued_scenes=g.loc[g.rescued, 'episode'].nunique(),
                               harmed_scenes=g.loc[g.harmed, 'episode'].nunique()))
        pd.DataFrame(groups).to_csv(folder / 'reference_change_groups.csv', index=False)
        metrics = pd.read_csv(folder / 'metrics.csv')
        k = ['dataset', 'study', 'cohort', 'budget', 'draw', 'cell', 'episode', 'port', 'mode']
        base = metrics[metrics.arm == 'class_mean'][k + ['oa', 'ba']]
        pair = metrics[metrics.arm.str.startswith('utility_')].merge(base, on=k, suffixes=('', '_reference'), validate='many_to_one')
        pair['ba_delta'] = pair.ba - pair.ba_reference
        pair['oa_delta'] = pair.oa - pair.oa_reference
        scene = pair[pair['mode'] == 'full'].groupby(['dataset', 'budget', 'cell', 'episode', 'port'])[['oa_delta', 'ba_delta']].mean().reset_index()
        scene.to_csv(folder / 'seed_draw_mean_scene_effects.csv', index=False)
        effects = pd.read_csv(folder / 'effects_summary.csv')
        subset = effects[(effects['mode'] == 'full') & (effects.metric == 'ba') & (effects.baseline == 'class_mean') & (effects.arm != 'utility_seed_mean')]
        seed_summary = subset.groupby(['dataset', 'cohort', 'budget', 'cell']).delta.agg(['mean', 'std', 'min', 'max']).reset_index()
        seed_summary.to_csv(folder / 'training_seed_summary.csv', index=False)
        print(folder.name, seed_summary[(seed_summary.dataset == 'oss2') & (seed_summary.budget == 16)].to_string(index=False))
        print(pd.DataFrame(groups).query("dataset=='oss2' and budget==16 and cell=='other_acquisition'").to_string(index=False))
        receipt = dict(learned_predictions_compared_to_core=len(candidates), changed_prediction_exposures=len(changed),
                       warning='Training seeds, support draws, repeated observations and ships are separate evidence units; seed mean is performance average, not ensemble')
        (folder / 'FINALIZATION_COMPLETE.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
