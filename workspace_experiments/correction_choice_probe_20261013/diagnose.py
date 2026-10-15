"""Describe the frozen failure without fitting or selecting another policy."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
units = json.loads((OUT/'PREDICTIONS_FROZEN.json').read_text(encoding='utf-8'))['units']
rows = []
for unit in units:
    with np.load(unit['path']) as z:
        counts = z['profile_counts']
        rows.append(dict(uid=unit['uid'],family=unit['family'],task=unit['task'],cohort=unit['cohort'],
            support_oof_events=int(z['profile_rescue'].sum()+z['profile_harm'].sum()),
            support_covered_classes=int((counts>0).sum()),max_class_support=int(counts.max()),
            profile_macro_gain=float(z['profile_mean'].mean()),
            profile_mean_sd=float(z['profile_sd'].mean())))
frame = pd.DataFrame(rows)
diagnostics = pd.read_csv(OUT/'diagnostics.csv')
frame = frame.merge(diagnostics[['uid','initial_gain_ba']],on='uid',validate='one_to_one')
frame.to_csv(OUT/'support_information.csv',index=False)
frame.groupby(['family','task','cohort']).mean(numeric_only=True).to_csv(OUT/'support_information_summary.csv')
diagnostics.groupby(['family','dataset','cohort'])[
    ['score_only_auc','contextual_auc','confidence_auc','oracle_gain_over_initial']
].mean().to_csv(OUT/'ranking_summary.csv')
pd.read_csv(OUT/'feature_auc.csv').groupby(['family','dataset','cohort','feature']).auc.mean().unstack().to_csv(
    OUT/'feature_auc_summary.csv')

timing = pd.read_csv(OUT/'decision_timings.csv').merge(
    pd.DataFrame(units)[['uid','family','dataset','cohort']],on='uid',validate='one_to_one')
timing.groupby(['family','dataset','cohort'])[['query_n','two_probes_seconds']].mean().to_csv(
    OUT/'decision_cost_summary.csv')

# Both sides share exactly the same frozen classifiers and source-only recipe.
m = pd.read_csv(OUT/'metrics.csv')
keys = ['uid','family','dataset','task','cohort','unit']
paired = m[m.arm=='contextual'].merge(m[m.arm=='score_only'],on=keys,suffixes=('_context','_score'),validate='one_to_one')
for metric in ('oa','ba','macro_brier'):
    paired['context_minus_score_'+metric] = paired[metric+'_context']-paired[metric+'_score']
paired[keys+['context_minus_score_oa','context_minus_score_ba','context_minus_score_macro_brier']].to_csv(
    OUT/'support_context_increment.csv',index=False)
print('Frozen support-information and ranking diagnostics written; no new fits.')
