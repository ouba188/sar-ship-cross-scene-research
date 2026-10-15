"""Posthoc cached-score path diagnosis; not a predeclared candidate result."""

import sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
import run as r

OUT = Path(__file__).resolve().parent


def main():
    rows, groups = [], []
    for directory in r.folders():
        manifest = pd.read_csv(r.c.read_json(directory/'PREDICTIONS_FROZEN.json')['target_manifest'],
                               usecols=['class_id'])
        for draw in range(5):
            for budget in r.c.BUDGETS:
                old = r.c.archive(directory/f'{draw}_{budget}.npz')
                new = r.c.archive(OUT/'predictions'/directory.name/f'{draw}_{budget}.npz')
                y = manifest.iloc[old['query_ids']].class_id.to_numpy(int)
                present = np.bincount(old['support_labels'], minlength=8)
                query_classes = np.unique(y)
                query_absent = np.setdiff1d(np.arange(8), query_classes)
                supported_only = new['count'].copy()
                supported_only[:, present==0] = new['raw'][:, present==0]
                # The absent-column-only path is independently replayed against guard.
                absent_only = new['raw'].copy()
                absent_only[:, present==0] = new['prior_only'][:, present==0]
                np.testing.assert_allclose(absent_only, new['guard'], atol=1e-10)
                scores = dict(raw=new['raw'], count=new['count'], guard=new['guard'],
                              count_supported_only=supported_only)
                low = (present>0)&(present<budget/8)
                high = present>=budget/8
                for name, columns in [('count_low_only',low),('count_high_only',high)]:
                    scores[name] = new['raw'].copy()
                    scores[name][:, columns] = new['count'][:, columns]
                reconstructed = (scores['count_low_only']+scores['count_high_only']+
                                 scores['guard']-2*scores['raw'])
                np.testing.assert_allclose(reconstructed, scores['count'], atol=1e-10)
                baseline = r.c.late(old['base'], scores['raw'])
                bm = r.metric(y, baseline)
                bp = baseline.argmax(1)
                for arm, score in scores.items():
                    p = r.c.late(old['base'], score)
                    pred = p.argmax(1)
                    m = r.metric(y, p)
                    rows.append(dict(task=directory.name, budget=budget, draw=draw, arm=arm,
                        n_query=len(y), query_classes=len(query_classes),
                        support_classes=int((present>0).sum()), **m,
                        delta_oa=m['oa']-bm['oa'], delta_ba=m['ba']-bm['ba'],
                        delta_macro_brier=m['macro_brier']-bm['macro_brier'],
                        predicted_query_absent_pct=100*np.isin(pred, query_absent).mean(),
                        predicted_support_absent_pct=100*(present[pred]==0).mean()))
                    for name in ('zero', 'below_average', 'at_or_above_average'):
                        classes = [k for k in query_classes if
                            ('zero' if present[k]==0 else 'below_average' if present[k]<budget/8 else 'at_or_above_average')==name]
                        rescue, harm = 0., 0.
                        for k in classes:
                            mask = y==k
                            rescue += 100*np.mean((bp[mask]!=k)&(pred[mask]==k))/len(query_classes)
                            harm += 100*np.mean((bp[mask]==k)&(pred[mask]!=k))/len(query_classes)
                        groups.append(dict(task=directory.name,budget=budget,draw=draw,arm=arm,group=name,
                            classes=len(classes), rescue_ba=rescue,harm_ba=harm,delta_ba=rescue-harm))
    f, g = pd.DataFrame(rows), pd.DataFrame(groups)
    f.to_csv(OUT/'diagnostic_paths.csv',index=False)
    p = f.groupby(['task','budget','arm']).mean(numeric_only=True).reset_index()
    p.to_csv(OUT/'diagnostic_ports.csv',index=False)
    summary = p.groupby(['budget','arm']).mean(numeric_only=True).reset_index()
    summary.to_csv(OUT/'diagnostic_summary.csv',index=False)
    g.to_csv(OUT/'support_groups.csv',index=False)
    g.groupby(['task','budget','arm','group']).mean(numeric_only=True).groupby(
        ['budget','arm','group']).mean().reset_index().to_csv(OUT/'support_group_summary.csv',index=False)
    p.assign(query_scope=np.where(p.query_classes==8,'all8','fewer8')).groupby(
        ['budget','arm','query_scope']).agg(ports=('task','size'),delta_oa=('delta_oa','mean'),
        delta_ba=('delta_ba','mean'),delta_macro_brier=('delta_macro_brier','mean'),
        absent_predictions=('predicted_query_absent_pct','mean')).reset_index().to_csv(OUT/'class_coverage_diagnostic.csv',index=False)
    r.c.write(OUT/'DIAGNOSTIC_RECEIPT.json',dict(posthoc=True,new_fits=0,
        purpose='Separate low-count, high-count and absent-support class score paths; nonlinear metric effects are not additive',
        original_candidates_unchanged=True,deployable_policy_selected=False,
        query_labels_used_only_for_evaluation=True,prediction_units=345))
    print(summary[['budget','arm','delta_oa','delta_ba','delta_macro_brier',
                   'predicted_query_absent_pct']].to_string(index=False),flush=True)


if __name__=='__main__':
    with r.c.threadpool_limits(limits=2):
        main()
