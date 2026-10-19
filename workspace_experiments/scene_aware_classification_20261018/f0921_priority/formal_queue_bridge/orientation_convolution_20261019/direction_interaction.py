"""Separate hard-relation gain from a generic class-separation improvement."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    p = pd.read_csv(HERE / 'direction_relations.csv')
    key = ['task', 'seed', 'draw', 'row', 'kind']
    p['easy_gap'] = p.same_near-p.other_far
    p['orientation_contrast'] = p.hard_relation_gap-p.easy_gap
    records = []
    for arm, comparator in [('B', 'A'), ('C', 'B'), ('C', 'D')]:
        a = p[p.arm.eq(arm)].set_index(key).sort_index()
        b = p[p.arm.eq(comparator)].set_index(key).sort_index()
        z = a[['line', 'budget']].copy()
        z['delta_hard'] = a.hard_relation_gap-b.hard_relation_gap
        z['delta_easy'] = a.easy_gap-b.easy_gap
        z['delta_orientation_contrast'] = a.orientation_contrast-b.orientation_contrast
        z['arm'], z['comparator'] = arm, comparator
        records.append(z.reset_index().dropna(subset=['delta_orientation_contrast']))
    z = pd.concat(records)
    z.to_csv(HERE / 'direction_interaction_events.csv', index=False)
    summary = []
    for keys, group in z.groupby(['arm', 'comparator', 'line', 'budget', 'kind']):
        byquery = group.groupby('row')[['delta_hard', 'delta_easy', 'delta_orientation_contrast']].mean()
        summary.append(dict(zip(['arm', 'comparator', 'line', 'budget', 'kind'], keys),
                            queries=len(byquery), repeated_events=len(group), **byquery.mean().to_dict(),
                            positive_queries=int((byquery.delta_orientation_contrast > 0).sum())))
    out = pd.DataFrame(summary)
    out.to_csv(HERE / 'direction_interaction_summary.csv', index=False)
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
