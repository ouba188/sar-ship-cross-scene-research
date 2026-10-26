"""Add the original predicted-class references for the wrong-to-wrong case."""
import ast
import json
import numpy as np
import pandas as pd
import evaluate as ev
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def main():
    cases = json.loads((ev.HERE / 'cases.json').read_text(encoding='utf-8'))
    helper = ev.ROOT / 'new_version_ablations_20261026/analyze_readout.py'
    node = next(n for n in ast.parse(helper.read_text(encoding='utf-8')).body if isinstance(n, ast.FunctionDef) and n.name == 'display')
    scope = {'tr': ev.tr}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(helper), 'exec'), scope)
    facts = []
    extras = []
    for case in cases:
        fact = {k: case[k] for k in ('source', 'seed', 'obs_id', 'outcome', 'class_id', 'before_pred', 'after_pred', 'fixed_rival', 'base_margin', 'before_margin', 'after_margin')}
        for side in ('before', 'after'):
            scores = np.asarray(case[f'{side}_scores'])
            true = int(case['class_id'])
            fact[f'{side}_strongest_rival_margin'] = float(scores[true] - np.delete(scores, true).max())
            fact[f'{side}_VV_contribution'], fact[f'{side}_VH_contribution'] = case[f'{side}_polar_contribution']
        facts.append(fact)
        if case['before_pred'] in (case['class_id'], case['fixed_rival']):
            continue
        source, seed, obs = case['source'], int(case['seed']), int(case['obs_id'])
        table, held, data, _ = ev.load_source(source)
        j = int(np.flatnonzero(held.obs_id.to_numpy() == obs)[0])
        refs = data['both', seed]['refs'][j, int(case['before_pred'])].tolist()
        ev.tr.extract.VIEW_TABLE = ev.ROOT / f'{source}_m123_replication_20261026/view_inventory.csv'
        ev.tr.extract.RADAR_TABLE = ev.ROOT / f'{source}_m123_replication_20261026/geometry.csv'
        records = ev.tr.extract.prepare_inventory(table)
        fig, axes = plt.subplots(2, 3, figsize=(10, 7))
        provenance = []
        for col, row in enumerate([int(held.source_row.iloc[j]), *refs]):
            for pol in range(2):
                p = scope['display'](axes[pol, col], records[row], pol)
                p.update(source_row=row, obs_id=int(table.obs_id.iloc[row]), sample_id=table.sample_id.iloc[row],
                         mmsi=table.mmsi.iloc[row], pass_group=table.pass_group.iloc[row], role=table.role.iloc[row])
                provenance.append(p)
                axes[pol, col].set_title(f'{("QUERY", "OLD RIVAL1", "OLD RIVAL2")[col]} {("VV", "VH")[pol]} obs{int(table.obs_id.iloc[row])}\n{ev.CLASSES[int(table.class_id.iloc[row])]}', fontsize=9)
                axes[pol, col].set(xticks=[], yticks=[])
        fig.suptitle(f'{source.upper()} obs{obs} WW: original predicted container references\nTrue bulk margin to strongest rival {fact["before_strongest_rival_margin"]:+.4f} -> {fact["after_strongest_rival_margin"]:+.4f}')
        fig.text(.02, .02, 'Native160 crop/pad; same fixed grayscale. Both arms retrieve these same references.', fontsize=8)
        fig.subplots_adjust(top=.82, bottom=.08, hspace=.2)
        name = f'panels/{source}_seed{seed}_WW_obs{obs}_old_rival.png'
        fig.savefig(ev.HERE / name, dpi=120)
        plt.close(fig)
        extras.append(dict(source=source, seed=seed, obs_id=obs, image=name, refs=refs,
                           reference_obs_ids=table.iloc[refs].obs_id.astype(int).tolist(), provenance=provenance))
    ev.save('case_evidence.csv', facts)
    ev.write('supplementary_references.json', extras)
    print(pd.DataFrame(facts).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
