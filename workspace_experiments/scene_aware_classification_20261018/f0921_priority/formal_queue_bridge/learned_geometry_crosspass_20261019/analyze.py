"""Paired outcomes and bounded original-chip review, without selecting a new model."""
import json
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
from prepare import HERE,BRIDGE,load

KEYS = ['task','seed','draw','row']


def signed_case_panels():
    meta = pd.read_csv(HERE/'cohort.csv')
    table = pd.read_csv(HERE/'case_support_contributions.csv')
    review = load('signed_native_review',BRIDGE/'review_cases.py')
    folder = HERE/'case_review'
    for row in [208,197]:
        part = table[table.row.eq(row)&table.comparator.eq('base')].copy()
        part['magnitude'] = part.total.abs()
        part = part.sort_values('magnitude',ascending=False).head(3)
        ids = [row]+part.support_row.astype(int).tolist()
        panel = Image.new('RGB',(240*len(ids),550),'white')
        draw = ImageDraw.Draw(panel)
        for j,i in enumerate(ids):
            r = meta.iloc[i]
            text = f'row{i} class{r.label}'
            if j:
                item = part.iloc[j-1]
                text += f'\nsim {item.old_similarity:.4f}->{item.new_similarity:.4f}\nmargin contribution {item.total:+.5f}'
            draw.text((j*240+5,5),text,fill='black')
            for pol,col in enumerate(['png_vv','png_vh']):
                panel.paste(Image.open(r[col]).convert('RGB'),(j*240+5,65+pol*240))
            path=folder/f'row{i}_native.png'
            if not path.exists(): review.native_query(r,path)
        panel.save(folder/f'row{row}_signed.png')


def main():
    meta = pd.read_csv(HERE/'cohort.csv',dtype={'date':str,'mmsi':str})
    pred = pd.read_csv(HERE/'predictions.csv')
    met = pd.read_csv(HERE/'metrics.csv')
    source = json.loads((HERE/'tasks.json').read_text())
    all_events,all_deltas,selected = [],[],[]
    for comp in ['base','A','B','D','C_test_mismatch']:
        a = pred[pred.arm.eq('C')].set_index(KEYS).sort_index()
        b = pred[pred.arm.eq(comp)].set_index(KEYS).sort_index()
        assert a.index.equals(b.index)
        e = a[['line','budget','truth','prediction','margin','top_same','top_wrong','same_support','readout_valid']].copy()
        e['comparator'] = comp
        e['old_prediction'] = b.prediction
        e['delta_margin'] = a.margin-b.margin
        e['delta_brier'] = a.brier-b.brier
        e['rescue'] = a.correct & ~b.correct
        e['harm'] = ~a.correct & b.correct
        e['persistent_correct'] = a.correct & b.correct
        e['persistent_wrong'] = ~a.correct & ~b.correct
        e = e.reset_index()
        all_events.append(e)
        idx = ['task','line','budget','target_pass','seed','draw']
        cols = ['oa','ba','nll','brier','map_r']
        d = met[met.arm.eq('C')].set_index(idx)[cols]-met[met.arm.eq(comp)].set_index(idx)[cols]
        d = d.reset_index(); d['comparator'] = comp
        all_deltas.append(d)
        if comp in ['base','A']:
            for line in ['cross_pass','same_pass']:
                for category in ['rescue','harm','persistent_correct','persistent_wrong']:
                    pool = e[e.line.eq(line)&e.budget.eq(16)&e[category]].sort_values(['delta_margin','row','seed','draw'])
                    if not pool.empty:
                        case = pool.iloc[(len(pool)-1)//2].to_dict()
                        case['category'] = category
                        selected.append(case)
    events,delta = pd.concat(all_events),pd.concat(all_deltas)
    events.to_csv(HERE/'paired_events.csv',index=False)
    delta.to_csv(HERE/'paired_metrics.csv',index=False)
    seed = delta.groupby(['comparator','line','budget','seed'])[['oa','ba','nll','brier','map_r']].mean().reset_index()
    seed.to_csv(HERE/'paired_seed.csv',index=False)
    summary = []
    for keys,g in seed.groupby(['comparator','line','budget']):
        comp,line,budget = keys
        ev = events[events.comparator.eq(comp)&events.line.eq(line)&events.budget.eq(budget)]
        summary.append(dict(comparator=comp,line=line,budget=budget,
            oa_pp=100*g.oa.mean(),ba_pp=100*g.ba.mean(),oa_seed_sd_pp=100*g.oa.std(),ba_seed_sd_pp=100*g.ba.std(),
            positive_oa_seeds=int((g.oa>1e-10).sum()),positive_ba_seeds=int((g.ba>1e-10).sum()),
            brier_delta=g.brier.mean(),map_delta=g.map_r.mean(),
            rescue_events=int(ev.rescue.sum()),harm_events=int(ev.harm.sum()),
            rescued_unique_rows=ev[ev.rescue].row.nunique(),harmed_unique_rows=ev[ev.harm].row.nunique()))
    pd.DataFrame(summary).to_csv(HERE/'paired_summary.csv',index=False)
    events.groupby(['comparator','line','budget','truth'])[['rescue','harm','delta_margin','delta_brier']].agg(['sum','mean']).to_csv(HERE/'class_effects.csv')
    cases = pd.DataFrame(selected)
    cases.to_csv(HERE/'selected_cases.csv',index=False)
    review = load('old_native_review',BRIDGE/'review_cases.py')
    folder = HERE/'case_review';folder.mkdir(exist_ok=True)
    native_rows = set()
    for case in cases.itertuples():
        ids = list(dict.fromkeys([int(i) for i in [case.row,case.top_same,case.top_wrong] if i>=0]))
        panel = Image.new('RGB',(240*len(ids),550),'white')
        draw = ImageDraw.Draw(panel)
        draw.text((5,5),f'C vs {case.comparator} {case.category}: {case.task} s{case.seed} d{case.draw}',fill='black')
        draw.text((5,22),f'query r{case.row}: {case.old_prediction}->{case.prediction}; truth {case.truth}',fill='black')
        for j,i in enumerate(ids):
            r = meta.iloc[i]
            draw.text((j*240+5,42),f'row{i} class{r.label} {r.date}',fill='black')
            for pol,col in enumerate(['png_vv','png_vh']):
                panel.paste(Image.open(r[col]).convert('RGB'),(j*240+5,65+pol*240))
            native_rows.add(i)
        name = f'{case.comparator}_{case.line}_{case.category}_r{case.row}_s{case.seed}_d{case.draw}'
        panel.save(folder/(name+'.png'))
    for i in sorted(native_rows):
        review.native_query(meta.iloc[i],folder/f'row{i}_native.png')
    task_report = []
    for t in source['tasks']:
        q = np.array(t['query'])
        for d,s in enumerate(t['supports']):
            s = np.array(s)
            same = meta.label.to_numpy()[q,None]==meta.label.to_numpy()[s][None,:]
            cross = meta.pass_key.to_numpy()[q,None]!=meta.pass_key.to_numpy()[s][None,:]
            task_report.append(dict(task=t['name'],draw=d,query_n=len(q),qs_events=len(q)*len(s),
                cross_pass_events=int(cross.sum()),same_class_events=int(same.sum()),same_class_cross_events=int((same&cross).sum()),
                queries_with_same_class=int(same.any(1).sum())))
    pd.DataFrame(task_report).to_csv(HERE/'relation_coverage.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print('NATIVE ROWS',sorted(native_rows))


if __name__=='__main__': main()
