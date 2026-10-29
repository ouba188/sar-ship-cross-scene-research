"""Two requested development transplant cases, four native panels maximum."""
import importlib.util
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('joint_case_analysis', Path(__file__).resolve().parents[1]/'analysis.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def main():
    d = a.load_dataset('private', [29, 43])
    assert d['seeds'] == [29, 43]
    folder = a.OUT/'private/transplant_cases'
    folder.mkdir(parents=True, exist_ok=True)
    helper = a.coverage.module('transplant_native_panel', a.ROOT/'spatial_m12_refit_20261029/analysis.py')
    helper.CLASSES = tuple(d['classes'])
    native = a.coverage.module('transplant_native_input', a.WORK/'sar_m123_standalone/sar_m123/input.py')
    cases, changes, margins, sources = [], [], [], []
    lookup = {r['id']:i for i,r in enumerate(d['group'])}
    for seed, state in ((43,'rescued'), (29,'harmed')):
        scores = {arm:d['scores'][arm,seed] for arm in ('saved_baseline', *a.ARMS)}
        for arm in a.ARMS:
            for part in ('m3_only','live_only'):
                name = arm+'_'+part
                pack = a.coverage.load_npz(a.HERE/'private/transplant'/f'{name}_s{seed}.npz',d['group'],d['classes'])
                assert np.array_equal(pack['baseline'],scores['saved_baseline'])
                scores[name] = pack['score']
        pool = [e for e in a.events(scores['saved_baseline'],scores['center_joint_m3_only'],d['group'],seed) if e['state']==state]
        changes.extend(pool)
        pool = [e for e in pool if e['id'] not in {'10407','2042','27525'}]
        selected = sorted(pool,key=lambda e:(-abs(e['margin_change']),e['id']))[0]
        i = lookup[selected['id']]
        query = d['group'][i]
        truth, rival = selected['label'], selected['rival']
        case = dict(selected, truth_class=d['classes'][truth], rival_class=d['classes'][rival],
                    mmsi=query.get('mmsi'),pass_group=query.get('pass_group'),port=query.get('port'),
                    scores={},panels={},visual_status='not_reviewed')
        for name,z in scores.items():
            row = dict(id=query['id'],seed=seed,arm=name,prediction=int(z[i].argmax()),
                       predicted_class=d['classes'][int(z[i].argmax())],truth=truth,rival=rival,
                       truth_rival_margin=float(z[i,truth]-z[i,rival]))
            margins.append(row)
            case['scores'][name] = dict(row,logits=z[i].tolist())
        legal = [r for r in d['refs'] if helper.eligible(query,r)]
        case['all_legal_reference_ids'] = [r['id'] for r in legal]
        for pol in ('vv','vh'):
            available = lambda r: bool(r.get(pol) and r[pol].get('alignment_valid',True))
            own = [r for r in legal if r['label']==truth and available(r)][:2]
            other = [r for r in legal if r['label']==rival and available(r)][:2]
            members = [query,*own,*other]
            roles = ['QUERY']+['TRUE REF']*len(own)+['RIVAL REF']*len(other)
            filename = f"{state}_{query['id']}_s{seed}_{pol}.png"
            receipts = helper.panel(members,roles,pol,folder/filename,native)
            for r,entry in zip(members,receipts):
                entry.update(case_id=query['id'],seed=seed,panel=filename,label=r['label'],
                             pair_legal=entry['role']!='QUERY',
                             usable_with_query=bool(receipts[0]['available'] and entry['available']))
            sources.extend(receipts)
            case['panels'][pol] = dict(file=filename,reference_ids=[r['id'] for r in own+other])
        cases.append(case)
    a.dump(folder/'cases.json',cases)
    a.dump(folder/'changed_ids.json',changes)
    a.dump(folder/'native_sources.json',sources)
    a.write_csv(folder/'margins.csv',margins)
    lines = ['# Requested private transplant cases','',
             'Comparison: center_joint_m3_only versus saved full baseline; development only.',
             'Selection: seed43 rescue / seed29 harm; exclude 10407, 2042, 27525; largest absolute fixed-rival margin change, then ID. No tuning.',
             'References: actual source bank, exclude identical ID/MMSI; first two available true/rival class entries in bank order. Posthoc labels only.']
    for c in cases:
        lines += ['',f"## {c['state']} {c['id']} seed{c['seed']}",'',f"Truth: {c['truth_class']}; fixed rival: {c['rival_class']}."]
        a.prior.table(lines,['Arm','Prediction','Truth minus fixed rival'],
                      [[name,r['predicted_class'],f"{r['truth_rival_margin']:+.9f}"] for name,r in c['scores'].items()])
        for pol in ('vv','vh'):
            lines += ['',f"{pol.upper()} references: {', '.join(c['panels'][pol]['reference_ids'])}",'',f"![{pol}]({c['panels'][pol]['file']})"]
    (folder/'CASES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    assert 'torch' not in sys.modules
    print([(c['id'],c['seed'],c['state']) for c in cases])


if __name__=='__main__':
    main()
