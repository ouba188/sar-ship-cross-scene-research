"""Post-fit paired descriptive strata. Group composition is not a causal effect."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
ARMS=('baseline','disentangle','full_h')
SEEDS=(17,29,43)


def stats(values):
    return dict(mean=statistics.mean(values),seed_sd=statistics.stdev(values),n_seeds=len(values))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=('validation','test'),default='test')
    args=parser.parse_args()
    suffix={'baseline':'baseline_base','disentangle':'disentangle_base','full_h':'full_h'}
    paths={(s,a):ROOT/'runs'/f'seed{s}'/((a+'_test.json') if args.split=='test' else suffix[a]+'_validation.json')
           for s in SEEDS for a in ARMS}
    assert all(p.exists() for p in paths.values()),'All nine results must exist before analysis'
    metadata=json.loads((ROOT/'atom_metadata_local.json').read_text(encoding='utf8'))
    metadata={r['id']:r for r in metadata}
    results={k:{r['id']:r for r in json.loads(p.read_text(encoding='utf8'))['predictions']} for k,p in paths.items()}
    ids=set(results[17,'baseline'])
    assert all(set(r)==ids for r in results.values())
    def bucket(r):return '1_atom' if r['atom_count']==1 else '2_atoms' if r['atom_count']==2 else 'more_than_2_atoms'
    groups={g:sorted(i for i in ids if bucket(metadata[i])==g) for g in ('1_atom','2_atoms','more_than_2_atoms')}
    classes=sorted({metadata[i]['class_name'] for i in ids})
    for cls in classes:
        for g in ('1_atom','2_atoms','more_than_2_atoms'):
            groups[cls+'/'+g]=[i for i in groups[g] if metadata[i]['class_name']==cls]
    output={}
    for name,members in groups.items():
        if not members:
            output[name]=dict(n=0,metrics=None);continue
        accuracy={a:[] for a in ARMS}
        for s in SEEDS:
            for a in ARMS:
                accuracy[a].append(sum(str(results[s,a][i]['prediction'])==str(results[s,a][i]['label'])
                                       for i in members)/len(members)*100)
        output[name]=dict(n=len(members),class_composition=dict(Counter(metadata[i]['class_name'] for i in members)),
                          accuracy_percent={a:stats(v) for a,v in accuracy.items()},
                          paired_delta_pp={a+'_minus_'+b:stats([x-y for x,y in zip(accuracy[a],accuracy[b])])
                              for a,b in (('disentangle','baseline'),('full_h','disentangle'),('full_h','baseline'))})
    payload=dict(split=args.split,groups=output,
                 interpretation='Descriptive paired accuracy. Atom groups differ in class composition; seed SD is training variability, not a vessel-level confidence interval.')
    destination=ROOT/f'aggregate_{args.split}_subgroups.json'
    destination.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
    print(destination)


if __name__=='__main__':main()
