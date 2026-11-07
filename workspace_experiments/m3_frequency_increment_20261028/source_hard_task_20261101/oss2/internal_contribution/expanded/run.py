"""Frozen extension of E/R/none: no optimization or checkpoint selection."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(HERE.parents[1]))
from model_task import *
import prepare_task
from reference_writer import WriterModel
from review_repeats import load_scores


def expanded_records():
    old,train,dev,rid=records()
    sel=data.read_manifest(PUBLIC/'manifests/selection.json')
    test=data.read_manifest(PUBLIC/'manifests/evaluation.json')
    rows=old+sel+test
    groups=dict(development=dev,selection=np.arange(len(old),len(old)+len(sel)),
                evaluation=np.arange(len(old)+len(sel),len(rows)))
    return rows,train,rid,groups


def main():
    current.setup(DEVICE)
    rows,train,rid,groups=expanded_records()
    audit={}
    for name,ids in groups.items():
        audit[name]=dict(n=len(ids),classes=np.bincount([rows[i]['label'] for i in ids],minlength=4).tolist(),
                        vessels=len({rows[i]['mmsi'] for i in ids}),passes=len({orbit(rows[i]) for i in ids}))
        for key,fn in [('id',lambda r:r['id']),('mmsi',lambda r:r['mmsi']),('orbit',orbit)]:
            overlap={fn(rows[i]) for i in train}&{fn(rows[i]) for i in ids}
            audit[name]['source_overlap_'+key]=sorted(overlap)
            if key!='orbit':assert not overlap,(name,key,overlap)
    assert len({r['id'] for r in rows})==len(rows)
    source_orbits={orbit(rows[i]) for i in train}
    strict={k:[int(i) for i in ids if orbit(rows[i]) not in source_orbits] for k,ids in groups.items()}
    write(HERE/'cohorts.json',dict(audit=audit,source=len(train),references=[rows[i]['id'] for i in rid],
                                 groups={k:v.tolist() for k,v in groups.items()},strict_groups=strict))
    print('COHORTS',audit,flush=True)
    # Reuse the exact frozen feature extractor, with expanded rows and an isolated destination.
    hold=np.concatenate(list(groups.values()))
    prepare_task.ROOT=HERE
    prepare_task.records=lambda:(rows,train,hold,rid)
    prepare_task.main()
    meta=json.loads((HERE/'cache/complete.json').read_text())
    meta.update(evaluation_manifest_read=True,scope='Historical frozen extension; all query cohorts; no training')
    write(HERE/'cache/complete.json',meta)
    a={p.stem:np.load(p,mmap_mode='r') for p in (HERE/'cache').glob('*.npy')}
    checks=[]
    for seed in (17,29,43):
        for name in ('evidence','reference','none'):
            arm='none' if name=='none' else 'aggregate'
            model=WriterModel(arm,seed) if name=='reference' else Model(arm,seed)
            folder=HERE.parent/'reference_message'/f'aggregate_{seed}' if name=='reference' else ROOT/f'{arm}_{seed}'
            model.restore(torch.load(folder/'last.pt',map_location='cpu',weights_only=True)['trainable_state'])
            model.eval()
            for suffix,override in [('',None)]+([('_off','off')] if name!='none' else []):
                pred=predict(model,a,rows,hold,rid,override)
                np.savez_compressed(HERE/f'{name}{suffix}_{seed}.npz',**pred)
                dev=groups['development']
                filename='last_off' if override else 'last_validation'
                old=np.load(folder/(filename+'.npz'))['score'] if name=='reference' else load_scores(seed,arm,filename,dev)
                error=float(abs(pred['score'][:len(dev)]-old).max())
                assert error<1e-4,(seed,name,suffix,error)
                checks.append(dict(seed=seed,variant=name+suffix,development_replay_error=error))
            del model
            print('INFERRED',seed,name,flush=True)
    write(HERE/'checks.json',checks)
    print('DONE',checks,flush=True)


if __name__=='__main__':main()
