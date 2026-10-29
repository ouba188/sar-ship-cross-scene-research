"""Frozen native-input evaluation; does not select or update any parameter."""
from train_center import *
from sar_m123.current.model import reference_bank


@torch.no_grad()
def main(dataset):
    setup()
    out=RUN_ROOT/dataset/'evaluation';out.mkdir(parents=True,exist_ok=True)
    refs=read_manifest(refit.RELEASE/'manifests/references.json' if dataset=='private' else PUB/'manifests/references.json')
    development=read_manifest(refit.RELEASE/'manifests/development.json' if dataset=='private' else PUB/'manifests/validation.json')
    models={(arm,s):load_candidate(RUN_ROOT/dataset/'trained_center'/f'{arm}_s{s}.pt')
            for arm in ('control','center') for s in (17,29,43)}
    first=next(iter(models.values()));bank=reference_bank(first,refs,'cuda')
    sample=development[:4];data=batch([prepare_record(r) for r in sample],'cuda')
    data['prefix']=first.prefix(data.pop('images'),stop=2)
    errors={}
    for (arm,s),net in models.items():
        native=net.forward_from(**data,reference=bank,pair_valid=pair_mask(sample,refs,'cuda')).cpu().numpy()
        with np.load(RUN_ROOT/dataset/'trained_center'/f'{arm}_s{s}.npz') as f:expected=f['score'][:4]
        error=float(abs(native-expected).max());assert error<1e-4,error
        errors[f'{arm}_{s}']=error
    queries=read_manifest(ROOT/'m2_preadd_expansion_20261028/query_records.json' if dataset=='private' else PUB/'manifests/evaluation.json')
    assert len(queries)==(3075 if dataset=='private' else 378)
    result={key:[] for key in models};tick=time.perf_counter()
    for start in range(0,len(queries),24):
        q=queries[start:start+24];data=batch([prepare_record(r) for r in q],'cuda')
        data['prefix']=first.prefix(data.pop('images'),stop=2)
        legal=pair_mask(q,refs,'cuda')
        for key,net in models.items():result[key].append(net.forward_from(**data,reference=bank,pair_valid=legal).cpu().numpy())
        if start%240==0:print('EVALUATION',dataset,start,len(queries),round(time.perf_counter()-tick,1),flush=True)
    ids=np.array([str(r['id']) for r in queries]);y=np.array([r['label'] for r in queries]);report=[]
    for (arm,s),parts in result.items():
        score=np.concatenate(parts)
        base=(ROOT/'spatial_m12_full_20261029/predictions'/f'expanded_after_spatial_s{s}.npz' if dataset=='private'
              else PUB/'predictions'/f'evaluation_full_adapt_s{s}.npz')
        with np.load(base) as f:
            assert np.array_equal(f['ids'],ids) and np.array_equal(f['y'],y)
            baseline=f['score']
        np.savez(out/f'{arm}_s{s}.npz',score=score,baseline=baseline,y=y,ids=ids)
        report.append(dict(arm=arm,seed=s,baseline=metrics(baseline,y),after=metrics(score,y)))
    write(out/'results.json',dict(records=report,native_fresh_checkpoint_errors=errors,seconds=time.perf_counter()-tick,
        historical_development_exposure=True,parameter_selection_on_test=False))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['private','oss2'],required=True)
    args=parser.parse_args();main(args.dataset)
