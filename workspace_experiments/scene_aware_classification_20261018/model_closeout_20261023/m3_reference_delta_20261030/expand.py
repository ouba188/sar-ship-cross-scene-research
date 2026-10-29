"""One frozen private expanded evaluation, baseline and paired controls retained."""
from run import *
from sar_m123.current.model import reference_bank


@torch.no_grad()
def main():
    setup()
    out=HERE/'joint_adaptation/expanded'
    out.mkdir(parents=True,exist_ok=True)
    queries=read_manifest(ROOT/'m2_preadd_expansion_20261028/query_records.json')
    refs=read_manifest(refit.RELEASE/'manifests/references.json')
    assert len(queries)==3075
    models={(arm,s):load_model(HERE/'joint_adaptation/training'/f'{arm}_s{s}/model.pt','cuda')
        for arm in ('control','joint_low') for s in (17,29,43)}
    first=next(iter(models.values()))
    bank=reference_bank(first,refs,'cuda')
    result={k:[] for k in models}
    tick=time.perf_counter()
    for start in range(0,len(queries),24):
        group=queries[start:start+24]
        values=batch([prepare_record(r) for r in group],'cuda')
        prefix=first.prefix(values.pop('images'),stop=2)
        legal=pair_mask(group,refs,'cuda')
        for k,net in models.items():
            result[k].append(net.forward_from(prefix,**values,reference=bank,pair_valid=legal).cpu().numpy())
        if start%240==0 or start+24>=len(queries):
            print('EXPANDED',min(start+24,len(queries)),len(queries),round(time.perf_counter()-tick,1),flush=True)
    y=np.array([r['label'] for r in queries])
    ids=np.array([str(r['id']) for r in queries])
    report=[]
    for (arm,seed),parts in result.items():
        score=np.concatenate(parts)
        with np.load(ROOT/'spatial_m12_full_20261029/predictions'/f'expanded_after_spatial_s{seed}.npz') as f:
            assert np.array_equal(f['ids'],ids) and np.array_equal(f['y'],y)
            baseline=f['score']
        np.savez(out/f'{arm}_s{seed}.npz',score=score,baseline=baseline,y=y,ids=ids)
        report.append(dict(arm=arm,seed=seed,before=metrics(baseline,y),after=metrics(score,y)))
    write(out/'results.json',dict(records=report,query_count=len(queries),seconds=time.perf_counter()-tick,
        frozen_candidate='joint_low; M3 output LR3e-5 during original adaptation; no post-test choice',
        historical_development_exposure=True))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
