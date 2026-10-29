"""Frozen broad-cohort evaluation, including the exact prior CE continuation."""
from experiment import *
from sar_m123.current.model import reference_bank


@torch.no_grad()
def main(dataset):
    previous.setup()
    out = HERE/dataset/'evaluation'
    out.mkdir(parents=True, exist_ok=True)
    refs = previous.read_manifest(previous.refit.RELEASE/'manifests/references.json' if dataset=='private'
                                  else previous.PUB/'manifests/references.json')
    dev = previous.read_manifest(previous.refit.RELEASE/'manifests/development.json' if dataset=='private'
                                 else previous.PUB/'manifests/validation.json')
    queries = previous.read_manifest(previous.ROOT/'m2_preadd_expansion_20261028/query_records.json' if dataset=='private'
                                     else previous.PUB/'manifests/evaluation.json')
    assert len(queries) == (3075 if dataset=='private' else 378)
    ids = np.array([str(r['id']) for r in queries]); y = np.array([r['label'] for r in queries])
    records=[]; tick=previous.time.perf_counter()
    for seed in (17,29,43):
        if (out/f'aligned_s{seed}.npz').exists(): raise FileExistsError(out/f'aligned_s{seed}.npz')
        models = {arm:restore(HERE/dataset/'training'/f'{arm}_s{seed}.pt') for arm in ('aligned','pooled')}
        models['ce'] = previous.restore(PREVIOUS/dataset/'training'/f'ce_s{seed}.pt')
        models['original'] = previous.original(dataset,seed)
        first=models['original']; bank=reference_bank(first,refs,'cuda')
        small=dev[:4]; values=previous.batch([previous.prepare_record(r) for r in small],'cuda')
        values['prefix']=first.prefix(values.pop('images'),stop=2)
        errors={}; latency={}
        for arm,net in models.items():
            result=net.forward_from(**values,reference=bank,pair_valid=previous.pair_mask(small,refs,'cuda')).cpu().numpy()
            root=PREVIOUS if arm in ('ce','original') else HERE
            key='baseline' if arm=='original' else 'score'
            name='ce' if arm=='original' else arm
            with np.load(root/dataset/'predictions'/f'{name}_s{seed}.npz') as saved:
                error=float(abs(result-saved[key][:4]).max())
            assert error<1e-4,(arm,error)
            errors[arm]=error
        native_q=previous.batch([previous.prepare_record(r) for r in small],'cuda')
        legal_small=previous.pair_mask(small,refs,'cuda')
        for arm,net in models.items():
            # Query encoding is included; the unchanged reference prefix is cached.
            previous.forward_batch(net,native_q,bank,legal_small)
            torch.cuda.synchronize(); begin=previous.time.perf_counter()
            for _ in range(10): previous.forward_batch(net,native_q,bank,legal_small)
            torch.cuda.synchronize()
            latency[arm]=1000*(previous.time.perf_counter()-begin)/10
        parts={arm:[] for arm in models}
        for start in range(0,len(queries),16):
            group=queries[start:start+16]
            values=previous.batch([previous.prepare_record(r) for r in group],'cuda')
            values['prefix']=first.prefix(values.pop('images'),stop=2)
            legal=previous.pair_mask(group,refs,'cuda')
            for arm,net in models.items():
                parts[arm].append(net.forward_from(**values,reference=bank,pair_valid=legal).cpu().numpy())
            if start%480==0: print('FROZEN_EVAL',dataset,seed,start,len(queries),round(previous.time.perf_counter()-tick,1),flush=True)
        scores={arm:np.concatenate(rows) for arm,rows in parts.items()}
        basepath=(previous.ROOT/'spatial_m12_full_20261029/predictions'/f'expanded_after_spatial_s{seed}.npz' if dataset=='private'
                  else previous.PUB/'predictions'/f'evaluation_full_adapt_s{seed}.npz')
        with np.load(basepath) as saved:
            assert np.array_equal(saved['ids'],ids) and np.array_equal(saved['y'],y)
            base_error=float(abs(scores['original']-saved['score']).max())
        assert base_error<1e-4,base_error
        for arm,score in scores.items():
            np.savez(out/f'{arm}_s{seed}.npz',score=score,baseline=scores['original'],y=y,ids=ids)
            records.append(dict(arm=arm,seed=seed,baseline=previous.metrics(scores['original'],y),
                                after=previous.metrics(score,y),native_development_error=errors[arm],baseline_replay_error=base_error,
                                native_batch4_ms=latency[arm],latency_reference_prefix_cached=True))
        previous.write(out/'results.json',dict(records=records,query_count=len(queries),
            seconds=previous.time.perf_counter()-tick,historical_cohort=True,parameter_selection_on_test=False,
            completed_seeds=sorted({r['seed'] for r in records})))
        print('COMPLETE_FROZEN_EVAL',dataset,seed,[r for r in records if r['seed']==seed],flush=True)
        del models,first,bank,values,net


if __name__=='__main__':
    p=previous.argparse.ArgumentParser();p.add_argument('--dataset',choices=['private','oss2'],required=True)
    main(p.parse_args().dataset)
