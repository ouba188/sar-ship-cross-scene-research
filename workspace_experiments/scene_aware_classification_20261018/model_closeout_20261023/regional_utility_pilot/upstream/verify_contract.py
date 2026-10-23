"""Bounded handoff checks: membership, raw inputs, model replay, no external metrics."""
from pipeline import *


def main():
    table,tr,va=load()
    external=table[table.role.ne('strategy_fit')]
    upstream=table.iloc[np.r_[tr,va]]
    for field in ('mmsi','pass_group','row_index'):
        assert not set(external[field]) & set(upstream[field])
    with np.load(HERE/'summaries.npz') as z:
        data={k:z[k] for k in z.files}
    # Compare the raw 69-D contract against the existing implementation on real rows.
    sys.path.insert(0,str(ROOT/'obb_dynamic_readout'))
    import run_heads as original
    sample=np.unique(np.r_[tr[:4],va[:4],np.flatnonzero(table.role.eq('utility_train'))[:4],
                            np.flatnonzero(table.role.eq('selection'))[:4],np.flatnonzero(table.role.eq('evaluation'))[:4]])
    summary=np.zeros((len(sample),2,7,8,3))
    summary[:,:,1:4]=data['summary1'][sample]
    info=dict(summary=summary,quality=data['quality'][sample],geometry=data['geometry'][sample])
    reports=[]
    for seed in SEEDS:
        with np.load(HERE/f'strategies_seed{seed}.npz') as z:
            indices,role,logits,raw=[z[k] for k in ('indices','role','logits','regional_raw')]
            assert z['strategy_names'].tolist()==['s0','body','full']
        np.testing.assert_array_equal(indices,table.row_index)
        np.testing.assert_array_equal(role,table.role)
        assert logits.shape==(len(table),3,8) and raw.shape==(len(table),8,69)
        assert np.isfinite(logits).all() and np.isfinite(raw).all()
        original_raw=original.raw_inputs(logits[sample,0],info,'dynamic')
        raw_error=float(abs(raw[sample]-original_raw).max())
        assert raw_error==0
        if seed==SEEDS[0]:
            for arm,typ in [('body',RegionHead),('full',RegionHead),('attention',AttentionHead),('calibration',Calibration),('five',Five),('neighborhood',Neighborhood)]:
                ck=torch.load(HERE/'models'/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=False)
                np.testing.assert_array_equal(ck['fit_indices'],table.iloc[tr].row_index)
                np.testing.assert_array_equal(ck['validation_indices'],table.iloc[va].row_index)
        s0=logits[:,0]
        for j,arm in enumerate(('body','full'),1):
            ck=torch.load(HERE/'models'/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=False)
            model=RegionHead().eval();model.load_state_dict(ck['state'])
            r=regional_raw(s0,data['summary1'],data['quality'],data['geometry'],arm)
            x=((r-ck['mu'])/ck['scale']).astype(np.float32)
            replay=s0+predict(model,x)
            error=float(abs(replay-logits[:,j]).max())
            assert error==0
            one=s0[sample]+predict(model,x[sample])
            one_error=float(abs(one-logits[sample,j]).max())
            assert one_error<2e-5
            np.testing.assert_array_equal(one.argmax(1),logits[sample,j].argmax(1))
            reports.append(dict(seed=seed,arm=arm,selected_epoch=ck['epoch'],full_replay_error=error,
                                small_batch_replay_error=one_error,raw_contract_error=raw_error))
    pd.DataFrame(reports).to_csv(HERE/'contract_replay_checks.csv',index=False)
    selections=[json.loads(p.read_text()) for p in sorted(HERE.glob('selection_*_*.json'))]
    receipts={mode:json.loads((HERE/f'extraction_{mode}.json').read_text()) for mode in ('scale','summaries')}
    reader_cost=sum(json.loads((HERE/f'reader_{layer}_receipt.json').read_text())['seconds'] for layer in ('layer1','layer3'))
    save_json('IMPLEMENTATION_RECEIPT.json',dict(status='complete',completed_utc=utc(),source_pool=4702,retained=len(table),
        roles=table.groupby('role').size().to_dict(),internal_fit=len(tr),internal_validation=len(va),
        internal_excluded=int(table.upstream_role.eq('fit_excluded_mmsi').sum()),
        seed=list(SEEDS),axis_readers=1,aux_readers=2,neural_heads=len(selections),
        original_historical_models_unchanged=True,full_stack_oof_within_strategy_fit=False,
        external_role_labels_used_for_supervision_or_selection=False,external_label_usage='Metadata coverage only; no upstream performance-based split/fit/selection.',
        historical_target_overlap=0,backbone_fits=0,source_extraction_passes=2,
        seconds=dict(successful_head_train_selection_and_replay=sum(r['seconds'] for r in selections),
            auxiliary_reader_fit=reader_cost,scale_pass=receipts['scale']['seconds'],summary_pass=receipts['summaries']['seconds'],
            backbone_gpu_wall=sum(r['timing']['backbone_gpu_wall'] for r in receipts.values())),
        cost_scope='Successful function/stage wall times; not full end-to-end wall time. Debugging, imports, split preparation and failed checks not included.',
        corrected_failures=['Five-head continuous-risk equality changed to same-batch exact selection replay plus bounded batch-score error/top1 equality.',
            'Trusted feature ID object-array loading enabled; existing fitted A/D/T were not refitted for this loader fix.',
            'Rotated single-channel CUDA area mask path disagreed at 180 degrees; exact CPU 16x16 area-bin weights restored original layer3 GAP. No failed extraction trained later heads.'],
        validation_checks=reports,outputs=[f'strategies_seed{s}.npz' for s in SEEDS]))
    print('CONTRACT COMPLETE',len(table),'rows, three seeds, no external performance evaluation',flush=True)


if __name__=='__main__':
    main()
