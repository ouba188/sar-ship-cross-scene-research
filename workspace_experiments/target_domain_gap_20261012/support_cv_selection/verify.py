"""Check fold isolation, result accounting, and full64 parity of the CV predictor."""

from run import *


def main():
    checked=0
    for path in (OUT/'oof').glob('*/draw_*.npz'):
        z=old.load_npz(path)
        np.testing.assert_allclose(z['oof'].sum(-1),1.,atol=1e-12)
        for repeat in range(3):
            for fold in range(4):
                val=z['fold_ids'][repeat]==fold
                if not val.any() or set(z['groups'][val])&set(z['groups'][~val]):
                    raise ValueError('Fold group isolation failed')
        checked+=1
    frame=pd.read_csv(OUT/'metrics.csv').set_index(['task','draw','scope','policy'])
    classes=pd.read_csv(OUT/'class_recalls.csv').groupby(['task','draw','scope','policy']).recall_delta.mean()
    np.testing.assert_allclose(classes.reindex(frame.index),frame.delta_ba,atol=1e-10)
    folder=old.FOLDERS[-1]
    old.previous.MATCHED=True
    ctx=old.ExpertContext(folder)
    data=old.load_task(folder)
    hs,ht=old.head_inputs(ctx,data)
    stats=old.source_statistics(hs,ctx.source_labels)
    support=data['support'][0]
    head=old.load_npz(old.ROOT/'backbone_ablation_20261010/targets/oss2/oss2/coefficients.npz')
    pe=(ht[:,:-1]@head['dual_coef']+head['dual_intercept']).astype(np.float32).astype(float)
    _,ke=ctx.target_components(data,support)
    kb=old.base_cross(data,support,support,ctx.info['eta'])
    p=fold_predict(stats,ht[support],pe[support],ke[support],data['prior'][support],kb,
                   np.arange(64),data['support_labels'][0])
    expected=old.load_npz(PREVIOUS/'target/oss2/predictions.npz')['full'][0][:,support]
    error=float(np.max(np.abs(p-expected)))
    np.testing.assert_allclose(p,expected,atol=1e-11,rtol=1e-9)
    old.write(OUT/'VERIFIED.json',dict(support_sets_checked=checked,
        actual_full64_parity=dict(task='oss2',draw=0,arms=list(ARMS),probability_max_error=error),
        class_recall_accounting=True,all_fold_groups_isolated=True))
    print('VERIFIED',checked,'support sets; full64 probability error',error,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
