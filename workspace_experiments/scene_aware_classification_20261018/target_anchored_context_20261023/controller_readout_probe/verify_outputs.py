"""Check saved input scale, flags, source fits, baseline identity and pooled rows."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
EXP = PARENT.parent/'context_expansion_20261022'
READOUTS = {'whole64':'target','foreground64':'target_fg','background64':'target_bg'}


def main():
    targets = pd.read_csv(EXP/'readout/targets.csv', dtype={'mmsi':str})
    f = np.load(EXP/'readout/features.npz')
    p = pd.read_csv(HERE/'predictions.csv')
    pooled = pd.read_csv(HERE/'pooled.csv')
    assert len(p) == 5695*4*3 and not p.duplicated(['arm','seed','instance_id']).any()
    assert np.isfinite(p[['s0','s1','s2','b0','b1','b2']]).all().all()
    assert p.active.all()
    assert len(list((HERE/'models').glob('*.pt'))) == 72
    assert len(pd.read_csv(HERE/'metrics.csv')) == 72
    assert len(pd.read_csv(HERE/'curves.csv')) == 2880
    checks = []
    for fold in range(6):
        parent = np.load(PARENT/f'fold{fold}_inputs.npz')
        own = np.load(PARENT/f'fold{fold}_own_input.npz')['env']
        projection = np.load(HERE/'projections'/f'fold{fold}.npz')
        np.testing.assert_array_equal(projection['train'], parent['train'])
        np.testing.assert_array_equal(projection['instance_ids'], targets.instance_id)
        np.testing.assert_allclose(projection['scale_mean'], f['target'][parent['train']].mean(0,dtype=np.float64), atol=1e-12)
        for arm, feature in READOUTS.items():
            saved = np.load(HERE/arm/f'fold{fold}_env.npz')
            valid = np.ones(len(targets),bool) if feature == 'target' else f[feature+'_valid']
            np.testing.assert_array_equal(saved['valid_features'], valid)
            np.testing.assert_array_equal(saved['env'][:,64:], own[:,64:])
            assert not saved['env'][~valid,:64].any()
            error = float(np.abs(np.linalg.norm(saved['env'][valid,:64],axis=1)-np.sqrt(32)).max())
            assert error < 3e-6
            # Reconstruct a small fixed sample directly from serialized PCA/scaler.
            ix = np.unique(np.r_[parent['train'][0], parent['validation'][0], parent['test'][0]])
            # Match the saved float32 scaler operations and PCA projection order.
            # Near-constant stem channels amplify rounding if evaluated in float64 instead.
            z = f[feature][ix].copy()
            z -= projection['scale_mean'].astype(z.dtype)
            z /= projection['scale_scale'].astype(z.dtype)
            z = z @ projection['pca_components'].T
            z -= projection['pca_mean'] @ projection['pca_components'].T
            expected = z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-8)*np.sqrt(32)
            expected[~valid[ix]] = 0
            transform_error = float(np.abs(saved['env'][ix,:64]-expected).max())
            assert transform_error < 1e-5, (fold, arm, transform_error)
            checks.append(dict(fold=fold, arm=arm, valid=int(valid.sum()), norm_error=error,
                               projection_replay_error=transform_error, layout_exact=True))
        constant = np.load(HERE/'constant_own_full'/f'fold{fold}_env.npz')['env']
        expected = own[parent['train']].mean(0,dtype=np.float64).astype(np.float32)
        np.testing.assert_array_equal(constant, np.broadcast_to(expected,constant.shape))
        base = pd.DataFrame(parent['base'][parent['test']], columns=['b0','b1','b2'], index=targets.iloc[parent['test']].instance_id)
        q = p[p.fold.eq(fold)]
        np.testing.assert_allclose(q[['b0','b1','b2']],base.loc[q.instance_id],atol=1e-12)
        np.testing.assert_array_equal(q.base_prediction,base.loc[q.instance_id].to_numpy().argmax(1))
    for (arm, seed), q in p.groupby(['arm','seed']):
        assert len(q) == len(targets)
        q = q.set_index('instance_id').loc[targets.instance_id]
        np.testing.assert_array_equal(q.class_id,targets.class_id)
        np.testing.assert_array_equal(q.prediction,q[['s0','s1','s2']].to_numpy().argmax(1))
        correct = q.prediction.eq(q.class_id)
        result = pooled[pooled.arm.eq(arm)&pooled.seed.eq(seed)&pooled.scope.eq('all')].iloc[0]
        np.testing.assert_allclose([result.OA,result.BA],
            [correct.mean()*100,correct.groupby(q.class_id).mean().mean()*100],atol=1e-10)
    constant_gates = pd.read_csv(HERE/'constant_own_full/gates.csv')
    assert constant_gates.minimum.eq(constant_gates.maximum).all()
    pd.DataFrame(checks).to_csv(HERE/'input_verification.csv',index=False)
    summary = dict(heads=72, predictions=len(p), validation_points=2880,
        max_norm_error=max(c['norm_error'] for c in checks),
        max_projection_replay_error=max(c['projection_replay_error'] for c in checks),
        exact_parent_layout=True, constant_source_mean_exact=True,
        baseline_top1_unchanged=True, pooled_recomputed=True, finite=True)
    (HERE/'verification_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
