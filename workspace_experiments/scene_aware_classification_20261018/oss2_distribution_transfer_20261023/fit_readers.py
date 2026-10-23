"""Source-only OSS2 ridge readers; no calibration or test-label evaluation."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
OUT = HERE / 'readers'
CLASSES = 8
LOGIT_SCALE = 3.


class View:
    def __init__(self, x, fit):
        self.scale = StandardScaler().fit(x[fit])
        self.pca = PCA(n_components=128, svd_solver='randomized',
                       random_state=17, iterated_power=4)
        self.pca.fit(self.scale.transform(x[fit]))
        self.post = StandardScaler().fit(
            self.pca.transform(self.scale.transform(x[fit])))

    def transform(self, x):
        return self.post.transform(self.pca.transform(self.scale.transform(x)))


def weights(table, fit):
    part = table.iloc[fit]
    assert set(part.class_id) == set(range(CLASSES))
    count = part.groupby('mmsi').mmsi.transform('size').to_numpy()
    ships = part.groupby('class_id').mmsi.transform('nunique').to_numpy()
    w = len(fit) / (CLASSES * count * ships)
    np.testing.assert_allclose(
        [w[part.class_id.eq(c)].sum() for c in range(CLASSES)],
        np.full(CLASSES, len(fit) / CLASSES))
    return w


def balanced_nll(y, score):
    assert set(y) == set(range(CLASSES))
    loss = logsumexp(score, axis=1) - score[np.arange(len(y)), y]
    return float(np.mean([loss[y == c].mean() for c in range(CLASSES)]))


def check_isolation(table, left, right):
    assert not np.intersect1d(left, right).size
    for column in ('mmsi', 'pass_group'):
        assert not set(table.iloc[left][column]) & set(table.iloc[right][column])


def save_reader(name, h, fit, held, view, head, w, alpha):
    W = ((head.coef_.astype(np.float64) / view.post.scale_)
         @ view.pca.components_.astype(np.float64)) / view.scale.scale_ * LOGIT_SCALE
    h0 = h[fit].mean(0, dtype=np.float64)
    b = head.predict(view.transform(h0[None]))[0] * LOGIT_SCALE
    intercept = b - W @ h0
    path = OUT / 'models' / f'{name}.npz'
    np.savez_compressed(path, W=W, intercept=intercept, h0=h0, b=b,
        head_coef=head.coef_, head_intercept=head.intercept_, fit=fit, held=held,
        fit_weights=w, scale_mean=view.scale.mean_, scale_scale=view.scale.scale_,
        pca_mean=view.pca.mean_, pca_components=view.pca.components_,
        post_mean=view.post.mean_, post_scale=view.post.scale_,
        alpha=alpha, logit_scale=LOGIT_SCALE)
    x = h.astype(np.float64)
    direct = head.predict(view.transform(x)) * LOGIT_SCALE
    with np.load(path) as saved:
        linear = x @ saved['W'].T + saved['intercept']
        z = (x - saved['scale_mean']) / saved['scale_scale']
        z = (z - saved['pca_mean']) @ saved['pca_components'].T
        z = (z - saved['post_mean']) / saved['post_scale']
        replay = (z @ saved['head_coef'].T + saved['head_intercept']) * LOGIT_SCALE
    linear_error = float(np.max(abs(linear - direct)))
    projection_error = float(np.max(abs(replay - direct)))
    assert linear_error < 1e-7 and projection_error < 1e-7
    assert np.array_equal(linear.argmax(1), direct.argmax(1))
    assert np.array_equal(replay.argmax(1), direct.argmax(1))
    fp32 = head.predict(view.transform(h)) * LOGIT_SCALE
    check = dict(reader=name, fit_rows=len(fit), held_rows=len(held), alpha=alpha,
        linear_max_error=linear_error, saved_projection_max_error=projection_error,
        linear_top1_changes=0, saved_projection_top1_changes=0,
        float32_max_error=float(np.max(abs(fp32-direct))),
        float32_top1_changes=int(np.sum(fp32.argmax(1) != direct.argmax(1))))
    return direct, check


def main():
    started = time.perf_counter()
    table = pd.read_csv(HERE / 'cohort.csv', dtype={'mmsi': str})
    ids = table.sample_id.to_numpy(str)
    with np.load(HERE / 'features.npz') as raw:
        np.testing.assert_array_equal(raw['sample_ids'], ids)
        h, valid = raw['layer3'], raw['valid']
    assert h.shape == (len(table), 2048) and h.dtype == np.float32
    assert valid.shape == (len(table), 2) and valid.all()
    assert np.isfinite(h).all()
    with np.load(HERE / 'indices.npz') as source:
        splits = {key: source[key] for key in source.files}
    np.testing.assert_array_equal(splits['sample_ids'], ids)
    (OUT / 'models').mkdir(parents=True, exist_ok=True)
    candidates, selections, checks, isolation = [], [], [], []
    all_test = []
    for fold in range(6):
        tick = time.perf_counter()
        train, val, test = [splits[f'fold{fold}_{part}']
                            for part in ('train', 'validation', 'test')]
        check_isolation(table, train, val)
        check_isolation(table, train, test)
        check_isolation(table, val, test)
        all_test.extend(test.tolist())
        ytrain = table.iloc[train].class_id.to_numpy()
        yval = table.iloc[val].class_id.to_numpy()
        w = weights(table, train)
        view = View(h, train)
        ztrain = view.transform(h[train])
        zval = view.transform(h[val].astype(np.float64))
        choices = []
        for alpha in (1., 10., 100.):
            head = Ridge(alpha=alpha).fit(ztrain, np.eye(CLASSES)[ytrain],
                                         sample_weight=w)
            risk = balanced_nll(yval, head.predict(zval) * LOGIT_SCALE)
            candidates.append(dict(fold=fold, alpha=alpha, validation_balanced_nll=risk))
            choices.append((risk, alpha, head))
        risk, alpha, head = min(choices, key=lambda item: item[0])
        base, check = save_reader(f'fold{fold}_outer', h, train,
            np.sort(np.concatenate([val, test])), view, head, w, alpha)
        checks.append(check)
        np.testing.assert_allclose(balanced_nll(yval, base[val]), risk, atol=1e-12)
        oof = np.zeros((len(table), CLASSES), dtype=np.float64)
        inner_bin = np.full(len(table), -1, dtype=np.int64)
        coverage = np.zeros(len(table), dtype=np.int64)
        for inner in range(4):
            prefix = f'fold{fold}_inner{inner}'
            fit, held = splits[prefix + '_fit'], splits[prefix + '_held']
            assert np.isin(fit, train).all() and np.isin(held, train).all()
            check_isolation(table, fit, held)
            iw = weights(table, fit)
            iv = View(h, fit)
            ih = Ridge(alpha=alpha).fit(iv.transform(h[fit]),
                np.eye(CLASSES)[table.iloc[fit].class_id.to_numpy()], sample_weight=iw)
            score, check = save_reader(prefix, h, fit, held, iv, ih, iw, alpha)
            checks.append(check)
            oof[held], inner_bin[held] = score[held], inner
            coverage[held] += 1
            isolation.append(dict(fold=fold, inner=inner, fit_rows=len(fit),
                held_rows=len(held), fit_classes=8,
                held_classes=int(table.iloc[held].class_id.nunique()),
                mmsi_overlap=0, pass_overlap=0))
        np.testing.assert_array_equal(np.flatnonzero(coverage), train)
        assert (coverage[train] == 1).all()
        assert np.isfinite(base).all() and np.isfinite(oof).all()
        path = OUT / f'fold{fold}_inputs.npz'
        np.savez_compressed(path, sample_ids=ids, train=train, validation=val, test=test,
            base=base, oof=oof, inner_bin=inner_bin, selected_alpha=alpha,
            valid=valid, logit_scale=LOGIT_SCALE)
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved['base'], base)
            np.testing.assert_array_equal(saved['oof'], oof)
        selections.append(dict(fold=fold, selected_alpha=alpha,
            validation_balanced_nll=risk, train_rows=len(train),
            validation_rows=len(val), test_rows=len(test),
            seconds=time.perf_counter()-tick))
        print(json.dumps(selections[-1]), flush=True)
    np.testing.assert_array_equal(np.sort(all_test), np.arange(len(table)))
    pd.DataFrame(candidates).to_csv(OUT / 'alpha_candidates.csv', index=False)
    pd.DataFrame(selections).to_csv(OUT / 'selections.csv', index=False)
    pd.DataFrame(checks).to_csv(OUT / 'linear_checks.csv', index=False)
    pd.DataFrame(isolation).to_csv(OUT / 'inner_isolation.csv', index=False)
    summary = dict(rows=len(table), feature_shape=list(h.shape), readers=len(checks),
        pca_fits=30, ridge_fits=42, selected_alphas=[s['selected_alpha'] for s in selections],
        max_linear_error=max(c['linear_max_error'] for c in checks),
        max_saved_projection_error=max(c['saved_projection_max_error'] for c in checks),
        linear_top1_changes=sum(c['linear_top1_changes'] for c in checks),
        float32_max_error=max(c['float32_max_error'] for c in checks),
        float32_top1_changes=sum(c['float32_top1_changes'] for c in checks),
        seconds=time.perf_counter()-started, outer_mmsi_pass_isolation=True,
        inner_mmsi_pass_isolation=True, inner_held_covers_train_once=True,
        test_labels_evaluated=False, calibration_trained=False,
        scope='source ridge OOF; outer source validation selects shared alpha')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
