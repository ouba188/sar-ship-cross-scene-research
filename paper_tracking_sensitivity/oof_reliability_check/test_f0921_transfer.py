"""Check that memory-bounded target projection preserves the frozen recipe."""

import unittest

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits

from run_f0921_transfer import project_target_batches
from run_opensarship2_transfer import target_geometry


class BatchedProjectionTest(unittest.TestCase):
    def test_partial_final_batch_and_float16_extra_match_full_projection(self):
        rng = np.random.default_rng(31)
        source = rng.normal(size=(192, 160)).astype(np.float32)
        extra = rng.normal(size=(192, 80)).astype(np.float32)
        labels = np.arange(192) % 8
        with threadpool_limits(limits=1):
            pca = PCA(128, random_state=0).fit(source)
            pca_ssl = PCA(64, random_state=0).fit(extra)
            x = pca.transform(source)
            center, scale = x.mean(0), x.std(0)+1e-6
            xs = (x-center)/scale
            joint = np.column_stack((xs, pca_ssl.transform(extra)))
            ec, es = joint.mean(0), joint.std(0)+1e-6
            model = dict(pca=pca, pca_ssl=pca_ssl, center=center, scale=scale,
                expert_center=ec, expert_scale=es, residual_center=xs.mean(0),
                residual_scale=xs.std(0)+1e-6,
                base=RidgeClassifier(alpha=1., class_weight='balanced').fit(xs, labels),
                expert=RidgeClassifier(alpha=1., class_weight='balanced').fit((joint-ec)/es, labels))
            target = rng.normal(size=(53, 160)).astype(np.float32)
            second = rng.normal(size=(53, 80)).astype(np.float16)
            expected = target_geometry(model, target, second.astype(np.float32))
            actual = project_target_batches(model, target, second, batch_size=17)
        for one_shot, batched in zip(expected, actual):
            # Different float32 BLAS batch shapes alter rounding at a few e-6.
            np.testing.assert_allclose(batched, one_shot, atol=1e-5, rtol=1e-5)
        for one_shot, batched in zip(expected[:2], actual[:2]):
            np.testing.assert_array_equal(batched.argmax(1), one_shot.argmax(1))


if __name__ == '__main__':
    unittest.main()
