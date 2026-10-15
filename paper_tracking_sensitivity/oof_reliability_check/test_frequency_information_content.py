import numpy as np
import pytest

from frequency_information_content import pooled_intervention


@pytest.mark.parametrize('mode', ['no_quadrant_layout', 'no_channel_identity'])
def test_pooled_projection_keeps_band_mass_and_is_idempotent(mode):
    rng = np.random.default_rng(21)
    z = rng.normal(size=(5, 4, 64))
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    w = rng.dirichlet(np.ones(4), size=5)
    e = (z*np.sqrt(w[:, :, None])).reshape(5, -1)
    changed = pooled_intervention(e, w, mode)
    np.testing.assert_allclose(np.sum(changed.reshape(5, 4, 64)**2, axis=-1), w, atol=1e-12)
    np.testing.assert_allclose(pooled_intervention(changed, w, mode), changed, atol=1e-12)
    b = changed.reshape(5, 4, 16, 4)
    axis = -1 if mode == 'no_quadrant_layout' else -2
    assert np.max(np.std(b, axis=axis)) < 1e-12


def test_haar_common_mode_cancellation_and_band_interventions():
    torch = pytest.importorskip('torch')
    from learned_frequency_model import haar
    from frequency_information_content import band_intervention
    x = torch.randn(2, 8, 16, 16, dtype=torch.float64)
    offset = torch.randn(2, 8, 1, 1, dtype=torch.float64)
    original, shifted = haar(x), haar(x+offset)
    torch.testing.assert_close(original[:, 1:], shifted[:, 1:])
    torch.testing.assert_close(shifted[:, 0]-original[:, 0], 2*offset.expand_as(original[:, 0]))
    mean = band_intervention(original, 'no_local_band_layout')
    torch.testing.assert_close(mean.mean((-1, -2)), original.mean((-1, -2)))
    assert torch.max(mean.std((-1, -2))) < 1e-12
    unsigned = band_intervention(original, 'unsigned_bands')
    torch.testing.assert_close(unsigned.square(), original.square())


def test_retrained_model_only_changes_the_declared_band_transform(monkeypatch):
    torch = pytest.importorskip('torch')
    import train_frequency_information as study
    from learned_frequency_model import FrequencyResidual
    torch.manual_seed(12)
    original = FrequencyResidual('selective')
    changed = study.InformationResidual('unsigned_bands')
    changed.load_state_dict(original.state_dict())
    monkeypatch.setattr(study, 'band_intervention', lambda bands, mode: bands)
    x, c = torch.randn(3, 512, 16, 16), torch.randn(3, 2)
    a, ga = original(x, c)
    b, gb = changed(x, c)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(ga, gb, rtol=0, atol=0)
    assert sum(p.numel() for p in changed.parameters()) == 35876
    b[:, 0].sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in changed.parameters())
