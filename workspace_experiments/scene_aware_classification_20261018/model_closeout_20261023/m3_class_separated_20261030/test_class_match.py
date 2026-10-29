import sys
from pathlib import Path
sys.path.insert(0, str(Path('E:/临时会话/sar_m123_standalone')))
import torch
from class_match import region_evidence


def inputs():
    q = torch.tensor([[[1., 0.], [0., 1.]]])
    r = torch.tensor([[[0., 1.], [1., 0.]], [[1., 0.], [1., 0.]]])
    t = torch.tensor([[[[0., .5], [.5, 0.]], [[.5, 0.], [0., .5]]]])
    w = torch.full((1, 2, 2), .5)
    return q, r, t, w, torch.ones(2, 2), torch.tensor([0, 1])


def test_aligned_identity():
    a = inputs()
    field, d, live = region_evidence(*a, 2, 'aligned')
    assert live.all()
    torch.testing.assert_close(d[0, 0], torch.zeros(2))
    assert d[0, 1, 1] == 2
    _, pooled, _ = region_evidence(*a, 2, 'pooled')
    torch.testing.assert_close(pooled[0, 0], torch.tensor([.5, .5]))
    torch.testing.assert_close(field.sum(1), torch.zeros(1, 2))


def test_reference_permutation():
    q, r, t, w, mass, labels = inputs()
    result = region_evidence(q, r, t, w, mass, labels, 2, 'aligned')
    ix = torch.tensor([1, 0])
    swapped = region_evidence(q, r[ix], t[:, ix], w[:, ix], mass[ix], labels[ix], 2, 'aligned')
    for a, b in zip(result, swapped):
        torch.testing.assert_close(a, b)


def test_empty_class_and_no_reference():
    q, r, t, w, mass, labels = inputs()
    field, _, live = region_evidence(q, r, t, w, mass, labels, 3, 'aligned')
    assert not live[:, 2].any() and not field[:, 2].any()
    for mode in ('aligned', 'pooled'):
        field, _, live = region_evidence(q, r, t*0, w, mass, labels, 3, mode)
        assert not live.any() and not field.any() and torch.isfinite(field).all()


if __name__ == '__main__':
    for test in (test_aligned_identity, test_reference_permutation, test_empty_class_and_no_reference):
        test()
    print('3 class-distance tests passed')
