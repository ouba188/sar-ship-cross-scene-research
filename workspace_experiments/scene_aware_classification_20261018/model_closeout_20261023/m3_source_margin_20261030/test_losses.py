import torch
from losses import focal_logits, margin_keep, objective


def test_initial_and_correct_focus():
    old = torch.tensor([[2., 0.], [0., 2.]])
    y = torch.tensor([0, 0])
    assert focal_logits(old, old, y) == 0
    assert margin_keep(old, old, y) == 0
    moved = old + torch.tensor([[1., 0.], [1., 0.]])
    torch.testing.assert_close(focal_logits(moved, old, y), torch.tensor(1.75))


def test_margin_direction_and_shift():
    old = torch.tensor([[2., 0., -1.], [0., 2., -1.]])
    y = torch.tensor([0, 0])
    improved = old + torch.tensor([[1., 0., 0.], [-9., 0., 0.]])
    assert margin_keep(improved, old, y) == 0
    worse = torch.tensor([[1., 1., -1.], [0., 2., -1.]], requires_grad=True)
    loss = margin_keep(worse, old, y)
    torch.testing.assert_close(loss, torch.tensor(1.25))
    torch.testing.assert_close(loss, margin_keep(worse + 7, old - 3, y))
    loss.backward()
    assert worse.grad[0, 0] < 0 and worse.grad[0, 1] > 0
    assert not worse.grad[1].any()


def test_teacher_detached_and_wrong_learnable():
    old = torch.tensor([[0., 2.]], requires_grad=True)
    y = torch.tensor([0])
    for arm in ('fd_lm', 'margin'):
        score = old.detach().clone().requires_grad_(True)
        ce, guard = objective(score, old, y, arm)
        (ce + guard).backward()
        assert score.grad[0, 0] < 0 and score.grad[0, 1] > 0
        assert old.grad is None


if __name__ == '__main__':
    for test in (test_initial_and_correct_focus, test_margin_direction_and_shift,
                 test_teacher_detached_and_wrong_learnable):
        test()
    print('3 source-protection tests passed')
