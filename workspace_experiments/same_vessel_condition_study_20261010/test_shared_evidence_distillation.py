import unittest
import numpy as np
import pandas as pd
import torch
from shared_evidence_distillation import EvidenceStudent, centered, objective, vessel_targets


class SharedEvidenceTests(unittest.TestCase):
    def test_initial_student_preserves_prior_and_first_step_learns(self):
        torch.manual_seed(2)
        model = EvidenceStudent(torch.ones(8))
        x, prior = torch.randn(5, 512), torch.randn(5, 8)
        torch.testing.assert_close(model(x, prior), prior, rtol=0, atol=0)
        model(x, prior)[:, 0].sum().backward()
        self.assertGreater(float(model.network[-1].weight.grad.norm()), 0)

    def test_other_teacher_excludes_all_chips_in_focal_acquisition(self):
        meta = pd.DataFrame(dict(mmsi=[1, 1, 1, 2], product=['a', 'a', 'b', 'c']))
        x = np.array([[2.], [4.], [10.], [77.]], np.float32)
        mean, other, valid = vessel_targets(x, meta)
        np.testing.assert_allclose(mean[:3, 0], 6.5)
        np.testing.assert_allclose(other[:3, 0], [10, 10, 3])
        np.testing.assert_array_equal(valid, [True, True, True, False])

    def test_teacher_nonparticipants_and_common_offset_do_not_affect_loss(self):
        scores, teacher = torch.randn(3, 8), torch.randn(3, 8)
        y, valid = torch.tensor([0, 1, 2]), torch.tensor([True, False, True])
        a = objective(scores, y, teacher, valid, 'self_teacher')[0]
        changed = teacher.clone()+19
        changed[1] = torch.randn(8)*1000
        b = objective(scores, y, changed, valid, 'self_teacher')[0]
        torch.testing.assert_close(a, b)

    def test_supervised_arm_independent_of_teacher(self):
        scores = torch.randn(4, 8)
        y, valid = torch.arange(4), torch.ones(4, dtype=torch.bool)
        a = objective(scores, y, scores, valid, 'supervised')[0]
        b = objective(scores, y, scores+100, valid, 'supervised')[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
