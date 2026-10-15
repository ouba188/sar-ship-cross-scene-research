import unittest
import numpy as np
import pandas as pd
import torch
from local_relation_distillation import auxiliary_rows, positive_relations, matched_means, initial_gradient_scale


class LocalTeachingTests(unittest.TestCase):
    def test_does_not_pull_down_a_better_student(self):
        teacher = torch.tensor([[1.,0.,0.,0.,0.,0.,0.,0.]])
        score = torch.tensor([[2.,0.,0.,0.,0.,0.,0.,0.]], requires_grad=True)
        loss = auxiliary_rows(score,teacher,torch.tensor([0]),'local').sum()
        loss.backward()
        self.assertEqual(float(loss),0.)
        torch.testing.assert_close(score.grad,torch.zeros_like(score),rtol=0,atol=0)

    def test_wrong_top1_teacher_retains_local_positive_edges(self):
        teacher = torch.tensor([[1.,3.,0.,0.,0.,0.,0.,0.]])
        _,mask = positive_relations(teacher,torch.tensor([0]))
        self.assertFalse(bool(mask[0,1]))
        self.assertTrue(bool(mask[0,2]))
        score = torch.zeros_like(teacher,requires_grad=True)
        auxiliary_rows(score,teacher,torch.tensor([0]),'local').sum().backward()
        self.assertLess(float(score.grad[0,0]),0)
        self.assertEqual(float(score.grad[0,1]),0.)
        self.assertGreater(float(score.grad[0,2]),0)

    def test_cap_and_common_offsets(self):
        teacher = torch.tensor([[9.,0.,0.,0.,0.,0.,0.,0.]])
        score = torch.zeros_like(teacher)
        a = auxiliary_rows(score,teacher,torch.tensor([0]),'local')
        torch.testing.assert_close(a,torch.tensor([4.]))
        torch.testing.assert_close(a,auxiliary_rows(score+17,teacher-11,torch.tensor([0]),'local'))

    def test_matched_teachers_average_products_not_chip_counts(self):
        values = np.zeros((5,8),np.float32); values[:,0]=[0,2,4,10,12]
        frame = pd.DataFrame(dict(anchor=[0,0],same_indices=[(1,2),(3,)],other_indices=[(3,),(4,)]))
        same,other,valid = matched_means(values,frame)
        self.assertEqual(float(same[0,0]),6.5)
        self.assertEqual(float(other[0,0]),11.)
        self.assertEqual(int(valid.sum()),1)

    def test_initial_gradient_calibration_matches_magnitude(self):
        prior=torch.zeros(3,8); teacher=torch.zeros(3,8); teacher[:,0]=1
        ratio,norms=initial_gradient_scale(prior,teacher,torch.tensor([0,0,0]),np.ones(3,bool),np.ones(3)/3,'local')
        self.assertAlmostEqual(ratio*norms['local'],norms['full'])


if __name__ == '__main__':
    unittest.main()
