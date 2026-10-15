import unittest
import numpy as np
import pandas as pd
from causal_relation_audit import scoped_teacher, relation_components


class RelationAuditTests(unittest.TestCase):
    def metadata(self):
        return pd.DataFrame(dict(mmsi=[1]*4, product=['a','a','b','c'], port=['P','P','P','Q'],
            class_id=[0]*4, incidence_deg=[30.,30.,32.,40.], pass_name=['ASC']*3+['DESC']), index=[9,8,7,6])

    def test_scope_excludes_focal_scene_and_equal_weights_products(self):
        scores = np.zeros((4,8)); scores[:,0] = [1,3,8,14]
        meta = self.metadata()
        all_t, counts, *_ = scoped_teacher(scores, meta, 'all_other')
        same_t, same_n, *_ = scoped_teacher(scores, meta, 'same_port')
        cross_t, cross_n, *_ = scoped_teacher(scores, meta, 'cross_port')
        self.assertEqual(all_t[0,0], 11.)
        self.assertEqual(all_t[3,0], 5.)
        self.assertEqual(same_t[0,0], 8.)
        self.assertEqual(same_n[3], 0)
        self.assertEqual(cross_t[0,0], 14.)
        self.assertEqual(cross_n[3], 2)

    def test_pair_energy_decomposition(self):
        rng = np.random.default_rng(9)
        a,b = rng.normal(size=(2,8))
        _,_,e = relation_components(a,b,3)
        pair_sum = sum(((b[i]-b[j])-(a[i]-a[j]))**2 for i in range(8) for j in range(i+1,8))/64
        self.assertAlmostEqual(e['full_energy'], pair_sum)
        self.assertAlmostEqual(e['full_energy'], e['true_class_energy']+e['rival_only_energy'])
        self.assertAlmostEqual(e['true_class_energy'], e['beneficial_energy']+e['harmful_energy']+e['partial_recovery_energy'])

    def test_wrong_overall_teacher_can_have_beneficial_local_relation(self):
        base = np.array([0.,2.,1.,0.,0.,0.,0.,0.])
        teacher = np.array([1.,3.,0.,0.,0.,0.,0.,0.])
        bm,tm,e = relation_components(base,teacher,0)
        self.assertNotEqual(teacher.argmax(),0)
        self.assertLess(bm[2],0)
        self.assertGreater(tm[2],0)
        self.assertGreater(e['beneficial_energy'],0)

    def test_common_offset_does_not_change_relations(self):
        base,teacher = np.arange(8,dtype=float),np.arange(8,dtype=float)[::-1]
        a,b = relation_components(base,teacher,0),relation_components(base+19,teacher-33,0)
        np.testing.assert_allclose(a[0],b[0])
        np.testing.assert_allclose(a[1],b[1])
        self.assertEqual(a[2],b[2])


if __name__ == '__main__':
    unittest.main()
