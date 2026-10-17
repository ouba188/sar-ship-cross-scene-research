"""Check that recovered inputs change the intended gate, not frozen B."""
import unittest
import numpy as np
import torch
import run_effect as run


class RecoveryEffectTests(unittest.TestCase):
    def test_missing_radar_only_changes_strict_gate(self):
        old=torch.tensor([[.5,0.,.6,1,0,1,.8],[0,0,.6,0,0,1,0]])
        new=old.clone(); new[:,1]=.9; new[:,4]=1
        self.assertEqual(run.eligibility(old,'strict').tolist(),[False,False])
        self.assertEqual(run.eligibility(new,'strict').tolist(),[True,False])
        torch.testing.assert_close(run.eligibility(old,'optional'),run.eligibility(new,'optional'))

    def test_real_source_baseline_and_optional_sampler_unchanged(self):
        torch.set_num_threads(2)
        data=run.load('Singapore')
        run.prepare(data,0)
        cached=data['cache']['fit'][0]['old_prob'].clone()
        traces=[]
        for version in ('old','new'):
            run.set_input(data,'optional',version)
            _,groups,_,_=run.base.fit_inventory(data)
            rng=np.random.default_rng(20261016)
            trace=[]
            for _ in range(5):
                sampled=run.fit.sample_step(data,groups,rng,True)
                if len(sampled)==1:
                    sampled+=run.fit.sample_step(data,groups,rng,False)
                trace.append([(i,p.tolist()) for i,p in sampled])
            traces.append(trace)
            torch.testing.assert_close(cached,data['cache']['fit'][0]['old_prob'],atol=0,rtol=0)
        self.assertEqual(traces[0],traces[1])
        self.assertTrue(data['metadata_changed'].any())
        for col in range(3):
            valid=data['legacy_geometry'][:,col+3]>0
            torch.testing.assert_close(data['legacy_geometry'][valid,col],
                                      data['geometries']['old'][valid,col],rtol=0,atol=0)


if __name__=='__main__':
    unittest.main(verbosity=2)
