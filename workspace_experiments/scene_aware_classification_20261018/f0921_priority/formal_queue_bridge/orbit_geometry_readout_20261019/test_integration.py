import unittest
import numpy as np
from run_experiment import kernels


class IntegrationTests(unittest.TestCase):
    def test_unknown_links_preserved_and_no_geometry_exact_fallback(self):
        rng=np.random.default_rng(4)
        n=6
        f={'prepool':rng.normal(size=(n,2,12,5)),
           'mass12':np.ones((n,12)), 'grid2':rng.normal(size=(n,2,4,5))}
        spatial={'valid':np.array([1,1,0,1,0,1],bool)}
        for name in ['image','orbit']:
            spatial[name]=rng.normal(size=(n,2,8,5))
            spatial[name+'_mass']=np.ones((n,8))
        kg=np.eye(n)
        actual=kernels(f,spatial,np.array([0,1,2]),kg)
        for name in ['add_pooled','add_image','add_orbit']:
            np.testing.assert_array_equal(actual[name][~spatial['valid']],actual['base'][~spatial['valid']])
            self.assertGreater(np.linalg.eigvalsh(actual[name]-actual['base']).min(),-1e-10)
        spatial['valid'][:]=False
        fallback=kernels(f,spatial,np.array([0,1,2]),kg)
        for name in ['add_pooled','add_image','add_orbit']:
            np.testing.assert_array_equal(fallback[name],fallback['base'])


if __name__=='__main__':
    unittest.main()
