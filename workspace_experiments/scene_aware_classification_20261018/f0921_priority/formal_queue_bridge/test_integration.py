import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from run import adapted,global_descriptor,margin_parts

HERE=Path(__file__).resolve().parent


class IntegrationTests(unittest.TestCase):
    def test_weighted_primal_matches_residual_solver(self):
        x=np.array([[1.,0],[0,1],[1,1],[-1,2.]])
        w0=np.array([[.2,-.1],[.4,.3]])
        p=x@w0;s=np.array([0,1,2]);y=np.array([0,0,1,1]);ridge=8.
        weights=np.array([.75,.75,1.5]);desired=2*np.eye(2)[y[s]]-1
        correction=np.linalg.solve(x[s].T@(weights[:,None]*x[s])+ridge*np.eye(2),x[s].T@(weights[:,None]*(desired-p[s])))
        actual,_=adapted(x@x.T,p,s,y)
        np.testing.assert_allclose(actual,p+x@correction,atol=1e-12)

    def test_query_cannot_modify_support_solve(self):
        x=np.array([[1.,0],[0,1],[1,1.]])
        p=np.zeros((3,2));s=np.array([0,1]);y=np.array([0,1,0])
        _,a=adapted(x@x.T,p,s,y)
        x[2]=[100,-100];p[2]=[45,-20]
        _,b=adapted(x@x.T,p,s,y)
        np.testing.assert_array_equal(a,b)

    def test_source_centering_ignores_target(self):
        x=np.array([[1.,0],[0,1],[1,1.]])
        a=global_descriptor(x,np.array([0,1]));x[2]=[500,-500]
        b=global_descriptor(x,np.array([0,1]))
        np.testing.assert_array_equal(a[:2],b[:2])

    def test_decomposition_independent_direct_sum(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(5,3));z=rng.normal(size=(5,2))
        old=x@x.T;new=old+z@z.T;s=np.array([0,1]);q=np.array([2,3,4])
        p=rng.normal(size=(5,2));y=np.array([0,1,1,0,1]);r=1-y[q]
        a,aa=adapted(old,p,s,y);b,ab=adapted(new,p,s,y)
        d,f=margin_parts(old,new,aa,ab,q,s,y[q],r)
        difference=b[q]-a[q]
        np.testing.assert_allclose(d+f,difference[np.arange(3),y[q]]-difference[np.arange(3),r],atol=1e-12)

    def test_actual_enrollment_isolated_and_nested(self):
        meta=pd.read_csv(HERE/'cohort.csv',dtype={'mmsi':str,'date':str})
        tasks=json.loads((HERE/'episodes.json').read_text());previous={}
        for t in tasks:
            src=np.array(t['source']);q=np.array(t['query'])
            self.assertFalse(set(meta.iloc[src].mmsi)&set(meta[meta.date.eq(t['fold'])].mmsi))
            self.assertEqual(meta.iloc[src].label.nunique(),7)
            for draw,s in enumerate(t['supports']):
                self.assertFalse(set(meta.iloc[s].mmsi)&set(meta.iloc[q].mmsi))
                self.assertEqual(len(s),len(set(meta.iloc[s].mmsi)))
                key=t['fold'],draw
                if key in previous:self.assertEqual(s[:len(previous[key])],previous[key])
                previous[key]=s


if __name__=='__main__':unittest.main()
