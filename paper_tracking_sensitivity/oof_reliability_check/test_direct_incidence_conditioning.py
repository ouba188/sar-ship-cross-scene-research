import numpy as np

from direct_incidence_conditioning import condition_features, extend_transforms, predict_conditioned
from conditioned_vessel_drift import incidence_shift
from protected_drift_activation import fit_source_heads


def test_same_information_as_response_correction():
    angles = np.array([np.nan,38,33,43,48])
    u = condition_features(angles)
    np.testing.assert_allclose(u, [[0,0],[0,0],[-1,1],[1,1],[2,4]])
    coef = np.arange(20).reshape(4,5)
    np.testing.assert_allclose(u @ coef[:2], incidence_shift(angles,coef))


def test_extended_transform_keeps_original_and_independent_conditions():
    original = np.array([[2.,0.],[0.,3.],[1.,1.]])
    x = np.array([[1.,2.,3.]])
    u = np.array([[.5,.25]])
    new = extend_transforms({'head':original})['head']
    np.testing.assert_allclose(np.column_stack((x,u)) @ new,
                               np.column_stack((x @ original,u)))


def test_zero_condition_source_fit_reproduces_original():
    rng=np.random.default_rng(7)
    x=rng.normal(size=(80,5)); labels=np.tile(np.arange(8),10)
    original={'joint_base':np.eye(5), 'joint_expert':rng.normal(size=(5,3))}
    old=fit_source_heads(x,labels,original)
    new=fit_source_heads(np.column_stack((x,np.zeros((80,2)))),labels,extend_transforms(original))
    for name in original:
        np.testing.assert_allclose(old[name]['coef'],new[name]['coef'][:5],atol=1e-11)
        np.testing.assert_allclose(new[name]['coef'][5:],0,atol=1e-11)
        np.testing.assert_allclose(old[name]['intercept'],new[name]['intercept'],atol=1e-11)


def test_conditioned_prediction_equals_explicit_primal_ridge():
    # Independent primal solve verifies support-only weighting and the added kernel.
    rng=np.random.default_rng(3); n=12
    bank=rng.normal(size=(n,4)); u=rng.normal(size=(n,2))
    stable=rng.normal(size=(n,2)); retained=rng.normal(size=(n,3)); comp=rng.normal(size=(n,4))
    phi=np.column_stack((rng.normal(size=(n,3)),np.ones(n)))
    views={'phis':{(True,True):phi},'common':{True:{'retained':retained,'complement':comp}},'stable':stable}
    heads={f'joint_{h}':{'coef':rng.normal(size=(6,8)),'intercept':rng.normal(size=8)} for h in ['base','expert']}
    support=np.array([8,1,6,2,10]); labels=np.array([0,2,2,1,0])
    base,expert=predict_conditioned(bank,views,heads,support,labels,u)
    _,inv,count=np.unique(labels,return_inverse=True,return_counts=True)
    weights=len(labels)/(len(count)*count[inv])
    base_phi=np.column_stack((phi[:,:-1],u/np.sqrt(2),phi[:,-1]))
    expert_phi=np.column_stack((retained/np.sqrt(2),comp/np.sqrt(2),np.ones(n),stable*.5,u*np.sqrt(.25/2)))
    for h,features,actual in [('base',base_phi,base),('expert',expert_phi,expert)]:
        prior=np.column_stack((bank,u)) @ heads['joint_'+h]['coef']+heads['joint_'+h]['intercept']
        xs=features[support]; residual=2*np.eye(8)[labels]-1-prior[support]
        beta=np.linalg.solve(xs.T@(weights[:,None]*xs)+8*np.eye(xs.shape[1]),xs.T@(weights[:,None]*residual))
        np.testing.assert_allclose(actual,prior+features@beta,atol=1e-11)


def test_zero_condition_predictions_preserve_reference_formula():
    from run_incidence_correction_trial import predict_joint
    rng=np.random.default_rng(9); n=10
    bank=rng.normal(size=(n,4)); u=np.zeros((n,2))
    views={'phis':{(True,True):np.column_stack((bank,np.ones(n)))},
           'common':{True:{'retained':bank[:,:2],'complement':bank[:,2:]}},'stable':bank[:,:1]}
    heads={f'joint_{h}':{'coef':rng.normal(size=(4,8)),'intercept':rng.normal(size=8)} for h in ['base','expert']}
    aug={k:{'coef':np.vstack((v['coef'],np.zeros((2,8)))),'intercept':v['intercept']} for k,v in heads.items()}
    support=np.array([8,3,1,6]); labels=np.array([0,1,2,1])
    a=predict_joint(bank,views,heads,support,labels)
    b=predict_conditioned(bank,views,aug,support,labels,u)
    for before,after in zip(a,b):
        np.testing.assert_allclose(before,after,atol=1e-11)
