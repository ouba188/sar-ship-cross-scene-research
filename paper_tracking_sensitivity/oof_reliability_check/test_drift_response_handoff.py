import numpy as np

from drift_response_handoff import response_features, fit_net_gate, predict_net_gate
from evidence_aware_handoff import route_scores


def test_response_matches_direct_linear_feature_perturbation():
    rng = np.random.default_rng(13)
    x, wb, we = rng.normal(size=(19,7)),rng.normal(size=(7,3)),rng.normal(size=(7,3))
    scale = rng.uniform(.3,2,7)
    factors = rng.normal(size=(7,4))
    b,e = x@wb,x@we
    features = response_features(b,e,wb,we,scale,factors)
    pb,pe = [],[]
    for j in range(4):
        for sign in (1,-1):
            shifted = x+sign*2*scale*factors[:,j]
            pb.append((shifted@wb).argmax(1))
            pe.append((shifted@we).argmax(1))
    pb,pe = np.array(pb).T,np.array(pe).T
    expected = np.column_stack(((pb!=b.argmax(1)[:,None]).mean(1),
        (pe!=e.argmax(1)[:,None]).mean(1),(pb==e.argmax(1)[:,None]).mean(1),
        (pe==b.argmax(1)[:,None]).mean(1),(pb==pe).mean(1)))
    np.testing.assert_allclose(features[:,4:9],expected)


def test_common_score_shift_and_direction_sign_do_not_change_responses():
    rng = np.random.default_rng(3)
    b,e = rng.normal(size=(30,3)),rng.normal(size=(30,3))
    gb,ge = rng.normal(size=(8,3)),rng.normal(size=(8,3))
    factors = rng.normal(size=(8,4))
    a = response_features(b,e,gb,ge,np.ones(8),factors)
    z = response_features(b+7,e-3,gb,ge,np.ones(8),-factors)
    np.testing.assert_allclose(a,z)


def test_zero_drift_does_not_flip_predictions():
    b = np.array([[1.,0.,-1.],[0.,1.,-1.]])
    f = response_features(b,b,np.zeros((4,3)),np.zeros((4,3)),np.ones(4),np.zeros((4,2)))
    np.testing.assert_array_equal(f[:,4:6],0)
    np.testing.assert_array_equal(f[:,8],1)
    np.testing.assert_array_equal(f[:,9:],0)


def test_gate_retains_neutral_utility_and_predicts_net_benefit():
    features = np.repeat([[-1.],[0.],[1.]],100,axis=0)
    base = np.tile([2.,1.,0.],(300,1))
    expert = np.tile([1.,2.,0.],(300,1))
    labels = np.repeat([0,2,1],100)
    model = fit_net_gate(features,base,expert,labels,np.ones(300,dtype=bool),np.ones(300))
    assert model['counts']=={'-1':100,'0':100,'1':100}
    score = predict_net_gate(model,np.array([[-1.],[0.],[1.]]))
    assert score[0]<-.5 and abs(score[1])<.02 and score[2]>.5


def test_gate_does_not_train_on_ineligible_rows():
    b,e = np.tile([1.,0.,-1.],(8,1)),np.tile([0.,1.,-1.],(8,1))
    y = np.array([0,1,0,1,2,2,2,2])
    model = fit_net_gate(np.arange(8.)[:,None],b,e,y,np.arange(8)<4,np.ones(8))
    assert model['rows']==4 and model['counts']['0']==0


def test_route_keeps_supports_out_and_preserves_budget():
    b,e = np.tile([1.,0.],(120,1)),np.tile([0.,1.],(120,1))
    ev = np.arange(120)>=20
    result = route_scores(b,e,np.ones(120),np.array([0,1]),ev,np.arange(120))
    assert result['cap']==5 and result['selected'].sum()==5
    assert not result['selected'][:20].any()
