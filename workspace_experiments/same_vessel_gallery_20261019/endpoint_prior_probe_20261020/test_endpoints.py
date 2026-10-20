import numpy as np
from probe_endpoints import endpoint_strength,directed

pixels=np.zeros((4,12),np.uint8)
slots=np.zeros((3,4,12),np.float32)
slots[0,:,:4]=1; slots[1,:,4:8]=1; slots[2,:,8:]=1
pixels[:,:4]=50; pixels[:,8:]=200
c,stats=endpoint_strength(pixels,slots)
assert np.isclose(c,.6)
flipped,other=endpoint_strength(pixels,slots[::-1])
assert flipped==-c
x=np.random.default_rng(3).normal(size=(1,2,3,64))
condition=np.array([[c,0.]])
for a,b in zip(directed(x,condition),directed(x[:,:,::-1],-condition)):
    assert np.array_equal(a,b)
assert np.count_nonzero(directed(x,condition)[0][:,1])==0
# Matching saturated peaks must use the declared all-pixel fallback.
pixels[:,:4]=50; pixels[0,:4]=255
pixels[:,8:]=100; pixels[0,8:]=255
c,stats=endpoint_strength(pixels,slots)
assert stats['peak_tie'] and c>0
print('PASS: strength sign, peak tie fallback, exact endpoint relabeling invariance, ambiguous-end zero')
