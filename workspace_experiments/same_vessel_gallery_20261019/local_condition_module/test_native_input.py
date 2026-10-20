import numpy as np
from native_input import crop_pad


def sample(h,w,y,x,th,tw):
    image=(np.arange(h*w).reshape(h,w)%251).astype(np.uint8)
    masks=np.zeros((3,h,w),np.float32)
    masks[0,y:y+th,x:x+tw]=1
    masks[2]=1-masks[0]
    return image,masks


for case in [(64,64,20,20,10,15),(128,128,90,90,20,30),
             (64,128,20,100,15,20),(97,95,20,30,30,20)]:
    image,masks=sample(*case)
    out,region,info=crop_pad(image,masks)
    sx,sy,cw,ch=[info[k] for k in ('crop_x','crop_y','crop_width','crop_height')]
    dx,dy=info['pad_left'],info['pad_top']
    assert np.array_equal(out[dy:dy+ch,dx:dx+cw],image[sy:sy+ch,sx:sx+cw])
    assert np.array_equal(region[:,dy:dy+ch,dx:dx+cw],masks[:,sy:sy+ch,sx:sx+cw])
    assert region[0].sum()==masks[0].sum()
    assert np.all(out[region.sum(0)==0]==0)
image,masks=sample(128,128,10,10,100,20)
try:
    crop_pad(image,masks)
except ValueError:
    pass
else:
    raise AssertionError('Oversize target must not be silently truncated')
print('PASS: native pixels, synchronized masks, zero padding, target preservation, oversized target refusal')
image,masks=sample(184,184,20,20,133,60)
out,region,info=crop_pad(image,masks,160)
assert region[0].sum()==masks[0].sum()
assert out.shape==(160,160)
print('PASS: 160px canvas preserves the observed maximum target extent')
