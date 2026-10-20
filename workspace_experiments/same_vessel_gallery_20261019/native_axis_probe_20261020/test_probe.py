import copy
import json
import numpy as np
from PIL import Image
from probe import LOCAL, regions, symmetric

data=json.loads((LOCAL/'view_snapshot.json').read_text(encoding='utf-8'))
for s in data[:12]:
    v=s['images']['vv']; h,w=v['h'],v['w']
    raw=(np.arange(h*w).reshape(h,w)%251).astype(np.uint8)
    out,m,shape,info=regions(raw,v,s['obs_id'])
    assert np.array_equal(m[3:6].sum(0),m[0])
    assert np.array_equal(m[6:9].sum(0),m[0])
    assert np.array_equal(m[9:12].sum((1,2)),m[3:6].sum((1,2)))
    changed=copy.deepcopy(v); changed['priors']['ship_axis']['angle']+=np.pi
    _,flipped,other,_=regions(raw,changed,s['obs_id'])
    assert np.array_equal(m[3:6],flipped[3:6][::-1])
    assert np.array_equal(shape,other)
    sx,sy=info['crop_x'],info['crop_y']; dx,dy=info['pad_left'],info['pad_top']
    cw,ch=info['crop_width'],info['crop_height']
    assert np.array_equal(out[dy:dy+ch,dx:dx+cw],raw[sy:sy+ch,sx:sx+cw])
x=np.random.default_rng(0).normal(size=(10,2,3,64))
assert np.array_equal(symmetric(x),symmetric(x[:,:,::-1]))
print('PASS: actual geometry partitions, scrambled pixel counts, native pixel preservation, exact endpoint exchange')
