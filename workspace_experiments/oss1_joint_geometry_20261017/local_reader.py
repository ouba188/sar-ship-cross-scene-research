"""Ship/radar joint native-coordinate pooling and a shared local descriptor."""
import numpy as np
from scipy.ndimage import binary_dilation
import torch
from torch import nn

ARMS=('image_ring','ship','ship_radar_encoding','joint')


def regions(d,radar,incidence,radar_valid=True,incidence_valid=True,arm='joint',size=56):
    h,w=float(d['native_h']),float(d['native_w'])
    length=max(float(d['length'])*1.25,4.)
    width=max(float(d['width'])*1.25,3.)
    reach=length/2+4*max(2.,width/2)+7.5*max(1.5,width/3)
    pad=max(16,int(np.ceil(reach*size/min(h,w))))
    grid=np.arange(-pad,size+pad)
    # Conv1(k7,s2,p3) centers are 2*j in the 112px input, not 2*j+0.5.
    yy,xx=np.meshgrid((grid+.25)*h/size-.5,(grid+.25)*w/size-.5,indexing='ij')
    x,y=xx-float(d['cx']),yy-float(d['cy'])
    theta=float(d['ship_axis'])
    co,si=np.cos(theta),np.sin(theta)
    u,v=x*co+y*si,-x*si+y*co
    body_full=(abs(u)<=length/2)&(abs(v)<=width/2)
    crop=(slice(pad,pad+size),slice(pad,pad+size))
    body=body_full[crop]
    if not body.any():
        body.flat[np.argmin((x*x+y*y)[crop])]=True
    weights=[];positions=[];coverages=[]
    dirs=(np.arange(4)*np.pi/4 if arm=='image_ring' else theta+np.arange(4)*np.pi/4)
    if arm=='joint' and radar_valid:
        dirs=np.array([theta,theta+np.pi/2,radar,radar+np.pi/2])
    if arm=='image_ring':
        dil=[body_full]+[binary_dilation(body_full,iterations=k) for k in (2,8,16)]
        shells=[dil[j+1]&~dil[j] for j in range(3)]
    for scale in range(3):
        for angle in dirs:
            cs,sn=np.cos(angle),np.sin(angle)
            rel=angle-theta
            edge=min(length/(2*max(abs(np.cos(rel)),1e-6)),
                     width/(2*max(abs(np.sin(rel)),1e-6)))
            radius=edge+(1,2,4)[scale]*max(2.,width/2)
            sigma=max(1.5,width/3)*(1+.25*scale)
            if arm=='image_ring':
                phi=np.arctan2(y,x)
                weight=shells[scale]*np.exp(4*(np.cos(2*(phi-angle))-1))
            else:
                # Paired lobes avoid assigning an unsupported bow/stern direction.
                a=np.exp(-((x-radius*cs)**2+(y-radius*sn)**2)/(2*sigma*sigma))
                b=np.exp(-((x+radius*cs)**2+(y+radius*sn)**2)/(2*sigma*sigma))
                weight=(a+b)*(~body_full)
            full_mass=float(weight.sum())
            weight=weight[crop]
            mass=float(weight.sum())
            coverage=mass/max(full_mass,1e-20)
            weight=weight/max(mass,1e-20) if mass>1e-10 else np.zeros_like(weight)
            pos=np.zeros(16,np.float32)
            pos[:4]=[scale/2,np.log1p(radius)/5,coverage,float(mass>1e-10)]
            reference=0. if arm=='image_ring' else theta
            pos[4:6]=[np.cos(2*(angle-reference)),np.sin(2*(angle-reference))]
            pos[6:8]=[np.log1p(length)/5,np.log1p(width)/5]
            if arm in ('ship_radar_encoding','joint'):
                if radar_valid:
                    pos[8:12]=[np.cos(2*(angle-radar)),np.sin(2*(angle-radar)),
                               np.cos(2*(theta-radar)),np.sin(2*(theta-radar))]
                    pos[14]=1
                if incidence_valid:
                    pos[12:14]=[np.sin(incidence),np.cos(incidence)]
                    pos[15]=1
            weights.append(weight.astype(np.float32));positions.append(pos);coverages.append(coverage)
    return body,np.stack(weights),np.stack(positions),np.array(coverages)


class LocalReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,32),nn.GELU())
        self.context=nn.Sequential(nn.LayerNorm(64),nn.Linear(64,32),nn.GELU())
        self.position=nn.Sequential(nn.Linear(16,16),nn.GELU())
        self.relation=nn.Sequential(nn.Linear(112,32),nn.GELU(),nn.Linear(32,16))
        self.attention=nn.Linear(32,1)

    def forward(self,body,tokens,positions):
        b=self.body(body)[:,None].expand(-1,tokens.shape[1],-1)
        c=self.context(tokens);p=self.position(positions)
        pair=self.relation(torch.cat((b,c,b*c,p),dim=-1))
        attention=self.attention(torch.cat((pair,p),dim=-1)).squeeze(-1)
        active=positions[:,:,3]>0
        attention=attention.masked_fill(~active,-1e4).softmax(-1)*active
        attention=attention/attention.sum(-1,keepdim=True).clamp_min(1e-8)
        mean=(pair*attention[:,:,None]).sum(1)
        spread=((pair-mean[:,None]).square()*attention[:,:,None]).sum(1)
        return torch.cat((mean,torch.sqrt(spread+1e-6)),dim=-1),attention


def support_features(raw,valid,ns):
    mean=(raw[:ns]*valid[:ns,None]).sum(0)/valid[:ns].sum().clamp_min(1)
    centered=(raw-mean)*valid[:,None]
    return centered/torch.sqrt(1+centered.square().sum(-1,keepdim=True))
