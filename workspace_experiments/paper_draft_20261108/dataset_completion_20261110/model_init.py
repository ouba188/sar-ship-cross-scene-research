"""Class-count adaptation only; the four-class default preserves original initialization."""
import common_ablation as old

torch, nn, F = old.torch, old.nn, old.F


class RegionalGuidance(old.s.RegionalGuidance):
    def __init__(self, arm, classes):
        if classes == 4:
            super().__init__(arm)
            return
        nn.Module.__init__(self)
        self.arm = arm
        self.value = nn.ModuleList(nn.Conv2d(512,32,1,bias=False) for _ in range(2))
        self.output = nn.ModuleList(nn.Linear((classes+1)*32,512,bias=False) for _ in range(2))
        self.scorer = nn.ModuleList(nn.Conv2d(512,classes,1,bias=False) for _ in range(2))
        for layer in self.output: nn.init.zeros_(layer.weight)
        for layer in self.scorer:
            nn.init.normal_(layer.weight,std=.005)
            layer.requires_grad_(arm=='content')
        self.head = nn.Linear(7170,classes)

    def forward(self,b,mode=None,details=False):
        if self.head.out_features == 4:
            return super().forward(b,mode,details)
        classes = self.head.out_features
        deltas=[];traces=[]
        for p in range(2):
            x=b['x'][:,p];mask=b['mask'][:,p].flatten(2)
            mass=mask.sum(-1,keepdim=True);uniform=mask/mass.clamp_min(1e-8)
            normalized=F.layer_norm(x.permute(0,2,3,1),(512,)).permute(0,3,1,2)
            v=self.value[p](normalized).flatten(2)
            response=b['response'][:,p].flatten(2)
            if self.arm=='class':
                relative=(classes*response-response.sum(1,keepdim=True))/(classes-1)
                valid=b['valid'][:,p].flatten(2)
                vw=valid/valid.sum(-1,keepdim=True).clamp_min(1e-8)
                centered=relative-(relative*vw).sum(-1,keepdim=True)
                score=2*torch.tanh(centered/((centered.square()*vw).sum(-1,keepdim=True)+.01).sqrt())
            elif self.arm=='content': score=2*torch.tanh(self.scorer[p](normalized).flatten(2))
            else: score=torch.zeros_like(response)
            logits=score[:,None]+mask.clamp_min(1e-12).log()[:,:,None]
            logits=logits.masked_fill(mask[:,:,None]<=0,-1e4)
            weights=logits.softmax(-1)*(mask[:,:,None]>0)
            weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
            if mode=='uniform': weights=uniform[:,:,None].expand(-1,-1,classes,-1)
            elif mode=='shuffle':
                weights=weights.clone()
                for i in range(len(x)):
                    for r in range(3):
                        ix=torch.where(mask[i,r]>0)[0]
                        weights[i,r,:,ix]=weights[i,r,:,ix.flip(0)]
            mean_v=torch.einsum('bru,bdu->brd',uniform,v)
            token=torch.einsum('brku,bdu->brkd',weights,v)
            descriptors=torch.cat((mean_v,token.flatten(2)),-1)
            mean_x=torch.einsum('bru,bdu->brd',uniform,x.flatten(2))
            scale=mean_x.square().mean(-1,keepdim=True).sqrt().detach()
            delta=.1*scale*self.output[p](descriptors).tanh()*(mass>0)
            delta=delta*b['available'][:,p,None,None]
            if mode=='off': delta=delta*0
            deltas.append(delta)
            if details: traces.append(dict(weights=weights,uniform=uniform,delta=delta,mean=mean_x))
        parts=[]
        for p in range(2):
            z=b['features'][:,p*3584:(p+1)*3584]
            parts.extend((z[:,:2048],z[:,2048:]+deltas[p].flatten(1)))
        score=self.head(torch.cat(parts+[b['features'][:,-2:]],1))
        return (score,traces) if details else score


class BaseNetwork(old.Network):
    def __init__(self,arm,seed,weight_path,classes=4):
        if classes == 4:
            super().__init__(arm,seed,weight_path)
            self.classes=classes
            return
        nn.Module.__init__(self)
        self.arm,self.classes=arm,classes
        self.use1=arm not in ('plain','no_m1')
        self.use2=arm not in ('plain','no_m2')
        self.use3=arm not in ('plain','no_m3')
        torch.manual_seed(seed)
        self.base=old.SpatialM12ResNet50(weight_path,'pre3x3','local',classes)
        self.base.internal_m1=nn.ModuleList();self.base.m1=nn.ModuleDict()
        self.base.m3=nn.ModuleList();self.base.anchor_last=nn.Identity();self.base.head=nn.Identity()
        self.m1=nn.ModuleList(old.RelationFeedback(128,'class',.3,classes=classes) for _ in range(2))
        self.guide=RegionalGuidance('class' if self.use1 else 'mean',classes)
        self.guide.head.reset_parameters()
        self.matcher=old.task.MaskedCostLocalMatch(seed=seed,classes=classes)
        if not self.use1: self.m1=nn.ModuleList()
        if not self.use3: self.matcher=None
        if not self.use2:
            head=self.guide.head;self.head=nn.Linear(4098,classes)
            with torch.no_grad():
                self.head.weight.copy_(torch.cat((head.weight[:,:2048],head.weight[:,3584:5632],head.weight[:,-2:]),1))
                self.head.bias.copy_(head.bias)
            self.guide=None;self.base.internal_m2=nn.ModuleList();self.base.m2=nn.ModuleList()
            self.base.deep_pool=nn.ModuleList();self.base.deep_output=nn.ModuleList()
        self.to(old.task.DEVICE).eval()
        self.base.backbone.requires_grad_(False)
        self.base.backbone.model.layer2[3].conv3.requires_grad_(True)
        for i in (1,2,3):getattr(self.base.backbone.model.layer3[5],f'conv{i}').requires_grad_(True)
        for name,p in self.named_parameters():
            if '.scorer.' in name:p.requires_grad_(False)
