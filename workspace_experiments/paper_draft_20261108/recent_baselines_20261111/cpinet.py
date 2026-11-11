"""Independent CPINet implementation; assumptions are listed in PROTOCOL_ZH.md."""
import torch
from torch import nn
from torch.nn import functional as F


class DenseBlock(nn.Module):
    def __init__(self, channels, growth):
        super().__init__()
        self.layers = nn.ModuleList(nn.Sequential(
            nn.BatchNorm2d(channels+i*growth), nn.ReLU(),
            nn.Conv2d(channels+i*growth, growth, 3, padding=1, bias=False),
            nn.Dropout2d(.2)) for i in range(3))

    def forward(self, x):
        for layer in self.layers:
            x = torch.cat((x, layer(x)), 1)
        return x


class SARBackbone(nn.Module):
    def __init__(self, initial=1):
        super().__init__()
        self.stem = nn.Identity() if initial == 1 else nn.Conv2d(1, initial, 3, padding=1, bias=False)
        channels = initial
        self.blocks = nn.ModuleList()
        self.channels = []
        for growth in (3,6,9,12,15):
            self.blocks.append(DenseBlock(channels,growth))
            channels += 3*growth
            self.channels.append(channels)

    def forward(self, x):
        x = self.stem(x)
        result = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < 4:
                x = F.avg_pool2d(x,2)
            result.append(x)
        return result


def conv_bn_relu(cin, cout):
    return nn.Sequential(nn.Conv2d(cin,cout,1,bias=False),nn.BatchNorm2d(cout),nn.ReLU())


class MultiScale(nn.Module):
    def __init__(self, channels, target):
        super().__init__()
        self.target = target
        self.rescale = nn.ModuleList()
        strides = (2,4,8,16,16)
        for i,c in enumerate(channels):
            factor = strides[target]/strides[i]
            if factor >= 1:
                self.rescale.append(nn.MaxPool2d(int(factor)) if factor>1 else nn.Identity())
            else:
                factor = int(1/factor)
                self.rescale.append(nn.ConvTranspose2d(c,c,factor,stride=factor,bias=False))
        self.fuse = conv_bn_relu(sum(channels)-channels[target],channels[target])

    def forward(self, levels):
        others = [op(x) for i,(op,x) in enumerate(zip(self.rescale,levels)) if i!=self.target]
        return levels[self.target]+self.fuse(torch.cat(others,1))


class IMDFF(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.levels = nn.ModuleList(MultiScale(channels,t) for t in (2,3,4))
        self.merge = conv_bn_relu(channels[2]+channels[3],channels[4])

    def forward(self, levels):
        third,fourth,last = [m(levels) for m in self.levels]
        return self.merge(torch.cat((F.max_pool2d(third,2),fourth),1)),last


def group_bilinear(x, groups):
    b,c,h,w = x.shape
    split = x.reshape(b,groups,c//groups,h*w)
    return torch.cat([(split[:,i]@split[:,j].transpose(1,2)).flatten(1)/(h*w)
                      for i in range(groups) for j in range(i,groups)],1)


class Excitation(nn.Sequential):
    def __init__(self, inputs, channels, reduction=2):
        super().__init__(nn.Linear(inputs,max(1,channels//reduction),bias=False),
                         nn.BatchNorm1d(max(1,channels//reduction)),nn.ReLU(),
                         nn.Linear(max(1,channels//reduction),channels,bias=False),
                         nn.BatchNorm1d(channels),nn.Sigmoid())


class MixedOrder(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()
        assert channels%groups==0
        self.groups = groups
        second = groups*(groups+1)//2*(channels//groups)**2
        self.first = Excitation(channels,channels)
        self.second = Excitation(second,channels)

    def forward(self, x):
        return self.first(x.mean((-2,-1)))+self.second(group_bilinear(x,self.groups))


class FBC(nn.Module):
    def __init__(self, channels, atoms=2048, threshold=.0005):
        super().__init__()
        self.u = nn.Conv2d(channels,atoms,1,bias=False)
        self.v = nn.Conv2d(channels,atoms,1,bias=False)
        self.threshold = threshold

    def forward(self, x, y):
        z = self.u(x)*self.v(y)
        z = z.sign()*F.relu(z.abs()-self.threshold)
        z = z.amax((-2,-1))
        # FBC's original authors normalize pooled codes before classification.
        z = z.sign()*z.abs().clamp_min(1e-8).sqrt()
        return F.normalize(z,p=2,dim=1)


class CPINet(nn.Module):
    def __init__(self, initial=1, groups=8, atoms=2048, classes=4):
        super().__init__()
        self.backbones = nn.ModuleList(SARBackbone(initial) for _ in range(2))
        channels = self.backbones[0].channels
        self.multiscale = nn.ModuleList(IMDFF(channels) for _ in range(2))
        self.attention = nn.ModuleList(MixedOrder(channels[-1],groups) for _ in range(2))
        self.codes = nn.ModuleList(FBC(channels[-1],atoms) for _ in range(4))
        self.dropout = nn.Dropout(.2)
        self.classifier = nn.Linear(atoms,classes)

    def forward(self, pixels, available):
        intermediate,last = [],[]
        for p in range(2):
            a,b = self.multiscale[p](self.backbones[p](pixels[:,p:p+1]))
            intermediate.append(a); last.append(b)
        gates = [self.attention[p](last[p]) for p in range(2)]
        augmented = [last[p]*gates[1-p][...,None,None] for p in range(2)]
        pairs = [(intermediate[0],intermediate[1]), (augmented[0],augmented[1]),
                 (last[0],last[0]),(last[1],last[1])]
        vectors = [m(*pair) for m,pair in zip(self.codes,pairs)]
        # The paper specifies one shared FC and ensembling, not four classifiers.
        scores = torch.stack([self.classifier(self.dropout(v)) for v in vectors],1)
        both = available.prod(1)
        active = torch.stack((both,both,available[:,0],available[:,1]),1)
        probability = (scores.softmax(-1)*active[...,None]).sum(1)/active.sum(1,keepdim=True)
        return probability.clamp_min(1e-12).log(),scores,active


class GradNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(4))
        self.register_buffer('initial',torch.zeros(4))

    def objective(self, losses, classifier):
        norms = torch.stack([torch.autograd.grad(loss,classifier.weight,retain_graph=True)[0].detach().norm()
                             for loss in losses])
        if not bool((self.initial>0).all()):
            self.initial.copy_(losses.detach().clamp_min(1e-8))
        ratio = losses.detach()/self.initial
        relative = ratio/ratio.mean()
        weighted = self.weights*norms
        target = (weighted.mean()*relative.pow(.12)).detach()
        return (weighted-target).abs().sum()

    @torch.no_grad()
    def normalize(self):
        self.weights.clamp_(min=1e-3)
        self.weights.mul_(4/self.weights.sum())


def losses(scores, active, labels, sample_weights=None):
    target = labels[:,None].expand(-1,4)
    values = F.cross_entropy(scores.flatten(0,1),target.flatten(),reduction='none').reshape(-1,4)
    weight = active if sample_weights is None else active*sample_weights[:,None]
    return (values*weight).sum(0)/weight.sum(0).clamp_min(1)
