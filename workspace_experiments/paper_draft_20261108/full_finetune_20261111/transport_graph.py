"""Exact forward/backward CUDA graph replay of the unchanged 60-step transport."""
import torch
from sar_m123.m3 import sinkhorn


def transport(cost, qa, ra, allowed):
    cost = torch.where(allowed, cost, torch.zeros_like(cost))
    return sinkhorn(cost, qa[:, None], ra[None], steps=60)*allowed


class GraphTransport:
    def __init__(self):
        self.graphs = {}
        self.position = 0

    def reset(self, module, args):
        self.position = 0

    def __call__(self, cost, qa, ra, allowed):
        values = cost, qa, ra, allowed
        training = torch.is_grad_enabled() and cost.requires_grad
        key = (training, self.position, tuple(tuple(v.shape) for v in values))
        self.position += 1
        if not training:
            return transport(*values)
        if key not in self.graphs:
            # Separate captures for each regional invocation: no overwritten intermediates
            # while several transport nodes coexist in one training backward.
            samples = tuple(v.detach().clone().requires_grad_(v.requires_grad) for v in values)
            self.graphs[key] = torch.cuda.make_graphed_callables(transport, samples,
                                                               num_warmup_iters=3)
        return self.graphs[key](*values).clone()


def install(matcher):
    dispatch = GraphTransport()
    matcher.register_forward_pre_hook(dispatch.reset)
    matcher.transport = dispatch
    return dispatch


def verify():
    import json
    import time
    from pathlib import Path
    torch.manual_seed(20261111)
    torch.set_float32_matmul_precision('highest')
    results = []
    dispatch = GraphTransport()
    for trial in range(3):
        q = torch.rand(12,8,device='cuda');r = torch.rand(32,8,device='cuda')
        q[0]=0;r[0]=0
        allowed = (q[:,None,:,None]>0)&(r[None,:,None,:]>0)
        x = torch.rand(12,32,8,8,device='cuda',requires_grad=True)
        weight = torch.randn_like(x)
        torch.cuda.synchronize();start=time.perf_counter()
        a=transport(x,q,r,allowed)
        ga=torch.autograd.grad((a*weight).sum(),x)[0]
        torch.cuda.synchronize();eager=time.perf_counter()-start
        dispatch.position=0
        torch.cuda.synchronize();start=time.perf_counter()
        b=dispatch(x,q,r,allowed)
        gb=torch.autograd.grad((b*weight).sum(),x)[0]
        torch.cuda.synchronize();graphed=time.perf_counter()-start
        forward_error=float((a-b).abs().max());gradient_error=float((ga-gb).abs().max())
        assert forward_error<1e-6 and gradient_error<1e-5
        results.append(dict(trial=trial,forward_error=forward_error,gradient_error=gradient_error,
            eager_seconds=eager,graph_seconds=graphed,capture_included=trial==0))
    (Path(__file__).parent/'transport_graph_check.json').write_text(json.dumps(results,indent=2))
    print(results,flush=True)


if __name__=='__main__': verify()
