"""Inference-only graph replay of unchanged 60-step M3 transport."""
import torch
from scaled_operator import MaskedCostLocalMatch


class TransportGraph:
    """One transport shape, one stream; all query-dependent inputs refreshed."""
    def __init__(self, operation, example):
        self.static = tuple(value.detach().clone() for value in example)
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                operation(*self.static)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        with torch.inference_mode(), torch.cuda.graph(self.graph, stream=stream):
            self.result = operation(*self.static)

    @torch.inference_mode()
    def __call__(self, inputs):
        for target, source in zip(self.static, inputs):
            target.copy_(source)
        self.graph.replay()
        return self.result.clone()


class GraphMatcher(MaskedCostLocalMatch):
    """Same weights/forward; graph acceleration replaces only transport dispatch."""
    def __init__(self, matcher):
        super().__init__(seed=17, classes=matcher.classes)
        device = next(matcher.parameters()).device
        self.to(device).load_state_dict(matcher.state_dict(), strict=True)
        self.eval().requires_grad_(False)
        self.graphs = {}

    def transport(self, cost, qa, ra, allowed):
        if self.training or torch.is_grad_enabled():
            raise ValueError('GraphMatcher is inference-only; use the original module for training.')
        values = cost, qa, ra, allowed
        key = tuple((tuple(v.shape), v.dtype, v.device) for v in values)
        if key not in self.graphs:
            self.graphs[key] = TransportGraph(super().transport, values)
        return self.graphs[key](values)
