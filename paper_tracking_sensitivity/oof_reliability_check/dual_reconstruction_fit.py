"""Fixed GPU source-only regression; no target classification evaluation."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from dual_reconstruction_core import (OUT, Inputs, BATCH, folders, info_for,
                                      freeze_protocol, write, reconstruct)


def fit(folder):
    _, suffix = info_for(folder)
    dst = OUT/'targets'/suffix
    if (dst/'FIT.json').exists():
        print('REUSE FIT', suffix, flush=True)
        return
    started = time.perf_counter()
    inputs = Inputs(folder)
    n = len(inputs.source_rows)
    x = torch.empty((n, 4096), device='cuda')
    y = torch.empty((n, 512), device='cuda')
    for lo in range(0, n, BATCH):
        s = slice(lo, min(lo+BATCH, n))
        x[s] = torch.from_numpy(inputs.source_x(s)).cuda()
        y[s] = torch.from_numpy(np.array(inputs.source_extra[s], dtype=np.float32)).cuda()
    xm, xs = x.mean(0), x.std(0, correction=0).clamp_min(1e-4)
    ym, ys = y.mean(0), y.std(0, correction=0).clamp_min(1e-4)
    x.sub_(xm).div_(xs)
    y.sub_(ym).div_(ys)
    mapper = {name: value.cpu().numpy() for name, value in
              [('x_mean', xm), ('x_std', xs), ('y_mean', ym), ('y_std', ys)]}
    gram, cross = x.T@x/n, x.T@y/n
    gram.diagonal().add_(.01)
    weight = torch.linalg.solve(gram, cross)
    relative = float(torch.linalg.vector_norm(gram@weight-cross)/torch.linalg.vector_norm(cross))
    if relative > 1e-3 or not torch.isfinite(weight).all():
        raise ValueError(f'Inaccurate ridge solve: {relative}')
    mapper['linear_weight'] = weight.cpu().numpy()
    train_mse = {'linear': float(torch.mean((x@weight-y)**2))}
    del gram, cross
    torch.manual_seed(20261010)
    net = torch.nn.Sequential(torch.nn.Linear(4096, 512), torch.nn.GELU(),
                              torch.nn.Linear(512, 512)).cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.0001)
    losses = []
    for epoch in range(8):
        order = torch.randperm(n, device='cuda')
        total = 0.
        for ix in order.split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((net(x[ix])-y[ix])**2)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(ix)
        losses.append(total/n)
        print(f'FIT {suffix} epoch{epoch+1}: source_mse={losses[-1]:.6f}', flush=True)
    net.eval()
    with torch.no_grad():
        total = 0.
        for lo in range(0, n, BATCH):
            total += float(torch.sum((net(x[lo:lo+BATCH])-y[lo:lo+BATCH])**2))
        train_mse['mlp'] = total/(n*512)
        for i, layer in [(1, net[0]), (2, net[2])]:
            mapper[f'mlp_weight{i}'] = layer.weight.T.detach().cpu().numpy()
            mapper[f'mlp_bias{i}'] = layer.bias.detach().cpu().numpy()
        probe = inputs.source_x(slice(0, 16))
        for kind, prediction in [('linear', x[:16]@weight), ('mlp', net(x[:16]))]:
            expected = (prediction*ys+ym).cpu().numpy()
            actual = reconstruct(probe, mapper, kind)
            np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=1e-4)
    dst.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst/'mapper.npz', **mapper)
    write(dst/'FIT.json', dict(dataset=inputs.dataset, task=inputs.info['task'],
        source_n=n, source_standardized_mse=train_mse, training_losses=losses,
        linear_solve_relative_residual=relative, cpu_gpu_forward_agree=True,
        seconds=time.perf_counter()-started, target_features_fitted=False,
        input_dim=4096, teacher_dim=512, mlp_parameters=sum(p.numel() for p in net.parameters())))
    print('FIT COMPLETE', suffix, time.perf_counter()-started, flush=True)
    del net, optimizer, x, y, weight
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+')
    args = parser.parse_args()
    freeze_protocol()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    with threadpool_limits(limits=2):
        for folder in folders():
            if not args.tasks or info_for(folder)[0]['task'] in args.tasks:
                fit(folder)
