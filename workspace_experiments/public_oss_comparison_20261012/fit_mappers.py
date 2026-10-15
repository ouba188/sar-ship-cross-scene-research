"""Source-only extraction and the two predeclared reconstruction fits."""

from common import *
import time
import torch
from threadpoolctl import threadpool_limits
from compact_fusion_source import _external_features


def fit(regime, name):
    dst = fit_dir(regime, name)
    if (dst / "MAPPER_COMPLETE.json").exists():
        print("REUSE MAPPER", regime, name, flush=True)
        return
    rows, raw, c4, extra = source_arrays(regime, name)
    started = time.perf_counter()
    x = torch.empty((len(rows), 4096), device="cuda")
    y = torch.as_tensor(extra, device="cuda", dtype=torch.float32)
    for lo in range(0, len(rows), BATCH):
        ix = rows[lo:lo+BATCH]
        x[lo:lo+len(ix)] = torch.from_numpy(np.c_[raw[ix], c4[ix]].astype(np.float32)).cuda()
    xm, xs = x.mean(0), x.std(0, correction=0).clamp_min(1e-4)
    ym, ys = y.mean(0), y.std(0, correction=0).clamp_min(1e-4)
    x.sub_(xm).div_(xs)
    y.sub_(ym).div_(ys)
    result = {k: v.cpu().numpy() for k, v in (("x_mean", xm), ("x_std", xs), ("y_mean", ym), ("y_std", ys))}
    gram, cross = x.T@x/len(rows), x.T@y/len(rows)
    gram.diagonal().add_(.01)
    weight = torch.linalg.solve(gram, cross)
    residual = float(torch.linalg.vector_norm(gram@weight-cross)/torch.linalg.vector_norm(cross))
    if residual > 1e-3:
        raise ValueError("Inaccurate source ridge solve")
    result["linear_weight"] = weight.cpu().numpy()
    train_mse = {"linear": float(torch.mean((x@weight-y)**2))}
    del gram, cross
    torch.manual_seed(20261010)
    net = torch.nn.Sequential(torch.nn.Linear(4096, 512), torch.nn.GELU(), torch.nn.Linear(512, 512)).cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.0001)
    losses = []
    for epoch in range(8):
        total = 0.
        for ix in torch.randperm(len(rows), device="cuda").split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((net(x[ix])-y[ix])**2)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(ix)
        losses.append(total/len(rows))
        print("MAPPER", regime, epoch+1, losses[-1], flush=True)
    net.eval()
    with torch.no_grad():
        train_mse["mlp"] = sum(float(torch.sum((net(x[lo:lo+BATCH])-y[lo:lo+BATCH])**2))
                               for lo in range(0, len(rows), BATCH))/(len(rows)*512)
        for i, layer in ((1, net[0]), (2, net[2])):
            result[f"mlp_weight{i}"] = layer.weight.T.cpu().numpy()
            result[f"mlp_bias{i}"] = layer.bias.cpu().numpy()
        probe = np.c_[raw[rows[:16]], c4[rows[:16]]].astype(np.float32)
        for kind, pred in (("linear", x[:16]@weight), ("mlp", net(x[:16]))):
            np.testing.assert_allclose(reconstruct(probe, result, kind), (pred*ys+ym).cpu().numpy(), atol=5e-4, rtol=1e-4)
    dst.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst / "mapper.npz", **result)
    np.save(dst / "source_rows.npy", rows)
    write(dst / "MAPPER_COMPLETE.json", dict(regime=regime, source_n=len(rows), seconds=time.perf_counter()-started,
        train_mse=train_mse, training_losses=losses, linear_solve_residual=residual, cpu_gpu_forward_agree=True,
        target_features_fitted=False, source_only=True))
    del net, optimizer, x, y, weight
    torch.cuda.empty_cache()


if __name__ == "__main__":
    freeze()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with threadpool_limits(limits=2):
        _external_features(joint("oss1") / "joint", source_rows("task_trained", "oss1"))
        fit("task_trained", "oss1")
        fit("shared_frozen", "oss1")
