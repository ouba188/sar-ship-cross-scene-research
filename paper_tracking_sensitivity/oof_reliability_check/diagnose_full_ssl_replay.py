"""Compare original and staged forward under the two actual backend settings."""

import numpy as np
import torch
from torch.nn import functional as F
from general_domain_ssl import pixels_batch
from full_ssl_run import OUT, OLD_CACHE, PIXELS, SIZE, load_r50, normalize, rgb, views, write


def difference(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return dict(max_abs=float(np.max(np.abs(a-b))), rms=float(np.sqrt(np.mean((a-b)**2))),
                different=int(np.count_nonzero(a != b)),
                relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(b)))


@torch.inference_mode()
def main():
    records = []
    for domain in ('G', 'D'):
        net = load_r50(domain)
        for ds in ('source', 'oss2', 'f0921'):
            pixels = np.load(PIXELS[ds], mmap_mode='r')
            rows = (np.load(OUT/'episode_plan.npz')['source_rows'][:64]
                    if ds == 'source' else np.arange(64))
            pair = pixels_batch(pixels, rows, 'cuda')
            new_pair = torch.from_numpy(np.array(pixels[rows], copy=True)).cuda().float()/255.
            assert torch.equal(pair, new_pair)
            old = np.load(OLD_CACHE/'features'/f'{ds}_{domain}50.npy', mmap_mode='r')[rows]
            for benchmark, tf32 in ((False, True), (True, False)):
                torch.backends.cudnn.benchmark = benchmark
                torch.backends.cudnn.allow_tf32 = tf32
                torch.backends.cuda.matmul.allow_tf32 = False
                resized = F.interpolate(pair, size=(SIZE, SIZE), mode='bilinear', align_corners=False, antialias=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    direct = net(normalize(rgb(resized))).float().cpu().numpy()
                staged = views(net, new_pair)[0].cpu().numpy()
                row = dict(domain=domain, dataset=ds, benchmark=benchmark, cudnn_tf32=tf32,
                           pair_exact=True, direct_vs_staged=difference(direct, staged),
                           direct_vs_cache=difference(direct, old))
                print(row, flush=True)
                records.append(row)
        del net
        torch.cuda.empty_cache()
    write(OUT/'REPLAY_DIAGNOSIS.json', records)


if __name__ == '__main__':
    main()
