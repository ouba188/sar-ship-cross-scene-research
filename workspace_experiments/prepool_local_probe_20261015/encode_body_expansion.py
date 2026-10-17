"""Extract only missing frozen features for the prespecified expansion tasks."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from localization import spatial_masks, PIXELS

HERE = Path(__file__).resolve().parent
OUT = HERE / "body_context_expansion"
sys.path.insert(0, str(HERE.parent / "scene_cross_dataset_redesign_20261014/ports_full_expansion"))
from encode import encode_r50, encode_r18
from general_domain_ssl import backbone, normalize, rgb


@torch.inference_mode()
def local_encode(model, rows):
    ship, _, _, ring, _ = spatial_masks(rows)
    pixels = np.load(PIXELS, mmap_mode="r")
    captured = []
    hook = model.relu.register_forward_hook(lambda _, __, x: captured.append(x))
    peaks, stats = [], []
    reference_stats = None
    for start in range(0, len(rows), 32):
        batch = rows[start:start+32]
        pair = torch.from_numpy(np.array(pixels[batch], dtype=np.float32)).cuda()/255.
        x = normalize(rgb(F.interpolate(pair, (112,112), mode="bilinear", align_corners=False, antialias=True)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x)
        stem = captured.pop().float()
        smask = torch.from_numpy(ship[start:start+32]).cuda()[:,None]
        rmask = torch.from_numpy(ring[start:start+32]).cuda()[:,None]
        peak = stem.masked_fill(~smask, float("-inf")).amax((2,3))
        mask = F.interpolate(rmask.float(), (128,128), mode="nearest")
        mean = (pair*mask).sum((2,3))/mask.sum((2,3))
        std = ((pair.square()*mask).sum((2,3))/mask.sum((2,3))-mean.square()).clamp_min(0).sqrt()
        if start == 0:
            precise = []
            for i in range(min(16, len(batch))):
                values = pair[i].double()[:, mask[i,0].bool()]
                precise.append(torch.cat((values.mean(1), values.std(1,unbiased=False))))
            reference_stats = torch.stack(precise).cpu().numpy()
        peaks.append(peak.cpu().numpy())
        stats.append(torch.cat((mean,std),1).cpu().numpy())
    hook.remove()
    return np.concatenate(peaks), np.concatenate(stats), reference_stats


def main():
    torch.set_num_threads(2)
    frozen = json.loads((OUT/"frozen.json").read_text(encoding="utf-8"))
    required = pd.read_csv(OUT/"required_samples.csv")
    pixels = np.load(PIXELS, mmap_mode="r")
    for pool in frozen["pools"]:
        port = pool["port"]
        if not pool["selected"]:
            continue
        folder = OUT/port.replace(" ","_")
        folder.mkdir(exist_ok=True)
        if (folder/"encoded.json").exists():
            print("encoded cache",port,flush=True)
            continue
        source = Path(pool["source_folder"]).parent/port.replace(" ","_")
        rows = required.loc[required.port.eq(port),"row_index"].to_numpy(int)
        old = np.load(source/"rows.npy")
        missing = rows[~np.isin(rows,old)]
        np.save(folder/"rows.npy",missing)
        if len(missing):
            r50,_=backbone("r50")
            encode_r50(r50.cuda().eval(),pixels,missing,folder)
            del r50
            torch.cuda.empty_cache()
        r18,_=backbone("r18")
        r18.load_state_dict(torch.load(source/"source_only_r18_best.pt",map_location="cpu",weights_only=False)["backbone"])
        r18.cuda().eval()
        if len(missing):
            encode_r18(r18,pixels,missing,folder,"D18")
        with np.load(HERE/"localization"/f"features_{pool['reference']}.npz") as original:
            cached_rows=original["rows"]
            needed=rows[~np.isin(rows,cached_rows)]
            reference_rows=cached_rows[:16]
            encode_rows=np.r_[reference_rows,needed]
            peak,ring,precise=local_encode(r18,encode_rows)
            peak_error=float(np.abs(peak[:16]-original["ship_peak"][:16]).max())
            ring_error=float(np.abs(ring[:16]-original["raw_ring"][:16]).max())
            np.testing.assert_allclose(peak[:16],original["ship_peak"][:16],atol=1e-6,rtol=1e-6)
            # Second-moment float32 variance has cancellation error; verify both
            # implementations against direct float64 pixel standard deviations.
            np.testing.assert_allclose(ring[:16],precise,atol=1e-6,rtol=1e-6)
            np.testing.assert_allclose(original["raw_ring"][:16],precise,atol=1e-6,rtol=1e-6)
            np.savez_compressed(folder/"local.npz",rows=needed,ship_peak=peak[16:],raw_ring=ring[16:])
        del r18
        torch.cuda.empty_cache()
        (folder/"encoded.json").write_text(json.dumps(dict(rows=len(rows),new_global_features=len(missing),
            new_local_features=len(needed),stem_replay_max_error=peak_error,ring_replay_max_error=ring_error,
            ring_float64_reference_max_error=float(np.abs(ring[:16]-precise).max())),indent=2))
        print("encoded",port,len(rows),"missing",len(missing),"local",len(needed),flush=True)


if __name__=="__main__":
    main()
