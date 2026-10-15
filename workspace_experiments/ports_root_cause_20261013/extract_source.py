"""Re-encode both source conditions with identical frozen encoder execution."""

from settings import *
import time
import torch
import extract_inputs as extract


def main():
    frame = source_frame()
    rows = permitted()
    models = extract.encoders()
    for condition in CONDITIONS:
        dst = OUT / "features" / condition
        if (dst / "COMPLETE.json").exists():
            print("REUSE", condition, flush=True)
            continue
        dst.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        arrays = {k: np.lib.format.open_memmap(dst / (k+".npy"), mode="w+", dtype="float32",
            shape=(len(frame), d)) for k,d in (("raw",2048),("c4",2048),("extra",512))}
        reused = []
        for port in PORTS:
            path = INPUT / "features" / port / condition
            ids = np.load(path / "ids.npy")
            positions = pd.Index(frame.sample_id).get_indexer(ids)
            assert (positions >= 0).all()
            take = np.isin(positions, rows)
            for k in arrays:
                arrays[k][positions[take]] = np.load(path / (k+".npy"), mmap_mode="r")[take]
            reused.extend(positions[take])
        fresh = np.setdiff1d(rows, reused)
        details = []
        for lo in range(0, len(fresh), 32):
            ix = fresh[lo:lo+32]
            batch = [extract.read(row, condition) for row in frame.iloc[ix].itertuples()]
            output = extract.forward(batch, *models)
            for key, value in output.items():
                if not np.isfinite(value).all():
                    raise ValueError("Nonfinite source feature")
                arrays[key][ix] = value
            details.extend(dict(row=int(i), sample_id=str(frame.sample_id.iloc[i]), **b[2]) for i,b in zip(ix,batch))
            if lo % 1024 == 0 or lo+32 >= len(fresh):
                print("SOURCE",condition,min(lo+32,len(fresh)),len(fresh),round(time.perf_counter()-started,1),flush=True)
        replay = {}
        if condition == "original":
            old = {"raw": SOURCE / "pooled_features.npy", "c4": ROOT / "c4_base_trial_20261009/pooled_c4.npy",
                "extra": ROOT / "neural_complement_pilot_20261007/frozen_replay/source_extra.npy"}
            for k,p in old.items():
                replay[k] = extract.differences(arrays[k][rows], np.load(p,mmap_mode="r")[rows])
                if replay[k]["max_relative_l2"] > (.001 if k == "extra" else .02):
                    write(dst / "REPLAY_FAILED.json", replay)
                    raise ValueError("Source original replay drift exceeds prior frozen tolerance")
        for value in arrays.values():
            value.flush()
        np.save(dst / "ids.npy", frame.sample_id.to_numpy(str))
        np.save(dst / "rows.npy", rows)
        pd.DataFrame(details).to_csv(dst / "descriptors_fresh.csv", index=False)
        write(dst / "COMPLETE.json", dict(condition=condition, rows=len(rows), fresh_rows=len(fresh),
            reused_prior_target_feature_rows=len(reused), unextracted_rows_not_permitted=True,
            seconds=time.perf_counter()-started, original_replay=replay, encoder_training=False))
        del arrays
    write(OUT / "FEATURES_COMPLETE.json", dict(rows=len(rows), conditions=CONDITIONS, frozen_encoders=True))


if __name__ == "__main__":
    with torch.inference_mode():
        main()
