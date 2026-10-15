"""Bounded provenance check after one source row exceeded legacy replay tolerance."""

from settings import *
import torch
import extract_inputs as extract
from extract_full_six_spatial_features import read_one


def main():
    frame,rows = source_frame(),permitted()
    current = np.load(OUT / "features/original/raw.npy",mmap_mode="r")
    old = np.load(SOURCE / "pooled_features.npy",mmap_mode="r")
    difference = np.linalg.norm(current[rows].astype(float)-old[rows],axis=1)/np.maximum(np.linalg.norm(old[rows],axis=1),1e-12)
    worst = rows[np.argsort(difference)[-8:]]
    selected = np.unique(np.r_[worst,rows[np.linspace(0,len(rows)-1,24,dtype=int)]])
    models = extract.encoders()
    batch = [extract.read(row,"original") for row in frame.iloc[selected].itertuples()]
    for row,b in zip(frame.iloc[selected].itertuples(),batch):
        legacy = np.stack([read_one(Path(row.path_vv)),read_one(Path(row.path_vh))])
        np.testing.assert_array_equal(legacy,b[0])
    new = extract.forward(batch,*models)
    repeated64 = extract.forward(batch+batch,*models)
    checks = {}
    for key in ("raw","c4","extra"):
        saved = np.load(OUT / "features/original" / (key+".npy"),mmap_mode="r")[selected]
        checks[key] = dict(current_reencoding=extract.differences(new[key],saved),
            batch32_to64=extract.differences(repeated64[key][:len(batch)],new[key]))
        if checks[key]["current_reencoding"]["max_relative_l2"] > (.001 if key == "extra" else .01):
            raise ValueError("Current extraction is not reproducible on worst and spread source rows")
    oldpaths = {"raw":SOURCE / "pooled_features.npy","c4":ROOT / "c4_base_trial_20261009/pooled_c4.npy",
        "extra":ROOT / "neural_complement_pilot_20261007/frozen_replay/source_extra.npy"}
    full = {}
    for key,path in oldpaths.items():
        a = np.load(OUT / "features/original" / (key+".npy"),mmap_mode="r")[rows]
        b = np.load(path,mmap_mode="r")[rows]
        assert np.isfinite(a).all()
        full[key] = extract.differences(a,b)
    write(OUT / "SOURCE_REPLAY_DIAGNOSIS.json",dict(rows=selected.tolist(),sample_ids=frame.sample_id.iloc[selected].tolist(),
        pixel_inputs_match_legacy=True,checks=checks,full_legacy_drift=full,raw_rows_over_old_two_percent=int((difference>.02).sum()),
        interpretation="Current frozen preprocessing independently reproduced. Legacy raw feature drift retained; exact historical numeric cause not identified.",
        next_action="Use fresh original source fit as factorial baseline; separately evaluate its legacy-model score drift. Do not claim old cache bit exact."))
    # The failed run completed every extraction batch before its legacy-comparison gate.
    dst = OUT / "features/original"
    np.save(dst / "rows.npy",rows)
    np.save(dst / "ids.npy",frame.sample_id.to_numpy(str))
    write(dst / "COMPLETE.json",dict(condition="original",rows=len(rows),fresh_rows=30827,reused_prior_target_feature_rows=5268,
        extraction_seconds_before_replay=199.2,original_replay=full,legacy_tolerance_exceeded=True,
        recovery_basis="SOURCE_REPLAY_DIAGNOSIS.json current-input/reencoding check; all batches completed in captured original run",
        descriptors_fresh_not_saved_due_to_original_gate=True,encoder_training=False,unextracted_rows_not_permitted=True))
    print(json.dumps(checks),flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
