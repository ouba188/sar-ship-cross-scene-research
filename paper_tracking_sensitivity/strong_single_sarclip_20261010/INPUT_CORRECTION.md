# Input Correction Before Target Readout

The first real smoke followed the historical local grayscale wrapper: independently
repeat each polarization into RGB, run the same backbone twice, concatenate.
That is a two-view adaptation, not the official dual-polarization Byte input.

The official repository `sar_clip/data.py`, lines83-96, explicitly constructs
VV,VH,(VV+VH)/2 for a two-band Byte image, divides by255 and maps to[-1,1].
The new primary extraction uses that native layout and a single forward.
This correction was made from official implementation evidence, before any new
SARCLIP target predictions or target accuracy were available. Target results did
not determine the input choice. The current128 cached pixels and224 resize remain.

The interrupted first attempt processed at least19,392 of38,091 source rows.
Its incomplete file D:/SAR_strong_single_cache_20261010/source_sarclip.npy and
SMOKE.json are retained but MUST NOT be used for fitting/evaluation. It has no
completion receipt; unfilled trailing rows do not represent extracted features.
The valid native-layout outputs have `_sarclip_joint.npy` names and separate
SMOKE_joint.json plus dataset extraction completion receipts.

No target labels or new neural training were involved in this correction.
The common readout, hyperparameter grid, source splits and R50 control are unchanged.
