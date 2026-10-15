# User-provided feature inventory, 2026-10-07

Paths below are relative to E:/Hermes/paper_tracking unless absolute.
Listed availability does not by itself establish representation compatibility,
sample alignment, or untouched evaluation status.

| Dataset | Location | User-provided description | Status in this run |
| --- | --- | --- | --- |
| F_0921 | f0921_feats/ | VV/VH clean+all, R50 2048, 1.4GB | Prior usage known; feature README previously inspected; not an untouched dataset |
| OpenSARShip1 | opensarship_external/external_feats.npz | 6.2MB | Recorded; not inspected or launched in this run |
| OpenSARShip2 | external_os2_fusar/feats_os2.npz | R50 2048 + SSL 512, 5006 chips | Shapes and producer inspected; VV-only base incompatible with current paired base |
| FUSARShip | external_os2_fusar/feats_fusar.npz | R50 2048 + SSL 512, 3166 chips | Shared producer inspected; single-image dual encoders, not paired polarization; not launched |
| PACS | xdom_pacs/featsA.npy and featsB_{art_painting,cartoon,photo,sketch}.npy | Four domains | Recorded only |
| Office-Home | xdom_generic/featsA_officehome.npy and featsB_officehome_{Art,Clipart,Product,Real}.npy | Four domains | Recorded only |
| FMoW | xdom_generic/featsA_fmow_sub.npy and featsB_fmow_clip.npy | CLIP second view | Recorded only |
| Camelyon17 | xdomain_data/camelyon17_v1.0/ | HF npz shards, 1.3GB | Recorded only |

Priority: complete current source-to-OpenSARShip2 transfer first. Additional
datasets are possible follow-ups, not automatically added to this run.
