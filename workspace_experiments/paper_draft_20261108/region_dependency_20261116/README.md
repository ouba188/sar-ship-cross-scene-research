# Regional dependency diagnosis

Fixed-weight, OSS2 validation diagnosis inspired by Li et al., GRSL 2023, DOI 10.1109/LGRS.2023.3266493. Read `FINDINGS_ZH.md` for conclusions, `PROTOCOL_ZH.md` for controls and `CASES_ZH.md` for native image observations. This study does not change the selected model or manuscript.

## Local reproduction

Use the project's existing ClearSAR environment and original workspace layout. Model code is imported from `../full_finetune_20261111/train.py`; selected checkpoints, native image cache, geometry cache and the existing OSS2 manifest/reference split are required. This archived diagnosis is not a standalone model distribution.

```powershell
python run.py smoke --kind full
python run.py run --kind full
python run.py run --kind r50
python analyze.py
python plots.py
python gallery.py
```

Run Full and R50 sequentially on the local GPU. Existing prediction files are resumed; delete or move only the specific study outputs deliberately if the protocol is changed. Do not reuse predictions under a different protocol.

`analyze.py --self-test` checks additive, pairwise and triple interaction games without loading weights. Analysis additionally verifies prediction IDs and Shapley efficiency on all samples.

## Archived vs local

Repository: scripts, protocol, aggregate/seed-level metrics, analytical validation receipt, Chinese conclusions, and editable aggregate SVG/PNG plots.

Local only: original chips, model weights, native/geometry caches, individual prediction NPZs, native region masks/metadata, sample-level Shapley values and the image gallery. `gallery/index.html` is regenerated locally; the repository intentionally does not include raw chip panels.

All intervals condition on three fixed trained seeds and the four observed passes. They resample physical vessel IDs, not seed-events. The polar-paired tables compare VV/VH on the same 206 eligible observations. System comparison to ordinary R50 is not a matched-history module ablation.
