# Figure QA completed

Date: 2026-10-08. Scope: five architecture/process diagrams only.

- Viewed all five PDF-rendered pages with `view_image` at 150 dpi. The exported PNG masters remain 300 dpi. Labels and arrows are readable; no text overlaps or clipped node labels remain.
- M1: moved the left `prior + map` label clear of its vertical connector, then inspected the revised rendering.
- M4: moved the inherited-gate connector away from the source LR box's bottom border, then inspected the revised rendering.
- Export checks: five single-page vector PDFs, five editable-text SVGs, five 300 dpi PNGs, and `module_pipeline_diagrams.pdf` containing exactly five pages. Individual PDFs contain zero raster images. Width is 183 mm; heights are 239-247 mm. Main diagram text is at least 8 pt; only the two short M1 connector labels use 7.5 pt.
- Layout checks: 114 text elements in total, zero text-to-text collisions and zero out-of-bounds labels. Detailed dimensions and resolutions are in `figure_export_qa.json`.
- Implementation review: shared 64 supports, ridge-1 source heads, ridge-8 residual solve, 192/2560/2753 dimensions, half-kernel weights, one penalized bias, differentiable dynamic prior, layer4-only training, and fixed-gate eligibility/ranking match the inspected implementation. The expert source prior reads retained features; M2's full joint map is consumed by the expert residual solver.
- R18 naming is confirmed as **SimCLR-pretrained** using local `e206_ssl_pretrain.py`. The captions specify NT-Xent, initialization from scratch, the pretraining-only 512->256->128 projection MLP, deployed 512-D trunk features, and historical unlabeled F0921 exposure.
- The figures state that all expert features are computed before admission; the 5% cap is not a compute-saving claim. No performance tables or claims were introduced; no model training was run.
- All generated or edited files are inside this `figures/` directory. Main-report and table files were not edited.

Regenerate using the existing environment:

```powershell
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' 'E:/Hermes/paper_tracking/sensitivity/module_evidence_20261008/figures/draw_module_figures.py'
```

Caption file: `figure_captions_zh.md`. Page previews used for visual inspection: `qa_previews/`.
