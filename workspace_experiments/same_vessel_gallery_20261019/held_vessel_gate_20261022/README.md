# Held-Vessel Gate Probe

Completed 2026-10-22. Read `FINDINGS_ZH.md` and `PROTOCOL.md`.

Run `run.py`, then `analyze.py`, then `diagnose.py` in the existing ClearSAR
environment. The scripts depend on sibling experiment modules and local frozen
feature/model caches; those large files and raw imagery are intentionally not
included in the GitHub archive. This is a research archive, not a standalone
dataset redistribution.

The old reference remains unchanged. OOF score gating achieved OA/BA
41.526%/42.748%, compared with 41.512%/42.707% for the reference, and did not
outperform the matched in-sample score gate. Zero-selected inner residuals limit
the effective training signal. All negative and positive outcomes are retained.

The quadratic oracle uses true labels only after training for diagnosis. Its
accuracy is not a deployable result and must not be reported as the proposed model.
