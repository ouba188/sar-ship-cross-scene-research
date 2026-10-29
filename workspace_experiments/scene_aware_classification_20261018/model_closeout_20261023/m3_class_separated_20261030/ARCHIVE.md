# Archive Scope

Included: experiment code, protocol, interface review, findings, training/evaluation scalar receipts, paired/class/pass summary tables, bounded source-offset events and 16-query derived VV/VH galleries.

Local only: checkpoints (`.pt`), prediction tensors (`.npz`), bytecode, complete per-query `events.csv`, original datasets and complete training/exposure manifests. The analysis code regenerates local detailed events from the retained local predictions. No authentication files are part of this archive.

The code imports existing sibling experiment helpers and the local standalone SAR package. It is not a self-contained dataset or model-weight release. The formal model is unchanged; this branch remains an experimental candidate with mixed OA/BA results.
