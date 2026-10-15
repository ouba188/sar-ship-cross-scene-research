# Bounded Replication Audit

- pixels: passed
- public: passed
- aggregation: passed
- ports: passed

Independent metric calculations do not import the parent evaluator. Pixel tests execute the actual transform under test against independently computed masks, pixel invariants and direct XML fields. All audit work is CPU-only. Only audit_replication.py and AUDIT*.json/md are written.

Primary same-pipeline originals and legacy archived baselines remain separate. Tiny legacy/primary numerical differences and prediction flips are recorded, not treated as failed exact replay between deliberately different inputs. Cache lineage checks require exact ids.npy/frame/archived-target alignment.

Limits: bbox neighborhoods can be large and include surrounding water, wake or sidelobes. Original saturation is not by itself a preprocessing error. Native per-polarization shuffles preserve each region's grayscale multiset, not VV/VH joint tuples on different masks. Shared public vessels/acquisitions prevent an independent-data replication claim. Aggregate directional repetition does not imply class-universal reliance; container-driven effects and counterexamples must remain in the main report. Removing shared query identities does not redesign supports. No class exclusions or new protocol choices are introduced here.

Interpretation boundaries supplied by the main analysis (not additional audit recomputations): ports' B display contrast reverses the public pattern; matched target shuffling can increase T-R while reducing absolute T accuracy. Ports' sham-valid shuffle contrast reverses the full-cohort contrast (+0.745 versus -0.778 pp; 13 negative/8 positive/2 zero ports), with lower average class coverage (about 4.31 versus 6.07). Do not claim complete-mechanism replication or stability across all subsets. Public class counterexamples also remain part of the result.

Audit environment: Pillow exited decoding the first OSS1 TIFF; four OSS1 native pairs were independently read with tifffile, then checked against original XML. This was confined to the auditor; parent readers and preprocessing were not changed. R50 bilinear128 pixels and the R18 native VH-to-RGB handoff were verified; the frozen Resize(128)/ToTensor configuration was checked without loading model libraries.

Run with D:/Program_files/anaconda3/python.exe -B audit_replication.py pixels|public|aggregation|ports. Ports may be run later after the two named tasks finish; pending is not a pass.
