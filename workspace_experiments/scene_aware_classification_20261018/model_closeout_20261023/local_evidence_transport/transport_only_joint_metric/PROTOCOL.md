# Separate evidence assignment from evidence scoring

Frozen before training, 2026-10-24. Follow-up of pair_conditioned_joint_metric; historical development, not independent confirmation.

Motivation: matched-query examples showed both useful and harmful changes to content scoring. Previous frozen cross interventions were diagnostic, not trained alternatives. This round trains the alternatives with the identical 640-parameter pair network, joint descriptors, original contents, bank, loss, source selection and update budget.

- transport_only: learned normalized K1 determines T1=OT(K1); local features are [sum(T1*K0),sum(T1)]. K0 is the original content cosine. Gradients flow through T1, not through K0. This changes assignment AND real/rejection mass; it does not guarantee protection of body evidence.
- metric_only: frozen T0=OT(K0); local features [sum(T0*K1),sum(T0)]. Gradients flow only through K1. T0 is precomputed from unchanged contents and masses.
- both: reuse the completed prior pair_joint three fits, since same network, initialization and training/selection budget. No fresh fitting or tuning of this comparator.

Both new arms keep old strong head, final strong+head(newlocal)-head(oldlocal). Same native PNG/C4/20OBB slots, 96-dimensional C, old PCA32, source normalization. No new pixels, angles, labels or model capacity. No per-class safeguard based on development examples. PNG frequency information is processed-image information, not calibrated radar scattering.

Source reference64, utility644, selection612, historical development485/445ships/29acquisitions. Seeds17/29/43 also index frozen upstream configurations, not isolated new-layer random seeds. Train100 full644 updates, AdamW.003/.01, half empirical/half class-balanced CE. Choose source J at0/10/20/40/60/80/100, earlier ties. Complete all6 fits before target metrics. Original strong retained regardless of outcome.

Before fitting check initial full/local replay and finite-difference gradient for each changed path; also replay old T0K0 over source utility against cached locals to detect reference/mass mistakes. No need to repeat unchanged geometric or frequency extraction checks. Invalid tokens remain zero mass and old dustbin normalization unchanged.

Evaluate OA/BA, fixed T=1 NLL/Brier, per-class/acquisition rescue/harm, paired acquisition bootstrap. Inspect actual native VV/VH and true/rival references for rescue/harm/CC/WW and fixed previous probes28839,7875,2080,29687. Exact six-region frozen-head decomposition is a model computation diagnostic, not physical cause. Do not delete losses, tune on evaluation, or infer frequency mechanism from display images.
