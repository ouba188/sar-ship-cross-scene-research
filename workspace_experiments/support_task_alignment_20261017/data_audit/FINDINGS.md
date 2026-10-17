# Fixed-prior support-pool audit

2026-10-17. Independent source-only census; no training, target labels, policy selection, or sampler tests. Writes are confined to this directory. Read the implemented `../sampling.py` and `../PROTOCOL_ZH.md`; their independent class-quota draws match the census assumptions.

**Result:** membership resampling is a substantive intervention for 107/110 fit tasks and 27/27 selection tasks, but its freedom is concentrated in common classes. It cannot create any newly auxiliary-eligible query, even at the individual task-context level, under the fixed acquisition, prior exclusions, class quotas, query cohort, and current validity masks.

## Available diversity

Pool size counts distinct MMSIs after removing query and fixed-prior MMSIs within the original support acquisition and source role. The original 16 members remain in every pool.

| Source role / origin | Tasks | Changeable | Legal pool min / median / max | Forced original slots / total slots |
| --- | ---: | ---: | --- | --- |
| Fit, all | 110 | 107 | 16 / 33 / 161 | 700 / 1760 (39.8%) |
| Fit, same | 48 | 45 | 16 / 26.5 / 131 | 372 / 768 |
| Fit, cross | 62 | 62 | 18 / 45.5 / 161 | 328 / 992 |
| Selection, all | 27 | 27 | 17 / 39 / 168 | 167 / 432 (38.7%) |
| Selection, same | 12 | 12 | 17 / 36.5 / 136 | 89 / 192 |
| Selection, cross | 15 | 15 | 23 / 70 / 168 | 78 / 240 |

Before query/prior exclusions, acquisition-pool medians are 60.5 fit and 71 selection. Fixed-prior MMSIs remove additional candidates from 96/110 fit tasks and 0/27 selection tasks; the latter source split already separates train/validation vessels. All legal-pool classes have nonzero original quotas, so there are **zero available candidates excluded solely by a zero quota**. Missing support classes are also absent from the legal pool.

Per-fold changeable/total tasks: pilot 26/27 fit and 7/7 selection; 010685 27/27 and 7/7; 011246 25/26 and 8/8; 011421 29/30 and 5/5. The three immutable member sets are `pilot/fit/002`, `S1A_011246_011040/fit/018`, and `S1A_011421_0115FD/fit/022`; each is a same-acquisition task with exactly 16 legal ships.

For one independent uniform class-matched redraw, the median expected number of members outside the original support is **4.73 fit / 5.22 selection**. For original + four redraws, the median expected within-task union is **27.90 / 30.19 ships**, not 80 independent ships. These are exact expectations from pool sizes, not measurements of the realized bank. For class pool n and quota k, expected new members are k(1-k/n); expected union with four redraws is k+(n-k)[1-(1-k/n)^4]. Realized sets and training exposure remain the main run's responsibility.

Across all folds, the fit support union can expand from 738 observations / 660 ships to at most 1102 / 1019, adding 364 observations / 359 ships outside the old support union. Selection can expand from 288 / 278 to 687 / 653, adding 399 / 375. These are full-pool ceilings, not what four draws necessarily realize. All are existing source observations; some already participate as queries or priors in other tasks. No new acquisition or query cohort is introduced, and overlapping folds/tasks are not independent evidence.

## Class bottlenecks

Forced slots mean n=k, so every member of that class must remain. The denominator below is support slots across original tasks, not unique vessels.

| Class | Fit forced / total slots | Fit tasks with that class changeable / present | Selection forced / total slots |
| --- | --- | --- | --- |
| Bulk carrier | 37 / 687 | 102 / 110 | 9 / 140 |
| Fishing vessel | 72 / 78 | 2 / 55 | 19 / 25 |
| General cargo | 176 / 341 | 42 / 108 | 32 / 77 |
| Product/chemical tanker | 128 / 242 | 33 / 96 | 36 / 67 |
| Container ship | 147 / 268 | 37 / 101 | 27 / 77 |
| Crude oil tanker | 40 / 40 | 0 / 35 | 19 / 19 |
| Tug/towing | 61 / 65 | 2 / 39 | 15 / 17 |
| Offshore supply | 39 / 39 | 0 / 36 | 10 / 10 |

Of the 359 additional possible fit-support ships, 209 are bulk carriers and 92 container ships. Crude tankers and offshore supply have no membership freedom; tug changes mostly rearrange existing source support evidence. Fishing gains 16 possible ships globally but only two fit task-class pools permit changes. More overall pool diversity does not imply more rare-class supervision.

## Rare-class local supervision

Counts below deduplicate by source observation within role and **cross** origin across folds. They use the actual current mask `local_features.valid & quality.valid`, not just class presence. Auxiliary eligibility requires a valid query and at least one valid same-class support; all original tasks contain at least two valid support classes. "Possible" is an upper bound over all legal class-matched subsets, not just the four redraws.

| Class | Fit queries / valid / original eligible / possible eligible | Cross-selection queries / valid / original eligible / possible eligible |
| --- | --- | --- |
| Fishing vessel | 12 / 3 / 1 / 1 | 5 / 4 / 0 / 0 |
| Crude oil tanker | 8 / 8 / 7 / 7 | 2 / 2 / 1 / 1 |
| Tug/towing | 7 / 0 / 0 / 0 | 5 / 0 / 0 / 0 |
| Offshore supply | 9 / 1 / 0 / 0 | 2 / 0 / 0 / 0 |

Three rare classes retain invalid-local queries: fishing 9 fit / 1 selection, tug 7 / 5, offshore 8 / 2, totaling **24 fit / 8 selection observations**. Their local validity cannot be repaired by support resampling with the current query cohort and masks. Do not add fit and selection as independent samples: fold roles overlap.

Valid queries can also remain blocked. Fit fishing rows 834 and 1114 have no usable same-class candidates in their existing contexts; offshore row 283 has no valid same-class support; crude row 1016 lacks its class. Cross-selection fishing rows 13, 606, 622, 834 and crude row 64 have zero same-class quota **and zero same-class legal candidates**. Thus all 12 cross-selection fishing/tug/offshore queries remain auxiliary-ineligible. This does not imply their full-model predictions cannot change through support refitting.

Same-origin pooling changes historical coverage counts: selection fishing has 2 eligible queries and crude has 2 when same and cross are combined, versus 0 and 1 for cross alone. The old report's cross-selection zeros are consistent. Across every class and origin, there are **zero newly eligible unique queries and zero newly eligible task-query contexts** in the full candidate pools.

## Reproduction and evidence

Run `python -B audit_source_pools.py` from this directory; only NumPy and the standard library are needed. Runtime here was under one second. The script reads the saved source task rows/task metadata, source train/validation CSVs, sample-ID axes, and source-indexed local validity masks. It never imports a training or target loader. Row-axis alignment and retention of all original supports passed; no sampler draws or sampler tests were run.

`pool_summary.csv` and `task_pools.csv` contain role/origin/fold and task results; `class_pools.csv` gives quotas and bottlenecks; `support_census.csv` distinguishes observations from ships; `query_class_summary.csv`, `unique_queries.csv`, and `query_contexts.csv` record eligibility and blocking reasons; `pool_candidates.csv` lists the eligible and original identities. These outputs support interpretation of the authorized 2x2 experiment, not selection of an arm or a claim of classification benefit.
