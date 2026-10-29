# private pairwise development analysis

Requested seeds: [17]; available: {'ce': [17], 'rank': [17], 'hybrid': [17]}.
Queries per paired seed: 160. Three-seed complete: False.

| Candidate - reference | Seed | Delta OA | Delta BA | Delta NLL | Delta Brier | Rescue | Harm |
|---|---|---|---|---|---|---|---|
| ce - saved_baseline | 17 | -1.25 | -0.45408163265305745 | -0.06019694322177438 | 0.00033787713404043185 | 2 | 4 |
| rank - saved_baseline | 17 | -1.875 | -1.6683673469387656 | -0.017224181176558773 | -0.003505251914135976 | 0 | 3 |
| hybrid - saved_baseline | 17 | -0.625 | -0.2499999999999929 | -0.06127769043093645 | -0.002726988860715096 | 2 | 3 |
| rank - ce | 17 | -0.625 | -1.2142857142857082 | 0.04297276204521561 | -0.003843129048176408 | 3 | 4 |
| hybrid - ce | 17 | 0.625 | 0.20408163265306456 | -0.0010807472091620696 | -0.003064865994755528 | 1 | 0 |

| Candidate - reference | Seed | Common energy fraction | Same prediction | Same changed |
|---|---|---|---|---|
| ce - saved_baseline | 17 | 0.9389948900415366 | 156/160 | 11/14 |
| rank - saved_baseline | 17 | 0.7317153863735171 | 159/160 | 5/6 |
| hybrid - saved_baseline | 17 | 0.9305705011036502 | 156/160 | 10/13 |
| rank - ce | 17 | 0.9491366689308649 | 155/160 | 12/14 |
| hybrid - ce | 17 | 0.9059853741975878 | 160/160 | 1/1 |

| Arm | Seed | Pair logistic | Rank accuracy | Class pairs | Missing class pairs |
|---|---|---|---|---|---|
| ce | 17 | 1.011240410397264 | 0.7836103937617421 | 21 | 7 |
| hybrid | 17 | 1.0135391294111278 | 0.7826822188729923 | 21 | 7 |
| rank | 17 | 1.0262776241440652 | 0.7807113809269365 | 21 | 7 |
| saved_baseline | 17 | 1.0294604395614233 | 0.7815465111455309 | 21 | 7 |

Equal over available class pairs; uniform over legal query pairs within each class pair.
Development diagnostic only; labels organize evaluation, never training or sample selection. Pairs share queries and are not independent evidence.
Query-common class offsets cancel exactly in pair gaps; parameter updates may still introduce such offsets.

Only exact current-round development NPZ names are loaded; source NPZs and prior results are excluded.
Each comparison uses its own complete paired seeds; unavailable arms/seeds are pending, never zero gain.
Single-seed results are a screening result, not three-seed confirmation. SD is null for one seed.
OA/BA differences are percentage points; NLL/Brier are unscaled and lower is better.
BA averages present query classes. Missing classes have null recall.
Repeated query events across seeds are not independent observations or vessels.
Shared shift is the class-centered new logit increment, not the whole M3 contribution.
Common-only and residual-only are label-free offsets fitted on development logits: descriptive diagnostics, not deployable candidates.
Source/pair objectives and gradients are reported training evidence, not proof of development benefit or physical causality.
Generated galleries require visual review; selected cases do not estimate population proportions.

## Pending

```json
[]
```
