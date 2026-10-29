# oss2 pairwise development analysis

Requested seeds: [17]; available: {'ce': [17], 'rank': [17], 'hybrid': [17]}.
Queries per paired seed: 228. Three-seed complete: False.

| Candidate - reference | Seed | Delta OA | Delta BA | Delta NLL | Delta Brier | Rescue | Harm |
|---|---|---|---|---|---|---|---|
| ce - saved_baseline | 17 | -0.877192982456144 | -1.512165265681766 | 0.006584851398665936 | -0.006104307176652446 | 4 | 6 |
| rank - saved_baseline | 17 | 0.0 | -0.006457445434591591 | 0.004748861305989394 | -0.0017400818274502239 | 1 | 1 |
| hybrid - saved_baseline | 17 | -0.43859649122807554 | -1.224808943842696 | 0.006538356946026891 | -0.005630735868060621 | 4 | 5 |
| rank - ce | 17 | 0.877192982456144 | 1.5057078202471743 | -0.0018359900926765427 | 0.004364225349202222 | 5 | 3 |
| hybrid - ce | 17 | 0.43859649122806843 | 0.28735632183907 | -4.649445263904539e-05 | 0.0004735713085918247 | 1 | 0 |

| Candidate - reference | Seed | Common energy fraction | Same prediction | Same changed |
|---|---|---|---|---|
| ce - saved_baseline | 17 | 0.9468916918756668 | 225/228 | 13/14 |
| rank - saved_baseline | 17 | 0.8127720094623285 | 228/228 | 3/3 |
| hybrid - saved_baseline | 17 | 0.9573200054882257 | 226/228 | 13/13 |
| rank - ce | 17 | 0.9165429679485023 | 225/228 | 10/11 |
| hybrid - ce | 17 | 0.6665013191734628 | 226/228 | 0/1 |

| Arm | Seed | Pair logistic | Rank accuracy | Class pairs | Missing class pairs |
|---|---|---|---|---|---|
| ce | 17 | 0.5509989614904176 | 0.8396325157224713 | 6 | 0 |
| hybrid | 17 | 0.5488211667722661 | 0.840178307507025 | 6 | 0 |
| rank | 17 | 0.5422092903182127 | 0.8397540680164691 | 6 | 0 |
| saved_baseline | 17 | 0.5473356561275037 | 0.8404607297405741 | 6 | 0 |

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
