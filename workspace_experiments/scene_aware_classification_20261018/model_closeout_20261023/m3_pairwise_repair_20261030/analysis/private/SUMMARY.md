# private pairwise development analysis

Requested seeds: [17, 29, 43]; available: {'ce': [17, 29, 43], 'rank': [17, 29, 43], 'hybrid': [17, 29, 43]}.
Queries per paired seed: 160. Three-seed complete: True.

| Candidate - reference | Seed | Delta OA | Delta BA | Delta NLL | Delta Brier | Rescue | Harm |
|---|---|---|---|---|---|---|---|
| ce - saved_baseline | 17 | -1.25 | -0.45408163265305745 | -0.06019694322177438 | 0.00033787713404043185 | 2 | 4 |
| rank - saved_baseline | 17 | -1.875 | -1.6683673469387656 | -0.017224181176558773 | -0.003505251914135976 | 0 | 3 |
| hybrid - saved_baseline | 17 | -0.625 | -0.2499999999999929 | -0.06127769043093645 | -0.002726988860715096 | 2 | 3 |
| rank - ce | 17 | -0.625 | -1.2142857142857082 | 0.04297276204521561 | -0.003843129048176408 | 3 | 4 |
| hybrid - ce | 17 | 0.625 | 0.20408163265306456 | -0.0010807472091620696 | -0.003064865994755528 | 1 | 0 |
| ce - saved_baseline | 29 | -2.5 | -0.816326530612244 | 0.03888367168356632 | 0.03443773670168859 | 0 | 4 |
| rank - saved_baseline | 29 | -0.625 | -1.7857142857142847 | 0.004225247275374766 | -0.0015035557074046801 | 0 | 1 |
| hybrid - saved_baseline | 29 | -2.5 | -0.816326530612244 | 0.0198792460427355 | 0.024381333892247725 | 0 | 4 |
| rank - ce | 29 | 1.875 | -0.9693877551020407 | -0.03465842440819156 | -0.03594129240909327 | 4 | 1 |
| hybrid - ce | 29 | 0.0 | 0.0 | -0.019004425640830824 | -0.010056402809440868 | 0 | 0 |
| ce - saved_baseline | 43 | 0.625 | -0.8282312925170103 | -0.0924454884951822 | -0.0252390565172943 | 5 | 4 |
| rank - saved_baseline | 43 | 1.25 | 1.0969387755102034 | -0.01239074938512319 | -0.01005650734641339 | 2 | 0 |
| hybrid - saved_baseline | 43 | 2.5 | 1.1377551020408134 | -0.09329406260275741 | -0.02787230052708778 | 5 | 1 |
| rank - ce | 43 | 0.625 | 1.9251700680272137 | 0.080054739110059 | 0.01518254917088091 | 5 | 4 |
| hybrid - ce | 43 | 1.875 | 1.9659863945578238 | -0.0008485741075752173 | -0.0026332440097934784 | 3 | 0 |

| Candidate - reference | Seed | Common energy fraction | Same prediction | Same changed |
|---|---|---|---|---|
| ce - saved_baseline | 17 | 0.9389948900415366 | 156/160 | 11/14 |
| ce - saved_baseline | 29 | 0.9317365653138697 | 155/160 | 15/18 |
| ce - saved_baseline | 43 | 0.9419687924895176 | 156/160 | 13/15 |
| rank - saved_baseline | 17 | 0.7317153863735171 | 159/160 | 5/6 |
| rank - saved_baseline | 29 | 0.8751805097319155 | 157/160 | 1/3 |
| rank - saved_baseline | 43 | 0.7999339489953358 | 159/160 | 3/4 |
| hybrid - saved_baseline | 17 | 0.9305705011036502 | 156/160 | 10/13 |
| hybrid - saved_baseline | 29 | 0.9327510111168092 | 155/160 | 12/15 |
| hybrid - saved_baseline | 43 | 0.93168607583903 | 156/160 | 10/12 |
| rank - ce | 17 | 0.9491366689308649 | 155/160 | 12/14 |
| rank - ce | 29 | 0.9398081227469793 | 152/160 | 15/20 |
| rank - ce | 43 | 0.9330748590971342 | 159/160 | 17/17 |
| hybrid - ce | 17 | 0.9059853741975878 | 160/160 | 1/1 |
| hybrid - ce | 29 | 0.8919782724382485 | 159/160 | 3/3 |
| hybrid - ce | 43 | 0.878404817698375 | 160/160 | 3/3 |

| Arm | Seed | Pair logistic | Rank accuracy | Class pairs | Missing class pairs |
|---|---|---|---|---|---|
| ce | 17 | 1.011240410397264 | 0.7836103937617421 | 21 | 7 |
| ce | 29 | 0.8835926807656233 | 0.7914830356294831 | 21 | 7 |
| ce | 43 | 0.9083097315692543 | 0.7881923899048081 | 21 | 7 |
| hybrid | 17 | 1.0135391294111278 | 0.7826822188729923 | 21 | 7 |
| hybrid | 29 | 0.88196557145207 | 0.7931362640806299 | 21 | 7 |
| hybrid | 43 | 0.9070794864962467 | 0.7881395087267 | 21 | 7 |
| rank | 17 | 1.0262776241440652 | 0.7807113809269365 | 21 | 7 |
| rank | 29 | 0.884131192629843 | 0.7929227519215916 | 21 | 7 |
| rank | 43 | 0.9045224523877307 | 0.7876770092683503 | 21 | 7 |
| saved_baseline | 17 | 1.0294604395614233 | 0.7815465111455309 | 21 | 7 |
| saved_baseline | 29 | 0.8862750302731511 | 0.7912830911836951 | 21 | 7 |
| saved_baseline | 43 | 0.909949704976698 | 0.7865648141955655 | 21 | 7 |

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
