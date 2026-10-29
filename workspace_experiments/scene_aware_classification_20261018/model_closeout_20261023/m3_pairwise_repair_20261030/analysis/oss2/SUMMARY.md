# oss2 pairwise development analysis

Requested seeds: [17, 29, 43]; available: {'ce': [17, 29, 43], 'rank': [17, 29, 43], 'hybrid': [17, 29, 43]}.
Queries per paired seed: 228. Three-seed complete: True.

| Candidate - reference | Seed | Delta OA | Delta BA | Delta NLL | Delta Brier | Rescue | Harm |
|---|---|---|---|---|---|---|---|
| ce - saved_baseline | 17 | -0.877192982456144 | -1.512165265681766 | 0.006584851398665936 | -0.006104307176652446 | 4 | 6 |
| rank - saved_baseline | 17 | 0.0 | -0.006457445434591591 | 0.004748861305989394 | -0.0017400818274502239 | 1 | 1 |
| hybrid - saved_baseline | 17 | -0.43859649122807554 | -1.224808943842696 | 0.006538356946026891 | -0.005630735868060621 | 4 | 5 |
| rank - ce | 17 | 0.877192982456144 | 1.5057078202471743 | -0.0018359900926765427 | 0.004364225349202222 | 5 | 3 |
| hybrid - ce | 17 | 0.43859649122806843 | 0.28735632183907 | -4.649445263904539e-05 | 0.0004735713085918247 | 1 | 0 |
| ce - saved_baseline | 29 | 0.43859649122806843 | 1.6616691654172868 | 0.04403320018878998 | 0.009449629407072502 | 3 | 2 |
| rank - saved_baseline | 29 | 1.3157894736841982 | 1.4238667183262308 | 5.411192069670889e-05 | -0.002648813640910075 | 3 | 0 |
| hybrid - saved_baseline | 29 | 0.43859649122806843 | 1.6616691654172868 | 0.03453128878267897 | 0.007007602126203993 | 3 | 2 |
| rank - ce | 29 | 0.8771929824561298 | -0.23780244709105602 | -0.04397908826809327 | -0.012098443047982577 | 4 | 2 |
| hybrid - ce | 29 | 0.0 | 0.0 | -0.009501911406111008 | -0.0024420272808685084 | 0 | 0 |
| ce - saved_baseline | 43 | 0.43859649122806843 | 0.27444143096990814 | 0.012064728512946044 | 0.001524648053016664 | 2 | 1 |
| rank - saved_baseline | 43 | -0.43859649122807554 | -1.0869565217391397 | -0.007526038103168542 | -0.0008667666000985452 | 1 | 2 |
| hybrid - saved_baseline | 43 | 0.43859649122806843 | 0.27444143096990814 | 0.009132242444874317 | 0.000837411889418882 | 2 | 1 |
| rank - ce | 43 | -0.877192982456144 | -1.3613979527090478 | -0.019590766616114585 | -0.002391414653115209 | 2 | 4 |
| hybrid - ce | 43 | 0.0 | 0.0 | -0.002932486068071727 | -0.000687236163597782 | 0 | 0 |

| Candidate - reference | Seed | Common energy fraction | Same prediction | Same changed |
|---|---|---|---|---|
| ce - saved_baseline | 17 | 0.9468916918756668 | 225/228 | 13/14 |
| ce - saved_baseline | 29 | 0.9600433778035153 | 227/228 | 5/5 |
| ce - saved_baseline | 43 | 0.9595543727637194 | 227/228 | 5/5 |
| rank - saved_baseline | 17 | 0.8127720094623285 | 228/228 | 3/3 |
| rank - saved_baseline | 29 | 0.8925608716293935 | 227/228 | 3/4 |
| rank - saved_baseline | 43 | 0.9242610143379638 | 227/228 | 2/3 |
| hybrid - saved_baseline | 17 | 0.9573200054882257 | 226/228 | 13/13 |
| hybrid - saved_baseline | 29 | 0.9589260926914148 | 227/228 | 5/5 |
| hybrid - saved_baseline | 43 | 0.9567873224220841 | 224/228 | 3/4 |
| rank - ce | 17 | 0.9165429679485023 | 225/228 | 10/11 |
| rank - ce | 29 | 0.9605235213476556 | 225/228 | 4/7 |
| rank - ce | 43 | 0.9431320941608899 | 227/228 | 7/8 |
| hybrid - ce | 17 | 0.6665013191734628 | 226/228 | 0/1 |
| hybrid - ce | 29 | 0.9416102554751122 | 228/228 | 0/0 |
| hybrid - ce | 43 | 0.8727264812717425 | 227/228 | 1/1 |

| Arm | Seed | Pair logistic | Rank accuracy | Class pairs | Missing class pairs |
|---|---|---|---|---|---|
| ce | 17 | 0.5509989614904176 | 0.8396325157224713 | 6 | 0 |
| ce | 29 | 0.5861650909615091 | 0.8339472631879118 | 6 | 0 |
| ce | 43 | 0.6029030768600315 | 0.8343610993753775 | 6 | 0 |
| hybrid | 17 | 0.5488211667722661 | 0.840178307507025 | 6 | 0 |
| hybrid | 29 | 0.5840856534104685 | 0.8335227171232321 | 6 | 0 |
| hybrid | 43 | 0.6014062004445941 | 0.8349772851445681 | 6 | 0 |
| rank | 17 | 0.5422092903182127 | 0.8397540680164691 | 6 | 0 |
| rank | 29 | 0.5769401981077328 | 0.8342663356504681 | 6 | 0 |
| rank | 43 | 0.5970355178645469 | 0.8366283945201354 | 6 | 0 |
| saved_baseline | 17 | 0.5473356561275037 | 0.8404607297405741 | 6 | 0 |
| saved_baseline | 29 | 0.578563939983041 | 0.8339793118442872 | 6 | 0 |
| saved_baseline | 43 | 0.6029246778773658 | 0.8344473806687368 | 6 | 0 |

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
