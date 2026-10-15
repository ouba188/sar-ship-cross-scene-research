# Bounded sourceFIT budget interference diagnosis

Exact original mixed-budget task replay; no new method or model selection. Three geometry contexts share source vessels; they are not independent target ports.

All probes use the original 1296-row source fitting pool and fixed seed 125012, rounds=1. They are NOT independent heldout data. Within a fold, B16/64/128 supports are nested and share the same 32 query vessels, disjoint from support vessels.

Each driver starts from the checkpoint model AND a deep copy of its actual AdamW state. The nine driver steps are independent branches. Positive CE gain means before-minus-after improvement; negative means harm. Gradient norms alone do not prove budget dominance. Late effects include the inherited mixed-budget optimizer moments and weight decay.

Means are equally weighted over contexts, seeds and applicable fold pairs. Initial seeds share initialization and probes, so these duplicated rows are not independent replicates. Raw CSVs retain every direction and sign; summaries include minima, maxima and sign counts. Task-scoped benefit does not require universal wins, and these one-step effects do not establish heldout, target or full-training benefit.

## Initial / late pairwise gradient cosines

```text
 step  fold_relation  driver_budget  probe_budget       mean     minimum    maximum  count  positive  negative  zero
    0 different_fold             16            64  0.0169349  -0.0291661 0.08101304     36        28         8     0
    0 different_fold             16           128 0.01786027 -0.03063489 0.09659225     36        28         8     0
    0 different_fold             64           128 0.01719226 -0.02457748 0.08109418     36        24        12     0
    0      same_fold             16            64  0.1921719   0.1333614  0.2371407     18        18         0     0
    0      same_fold             16           128   0.119368   0.0686036  0.1662793     18        18         0     0
    0      same_fold             64           128  0.6834958   0.6170117  0.7793218     18        18         0     0
   90 different_fold             16            64  0.0125992 -0.03686434 0.07645443     36        23        13     0
   90 different_fold             16           128 0.01351289 -0.03192155 0.09181732     36        25        11     0
   90 different_fold             64           128 0.01746243 -0.02387389 0.08335533     36        26        10     0
   90      same_fold             16            64  0.1882175   0.1120338  0.2491494     18        18         0     0
   90      same_fold             16           128   0.116897  0.04999685  0.1806491     18        18         0     0
   90      same_fold             64           128  0.6821073   0.6170834  0.7784115     18        18         0     0
```

## Actual cross-budget CE gain (rows: driver; columns: probe)

```text
probe_budget                                16            64            128
step fold_relation  driver_budget                                          
0    different_fold 16              same-budget +5.288396e-05 +1.628197e-05
                    64            +0.0001646008   same-budget +1.170646e-05
                    128           +0.0002199211 +4.343118e-05   same-budget
     same_fold      16              same-budget +0.0006596349 +0.0001293548
                    64             +0.003850734   same-budget +0.0007070523
                    128            +0.002498091  +0.002335771   same-budget
90   different_fold 16              same-budget  +2.12631e-05 +5.278694e-06
                    64            +8.758501e-06   same-budget +4.170698e-06
                    128           +5.947025e-06 +1.739678e-05   same-budget
     same_fold      16              same-budget +0.0001355933 +2.731883e-05
                    64            +0.0001298025   same-budget +3.270982e-05
                    128           +2.943793e-05 +4.601209e-05   same-budget
```

Same-budget effects and step 30 remain in the complete CSVs.

## Replay and isolation

```text
    context  seed  max_error  image_mean_x_max_error  image_coef_max_error  image_mean_d_max_error
Los Angeles 31415        0.0                     0.0                   0.0                     0.0
Los Angeles 27182        0.0                     0.0                   0.0                     0.0
 Port Klang 31415        0.0                     0.0                   0.0                     0.0
 Port Klang 27182        0.0                     0.0                   0.0                     0.0
     Santos 31415        0.0                     0.0                   0.0                     0.0
     Santos 27182        0.0                     0.0                   0.0                     0.0
```

Maximum branch model/optimizer/gradient mutation: 0
Maximum final archive replay error: 0 (atol=1e-10, rtol=0).
