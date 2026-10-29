# Centered logit increment decomposition

For each query, subtract the across-class mean from new-minus-original logits; common is the across-query mean of that centered increment.
Residual is each query centered increment minus common. All energies are sums of squared logit increments, not averages or physical energy.
Absolute total/common/residual energies accompany fractions: a larger residual fraction alone does not establish a larger useful update.
Means and sample SD are over available orders; energy fractions are averaged per order, not pooled. Summed decision counts are repeated query-order events.
Common-only and residual-only scores are descriptive decompositions of development predictions, not independently trained or deployable interventions.
These fractions are not causal information fractions or proof of useful query/reference dependence.


| Arm | Orders | Mean total energy | Mean common energy | Mean residual energy | Mean common fraction | Mean full-common OA | Mean full-common BA |
|---|---|---|---|---|---|---|---|
| query_only | [17, 29, 43] | 474.1190407433596 | 449.7838148945049 | 24.335225848854805 | 0.9487986653391126 | 1.25 | 1.1043083900226758 |
| ref_only | [17, 29, 43] | 478.48252944939856 | 455.3985150649001 | 23.084014384498463 | 0.9518337733672588 | 1.25 | 0.775510204081634 |
| interaction | [17, 29, 43] | 389.0485118479046 | 364.02001595208964 | 25.02849589581494 | 0.9360338893381099 | 0.8333333333333334 | 0.761904761904764 |

| Arm | Seed | Total energy | Common energy | Residual energy | Common fraction | Residual fraction |
|---|---|---|---|---|---|---|
| query_only | 17 | 479.9905515885341 | 451.6609951362965 | 28.329556452237647 | 0.9409789289424956 | 0.05902107105750449 |
| query_only | 29 | 481.70316455544935 | 456.3991475196641 | 25.30401703578535 | 0.9474696890166009 | 0.05253031098339936 |
| query_only | 43 | 460.6634060860955 | 441.291302027554 | 19.37210405854141 | 0.9579473780582413 | 0.04205262194175864 |
| ref_only | 17 | 486.7925101414422 | 459.50777614731703 | 27.28473399412521 | 0.9439499716497336 | 0.05605002835026646 |
| ref_only | 29 | 475.52476826369366 | 451.77865209205976 | 23.746116171633894 | 0.9500633452630886 | 0.04993665473691144 |
| ref_only | 43 | 473.1303099430598 | 454.90911695532355 | 18.221192987736284 | 0.9614880031889542 | 0.03851199681104592 |
| interaction | 17 | 399.1728631314204 | 369.7492358787047 | 29.423627252715704 | 0.9262885081368157 | 0.07371149186318438 |
| interaction | 29 | 423.84087924519247 | 397.30725748747005 | 26.5336217577224 | 0.9373972095259535 | 0.06260279047404643 |
| interaction | 43 | 344.1317931671009 | 325.0035544900942 | 19.128238677006724 | 0.9444159503515603 | 0.055584049648439715 |
