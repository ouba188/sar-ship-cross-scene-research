# Added-writer dependence controls

Every delta is intervention minus reference. Rescue/harm states follow that same direction.
off disables only the added writer; logits must recover the saved original within 1e-4. Any top-1 differences are reported separately.
uniform replaces only the added writer reference weights with legal uniform weights, retaining transport and the frozen original M3.
uniform versus normal tests sensitivity and utility of the added writer reference weighting. It does not remove reference content, transport dependence, or query dependence.
query_only uniform is a negative control for the changed reference weights; numerical differences and decision changes are reported without inferring a physical mechanism.
New-arm versus old CE comparisons belong in the main analysis. These controls do not update parameters or constitute separately trained candidates.
Missing controls remain pending. Available-seed means are provisional; repeated query events are not independent samples.


| Intervention - reference | Orders | Mean delta OA | SD OA | Mean delta BA | SD BA | Mean delta NLL | Mean delta Brier |
|---|---|---|---|---|---|---|---|
| query_only_off - query_only | [17, 29, 43] | 0.29239766081871466 | 0.5064476045523054 | 0.048103289179380226 | 1.4403327475090748 | -0.024859720823567095 | -0.003862515270014725 |
| query_only_off - saved_baseline | [17, 29, 43] | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| query_only_uniform - query_only | [17, 29, 43] | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| query_only_uniform - saved_baseline | [17, 29, 43] | -0.29239766081871466 | 0.5064476045523054 | -0.048103289179380226 | 1.4403327475090748 | 0.024859720823567095 | 0.003862515270014725 |
| ref_only_off - ref_only | [17, 29, 43] | 0.0 | 0.877192982456144 | -0.6765343919800415 | 2.596549001014806 | -0.026693297633277435 | -0.003515150574452212 |
| ref_only_off - saved_baseline | [17, 29, 43] | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| ref_only_uniform - ref_only | [17, 29, 43] | 0.0 | 0.0 | -0.2665333999666828 | 0.4616493906563715 | 0.0006810146984603064 | -0.0003188741144268074 |
| ref_only_uniform - saved_baseline | [17, 29, 43] | 0.0 | 0.877192982456144 | 0.4100009920133587 | 2.1584674611978754 | 0.027374312331737743 | 0.0031962764600254046 |
| interaction_off - interaction | [17, 29, 43] | 0.5847953216374245 | 0.5064476045523013 | 0.31248420733453725 | 1.4790144448305398 | -0.023656240225180536 | -0.0031888880814806195 |
| interaction_off - saved_baseline | [17, 29, 43] | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| interaction_uniform - interaction | [17, 29, 43] | 0.14619883040935613 | 0.25322380227615066 | 0.0957854406130257 | 0.16590524976713192 | -0.0009535434832856904 | -0.00026572022929072237 |
| interaction_uniform - saved_baseline | [17, 29, 43] | -0.43859649122806843 | 0.7596714068284519 | -0.21669876672151153 | 1.643082492194769 | 0.022702696741894846 | 0.002923167852189897 |

| Arm | Control | Seed | Max logit change vs normal | Changed vs normal | Max logit change vs baseline | Changed vs baseline |
|---|---|---|---|---|---|---|
| query_only | off | 17 | 0.6801676750183105 | 12 | 0.0 | 0 |
| query_only | uniform | 17 | 0.0 | 0 | 0.6801676750183105 | 12 |
| query_only | off | 29 | 0.9250473976135254 | 7 | 0.0 | 0 |
| query_only | uniform | 29 | 0.0 | 0 | 0.9250473976135254 | 7 |
| query_only | off | 43 | 0.4003511667251587 | 6 | 0.0 | 0 |
| query_only | uniform | 43 | 0.0 | 0 | 0.4003511667251587 | 6 |
| ref_only | off | 17 | 0.7119007110595703 | 12 | 0.0 | 0 |
| ref_only | uniform | 17 | 0.03296971321105957 | 0 | 0.7100973129272461 | 12 |
| ref_only | off | 29 | 0.7769923210144043 | 8 | 0.0 | 0 |
| ref_only | uniform | 29 | 0.13319307565689087 | 2 | 0.7416296005249023 | 6 |
| ref_only | off | 43 | 0.36048805713653564 | 6 | 0.0 | 0 |
| ref_only | uniform | 43 | 0.007140636444091797 | 0 | 0.36059772968292236 | 6 |
| interaction | off | 17 | 0.660365104675293 | 10 | 0.0 | 0 |
| interaction | uniform | 17 | 0.03440713882446289 | 0 | 0.6611275672912598 | 10 |
| interaction | off | 29 | 0.7860908508300781 | 6 | 0.0 | 0 |
| interaction | uniform | 29 | 0.0629281997680664 | 1 | 0.7299199104309082 | 5 |
| interaction | off | 43 | 0.37341558933258057 | 8 | 0.0 | 0 |
| interaction | uniform | 43 | 0.00879049301147461 | 0 | 0.36862337589263916 | 8 |

| Intervention - reference | Seed | Delta OA | Delta BA | Delta NLL | Delta Brier | Rescue | Harm | CC | WW |
|---|---|---|---|---|---|---|---|---|---|
| query_only_off - query_only | 17 | 0.877192982456144 | 1.5057078202471814 | -0.007246252270376008 | 0.004265115140998255 | 5 | 3 | 127 | 93 |
| query_only_off - saved_baseline | 17 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 132 | 96 |
| query_only_uniform - query_only | 17 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 130 | 98 |
| query_only_uniform - saved_baseline | 17 | -0.877192982456144 | -1.5057078202471814 | 0.007246252270376008 | -0.004265115140998255 | 3 | 5 | 127 | 93 |
| ref_only_off - ref_only | 17 | 0.877192982456144 | 1.5057078202471814 | -0.010351443539969862 | 0.0035135765770739624 | 5 | 3 | 127 | 93 |
| ref_only_off - saved_baseline | 17 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 132 | 96 |
| ref_only_uniform - ref_only | 17 | 0.0 | 0.0 | -0.0005843103595650767 | -0.0001794423080094809 | 0 | 0 | 130 | 98 |
| ref_only_uniform - saved_baseline | 17 | -0.877192982456144 | -1.5057078202471814 | 0.009767133180404786 | -0.0036930188850834433 | 3 | 5 | 127 | 93 |
| interaction_off - interaction | 17 | 0.877192982456144 | 0.9245377311344356 | -0.005795991723962501 | 0.0048135675159475255 | 4 | 2 | 128 | 94 |
| interaction_off - saved_baseline | 17 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 132 | 96 |
| interaction_uniform - interaction | 17 | 0.0 | 0.0 | -0.0006699264180896058 | -0.0001336050784701559 | 0 | 0 | 130 | 98 |
| interaction_uniform - saved_baseline | 17 | -0.877192982456144 | -0.9245377311344356 | 0.0051260653058728956 | -0.004947172594417681 | 2 | 4 | 128 | 94 |
| query_only_off - query_only | 29 | 0.0 | -1.3743128435782097 | -0.048711151441281464 | -0.013669750333644481 | 3 | 3 | 129 | 93 |
| query_only_off - saved_baseline | 29 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 132 | 96 |
| query_only_uniform - query_only | 29 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 132 | 96 |
| query_only_uniform - saved_baseline | 29 | 0.0 | 1.3743128435782097 | 0.048711151441281464 | 0.013669750333644481 | 3 | 3 | 129 | 93 |
| ref_only_off - ref_only | 29 | -0.877192982456144 | -3.548225887056475 | -0.05052559751024832 | -0.011689886704105201 | 3 | 5 | 129 | 91 |
| ref_only_off - saved_baseline | 29 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 132 | 96 |
| ref_only_uniform - ref_only | 29 | 0.0 | -0.7996001999000484 | 0.003010801569338417 | -0.0007591879756246867 | 1 | 1 | 133 | 93 |
| ref_only_uniform - saved_baseline | 29 | 0.877192982456144 | 2.7486256871564265 | 0.05353639907958674 | 0.010930698728480515 | 4 | 2 | 130 | 92 |
| interaction_off - interaction | 29 | 0.0 | -1.3743128435782097 | -0.047441952381730434 | -0.012497083105207696 | 3 | 3 | 129 | 93 |
| interaction_off - saved_baseline | 29 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 132 | 96 |
| interaction_uniform - interaction | 29 | 0.43859649122806843 | 0.2873563218390771 | -0.0018289548447645299 | -0.0006456519414413942 | 1 | 0 | 132 | 95 |
| interaction_uniform - saved_baseline | 29 | 0.43859649122806843 | 1.6616691654172868 | 0.045612997536965905 | 0.011851431163766302 | 3 | 2 | 130 | 93 |
| query_only_off - query_only | 43 | 0.0 | 0.012914890869168971 | -0.018621758759043816 | -0.0021829106173979484 | 2 | 2 | 134 | 90 |
| query_only_off - saved_baseline | 43 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 136 | 92 |
| query_only_uniform - query_only | 43 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 136 | 92 |
| query_only_uniform - saved_baseline | 43 | 0.0 | -0.012914890869168971 | 0.018621758759043816 | 0.0021829106173979484 | 2 | 2 | 134 | 90 |
| ref_only_off - ref_only | 43 | 0.0 | 0.012914890869168971 | -0.019202851849614122 | -0.002369141596325397 | 2 | 2 | 134 | 90 |
| ref_only_off - saved_baseline | 43 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 136 | 92 |
| ref_only_uniform - ref_only | 43 | 0.0 | 0.0 | -0.0003834471143924212 | -1.799205964625461e-05 | 0 | 0 | 136 | 92 |
| ref_only_uniform - saved_baseline | 43 | 0.0 | -0.012914890869168971 | 0.0188194047352217 | 0.0023511495366791424 | 2 | 2 | 134 | 90 |
| interaction_off - interaction | 43 | 0.8771929824561298 | 1.3872277344473858 | -0.017730776569848672 | -0.0018831486551816878 | 4 | 2 | 132 | 90 |
| interaction_off - saved_baseline | 43 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 136 | 92 |
| interaction_uniform - interaction | 43 | 0.0 | 0.0 | -0.00036174918700293546 | -1.7903667960617042e-05 | 0 | 0 | 134 | 94 |
| interaction_uniform - saved_baseline | 43 | -0.8771929824561298 | -1.3872277344473858 | 0.017369027382845736 | 0.0018652449872210708 | 2 | 4 | 132 | 90 |

Pending:
[]
