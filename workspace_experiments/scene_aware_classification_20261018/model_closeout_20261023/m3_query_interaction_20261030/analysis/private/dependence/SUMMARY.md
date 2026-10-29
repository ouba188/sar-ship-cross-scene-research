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
| query_only_off - query_only | [17, 29, 43] | 0.8333333333333334 | 2.1949278651776543 | 0.37595341166769697 | 0.8763365608881154 | 0.0348099434075948 | -0.005396685474477543 |
| query_only_off - saved_baseline | [17, 29, 43] | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| query_only_uniform - query_only | [17, 29, 43] | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| query_only_uniform - saved_baseline | [17, 29, 43] | -0.8333333333333334 | 2.1949278651776543 | -0.37595341166769697 | 0.8763365608881154 | -0.0348099434075948 | 0.005396685474477543 |
| ref_only_off - ref_only | [17, 29, 43] | 0.625 | 1.875 | 0.18547722119150478 | 0.5464768164023474 | 0.03946959238205494 | -0.005329338990089287 |
| ref_only_off - saved_baseline | [17, 29, 43] | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| ref_only_uniform - ref_only | [17, 29, 43] | 0.0 | 0.0 | 0.0 | 0.0 | 7.259229180966997e-05 | -0.0003202334472861927 |
| ref_only_uniform - saved_baseline | [17, 29, 43] | -0.625 | 1.875 | -0.18547722119150478 | 0.5464768164023474 | -0.03939700009024527 | 0.005009105542803094 |
| interaction_off - interaction | [17, 29, 43] | 0.4166666666666667 | 2.1949278651776543 | -0.12744794887652233 | 0.825333466729988 | 0.0267808754320793 | -0.006998866381137299 |
| interaction_off - saved_baseline | [17, 29, 43] | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| interaction_uniform - interaction | [17, 29, 43] | 0.0 | 0.0 | 0.0 | 0.0 | -0.00036985663918143175 | -0.00047560285463621454 |
| interaction_uniform - saved_baseline | [17, 29, 43] | -0.4166666666666667 | 2.1949278651776543 | 0.12744794887652233 | 0.825333466729988 | -0.02715073207126073 | 0.0065232635265010845 |

| Arm | Control | Seed | Max logit change vs normal | Changed vs normal | Max logit change vs baseline | Changed vs baseline |
|---|---|---|---|---|---|---|
| query_only | off | 17 | 2.0337276458740234 | 14 | 0.0 | 0 |
| query_only | uniform | 17 | 0.0 | 0 | 2.0337276458740234 | 14 |
| query_only | off | 29 | 2.040217399597168 | 20 | 0.0 | 0 |
| query_only | uniform | 29 | 0.0 | 0 | 2.040217399597168 | 20 |
| query_only | off | 43 | 1.4503674507141113 | 18 | 0.0 | 0 |
| query_only | uniform | 43 | 0.0 | 0 | 1.4503674507141113 | 18 |
| ref_only | off | 17 | 2.042814254760742 | 13 | 0.0 | 0 |
| ref_only | uniform | 17 | 0.043770790100097656 | 0 | 2.0206336975097656 | 13 |
| ref_only | off | 29 | 2.0366334915161133 | 19 | 0.0 | 0 |
| ref_only | uniform | 29 | 0.03792381286621094 | 0 | 2.0140581130981445 | 19 |
| ref_only | off | 43 | 1.4252147674560547 | 18 | 0.0 | 0 |
| ref_only | uniform | 43 | 0.04073596000671387 | 0 | 1.4272165298461914 | 18 |
| interaction | off | 17 | 1.7250595092773438 | 15 | 0.0 | 0 |
| interaction | uniform | 17 | 0.04215502738952637 | 0 | 1.7153949737548828 | 15 |
| interaction | off | 29 | 1.839651107788086 | 20 | 0.0 | 0 |
| interaction | uniform | 29 | 0.035157203674316406 | 0 | 1.8173332214355469 | 20 |
| interaction | off | 43 | 1.2260613441467285 | 17 | 0.0 | 0 |
| interaction | uniform | 43 | 0.03326058387756348 | 0 | 1.199859619140625 | 17 |

| Intervention - reference | Seed | Delta OA | Delta BA | Delta NLL | Delta Brier | Rescue | Harm | CC | WW |
|---|---|---|---|---|---|---|---|---|---|
| query_only_off - query_only | 17 | 0.625 | -0.11734693877551905 | 0.042912275787212906 | -0.005531967219505218 | 3 | 2 | 70 | 85 |
| query_only_off - saved_baseline | 17 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 73 | 87 |
| query_only_uniform - query_only | 17 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 72 | 88 |
| query_only_uniform - saved_baseline | 17 | -0.625 | 0.11734693877551905 | -0.042912275787212906 | 0.005531967219505218 | 2 | 3 | 70 | 85 |
| ref_only_off - ref_only | 17 | 0.625 | -0.11734693877551905 | 0.04802718337320311 | -0.005315543386300803 | 3 | 2 | 70 | 85 |
| ref_only_off - saved_baseline | 17 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 73 | 87 |
| ref_only_uniform - ref_only | 17 | 0.0 | 0.0 | -0.0005439397946047642 | -0.0005408057338351924 | 0 | 0 | 72 | 88 |
| ref_only_uniform - saved_baseline | 17 | -0.625 | 0.11734693877551905 | -0.04857112316780787 | 0.0047747376524656104 | 2 | 3 | 70 | 85 |
| interaction_off - interaction | 17 | 0.625 | -0.4846938775510239 | 0.03481186176196571 | -0.009028819025796442 | 3 | 2 | 70 | 85 |
| interaction_off - saved_baseline | 17 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 73 | 87 |
| interaction_uniform - interaction | 17 | 0.0 | 0.0 | -0.0012510640133567286 | -0.000825414823473869 | 0 | 0 | 72 | 88 |
| interaction_uniform - saved_baseline | 17 | -0.625 | 0.4846938775510239 | -0.03606292577532244 | 0.008203404202322573 | 2 | 3 | 70 | 85 |
| query_only_off - query_only | 29 | 3.125 | 1.3877551020408205 | -0.04481711476619843 | -0.037471656971497014 | 5 | 0 | 68 | 87 |
| query_only_off - saved_baseline | 29 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 73 | 87 |
| query_only_uniform - query_only | 29 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 68 | 92 |
| query_only_uniform - saved_baseline | 29 | -3.125 | -1.3877551020408205 | 0.04481711476619843 | 0.037471656971497014 | 0 | 5 | 68 | 87 |
| ref_only_off - ref_only | 29 | 2.5 | 0.816326530612244 | -0.03926611465525198 | -0.036373876205516575 | 4 | 0 | 69 | 87 |
| ref_only_off - saved_baseline | 29 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 73 | 87 |
| ref_only_uniform - ref_only | 29 | 0.0 | 0.0 | -0.00032673691447637765 | -0.0004670305746458103 | 0 | 0 | 69 | 91 |
| ref_only_uniform - saved_baseline | 29 | -2.5 | -0.816326530612244 | 0.038939377740775605 | 0.035906845630870765 | 0 | 4 | 69 | 87 |
| interaction_off - interaction | 29 | 2.5 | 0.816326530612244 | -0.05126799931007131 | -0.038632627227038085 | 4 | 0 | 69 | 87 |
| interaction_off - saved_baseline | 29 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 73 | 87 |
| interaction_uniform - interaction | 29 | 0.0 | 0.0 | -0.0008225132283898873 | -0.0006264877051522788 | 0 | 0 | 69 | 91 |
| interaction_uniform - saved_baseline | 29 | -2.5 | -0.816326530612244 | 0.05044548608168142 | 0.03800613952188581 | 0 | 4 | 69 | 87 |
| query_only_off - query_only | 43 | -1.25 | -0.14254792826221063 | 0.10633466920176993 | 0.0268135677675696 | 5 | 7 | 65 | 83 |
| query_only_off - saved_baseline | 43 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 70 | 90 |
| query_only_uniform - query_only | 43 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 72 | 88 |
| query_only_uniform - saved_baseline | 43 | 1.25 | 0.14254792826221063 | -0.10633466920176993 | -0.0268135677675696 | 7 | 5 | 65 | 83 |
| ref_only_off - ref_only | 43 | -1.25 | -0.14254792826221063 | 0.1096477084282137 | 0.02570140262154952 | 5 | 7 | 65 | 83 |
| ref_only_off - saved_baseline | 43 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 70 | 90 |
| ref_only_uniform - ref_only | 43 | 0.0 | 0.0 | 0.0010884535845101517 | 4.7135966622424696e-05 | 0 | 0 | 72 | 88 |
| ref_only_uniform - saved_baseline | 43 | 1.25 | 0.14254792826221063 | -0.10855925484370355 | -0.025654266654927094 | 7 | 5 | 65 | 83 |
| interaction_off - interaction | 43 | -1.875 | -0.7139764996907871 | 0.0967987638443435 | 0.02666484710942263 | 4 | 7 | 66 | 83 |
| interaction_off - saved_baseline | 43 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 70 | 90 |
| interaction_uniform - interaction | 43 | 0.0 | 0.0 | 0.0009640073242023206 | 2.5093964717504136e-05 | 0 | 0 | 73 | 87 |
| interaction_uniform - saved_baseline | 43 | 1.875 | 0.7139764996907871 | -0.09583475652014117 | -0.026639753144705125 | 7 | 4 | 66 | 83 |

Pending:
[]
