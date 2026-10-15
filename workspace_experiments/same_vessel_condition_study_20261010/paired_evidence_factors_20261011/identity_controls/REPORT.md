# Held Source Identity Controls

{
  "held_chips": 823,
  "held_ships": 302,
  "common_anchors": 233,
  "common_ships": 86,
  "matched_pairs": 359,
  "known_pairs": 165,
  "unknown_pairs": 194,
  "exports": 20,
  "cohort": "Metadata-only held-source cohort, identical across models and spaces; no correctness filter.",
  "matching": "Same MMSI, different port/product. Other MMSI, same class, donor port/product exact; known scene-mean angle gap <=2 degrees. No fallback.",
  "distances": "Squared Euclidean distances in supplied z/sqrt(d); no additional division by dimension. Ratio is weighted same/other distance.",
  "teachers": "Average chips per donor scene, then equal matched scenes. Reused other scenes retain pair weights. Multi-observation diagnostic, not deployment.",
  "relations": "All seven true-vs-rival relations; margin>1e-8 is positive. Rescue: nonpositive to positive; harm: positive to nonpositive. Percentages use all seven relations.",
  "aggregation": "Equal chips/acquisition, acquisitions/vessel, then either vessels or covered classes. Strata have separate cohorts; never compare strata as causal effects.",
  "limitations": "Natural repeats, shared scenes and reused held development data; unknown angles retained separately; no causal identification or target evaluation."
}

Two-seed descriptive means:

                                      distance_ratio  same_net_pct  other_net_pct  anchor_positive_pct
branch         factor      space                                                                      
split_relation class_only  candidate        1.014741     -3.664000      -5.202406            80.178302
                           remainder        0.984918      0.000000       0.000000             0.000000
               head        candidate        1.020206     -4.803048      -5.720826            81.801048
                           remainder        0.983957     -5.712307      -0.721367            46.854791
               paired      candidate        0.982219     -4.142521      -5.463180            81.395166
                           remainder        0.990655      3.544570       5.972843            57.727638
               pca         candidate        0.902201      4.094226       2.557433            62.846627
                           remainder        1.006286     -2.014867      -0.637030            77.214801
               stable_only candidate        0.868100      2.341652      -3.694813            79.176815
                           remainder        1.004504      3.805145       1.360845            61.464982
split_task     class_only  candidate        1.012686     -3.677046      -4.689817            80.452713
                           remainder        0.983833      0.000000       0.000000             0.000000
               head        candidate        1.011637     -5.166584      -5.449712            82.134492
                           remainder        0.983982     -4.829045      -0.183221            46.231136
               paired      candidate        0.978628     -3.418552      -3.817221            80.285018
                           remainder        0.989755      1.997243       5.349255            58.753806
               pca         candidate        0.913812      1.043855      -1.093413            65.403322
                           remainder        1.002076     -2.788235      -1.656910            78.053016
               stable_only candidate        0.859971      1.618748      -3.093713            78.761472
                           remainder        1.004396      3.084060       1.452580            62.303092

Positive net correction is teacher ordering rescue minus harm, not eight-class BA.
Candidate/remainder distance magnitudes have different supplied dimension scaling.
No training, target predictions, target metrics, or model selection performed.