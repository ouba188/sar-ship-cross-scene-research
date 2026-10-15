# Held Content Scale Diagnostic

Each export uses its own weighted centered covariance of all 823 held-source c7 rows. Weights are equal classes, vessels/class, acquisitions/vessel, chips/acquisition. Weighted variance V=trace(Cov_w(c)); effective rank=exp(-sum(p*log(p))) for covariance eigenvalues normalized to sum one (maximum 7).

Pair numerator is E_pair[||delta_c||^2] after acquisition aggregation, with equal covered classes, vessels/class, pairs/vessel. Divide by V, without a factor of 2 or a dimension divisor. The same full-held covariance denominator is used for all and cross_port; their pair coverage differs. This controls overall content scale, not class composition, identity effects, or causal factors.

Two-seed arithmetic means of per-seed metrics (not a ratio of pooled means):

    method      scope  content_weighted_variance  content_effective_rank  content_pair_delta_energy  content_delta_over_own_variance  seed_count
class_only        all                   6.837370                6.723446                   8.290680                         1.212560           2
class_only cross_port                   6.837370                6.723446                   8.589179                         1.256224           2
    paired        all                   6.045703                6.478926                   7.196573                         1.190202           2
    paired cross_port                   6.045703                6.478926                   7.444133                         1.231322           2

Individual seeds:

model  seed     method        role      scope  chips  ships  classes  pairs  covered_ships  covered_classes  content_weighted_variance  content_effective_rank  content_pair_delta_energy  content_delta_over_own_variance
 full 31415 class_only test_source        all    823    302        8   1492            200                8                   6.839675                6.741850                   8.166219                         1.193948
 full 31415 class_only test_source cross_port    823    302        8    479            116                7                   6.839675                6.741850                   8.332865                         1.218313
 full 31415     paired test_source        all    823    302        8   1492            200                8                   6.090143                6.673053                   7.380452                         1.211868
 full 31415     paired test_source cross_port    823    302        8    479            116                7                   6.090143                6.673053                   7.488738                         1.229649
 full 27182 class_only test_source        all    823    302        8   1492            200                8                   6.835065                6.705042                   8.415140                         1.231172
 full 27182 class_only test_source cross_port    823    302        8    479            116                7                   6.835065                6.705042                   8.845493                         1.294134
 full 27182     paired test_source        all    823    302        8   1492            200                8                   6.001264                6.284799                   7.012695                         1.168536
 full 27182     paired test_source cross_port    823    302        8    479            116                7                   6.001264                6.284799                   7.399527                         1.232995

Lower original-z-normalized pair energy alone does not establish greater stability. Compare the own-variance ratio before attributing the reduction to preferential suppression of within-vessel changes. No probe fitting or model training in this supplement.
