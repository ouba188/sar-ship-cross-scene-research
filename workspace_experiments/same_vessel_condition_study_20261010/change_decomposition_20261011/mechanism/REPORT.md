# Source Change Decomposition Diagnostics

Two-seed full-model held-source means. Observational, not causal.

## Independent Class Probes

    method    space        OA        BA  seed_count
class_only  content 43.317132 38.624618           2
class_only residual 11.117861 11.943938           2
class_only        z 43.317132 38.712617           2
    paired  content 43.803159 38.994224           2
    paired residual 17.739976 17.741319           2
    paired        z 43.317132 38.712617           2

## Same-Vessel Distinct-Product Deltas

    method      scope  residual_delta_mse_ratio  residual_explained_fraction  content_delta_energy_ratio  residual_delta_energy_ratio  full_reconstruction_delta_ratio
class_only        all                  0.144188                     0.855812                    0.153897                     0.855814                         0.005578
class_only cross_port                  0.142592                     0.857408                    0.155149                     0.857406                         0.005621
    paired        all                  0.135773                     0.864227                    0.133540                     0.864228                         0.029505
    paired cross_port                  0.126975                     0.873025                    0.134411                     0.873023                         0.025751

## Acquisition Input Versus Chip Aggregation

    method      scope  residual_explained_fraction  acquisition_input_residual_explained_fraction  residual_jensen_delta_rms_ratio  residual_jensen_endpoint_rms_ratio
class_only        all                     0.855812                                       0.855812                     7.216954e-09                        5.110331e-09
class_only cross_port                     0.857408                                       0.857408                     0.000000e+00                        0.000000e+00
    paired        all                     0.864227                                       0.864227                     7.758128e-09                        5.610738e-09
    paired cross_port                     0.873025                                       0.873025                     0.000000e+00                        0.000000e+00

Load exported selected model without training; run model(mean acquisition z). Compare with acquisition mean of exported model(chip z) components. acquisition_input_* metrics use the former, ordinary metrics the latter. Both use identical actual original z deltas and class/ship/pair weights. Jensen gap=model(mean z).residual-mean(model(z).residual). Delta RMS ratio is sqrt(E||gap_a-gap_b||^2/E||delta_z||^2). Endpoint RMS ratio is sqrt(E[(||gap_a||^2+||gap_b||^2)/2]/E||delta_z||^2), on the same covered pairs.

## Mechanical Partition Identity

Orthogonal decoder components impose delta_content dot delta_residual=0. Let e=delta_content+delta_residual-delta_z. Residual delta error energy =content delta energy+||e||^2-2 delta_content dot e. Exact reconstruction therefore forces residual explained fraction=1-content delta energy ratio. These are not independent evidence of separation. partition_identity_gap is residual error ratio minus content energy ratio; approximate reconstruction can change it. No fixed physical axes or causal factors are identified.

    method      scope  partition_identity_gap  full_reconstruction_delta_ratio  content_residual_delta_inner_product_ratio
class_only        all               -0.009710                         0.005578                                1.416772e-10
class_only cross_port               -0.012557                         0.005621                                3.112765e-10
    paired        all                0.002233                         0.029505                                8.850797e-10
    paired cross_port               -0.007436                         0.025751                               -5.541201e-10

## Selected Content Readout

    method        OA        BA  OA_defined_seeds  BA_defined_seeds  seed_count
class_only 43.377886 38.719315                 2                 2           2
    paired 43.803159 38.994224                 2                 2           2

## Actual Coverage

       role      scope  class_id  chips  ships  acquisitions  classes  covered_classes  covered_ships  pairs  covered_acquisitions  covered_chips
  sourcefit        all        -1   1296    525          1296        8                8            269   2281                  1040           1040
  sourcefit        all         0    376    149           376        1                1             86    652                   313            313
  sourcefit        all         1    197     73           197        1                1             40    368                   164            164
  sourcefit        all         2    331    129           331        1                1             70    604                   272            272
  sourcefit        all         3    141     48           141        1                1             26    308                   119            119
  sourcefit        all         4     71     40            71        1                1             16     58                    47             47
  sourcefit        all         5     40     25            40        1                1              7     32                    22             22
  sourcefit        all         6     79     35            79        1                1             13    158                    57             57
  sourcefit        all         7     61     26            61        1                1             11    101                    46             46
  sourcefit cross_port        -1   1296    525          1296        8                7            133    585                   474            474
  sourcefit cross_port         0    376    149           376        1                1             67    305                   256            256
  sourcefit cross_port         1    197     73           197        1                1              7     16                    20             20
  sourcefit cross_port         2    331    129           331        1                1             45    242                   162            162
  sourcefit cross_port         3    141     48           141        1                1              4      9                    13             13
  sourcefit cross_port         4     71     40            71        1                1              4      5                     9              9
  sourcefit cross_port         5     40     25            40        1                1              2      3                     5              5
  sourcefit cross_port         6     79     35            79        1                1              4      5                     9              9
  sourcefit cross_port         7     61     26            61        1                0              0      0                     0              0
test_source        all        -1    823    302           823        8                8            200   1492                   721            721
test_source        all         0    266     88           266        1                1             69    510                   247            247
test_source        all         1    136     43           136        1                1             31    255                   124            124
test_source        all         2    206     72           206        1                1             50    410                   184            184
test_source        all         3     35     16            35        1                1              8     75                    27             27
test_source        all         4     46     26            46        1                1             12     32                    32             32
test_source        all         5     27     19            27        1                1              6     10                    14             14
test_source        all         6     66     20            66        1                1             14    145                    60             60
test_source        all         7     41     18            41        1                1             10     55                    33             33
test_source cross_port        -1    823    302           823        8                7            116    479                   409            409
test_source cross_port         0    266     88           266        1                1             59    254                   206            206
test_source cross_port         1    136     43           136        1                1             11     68                    54             54
test_source cross_port         2    206     72           206        1                1             31    110                   104            104
test_source cross_port         3     35     16            35        1                1              5     31                    20             20
test_source cross_port         4     46     26            46        1                1              5     10                    14             14
test_source cross_port         5     27     19            27        1                1              4      5                     9              9
test_source cross_port         6     66     20            66        1                1              1      1                     2              2
test_source cross_port         7     41     18            41        1                0              0      0                     0              0

Classes without cross-port pair coverage:

       role      scope  class_id  chips  ships  acquisitions  classes  covered_classes  covered_ships  pairs  covered_acquisitions  covered_chips
  sourcefit cross_port         7     61     26            61        1                0              0      0                     0              0
test_source cross_port         7     41     18            41        1                0              0      0                     0              0

## Full Reconstruction

    method  original_energy  reconstruction_mse  full_reconstruction_ratio  original_energy_defined_seeds  reconstruction_mse_defined_seeds  full_reconstruction_ratio_defined_seeds  seed_count
class_only          1.00408            0.005358                   0.005357                              2                                 2                                        2           2
    paired          1.00408            0.031297                   0.031258                              2                                 2                                        2           2

## Independent Matched Identity Control

    method              space  same_other_ratio  seed_count
class_only            content          1.032672           2
class_only           residual          0.977738           2
class_only                  z          0.983123           2
class_only  content_component          1.032672           2
class_only residual_component          0.977738           2
    paired            content          1.020257           2
    paired           residual          0.985216           2
    paired                  z          0.983123           2
    paired  content_component          1.020257           2
    paired residual_component          0.985216           2

Actual held cohort: 359 matched pairs; 233 anchors.

Observational separation diagnostics, not causal identification. Residual class readability is not expected to be zero. Coverage is class dependent; zero coverage is not zero drift. No validation or target metrics, architecture comparisons, or neural training.

Class-wise recalls are in class_recall_held_summary.csv. A missing ratio remains undefined; explained fractions are not clipped. Decoder components are orthogonal, but their energy ratios relative to original z need not add to one because reconstruction is approximate.
