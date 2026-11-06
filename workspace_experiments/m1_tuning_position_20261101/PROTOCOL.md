# M1 tuning jointly with insertion position

2026-11-01. Local development experiment; no Git or formal-model replacement.

## Question and scope

Can source-trained M1 high/low class-response attention beat a reasonably tuned mean-context control, rather than merely beat one fixed default?

Reuse OSS2 four-class source719 / historical development228 / fixed32 source references and the E17 carrier. Native160 crop/pad and separate VV/VH remain unchanged. Source-calibrated class readers are reused, fitted without development labels. M1 and layer3[5].conv3 train; M2, M3, head, BN and other convolutions stay fixed. M3 delta and low readout remain the original frozen carrier cache: this isolates M1/downstream adaptation, NOT fully joint M1-M3 training or new reference extraction.

## Position and search

Compare pre/post layer3[5].conv2 (3x3); M2 stays before conv2. Both are 256-channel 10x10 inputs. Each has an independent calibrated reader. No new shallow position is presumed superior.

At both positions, test class_gate and mean_gate with:

| Config | Fixed gamma | Temperature | Internal LR | Output LR | conv3 LR | Auxiliary CE |
|---|---:|---:|---:|---:|---:|---:|
| extended | 0.1 | 1 | 0.0001 | 0.001 | 0.00001 | 0.1 |
| gain | 0.3 | 1 | 0.0001 | 0.001 | 0.00001 | 0.1 |
| sharp | 0.3 | 0.5 | 0.0001 | 0.001 | 0.00001 | 0.1 |
| task_lr | 0.3 | 0.5 | 0.0003 | 0.001 | 0.00001 | 0.02 |

Six epochs, 30 minibatches each, seed17, batch24, same source order/weights, AdamW wd0.01, clip5. Output zero start. Mean is mathematically independent of temperature: reuse gain/mean for sharp/mean after verifying forward parity. Thus 14 distinct fits, not 16 duplicate fits. The final config changes two optimization choices and is a performance candidate, not isolated causal evidence for either change.

Select checkpoint per run by development BA, then OA, then lower NLL, then earlier epoch (including epoch0). Compare identical configs/positions and the independently best class and mean candidates. Keep fixed-final-epoch tables too. The development set is explicitly used for tuning; best results are not independent confirmation. Preserve all negative runs.

If a class candidate wins, freeze each method's selected configuration and epoch before seed29/43 repeats; no per-seed re-selection. Check same-weight uniform-context, spatial-mean and M1-off interventions. If no win, inspect measured internal differences before a targeted design revision, not indefinite blind sweeps.

## Evidence

Record OA/BA/NLL, class counts, rescue/harm IDs, update magnitudes and clipped steps. Review raw VV/VH of rescues, harms and persistent correct/incorrect cases with actual legal source references. Reference thumbnails use pooled-cosine ranking, not an asserted maximum attention path. Do not claim SAR physical causes from appearance alone.
