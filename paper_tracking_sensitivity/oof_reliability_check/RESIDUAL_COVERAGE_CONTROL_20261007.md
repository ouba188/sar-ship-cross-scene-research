# Follow-up strong control, declared after residual expert readout

The original fixed residual screen passed its two positive-mean criteria.
The bias-only arm accounts for much of the improvement, raising a concrete
alternative explanation: the source classifier may simply predict classes
not represented in the 64 target labels.

Before this control's readout, fix ONE baseline: unchanged original fusion
scores, set every class absent from the same 64 support labels to -infinity,
then argmax. No changes to feature scaling, coefficients, support, test rows,
regularization, or candidate experts. This is a deliberately simple hard
observed-class restriction, not an assumption absent support means absent
query truth and not a proposed novel method.

Save all 23-port/5-draw predictions before reading their query outcomes.
Compare this baseline with the unchanged fusion, anchored bias and anchored
visual arms using identical full-query BA/accuracy. Preserve original screen
results and label this as a POST-READOUT explanatory follow-up. A stronger
baseline takes precedence over a superficially novel module in prioritization.
