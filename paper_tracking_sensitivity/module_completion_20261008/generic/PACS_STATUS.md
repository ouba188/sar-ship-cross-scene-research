# PACS retrained RGB status

Protocol: retrained_rgb_reference. Fixed ImageNet RGB initialization family,
12 source-only bootstrap epochs, batch 64, uint8 Resize224 cache, bf16 autocast
with float32 parameters/Adam. Aligned-only and joint each train layer4 for six
epochs. Every outer target domain is excluded from encoder/head/gate training.
The old five support draws and Base64 scores remain unchanged. No target tuning.

- art_painting: complete. Bootstrap training 177.36 s; aligned-only training
  20.19 s; joint training 19.38 s. These exclude imports, prefix/features,
  CPU preparation, and evaluation. Full OA/BA (%): reference 81.2399/81.1344,
  complement 81.3508/81.2609, aligned 80.1411/80.1275, joint 80.0403/79.9779.
- cartoon: complete. Bootstrap training 139.02 s; aligned-only training
  21.30 s; joint training 15.11 s. Full OA/BA (%): reference 74.1316/77.4182,
  complement 74.1053/77.3757, aligned 74.0526/77.3684, joint 74.0526/77.3579.
- photo: complete. Bootstrap training 124.38 s; aligned-only training
  26.38 s; joint training 29.42 s. Full OA/BA (%): reference 94.6077/93.3728,
  complement 94.5205/93.2559, aligned 93.9103/93.0107, joint 93.6986/92.8388.
- sketch: complete. Bootstrap training 136.06 s; aligned-only training
  19.16 s; joint training 16.99 s. Full OA/BA (%): reference 61.2574/66.6167,
  complement 61.2781/66.6529, aligned 61.2937/66.5615, joint 61.3299/66.5626.

All four domains and 80 readouts are complete (four arms x five draws x four
domains). Equal-domain means, with equal draw weights within each domain:

| Arm | Full OA % | Full BA % | Delta OA pp | Delta BA pp |
| --- | ---: | ---: | ---: | ---: |
| reference | 77.809164 | 79.635512 | 0 | 0 |
| complement_only | 77.813689 | 79.636341 | +0.004524 | +0.000829 |
| aligned_only | 77.349439 | 79.266995 | -0.459725 | -0.368517 |
| joint | 77.280367 | 79.184320 | -0.528798 | -0.451192 |

Joint BA decreases in all four domains. Complement-only is essentially flat;
these results do not support a general RGB-domain improvement from joint.
Total logged bootstrap/tail training: 744.720 s, excluding preparation,
imports, feature extraction, and evaluation. No target-based recipe changes.

Machine summary: rgb_retrained/pacs/summary.json; all 80 rows:
rgb_retrained/pacs/metrics.csv and metrics.json.
GPU was released to the main agent for six SAR ports after PACS completion.
Subsequently, all four Office-Home folds also completed under the same recipe;
see GENERIC_COMPLETION_ZH.md and rgb_retrained/officehome/summary.json.
The generic branch released the GPU again at 2026-10-08 11:29:44 Asia/Shanghai.
