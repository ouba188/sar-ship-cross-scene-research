# Archived initial implementation

This pass completed 48 fits (four arms, four folds, three seeds) before the independent review finding was repaired. Its metrics and checkpoints remain here and are not the main result.

- Old native coordinates: `(j + 0.5) * W / 56 - 0.5`. Correct conv1-center coordinates: `(j + 0.25) * W / 56 - 0.5`.
- Old `image_ring` coverage feature was shell area divided by image area; old Gaussian coverage was observed mass divided by the theoretical unmasked Gaussian mass. The corrected implementation uses observed/complete intended mask mass for every arm. This changes a reader input as well as its diagnostic interpretation.
- Initial training included image_ring, ship, ship_radar_encoding, joint. The corrected pass also includes a predeclared body_only control, with the same reader and no outer pixels or angle information.
- Original target performance was not inspected to decide these corrections. They follow the coordinate review and the need to distinguish body-readout gains from actual neighborhood content.
- Because center alignment AND coverage semantics changed together, pre/post differences are not an isolated estimate of the center correction alone.
