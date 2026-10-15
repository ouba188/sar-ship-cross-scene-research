"""One source-gradient-normalized follow-up, with no coefficient search."""

import pandas as pd
from train_split_evidence_kernel import OUT, main


if __name__=='__main__':
    diagnostics=pd.read_csv(OUT/'mechanism/source_objectives.csv')
    initial=diagnostics[diagnostics.arm.eq('pca') & diagnostics.snapshot.eq('initial')]
    weights={int(row.seed):.1/max(1.,row.stability_to_task_gradient_norm)
             for row in initial.itertuples()}
    main(out=OUT/'gradient_balanced',arms=('selected_directions',),stability_weights=weights)
