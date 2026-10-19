"""Describe region coverage of existing 3x3 windows, not a performance experiment."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
BRIDGE=HERE.parent


def window_sum(x):
    padded=np.pad(x,1)
    return sum(padded[i:i+x.shape[0],j:j+x.shape[1]] for i in range(3) for j in range(3))


def main():
    roles=np.load(BRIDGE/'masks.npz')['regions']
    roles=roles.reshape(len(roles),3,32,4,32,4).mean((3,5))
    meta=pd.read_csv(BRIDGE/'learned_geometry_crosspass_20261019/cohort.csv')
    records=[]
    for i in [197,227,219,215,45,63,94,120]:
        target,other,rest=roles[i]
        centers=target>=.5
        count=window_sum(np.ones_like(target))
        foreign=window_sum(other+rest)/count
        record=dict(row=i,label=int(meta.iloc[i].label),target_dominant_cells=int(centers.sum()),
            window_foreign_mass_mean=float(foreign[centers].mean()) if centers.any() else np.nan,
            target_cells_with_foreign_window=int((centers&(foreign>.01)).sum()),
            other_candidate_cell_mass=float(other.sum()),
            candidate_mass_in_target_windows=float((window_sum(other)/count)[centers].mean()) if centers.any() else np.nan)
        records.append(record)
    out=pd.DataFrame(records)
    out.to_csv(HERE/'region_window_coverage.csv',index=False)
    print(out.to_string(index=False))


if __name__=='__main__':main()
