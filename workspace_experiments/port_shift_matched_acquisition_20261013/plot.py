"""Render existing direction effects without rerunning metrics or fitting."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).resolve().parent
    pair=pd.read_csv(root/'direction_effects.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,b in zip(axs,[16,64]):
        sub=pair[(pair.budget==b)&(pair['mode']=='dual_late')&(pair.contrast=='port_same_acquisition')].copy()
        sub['label']=sub.query_port+' <- '+sub.support_port
        sub=sub.sort_values('delta_ba')
        ax.barh(sub.label,sub.delta_ba,color=['#b34454' if v<0 else '#268c83' for v in sub.delta_ba])
        ax.axvline(0,color='#444444',lw=.8);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        ax.set_xlabel('Other-port minus same-port BA (pp)')
        ax.set_title(f'B={b}: same acquisition\nCompact dual-head diagnostic')
    fig.savefig(root/'port_effects.png',dpi=170);fig.savefig(root/'port_effects.svg');plt.close(fig)


if __name__=='__main__':main()
