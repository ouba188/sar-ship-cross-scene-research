"""Keep the stronger existing OSS2 reference distinct from the legacy replay."""

from core import *


def main():
    for name in ('metrics','class_recalls','contrasts','factors','events',
                 'metrics_summary','contrasts_summary','factors_summary','cohort_inventory'):
        old=pd.read_csv(OUT/'full'/(name+'.csv'))
        matched=pd.read_csv(OUT/'matched_full'/(name+'.csv'))
        primary=pd.concat([old[old.task.ne('oss2')],matched],ignore_index=True)
        primary.to_csv(OUT/('primary_'+name+'.csv'),index=False)
    d=pd.read_csv(OUT/'primary_contrasts_summary.csv')
    d=d[d.scope.eq('original') & d.stage.eq('full') & d.comparison.eq('fixed_vs_original')]
    print('PRIMARY FULL GAIN',flush=True)
    print(d[['task','delta_oa','delta_ba','delta_oa_positive','delta_oa_negative',
             'delta_ba_positive','delta_ba_negative','delta_nll','delta_brier']].to_string(index=False),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    colors=('#187f90','#ba455d')
    x=np.arange(len(d))
    for i,score in enumerate(('oa','ba')):
        axes[0].bar(x+(i-.5)*.32,d['delta_'+score],width=.32,color=colors[i],label=score.upper())
    axes[0].set_xticks(x,['Los Angeles','Port Klang','Santos','OSS2'])
    axes[0].set_title('Full model: source-fixed adaptation')
    axes[0].set_ylabel('Change from matched baseline (pp)')
    m=pd.read_csv(OUT/'matched_full/contrasts_summary.csv')
    m=m[m.scope.eq('original') & m.comparison.eq('fixed_vs_original')].set_index('stage')
    stages=['expert_prior','expert','full']
    x=np.arange(3)
    for i,score in enumerate(('oa','ba')):
        vals=m.loc[stages,'delta_'+score].to_numpy()
        bars=axes[1].bar(x+(i-.5)*.32,vals,width=.32,color=colors[i],label=score.upper())
        axes[1].bar_label(bars,fmt='%+.2f',padding=3,fontsize=9)
    axes[1].set_xticks(x,['Expert prior','After support','Final fusion'])
    axes[1].set_title('OSS2: changes through the decision stages')
    axes[1].set_ylabel('Stage-specific gain (pp)')
    for ax in axes:
        ax.axhline(0,color='#555555',lw=.8)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(frameon=False)
    fig.savefig(OUT/'integration_results.png',dpi=170)
    plt.close(fig)
    write(OUT/'COMPLETE.json',dict(date='2026-10-12',primary_tasks=4,primary_support_sets=20,
        legacy_oss2_additional_sets=5,compact_diagnostic_sets=40,neural_fits=0,
        readout_equation_solves=546,main_model_changed=False,
        primary_oss2_reference='existing matched post-tail dual_refit_late',
        original_prediction_replay=True,transmission_diagnostics_complete=True))


if __name__=='__main__':
    main()
