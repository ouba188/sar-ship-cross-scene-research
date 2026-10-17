"""Scene-level descriptive associations, without fitting a recognition model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ENV = HERE/'environment'


def correlation(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method='spearman'))


def main():
    env = pd.read_csv(ENV/'sample_environment.csv')
    meta = pd.read_csv(HERE.parent/'oss1_joint_geometry_20261017/geometry.csv')
    desc = pd.read_csv(HERE/'image_descriptors.csv')
    desc = desc[(desc.dataset == 'OSS1') & (desc.representation == 'sigma0')].copy()
    columns = ['near_ends', 'near_sides', 'far_ends', 'far_sides']
    neighbor = desc[columns].mean(axis=1, skipna=False)
    desc['neighbor_db'] = np.where(neighbor > 0, 10*np.log10(neighbor), np.nan)
    ids = meta[['global_row', 'sample_id']].merge(env, on='sample_id', validate='one_to_one')
    measured = desc.merge(ids, on='global_row', suffixes=('', '_env'), validate='many_to_one')
    variables = ['era5_wind_speed_m_s', 'era5_hs', 'cmems_current_speed_m_s']
    masks = {
        'era5_wind_speed_m_s': measured.era5_u10_status.eq('actual') & measured.era5_v10_status.eq('actual'),
        'era5_hs': measured.era5_hs_status.eq('actual'),
        'cmems_current_speed_m_s': measured.cmems_uo_status.eq('actual') & measured.cmems_vo_status.eq('actual'),
    }
    for variable, mask in masks.items():
        measured.loc[~mask, variable] = np.nan
    # Predefined broad latitude groups; labels do not claim precise port membership.
    measured['geographic_band'] = np.where(measured.lat >= 35, 'north_35plus',
                                          np.where(measured.lat >= 27, 'middle_27to35', 'south_below27'))
    measured.to_csv(HERE/'image_environment_join.csv', index=False)
    values = variables+['neighbor_db', 'body_neighbor_db', 'ends_sides_balance', 'incidence_deg']
    per_class = measured.groupby(['product', 'pol', 'geographic_band', 'class_id'])[values].median().reset_index()
    # First summarize within class, then give the observed classes equal weight in each scene.
    scene = per_class.groupby(['product', 'pol', 'geographic_band'])[values].mean().reset_index()
    counts = measured.groupby(['product', 'pol']).agg(
        samples=('global_row', 'nunique'), classes=('class_id', 'nunique'),
        era5_cell_hours=('era5_shared_cell_time_id', 'nunique'),
        cmems_cell_days=('cmems_shared_cell_time_id', 'nunique')).reset_index()
    scene = scene.merge(counts, on=['product', 'pol'], validate='one_to_one')
    scene.to_csv(HERE/'environment_scene_summary.csv', index=False)
    correlations = []
    for pol, group in scene.groupby('pol'):
        for variable in variables:
            for response in ['neighbor_db', 'body_neighbor_db']:
                g = group.dropna(subset=[variable, response, 'incidence_deg']).reset_index(drop=True)
                if len(g) < 6:
                    continue
                ranked = g[[variable, response, 'incidence_deg']].rank()
                bands = pd.get_dummies(g.geographic_band, drop_first=True, dtype=float)
                x = np.column_stack([np.ones(len(g)), bands.to_numpy(), ranked.incidence_deg])
                r = ranked[[variable, response]].to_numpy()
                residual = r-x@np.linalg.lstsq(x, r, rcond=None)[0]
                jack = [correlation(g.loc[g.index != j, variable], g.loc[g.index != j, response]) for j in g.index]
                correlations.append(dict(pol=pol, variable=variable, image_response=response,
                                         acquisitions=len(g), geographic_bands=g.geographic_band.nunique(),
                                         scene_rank_correlation=correlation(g[variable], g[response]),
                                         geography_incidence_partial_rank_correlation=float(np.corrcoef(residual.T)[0, 1]),
                                         leave_one_acquisition_rho_min=min(jack), leave_one_acquisition_rho_max=max(jack)))
    out = pd.DataFrame(correlations)
    out.to_csv(HERE/'ENVIRONMENT_ASSOCIATIONS.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout='constrained')
    for i, pol in enumerate(['vv', 'vh']):
        for j, variable in enumerate(variables[:2]):
            ax = axes[i, j]
            for band, g in scene[scene.pol.eq(pol)].groupby('geographic_band'):
                ax.scatter(g[variable], g.neighbor_db, s=26, label=band, alpha=.8)
            ax.set(xlabel='10m wind speed (m/s)' if j == 0 else 'Significant wave height (m)',
                   ylabel=f'{pol.upper()} neighborhood sigma0 (dB)', title='One point per acquisition; class-balanced summary')
            ax.legend(fontsize=7)
    fig.suptitle('Environmental associations are exploratory, not causal or classification gains', fontsize=11)
    fig.savefig(HERE/'environment_scene_associations.png', dpi=160)
    plt.close(fig)
    status = dict(acquisitions_with_wind=int(scene.dropna(subset=['era5_wind_speed_m_s'])['product'].nunique()),
                  acquisitions_with_wave=int(scene.dropna(subset=['era5_hs'])['product'].nunique()),
                  acquisitions_with_current=int(scene.dropna(subset=['cmems_current_speed_m_s'])['product'].nunique()),
                  new_classifier_fits=0, selection='All available predefined OSS1 rows; no target-utility selection',
                  dependence='Correlations use acquisition summaries, not repeated chip/grid assignments',
                  statistics='Exploratory rank correlations; no significance/causal claim; leave-one-acquisition range is not a confidence interval',
                  inference='No environmental variables have been added to inference')
    (HERE/'ENVIRONMENT_ANALYSIS_DONE.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
