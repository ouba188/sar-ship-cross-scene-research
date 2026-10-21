"""Class-conditioned descriptive checks and independent saved-result verification."""
import json
import unittest

import numpy as np
import pandas as pd

from analyze_conditions import OUT, OLD, corr, pixel_statistics, polygon_mask


class Verification(unittest.TestCase):
    def test_region_statistics_exclude_other_candidate(self):
        raw=np.array([[255,0],[128,64]],dtype=np.uint8)
        target=np.array([[True,False],[False,False]])
        other=np.array([[False,False],[True,False]])
        values=pixel_statistics(raw,target,other)
        self.assertAlmostEqual(values['target_mean'],1)
        self.assertAlmostEqual(values['rest_mean'],32/255)
        self.assertAlmostEqual(values['contrast'],1-32/255)
        self.assertEqual(values['rest_pixels'],2)

    def test_saved_oa_ba_replays_existing_report(self):
        pred=pd.read_csv(OUT/'frozen_predictions.csv.gz')
        expected=pd.read_csv(OLD/'per_seed.csv').set_index(['arm','seed'])
        for (arm,seed),g in pred.groupby(['arm','seed']):
            self.assertAlmostEqual(g.correct.mean(),expected.loc[(arm,seed),'oa'],places=10)
            self.assertAlmostEqual(g.groupby('class_id').correct.mean().mean(),
                                   expected.loc[(arm,seed),'ba'],places=10)

    def test_one_pair_per_ship_and_seed_average(self):
        p=pd.read_csv(OUT/'pair_factors.csv',dtype={'mmsi':str})
        self.assertEqual(len(p),910)
        self.assertTrue(p.mmsi.is_unique)
        b=pd.read_csv(OLD/'pair_predictions.csv')
        b=b[b.arm.eq('baseline')]
        self.assertTrue(b.groupby('pair_id').size().eq(3).all())
        actual=b.assign(delta=b.margin_b-b.margin_a).groupby('pair_id').delta.mean()
        np.testing.assert_allclose(p.delta_margin,p.pair_id.map(actual))

    def test_diagnostic_oof_once_per_ship(self):
        d=pd.read_csv(OUT/'diagnostic_predictions.csv')
        self.assertFalse(d.duplicated(['inputs','pair_id']).any())
        self.assertTrue(d.groupby('inputs').size().eq(910).all())
        for _,g in d.groupby('inputs'):
            for fold,part in g.groupby('fold'):
                expected=g.loc[g.fold.ne(fold),'actual'].mean()
                np.testing.assert_allclose(part.constant,expected)

    def test_crude_followup_saved_errors(self):
        d=pd.read_csv(OUT/'crude_diagnostic_predictions.csv')
        metrics=pd.read_csv(OUT/'crude_diagnostic_metrics.csv',dtype={'fold':str})
        self.assertEqual(len(d),100)
        self.assertTrue(d.pair_id.is_unique)
        for fold,g in d.groupby('fold'):
            expected=d.loc[d.fold.ne(fold),'delta_margin'].mean()
            np.testing.assert_allclose(g.constant,expected)
            reported=metrics[metrics.fold.eq(str(fold))].iloc[0]
            self.assertAlmostEqual(np.mean((g.predicted-g.delta_margin)**2),reported.mse,places=10)


def main():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Verification)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    p=pd.read_csv(OUT/'pair_factors.csv')
    rows=[]
    for name,q in p.groupby('class_name'):
        for feat in ('vv_contrast','vh_contrast','vv_sin2','vh_sin2','vh_log_length'):
            row=dict(class_name=name,feature=feat,ships=len(q),rho=corr(q[feat],q.delta_margin))
            same=q[q.same_port.eq(1)]
            row.update(same_port_n=len(same),same_port_rho=corr(same[feat],same.delta_margin))
            for fold,g in q.groupby('fold'):
                row[f'fold{fold}_n']=len(g); row[f'fold{fold}_rho']=corr(g[feat],g.delta_margin)
            # Sensitivity to the repeated acquisition shared by multiple ships.
            loo=[]
            for product in set(q.product_a)|set(q.product_b):
                remaining=q[q.product_a.ne(product)&q.product_b.ne(product)]
                if len(remaining)>=20:
                    loo.append(corr(remaining[feat],remaining.delta_margin))
            row['leave_product_out_rho_min']=np.nanmin(loo) if loo else np.nan
            row['leave_product_out_rho_max']=np.nanmax(loo) if loo else np.nan
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'class_relation_checks.csv',index=False)
    b=pd.read_csv(OLD/'pair_predictions.csv')
    b=b[b.arm.eq('baseline')].merge(p[['pair_id','angle_bin','incidence_bin','same_port']],on='pair_id',validate='many_to_one')
    adjusted=[]
    for field in ('angle_bin','incidence_bin','same_port'):
        for value,g in b.groupby(field,observed=True):
            c=g.assign(CC=g.outcome.eq('CC'),WW=g.outcome.eq('WW'),one=g.outcome.isin(['CW','WC']))
            perclass=c.groupby('class_id')[['CC','WW','one']].mean()
            adjusted.append(dict(field=field,group=value,ships=g.pair_id.nunique(),classes=len(perclass),
                macro_both_correct=perclass.CC.mean(),macro_both_wrong=perclass.WW.mean(),
                macro_one_correct=perclass.one.mean()))
    pd.DataFrame(adjusted).to_csv(OUT/'class_standardized_groups.csv',index=False)
    pred=pd.read_csv(OUT/'frozen_predictions.csv.gz')
    confusion=pred.groupby(['arm','class_name','class_id','prediction']).size().rename('seed_events').reset_index()
    confusion['mean_observations']=confusion.seed_events/3
    confusion.to_csv(OUT/'confusions.csv',index=False)
    events=pd.read_csv(OLD/'pair_predictions.csv')
    selected=pd.read_csv(OUT/'selected_cases.csv')
    events=events[events.pair_id.isin(selected.pair_id)&events.arm.isin(['baseline','marginal','joint','joint_condition'])]
    events.to_csv(OUT/'case_predictions_all_seeds.csv',index=False)
    (OUT/'VERIFICATION.json').write_text(json.dumps(dict(tests=5,passed=True,
        checks=['region exclusion','OA/BA against old report','pair/seed mapping','OOF null prediction',
                'posthoc crude diagnostic error replay']),indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).query("class_name == 'crude_oil_tanker'").to_string(index=False))
    print(pd.DataFrame(adjusted).to_string(index=False))


if __name__=='__main__':
    main()
