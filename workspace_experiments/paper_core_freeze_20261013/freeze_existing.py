"""Index existing artifacts and extract matched tables; no fitting or inference."""

import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
SAR=Path('E:/Hermes/paper_tracking/sensitivity')
BASE=SAR/'backbone_ablation_20261010'
MODE='dual_refit_late'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(name, value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    protocol=read(BASE/'PROTOCOL.json')
    tasks=protocol['tasks']
    assert len(tasks)==25
    assets=[]
    for task in tasks:
        result=BASE/'targets'/task
        inputs=SAR/'learned_frequency_residual_20261009/tasks'/task
        # Public task inputs omit the repeated task-name directory.
        if not inputs.exists() and task.split('/')[0] in ('f0921','oss2'):
            inputs=SAR/'learned_frequency_residual_20261009/tasks'/task.split('/')[0]
        paths=dict(predictions=result/'predictions.npz',coefficients=result/'coefficients.npz',
                   completion=result/'COMPLETE.json',prepared=inputs/'PREPARED.json',
                   input_cache=inputs/'target.npz',frequency_checkpoint=inputs/'selective.pt',
                   frequency_embedding=inputs/'selective_embedding.npz')
        for path in paths.values():
            assert path.is_file(),path
        with np.load(paths['predictions'],allow_pickle=False) as z:
            assert MODE in z['modes']
            assert z['support'].shape==(5,64)
            rows=len(z['rows'])
        assets.append(dict(task=task,row_count=rows,**{k:str(v) for k,v in paths.items()}))
    metrics=pd.read_csv(BASE/'metrics.csv')
    selected=metrics[metrics['mode']==MODE]
    assert len(selected)==125
    bytask=selected.groupby(['dataset','task'])[['oa','ba']].mean().reset_index()
    actual=bytask.groupby('dataset')[['oa','ba']].mean().sort_index()
    summary=pd.read_csv(BASE/'summary.csv')
    published=summary[summary['mode']==MODE].set_index('dataset')[['oa','ba']].sort_index()
    np.testing.assert_allclose(actual,published,atol=1e-10,rtol=0)
    summary[summary['mode'].isin([MODE,'r50_late','dual_refit_unified','dual_original_late'])].to_csv(
        OUT/'FROZEN_REFERENCE_RESULTS.csv',index=False)
    effects=pd.read_csv(BASE/'effect_summary.csv')
    effects[(effects.candidate==MODE)&(effects.factor=='r18_increment')].to_csv(
        OUT/'MATCHED_SECOND_BRANCH_GAIN.csv',index=False)
    c4=pd.read_csv(SAR/'fusion_c4_ablation_20261010/effects.csv')
    c4[c4.readout=='fusion'].assign(lineage='dual_original_late, not new post-tail refit').to_csv(
        OUT/'C4_MATCHED_GAIN.csv',index=False)
    joint=pd.read_csv(ROOT/'joint_support_correction_20261013/paired_summary.csv')
    joint[(joint.candidate=='initial')&(joint.reference=='raw')].to_csv(
        OUT/'SUPPORT_CORRECTION_B16_GAIN.csv',index=False)
    write('FROZEN_REFERENCE.json',dict(
        version='paper_reference_20261013',status='frozen_existing_reference',
        created_date='2026-10-13',purpose='Evidence and recipe freeze; not a newly trained or pruned model',
        mode=MODE,output='0.5 * softmax(base) + 0.5 * softmax(expert)',
        classes=8,support_budget=64,draws_per_task=5,tasks=assets,
        source_protocol=str(BASE/'PROTOCOL.json'),
        source_code=str(SAR/'oof_reliability_check/backbone_ablation_run.py'),
        geometry_resolver=str(SAR/'oof_reliability_check/backbone_ablation_core.py'),
        parameters=dict(source_ridge=1.,support_ridge=8.,fusion_weight=.5),
        auxiliary_legacy_components_retained_for_exact_reference=[
            'learned-frequency base embedding and conditional kernel',
            'expert PCA complement',
            'original port-specific stable/drift/incidence transforms'],
        auxiliary_components_are_not_new_core_claims=True,
        default_recent_extensions=[],runtime_code_modified=False,
        metadata_only_index=True,self_contained_model_package=False,
        no_target_best_arm_splicing=True,independent_confirmation=False,
        provenance_limits=['Historical e206 F/domain exposure unchanged',
                           'Source-head port exclusion is not encoder-level unseen-port proof',
                           'Five support draws are not five neural training seeds']
    ))
    write('RESEARCH_SCOPE.json',dict(
        status='frozen',primary_reference='FROZEN_REFERENCE.json',
        core_components=['C4 global encoding','R50 + existing SAR-SSL R18 complementary evidence',
                         'source-anchored class-balanced support residuals','equal full probability fusion'],
        optional_frozen_candidate=dict(
            name='class-supervised support-direction regrouping',arm='initial',
            default_enabled=False,scope='Small-budget study, strongest mean signal F0921 B16',
            evidence=str(ROOT/'joint_support_correction_20261013/PREDICTIONS_FROZEN.json'),
            parameter_policy='Existing per-task initial.npz only; no source-coordinate transplantation',
            budget_sensitivity=[16,64,128],no_inference_dataset_winner_rule=True),
        diagnostics_only=['leave-one-support-vessel sensitivity','signed support contributions',
                          'budget/composition sensitivity','prior/kernel/fusion path attribution'],
        exclude_from_primary_extensions=['raw-vs-initial learned selectors','classwise utility gates',
            'support CV readout selection','count/absent-class guards','task-loss continuation',
            'budget-isolated training','causal-space suppression or distillation',
            'new stable/incidence external insertion','adaptive base/fusion handoff'],
        retire_as_core_claims=['5 percent handoff is necessary','effective dynamic band attention',
            'pure causal feature isolation','universal stable/drift physical separation',
            'two backbones beat every strong single','all gains from one shared physical mechanism'],
        first_architectural_pruning_candidate='frequency branch; existing original-prior ablations only',
        pruning_not_executed=True,
        cannot_assign_frozen_reference_scores_to_pruned_model=True,
        manuscripts_rewritten=False,historical_artifacts_deleted=False,new_experiments=0
    ))
    write('FREEZE_VERIFIED.json',dict(status='complete',tasks=25,draw_units=125,
        indexed_paths=len(assets)*7,registered_mode=MODE,metric_table_max_error=float(abs(actual-published).to_numpy().max()),
        inference_runs=0,training_runs=0,model_code_changes=0,
        check_purpose='Detect wrong model lineage, missing existing assets, or mixed numerical tables'))
    print(actual.to_string())
    print('FROZEN:25 tasks /125 support units; metadata and tables only')


if __name__=='__main__':
    main()
