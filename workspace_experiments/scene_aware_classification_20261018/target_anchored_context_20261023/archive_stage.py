"""Archive completed anchored-reader artifacts, never feature or weight caches."""
import gzip
import os
import shutil
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
ARCHIVE = Path('E:/临时会话/_sar_ship_cross_scene_incremental_git/workspace_experiments/scene_aware_classification_20261018')


def archive(folder, source=SOURCE):
    if not ((folder / 'FINDINGS_ZH.md').exists() or (folder / 'REVIEW_ZH.md').exists()):
        print('pending, not archived:', folder.name)
        return
    count = 0
    for path in sorted(folder.iterdir()):
        if not path.is_file() or (path.suffix not in {'.py', '.md', '.json', '.csv', '.png', '.diff'}
                                  and not path.name.endswith('.csv.gz')
                                  and path.name != 'stages.jsonl'):
            continue
        out = ARCHIVE / source.name / path.relative_to(source)
        if os.name == 'nt':
            out = Path('\\\\?\\' + str(out))
        out.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.csv' and path.stat().st_size > 1_000_000:
            compressed = Path(str(out) + '.gz')
            if compressed.exists():
                with gzip.open(compressed, 'rb') as previous:
                    if previous.read() == path.read_bytes():
                        count += 1
                        continue
            with path.open('rb') as a, gzip.open(str(out) + '.gz', 'wb', compresslevel=6) as b:
                shutil.copyfileobj(a, b)
        else:
            shutil.copy2(path, out)
        count += 1
    print(folder.name, count, 'artifacts')


if __name__ == '__main__':
    archive(SOURCE)
    for name in ['centering_probe', 'partition_replication', 'controller_readout_probe', 'common_gain_diagnosis',
                 'common_gain_training', 'score_scale_decoupling', 'linear_probe_control',
                 'spatial_evidence_review', 'global_calibration_control']:
        folder = SOURCE / name
        if folder.exists():
            archive(folder)
    eight = SOURCE.parent / 'eight_class_evidence_20261023'
    if eight.exists():
        archive(eight, source=eight)
        obb = eight / 'obb_dynamic_readout'
        if (obb / 'FINDINGS_ZH.md').exists():
            for folder in [obb, obb / 'data_review', obb / 'analysis']:
                if not folder.exists():
                    continue
                archive(folder, source=eight)
                for panel_dir in ['panels', 'source_panels', 'test_panels', 'coverage_panels']:
                    for path in (folder / panel_dir).glob('*.png'):
                        out = ARCHIVE / eight.name / path.relative_to(eight)
                        if os.name == 'nt':
                            out = Path('\\\\?\\' + str(out))
                        out.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, out)
        for name in ['baseline', 'gated_transfer', 'gated_transfer_aligned', 'reader_stability', 'spatial_std_transfer']:
            folder = eight / name
            if (folder / 'FINDINGS_ZH.md').exists():
                archive(folder, source=eight)
                review = folder / 'review'
                if review.exists():
                    for path in review.glob('*.png'):
                        out = ARCHIVE / eight.name / path.relative_to(eight)
                        out.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, out)
                norm=folder/'weight_norm_diagnosis'
                if norm.exists():archive(norm,source=eight)
        transfer = eight / 'calibrated_signed_transfer'
        if (transfer / 'FINDINGS_ZH.md').exists():
            archive(transfer, source=eight)
            for stage in ['calibration', 'increment']:
                part = transfer / stage
                # Stage tables are covered by the completed parent report.
                for path in part.iterdir():
                    if path.is_file() and path.suffix in {'.csv', '.json'}:
                        out = ARCHIVE / eight.name / path.relative_to(eight)
                        out.parent.mkdir(parents=True, exist_ok=True)
                        if path.suffix == '.csv' and path.stat().st_size > 1_000_000:
                            compressed = Path(str(out) + '.gz')
                            if compressed.exists():
                                with gzip.open(compressed, 'rb') as previous:
                                    if previous.read() == path.read_bytes():
                                        continue
                            with path.open('rb') as a, gzip.open(compressed, 'wb', compresslevel=6) as b:
                                shutil.copyfileobj(a, b)
                        else:
                            shutil.copy2(path, out)
                for path in (part / 'panels').glob('*.png'):
                    out = ARCHIVE / eight.name / path.relative_to(eight)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, out)
            for child in ['polarization_diagnosis', 'calibration_utility_diagnosis', 'polarization_readout']:
                sub = transfer / child
                if (sub / 'FINDINGS_ZH.md').exists():
                    archive(sub, source=eight)
                    for path in (sub / 'panels').glob('*.png'):
                        out = ARCHIVE / eight.name / path.relative_to(eight)
                        out.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, out)
                    for nested_name in ['standard_controls', 'spatial_cases', 'integration_feasibility',
                                        'free_information_control', 'full_model_integration', 'literature_comparison',
                                        'crude_case_diagnosis', 'score_strength_diagnosis',
                                        'score_matched_moment_diagnosis', 'strong_calibration_increment',
                                        'mean_only_control', 'module_novelty_review', 'mean_local_case_basis',
                                        'joint_distribution_readout']:
                        nested = sub / nested_name
                        if (nested / 'FINDINGS_ZH.md').exists():
                            archive(nested, source=eight)
                            for path in (nested / 'panels').glob('*.png'):
                                out = ARCHIVE / eight.name / path.relative_to(eight)
                                if os.name == 'nt':
                                    out = Path('\\\\?\\' + str(out))
                                out.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(path, out)
                            corrected = nested / 'centered_correction'
                            if (corrected / 'FINDINGS_ZH.md').exists():
                                archive(corrected, source=eight)
                                for path in (corrected / 'panels').glob('*.png'):
                                    out = ARCHIVE / eight.name / path.relative_to(eight)
                                    out.parent.mkdir(parents=True, exist_ok=True)
                                    shutil.copy2(path, out)
                            for diagnostic_name in ['conditional_distribution_diagnosis', 'component_isolation',
                                                    'matched_attention_readout', 'fixed_model_spatial_attribution']:
                                diagnostic = nested / diagnostic_name
                                if (diagnostic / 'FINDINGS_ZH.md').exists():
                                    archive(diagnostic, source=eight)
                                    for panel_dir in ['panels', 'source_panels', 'test_panels']:
                                        for path in (diagnostic / panel_dir).glob('*.png'):
                                            out = ARCHIVE / eight.name / path.relative_to(eight)
                                            if os.name == 'nt':
                                                out = Path('\\\\?\\' + str(out))
                                            out.parent.mkdir(parents=True, exist_ok=True)
                                            shutil.copy2(path, out)
                                    if diagnostic_name == 'matched_attention_readout':
                                        for name in ['sided_ablation', 'conditional_attention_diagnosis',
                                                     'response_attribution']:
                                            child = diagnostic / name
                                            if (child / 'FINDINGS_ZH.md').exists():
                                                archive(child, source=eight)
                                                for path in (child / 'panels').glob('*.png'):
                                                    out = ARCHIVE / eight.name / path.relative_to(eight)
                                                    if os.name == 'nt':
                                                        out = Path('\\\\?\\' + str(out))
                                                    out.parent.mkdir(parents=True, exist_ok=True)
                                                    shutil.copy2(path, out)
    oss2 = SOURCE.parent / 'oss2_distribution_transfer_20261023'
    if (oss2 / 'analysis/FINDINGS_ZH.md').exists():
        for folder in [oss2] + [oss2 / p for p in ['readers', 'analysis', 'geometry_coverage_diagnosis',
                                                  'robust_moment_readout', 'robust_moment_readout/analysis',
                                                  'pair_local_evidence', 'pair_local_evidence/review',
                                                  'pair_local_evidence/antisymmetric_readout',
                                                  'pair_local_evidence/antisymmetric_readout/analysis',
                                                  'reader_substitution', 'native_canvas_probe',
                                                  'next_mechanism_review', 'spatial_topology_probe',
                                                  'spatial_topology_probe/analysis',
                                                  'radiometric_source_inspection',
                                                  'radiometric_source_inspection/radiometric_scale_probe']]:
            archive(folder, source=oss2)
            for path in (folder / 'panels').glob('*.png'):
                out = ARCHIVE / oss2.name / path.relative_to(oss2)
                if os.name == 'nt':
                    out = Path('\\\\?\\' + str(out))
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, out)
    for name in ['contrastive_spatial_pool_20261023', 'contrastive_spatial_pool_20261023_review',
                 'contrastive_spatial_pool_20261023_oldsplit', 'signed_evidence_pool_20261023']:
        folder=SOURCE.parent/name
        if folder.exists():
            archive(folder, source=folder)
            controls=folder/'dispersion_controls'
            if controls.exists():archive(controls, source=folder)
            for child in ['signed_only_retrain', 'source_utility_diagnosis',
                          'source_utility_alignment', 'spatial_contributions', 'literature_design',
                          'old_partition_replication', 'calibrated_increment']:
                sub=folder/child
                if sub.exists():
                    archive(sub, source=folder)
                    matched=sub/'matched_controls'
                    if matched.exists():
                        archive(matched, source=folder)
                        diagnosis = matched / 'score_matched_diagnosis'
                        if diagnosis.exists():archive(diagnosis, source=folder)
                    constant = sub / 'constant_moments_control'
                    if (constant / 'FINDINGS_ZH.md').exists():
                        archive(constant, source=folder)
                        for part in ['new', 'old']:
                            for path in (constant / part).iterdir():
                                if path.suffix in {'.png', '.csv', '.json'}:
                                    out = ARCHIVE / folder.name / path.relative_to(folder)
                                    out.parent.mkdir(parents=True, exist_ok=True)
                                    shutil.copy2(path, out)
                    for part in ['new', 'old']:
                        images=sub/part
                        if images.exists() and (sub/'FINDINGS_ZH.md').exists():
                            for path in images.iterdir():
                                if path.suffix in {'.png','.csv','.json'}:
                                    out=ARCHIVE/folder.name/path.relative_to(folder)
                                    out.parent.mkdir(parents=True,exist_ok=True)
                                    shutil.copy2(path,out)
