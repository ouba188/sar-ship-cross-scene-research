"""Explicit text-only allowlist; default preview, never commit/push or replace ZIP.

Model/config source trees and the private experiment remain read-only. --apply
copies only the listed public files into their existing repository locations.
An optional --zip NEW_PATH creates a fresh portable code/docs archive (mode x).
"""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile


WORK = Path(__file__).resolve().parent.parent
CODE = WORK/'port_prototype_model_20261121'
REPO = Path('E:/临时会话/_sar_ship_cross_scene_incremental_git')
SAMPLE_ID = re.compile(r'pub2_opensarship2_\d+', re.IGNORECASE)
TEXT_SUFFIXES = {'.py', '.md', '.json', '.csv', '.toml', '.txt'}
PRIVATE_JSON_KEYS = {'id','sample_id','sample_ids','vessel_id','vessel_ids','mmsi',
                    'donor_id','donor_ids','query_id','query_ids','reference_ids',
                    'source_ids','predictions','destination_flat_indices',
                    'donor_source_flat_indices_shared_by_vv_vh'}

# Every file is explicit; directory scans never decide what gets copied.
MODEL_MODULES = '''__init__ backbone cli data disentangle exchange geometry matching
model priors protocol prototypes region_ops selection supervision teacher'''.split()
TWO_STAGE_MODULES = '''__init__ bank cache cli contracts data geometry h stage1
stage1_losses training'''.split()
TEST_MODULES = '''test_cli_contract test_disentangle test_geometry_exact_coverage
test_matching test_prototypes test_supervision test_teacher test_two_stage_bank
test_two_stage_cache test_two_stage_comparison test_two_stage_data test_two_stage_empirical
test_two_stage_encoder test_two_stage_h test_two_stage_h_batch test_two_stage_pipeline'''.split()
MODEL_FILES = (
    ['pyproject.toml','README_ZH.md','REFERENCE_NOTES.md','TWO_STAGE_README_ZH.md',
     'TWO_STAGE_VERIFICATION_ZH.md','VERIFICATION_ZH.md']
    + [f'sar_port/{name}.py' for name in MODEL_MODULES]
    + [f'sar_port/two_stage/{name}.py' for name in TWO_STAGE_MODULES]
    + [f'tests/{name}.py' for name in TEST_MODULES]
    + [f'scripts/{name}.py' for name in ('check_cli','check_model','check_native','make_ablation_configs')])
# These are source-code distribution templates, not the real OSS2 seed configs.
TEMPLATE_CONFIGS = ['configs/two_stage_native.json','configs/baseline.json',
                    'configs/two_layer.json','configs/full.json','configs/bootstrap.json']
ABLATION_TEMPLATES = '''aggregate cross_class_donor cross_layer fixed_coarse fixed_fine
frequency_effect gate_ce_only hard_selection hierarchical_ports learned_fusion no_axis
no_class_competition no_dustbin no_effect no_gate no_guidance no_m1 no_m2 no_m3 no_pcea
no_prior no_rejection_gate no_size_prior no_third_layer same_layer'''.split()
TEMPLATE_CONFIGS += [f'configs/ablations/{name}.json' for name in ABLATION_TEMPLATES]
EXPERIMENT_REQUIRED = '''EXPERIMENT_PROTOCOL_ZH.md CONTROLLED_QUALIFICATION_REVIEW_ZH.md
DATA_COMPOSITION_ZH.md prepare_experiment.py run_comparison.py progress.py
download_official_weights.py smoke_real_supervision.py make_intervention_gallery.py
record_gallery_review.py summarize_data_composition.py collect_results.py collect_subgroups.py
branch_diagnostics.py verify_finished_comparison.py review_cases.py archive_public_experiment.py run_post_training_diagnostics.py
PUBLIC_ARCHIVE_ALLOWLIST_ZH.md aggregate_data_composition.json aggregate_intervention_audit.json'''.split()
EXPERIMENT_OPTIONAL = '''FINDINGS_ZH.md
aggregate_training_integrity.json aggregate_branch_diagnostics.json aggregate_h_margin_diagnostics.json
aggregate_validation_subgroups.json aggregate_test_subgroups.json
collected/validation/aggregate_summary.json collected/validation/aggregate_summary.csv
collected/test/aggregate_summary.json collected/test/aggregate_summary.csv
real_smoke_summary.json real_smoke_original_loop.json'''.split()
TUNING_REQUIRED = ['TUNING_PLAN_ZH.md','scan_validation.py','test_scan_validation.py','collect_fusion_scan.py',
                   'evaluate_selected_gamma.py']
TUNING_OPTIONAL = ['FINDINGS_ZH.md','aggregate_fusion_scan_summary.json',
                   'fusion_candidate_table.csv','selected_gamma.json']
TUNING_OPTIONAL += [f'final_seed{seed}/aggregate_validation_fusion_scan.json' for seed in (17,29,43)]
DISENTANGLE_REQUIRED = ['TUNING_PLAN_ZH.md','STAGE1_HANDOFF_DIAGNOSIS_ZH.md',
    'aggregate_stage1_handoff_diagnosis.json','run_aux_scale_ablation.py',
    'test_aux_scale_ablation.py','analyze_head_reuse.py']
DISENTANGLE_OPTIONAL = ['test_head_reuse.py','aggregate_aux_scale_validation.json',
    'AUX_SCALE_RESULTS_ZH.md','aggregate_head_reuse_validation.json']
DESIGNS = ['STAGE1_MULTISCALE_DISENTANGLEMENT_DESIGN_ZH.md',
           'STAGE2_H_MULTIPORT_EVIDENCE_DESIGN_ZH.md']
WORKSPACE_PUBLIC = ['run_on_d.py', 'STORAGE_LAYOUT_ZH.md']
PREFLIGHT_KEYS = '''input_records retained split_counts split_class_counts canvas_counts
reviewed_q_observations independent_donors empirical_queries supplemental_query_exposures_per_epoch
strong_controlled_samples pure_q_cells_by_scale sensor_nodata_masks_recovered
availability_convention evaluation_scope seeds stage1_epochs base_epochs h_epochs
selection_metric test_selection'''.split()


@dataclass(frozen=True)
class Entry:
    source: Path
    destination: str
    required: bool = True
    projection: bool = False
    template: bool = False


def entries(work=WORK, code=CODE):
    result = []
    result += [Entry(work/name, 'workspace_experiments/'+name) for name in WORKSPACE_PUBLIC]
    model_dest = 'workspace_experiments/port_prototype_model_20261121/'
    result += [Entry(code/name, model_dest+name) for name in MODEL_FILES]
    result += [Entry(code/name, model_dest+name, template=True) for name in TEMPLATE_CONFIGS]
    for folder, required, optional in (
        ('oss2_three_arm_experiment_20261122',EXPERIMENT_REQUIRED,EXPERIMENT_OPTIONAL),
        ('oss2_h_tuning_20261122',TUNING_REQUIRED,TUNING_OPTIONAL),
        ('oss2_disentangle_tuning_20261122',DISENTANGLE_REQUIRED,DISENTANGLE_OPTIONAL)):
        for names, mandatory in ((required,True),(optional,False)):
            result += [Entry(work/folder/name,f'workspace_experiments/{folder}/{name}',mandatory) for name in names]
    result += [Entry(work/name,'workspace_experiments/two_stage_minimal_20261122/'+name) for name in DESIGNS]
    result.append(Entry(work/'oss2_three_arm_experiment_20261122/preflight_summary.json',
                        'workspace_experiments/oss2_three_arm_experiment_20261122/preflight_summary_public.json',
                        projection=True))
    return result


def check_json(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if key.lower() in PRIVATE_JSON_KEYS:
                raise ValueError(f'Private per-record JSON field: {key}')
            check_json(item)
    elif isinstance(value,list):
        for item in value:
            check_json(item)
    elif isinstance(value,str):
        if re.search(r'(?:[A-Za-z]:[\\/]|^/|^\\\\).*\.(?:pt|pth|ckpt)(?:\b|$)',value,re.IGNORECASE):
            raise ValueError('Public JSON contains an absolute checkpoint path; retain hashes instead')


def public_bytes(entry):
    path = Path(entry.destination)
    if path.is_absolute() or '..' in path.parts or any('_local' in part.lower() for part in path.parts):
        raise ValueError(f'Unsafe/private destination: {entry.destination}')
    if path.suffix not in TEXT_SUFFIXES:
        raise ValueError(f'Non-text output not allowed: {entry.destination}')
    if 'configs' in path.parts and not entry.template:
        raise ValueError('Real experiment configs are not public outputs')
    text = entry.source.read_text(encoding='utf-8-sig')
    if entry.projection:
        source = json.loads(text)
        # The original excluded list contains real sample IDs: never copy it.
        public = {key:source[key] for key in PREFLIGHT_KEYS}
        public['excluded_count'] = len(source['excluded'])
        reasons = {}
        for row in source['excluded']:
            reasons[row['reason']] = reasons.get(row['reason'],0)+1
        public['excluded_reason_counts'] = reasons
        text = json.dumps(public,ensure_ascii=False,indent=2)+'\n'
    if SAMPLE_ID.search(text):
        raise ValueError(f'Actual sample identifier found in {entry.destination}')
    if path.suffix == '.json':
        check_json(json.loads(text))
    return text.replace('\r\n','\n').replace('\r','\n').encode('utf8')


def repo_destination(repo, relative):
    root = repo.resolve()
    target = (root/relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f'Destination escapes repository: {relative}')
    return target


def destination_same(target, payload):
    """Preserve existing CRLF bytes when public text has not actually changed."""
    return target.is_file() and target.read_bytes().replace(b'\r\n', b'\n') == payload


def dirty_repo_paths(repo):
    result = subprocess.run(['git','-c','core.longpaths=true','-C',str(repo),'status','--porcelain=v1','-z','--untracked-files=all'],
                             check=True,capture_output=True)
    chunks = result.stdout.decode('utf8').split('\0')
    dirty = set()
    i = 0
    while i < len(chunks):
        item = chunks[i]
        if item:
            status, name = item[:2], item[3:]
            dirty.add(name.replace('\\','/'))
            if 'R' in status or 'C' in status:
                i += 1
                if i < len(chunks): dirty.add(chunks[i].replace('\\','/'))
        i += 1
    return dirty


def make_plan(work=WORK, code=CODE, repo=REPO):
    payloads, report = {}, dict(ready=[],missing_optional=[],missing_required=[],blocked=[])
    for entry in entries(work,code):
        if not entry.source.is_file():
            report['missing_required' if entry.required else 'missing_optional'].append(entry.destination)
            continue
        try:
            payload = public_bytes(entry)
            target = repo_destination(repo,entry.destination)
            if entry.destination in payloads: raise ValueError('Duplicate allowlist destination')
            payloads[entry.destination] = payload
            report['ready'].append(dict(path=entry.destination,bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),source_projection=entry.projection,
                destination_exists=target.exists(),destination_same=destination_same(target,payload)))
        except (ValueError,OSError,UnicodeError) as exc:
            report['blocked'].append(dict(path=entry.destination,reason=str(exc)))
    return payloads,report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=WORK)
    parser.add_argument('--code-root',type=Path,default=CODE)
    parser.add_argument('--repo',type=Path,default=REPO)
    parser.add_argument('--apply',action='store_true',help='Copy reviewed allowlist; default is read-only preview')
    parser.add_argument('--zip',type=Path,help='Optional NEW code/docs ZIP; never overwrite an existing file')
    args = parser.parse_args()
    payloads,report = make_plan(args.workspace,args.code_root,args.repo)
    report.update(mode='apply' if args.apply else 'preview',zip_requested=str(args.zip) if args.zip else None,
                  exclusions=['real OSS2 configs','manifests','*_local*','raw images','Q masks','model weights',
                              'prepared/feature caches','per-sample predictions','training state/ledger','unlisted files'])
    if args.zip and args.zip.exists():
        report['blocked'].append(dict(path=str(args.zip),reason='ZIP already exists; exclusive creation required'))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if report['blocked'] or report['missing_required']:
        raise SystemExit('Resolve listed blocked/missing-required files; no archive writes performed')
    if not args.apply:
        return
    dirty = dirty_repo_paths(args.repo)
    for name,payload in payloads.items():
        target = repo_destination(args.repo,name)
        if name in dirty and target.exists() and not destination_same(target,payload):
            raise SystemExit(f'Refusing to overwrite pre-existing repository changes: {name}')
    # All paths/content/conflicts are checked before writing. Unlisted files are
    # untouched: no delete, whole-tree copy, Git add/commit, or push is performed.
    receipt = dict(created_at=datetime.now(timezone.utc).isoformat(),
                   files=[dict(path=name,sha256=hashlib.sha256(payload).hexdigest()) for name,payload in payloads.items()],
                   missing_optional=report['missing_optional'],scope=report['exclusions'])
    if args.zip:
        args.zip.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(args.zip,mode='x',compression=zipfile.ZIP_DEFLATED) as archive:
            for name,payload in payloads.items():
                archive.writestr(name,payload)
            archive.writestr('PUBLIC_ARCHIVE_CONTENTS.json',json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    changed = 0
    for name,payload in payloads.items():
        target = repo_destination(args.repo,name)
        if destination_same(target,payload):
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(payload)
        changed += 1
    print(json.dumps(dict(copied_or_updated=changed,unchanged=len(payloads)-changed,
                         optional_missing=len(report['missing_optional']),zip_created=bool(args.zip)),indent=2))


if __name__=='__main__':main()
