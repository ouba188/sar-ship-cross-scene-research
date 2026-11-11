"""Check independent dimensions and incremental audit merging without torch."""
from pathlib import Path
import ast
import csv
import json
import math
import sys
import tempfile

sys.dont_write_bytecode=True
from obb_geometry import conditions, perturb
from experiment import preparation_plan, save_condition_audit

ROOT=Path(__file__).resolve().parent


def main():
    configs=conditions()
    assert len(configs)==31 and len({v['name'] for v in configs})==31
    original=[80.,79.,22.,8.,.73,.6]
    new=[v for v in configs if v['family'] in ('length','width')]
    assert len(new)==8
    for cfg in new:
        result=perturb(original,cfg)
        varied=2 if cfg['family']=='length' else 3
        fixed=3 if varied==2 else 2
        assert result[:2]==original[:2] and result[4:]==original[4:]
        assert result[fixed]==original[fixed]
        assert math.isclose(result[varied]-original[varied],original[varied]*cfg['value'],abs_tol=1e-12)
    assert original==[80.,79.,22.,8.,.73,.6]
    plan=preparation_plan()
    old_names={v['name'] for v in configs if v['family'] not in ('length','width')}
    assert old_names<=set(plan['reuse']),plan
    assert {v['name'] for v in plan['pending']}=={v['name'] for v in new}-set(plan['reuse']),plan
    old=[dict(condition='zero',index='0',value='1.00000000'),
         dict(condition='along_p05',index='0',value='0.12345678901234567')]
    with tempfile.TemporaryDirectory(prefix='_audit_check_',dir=ROOT) as folder:
        root=Path(folder)
        assert root.parent==ROOT
        save_condition_audit(old,root)
        added=[dict(condition='length_p10',index='0',value='0.875')]
        save_condition_audit(added,root)
        with (root/'mask_audit.csv').open(newline='',encoding='utf-8') as f:
            merged=list(csv.DictReader(f))
        assert merged==old+added
        # Retrying one unfinished condition must not duplicate it or touch prior fields.
        changed=[dict(condition='length_p10',index='0',value='0.900')]
        save_condition_audit(changed,root)
        with (root/'mask_audit.csv').open(newline='',encoding='utf-8') as f:
            merged=list(csv.DictReader(f))
        assert merged==old+changed
        assert not (root/'mask_audit.pending.csv').exists()
    for path in ROOT.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    result=dict(status='passed',condition_count=31,reused_geometry_conditions=len(plan['reuse']),
        pending_geometry_conditions=[v['name'] for v in plan['pending']],
        expected_audit_rows_per_condition=plan['expected_audit_rows_per_condition'],
        independent_dimension_invariants_passed=True,audit_merge_and_retry_passed=True,
        existing_field_strings_preserved=True,torch_imported='torch' in sys.modules,
        new_geometry_generated=False,model_inference_started=False)
    (ROOT/'extension_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
