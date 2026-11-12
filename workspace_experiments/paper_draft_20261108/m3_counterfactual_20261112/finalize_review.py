"""Record the completed native-image review; never alters predictions."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent
reviewed={(29,1294),(43,1245),(17,1303),(17,1323),(43,1483),(43,1162),
          (43,1344),(43,1473),(17,1180),(43,1472),(17,1143),(17,1310)}
cases=json.loads((ROOT/'private/cases.json').read_text(encoding='utf-8'))
assert {(c['seed'],c['index']) for c in cases}==reviewed
for case in cases:
    assert (ROOT/'gallery'/case['panel']).is_file()
    case['review']='native_vv_vh_and_displayed_legal_references_reviewed_20261112'
    case['review_notes']='../FINDINGS_ZH.md'
(ROOT/'private/cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
for filename in ('status.json','analysis_receipt.json'):
    path=ROOT/filename
    obj=json.loads(path.read_text(encoding='utf-8'))
    obj.update(state='complete',analysis_pending=False,visual_review_complete=True,
               formal_model_changed=False,manuscript_changed=False,git_pending=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
print('12 panels reviewed; replay and analysis complete.')
