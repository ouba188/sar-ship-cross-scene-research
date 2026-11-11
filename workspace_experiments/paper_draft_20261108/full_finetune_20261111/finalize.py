"""Record completed visual review and append the human findings link."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
path = ROOT/'analysis/cases_local.json'
cases = json.loads(path.read_text(encoding='utf-8'))
for case in cases:
    case['review_status'] = 'reviewed_native_vv_vh_and_displayed_source_references_20261112'
    case['review_notes'] = '../FINDINGS_ZH.md'
path.write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
index = ROOT/'analysis/index.html'
text = index.read_text(encoding='utf-8')
text = text.replace('Full-Tuned native chip cases: visual review pending',
                    'Full-Tuned native chip cases: reviewed 2026-11-12')
text = text.replace('pending_main_agent_visual_review',
                    'reviewed_native_vv_vh_and_displayed_source_references_20261112')
index.write_text(text,encoding='utf-8')
report = ROOT/'analysis/RESULTS.md'
text = report.read_text(encoding='utf-8').replace(
    'Computed from completed runs. Native panels still require main-agent visual review.',
    'Computed from completed runs. All 12 panels reviewed on 2026-11-12; see ../FINDINGS_ZH.md.')
text = text.replace('Main agent must inspect raw VV/VH and source references before claiming image-based explanations.',
                    'Native VV/VH and displayed legal source references have been inspected; interpretations are in ../FINDINGS_ZH.md.')
report.write_text(text,encoding='utf-8')
status = json.loads((ROOT/'status.json').read_text())
status.update(analysis_pending=False,visual_review_complete=True,formal_model_replaced=False,
              manuscript_changed=False,dataset='OSS2 four-class',completed_date='2026-11-12',git_pending=True)
(ROOT/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
print('Finalized 12 panel reviews; formal model unchanged.')
