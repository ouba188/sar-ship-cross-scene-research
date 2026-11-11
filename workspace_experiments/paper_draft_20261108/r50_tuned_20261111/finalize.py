"""Record the completed human-readable native-chip review without new inference."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'analysis'
REVIEWED = {
    'Full_rescued_1146_s17.png', 'Full_rescued_1337_s17.png',
    'Full_harmed_1341_s17.png', 'Full_harmed_1371_s17.png',
    'Full_stable_correct_1145_s17.png', 'Full_stable_correct_1432_s17.png',
    'Full_stable_wrong_1159_s17.png', 'Full_stable_wrong_1167_s17.png',
    'Plain_harmed_1242_s17.png', 'Plain_stable_wrong_1163_s17.png',
}


def update_json(path, changes):
    obj = json.loads(path.read_text(encoding='utf-8'))
    obj.update(changes)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


cases_path = OUT / 'cases_local.json'
cases = json.loads(cases_path.read_text(encoding='utf-8'))
for case in cases:
    if Path(case['file']).name in REVIEWED:
        case['review_status'] = 'main_agent_viewed_native_vv_vh; findings in ../FINDINGS_ZH.md'
cases_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
update_json(OUT / 'analysis_receipt.json', dict(status='complete',
    native_panels_reviewed=len(REVIEWED), native_panels_generated=len(cases),
    reviewed_panels=sorted(REVIEWED), findings='../FINDINGS_ZH.md'))
update_json(ROOT / 'status.json', dict(analysis_pending=False, native_review_complete=True,
    state='complete', manuscript_changed=False, formal_model_changed=False))
report = OUT / 'RESULTS.md'
text = report.read_text(encoding='utf-8')
text = text.replace('native panels await main-agent visual review.',
    '10 of 16 native panels visually reviewed; see ../FINDINGS_ZH.md.')
text = text.replace('Main agent must inspect raw VV/VH, Full references, and remaining failures before final conclusions.',
    'Main agent inspected 10 native panels, including all eight Full panels and their legal source references.')
report.write_text(text, encoding='utf-8')
page = OUT / 'index.html'
text = page.read_text(encoding='utf-8')
text = text.replace('<h1>R50-Tuned native cases: pending visual review</h1>',
    '<h1>R50-Tuned native cases</h1><p>10 of 16 panels reviewed; findings: ../FINDINGS_ZH.md.</p>')
text = text.replace('pending_main_agent_visual_review',
    'See cases_local.json and ../FINDINGS_ZH.md for completed review coverage')
page.write_text(text, encoding='utf-8')
print('Finalized: training, paired analysis and 10 native panel reviews complete.')
