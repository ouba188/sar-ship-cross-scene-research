"""Record completed native review and check generated local gallery links."""
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parent
cases=json.loads((ROOT/'cases_local.json').read_text(encoding='utf-8'))
for case in cases:
    case['review']='reviewed_native_query_and_actual_legal_references'
(ROOT/'cases_local.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
page=(ROOT/'index.html').read_text(encoding='utf-8')
links=re.findall(r'<img src="([^"]+)"',page)
assert len(links)==8 and all((ROOT/link).is_file() for link in links)
status=json.loads((ROOT/'status.json').read_text(encoding='utf-8'))
status.update(analysis_pending=False,visual_review_complete=True,panels_reviewed=8,
              source_pair_panels_reviewed=8,complete=True,git_pending=True)
(ROOT/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
print('Completed; 8 result panels and local image links verified.')
