"""Record the agent's completed inspection of the two rendered gallery pages."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
pages = [root/'gallery_local'/f'intervention_page_{p:02d}_local.png' for p in (1, 2)]
review = dict(status='completed', reviewer='Codex visual inspection via view_image',
              rendered_pages_checked=2,
              page_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in pages],
              observations=[
                  'Both pages show the same two bottom-corner Q patches, clear of the orange protection boundary.',
                  'Central ship structure remains visually unchanged in all eight original/intervened pairs in both polarizations.',
                  'The upper-right bright neighboring return in page 1 row 3 remains unchanged and is outside Q.',
                  'Replacement changes are subtle at the fixed 0..1 scale; numerical checks establish exact unchanged pixels outside Q.',
                  'This is local trusted-Q replacement, not replacement of the whole background or a certification of scene independence.'
              ])
for name in ('intervention_audit_local.json', 'aggregate_intervention_audit.json'):
    path = root/name
    value = json.loads(path.read_text(encoding='utf8'))
    value.pop('human_visual_review', None)
    value['visual_review'] = review
    if name == 'aggregate_intervention_audit.json':
        assert 'pub2_' not in json.dumps(value)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
print('Recorded inspection of both rendered pages; aggregate contains no sample IDs.')
