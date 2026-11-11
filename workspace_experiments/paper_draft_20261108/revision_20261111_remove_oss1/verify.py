"""Check scope removal, numbering and unchanged formulas after rebuilding."""
import json
import re
from pathlib import Path
import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
source = (ROOT / 'MANUSCRIPT_ZH.md').read_text(encoding='utf-8')
before = (HERE / 'before/MANUSCRIPT_ZH.md').read_text(encoding='utf-8')
tables = [int(n) for n in re.findall(r'^\*\*表(\d+) ', source, re.M)]
assert tables == list(range(1, 24)), tables
equations = re.findall(r'\$\$(.*?)\$\$', source, re.S)
assert equations == re.findall(r'\$\$(.*?)\$\$', before, re.S)
assert re.findall(r'\\tag\{(\d+)\}', source) == [str(n) for n in range(1, 19)]
figures = re.findall(r'^!\[[^\n]+\]\(([^)]+)\)', source, re.M)
assert figures == re.findall(r'^!\[[^\n]+\]\(([^)]+)\)', before, re.M)
pattern = re.compile(r'OSS1|OpenSARShip\s*1\.0', re.I)
for name in ('MANUSCRIPT_ZH.md', 'DATASET_PROFILE_ZH.md', 'index.html', 'manuscript_ieee.tex'):
    assert not pattern.search((ROOT / name).read_text(encoding='utf-8')), name
document = fitz.open(ROOT / 'manuscript_ieee.pdf')
texts = [p.get_text() for p in document]
assert not pattern.search('\n'.join(texts)), 'PDF still includes OSS1'
keys = ('数据集组成', '预处理对照', '跨场景分类比较', '组件消融',
        '场景级增量分布', '参数与解冻配方', '完整模型的逐类召回')
locations = {key: [i + 1 for i, t in enumerate(texts) if key in t] for key in keys}
assert all(locations.values()), locations
log = (ROOT / '_build_ieee/manuscript_ieee.log').read_text(encoding='utf-8', errors='replace')
assert 'undefined references' not in log.lower()
receipt = dict(pages=len(document), tables=len(tables), numbered_equations=18,
    formulas_unchanged=True, embedded_figure_paths_unchanged=True,
    oss1_absent_from_active_manuscript=True, changed_page_locations=locations,
    overfull_warnings=log.count('Overfull'), missing_characters=log.count('Missing character'),
    visual_review='pending', formal_model_unchanged=True, new_training=False)
(HERE / 'verification.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(receipt, ensure_ascii=False))
