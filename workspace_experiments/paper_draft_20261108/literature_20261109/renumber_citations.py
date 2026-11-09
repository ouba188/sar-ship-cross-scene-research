"""Keep numeric manuscript citations ordered by first occurrence."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "MANUSCRIPT_ZH.md"
text = path.read_text(encoding="utf-8")
body, bibliography = text.split("# 参考文献", 1)
entries = dict(re.findall(r"^\[(\d+)\]\s+(.+)$", bibliography, re.M))
pattern = re.compile(r"(?<!layer1)(?<!layer2)(?<!layer3)(?<!layer4)\[(\d+(?:,\d+)*)\]")
order = []
for match in pattern.finditer(body):
    for key in match.group(1).split(","):
        if key not in entries:
            raise ValueError(f"Missing bibliography entry: {key}")
        if key not in order:
            order.append(key)
if set(order) != set(entries):
    raise ValueError(f"Uncited entries: {set(entries) - set(order)}")
mapping = {old: str(i + 1) for i, old in enumerate(order)}
body = pattern.sub(lambda m: "[" + ",".join(mapping[k] for k in m.group(1).split(",")) + "]", body)
rows = [f"[{mapping[old]}] {entries[old]}" for old in order]
path.write_text(body + "# 参考文献\n\n" + "\n\n".join(rows) + "\n", encoding="utf-8")
note = "# 当前参考文献编号\n\n2026-11-09近期文献补充后，按正文首次引用顺序编号。\n\n"
note += "\n\n".join(rows) + "\n"
(ROOT / "literature_20261109/CITATION_NUMBER_MAP_ZH.md").write_text(note, encoding="utf-8")
print(f"Validated and renumbered {len(entries)} cited references.")
print(mapping)
