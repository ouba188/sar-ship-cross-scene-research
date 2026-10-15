"""Render the interpretation note with the existing Markdown parser."""

from pathlib import Path

from markdown_it import MarkdownIt


ROOT = Path('E:/Hermes/paper_tracking/sensitivity/latent_clue_cards_20261008')


def main():
    source = ROOT/'LATENT_CLUE_CARDS_ZH.md'
    body = MarkdownIt('commonmark', {'html': False}).enable('table').render(
        source.read_text(encoding='utf-8'))
    html = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>两条隐线索的行为说明书</title>
<style>
body { margin: 0; color: #242a30; background: #fff; font: 16px/1.8 "Microsoft YaHei", sans-serif; }
main { max-width: 1120px; margin: auto; padding: 28px 24px 64px; overflow-wrap: anywhere; }
h1 { font-size: 28px; line-height: 1.4; color: #135b64; }
h2 { margin-top: 36px; padding-bottom: 8px; border-bottom: 1px solid #dce3e5; font-size: 22px; }
img { display: block; max-width: 100%; height: auto; margin: 24px auto; }
table { border-collapse: collapse; width: 100%; font-size: 14px; margin: 20px 0; }
th, td { padding: 10px 12px; border: 1px solid #dce3e5; text-align: left; vertical-align: top; }
th { background: #edf6f5; }
a { color: #087f8c; }
code { font-size: 14px; }
li { margin: 8px 0; }
@media(max-width: 600px) { main { padding: 16px 12px 40px; } table { display: block; overflow-x: auto; } h1 { font-size: 25px; } }
</style></head><body><main>'''+body+'</main></body></html>'
    target = source.with_suffix('.html')
    target.write_text(html, encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
