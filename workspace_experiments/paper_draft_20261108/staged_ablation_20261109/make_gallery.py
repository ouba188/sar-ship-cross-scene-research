"""Assemble already-reviewed local panels without changing image pixels."""
from pathlib import Path
import csv
import html
import json

ROOT=Path(__file__).resolve().parent


def main():
    ordinary=json.loads((ROOT/'case_records.json').read_text(encoding='utf-8'))
    followup=json.loads((ROOT/'followup_cases.json').read_text(encoding='utf-8'))
    sections=[]
    for heading,items in [('针对收益组与M3直接路径',followup),('固定清单顺序的完整对照',ordinary)]:
        sections.append(f'<h2>{heading}</h2>')
        for item in items:
            path=ROOT/item['file'];assert path.is_file(),path
            name=item.get('name',item.get('arm','')+' / '+item.get('outcome',''))
            sections.append(f'<section><h3>{html.escape(name)} · {html.escape(item["id"])}</h3>'
                f'<p>真实类分差 {item["margin_before"]:+.3f} → {item["margin_after"]:+.3f}</p>'
                f'<img loading="lazy" src="{html.escape(item["file"],quote=True)}" '
                f'alt="{html.escape(item["id"])} 两极化及实际源参考对照"></section>')
    rows=list(csv.DictReader((ROOT/'evaluation_metrics.csv').open(encoding='utf-8')))
    names={'full':'完整模型','plain':'普通骨干','se':'匹配位置SE','no_m1':'去M1','no_m2':'去M2','no_m3':'去M3'}
    body=[]
    for arm,name in names.items():
        subset=[r for r in rows if r['arm']==arm]
        means=[sum(float(r[k]) for r in subset)/len(subset) for k in ('oa','ba')]
        body.append(f'<tr><td>{name}</td><td>{means[0]:.3f}</td><td>{means[1]:.3f}</td></tr>')
    content='''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OSS2 分阶段消融与原始图像证据</title>
<style>body{font-family:"Microsoft YaHei",sans-serif;max-width:1050px;margin:30px auto;padding:0 18px;color:#20272b;line-height:1.75;background:#fff}h1{font-size:25px}h2{font-size:21px;border-bottom:2px solid #39736b;padding-top:22px}h3{font-size:17px}section{border-bottom:1px solid #d7dce0;padding:12px 0 24px}img{display:block;width:100%;height:auto}table{border-collapse:collapse;width:100%;max-width:600px}td,th{padding:7px 12px;border-bottom:1px solid #cbd3d7;text-align:left}a{color:#24675e}.note{color:#555;font-size:14px}</style>
<h1>OSS2：分阶段配套重训与原图证据</h1>
<p>四分类，评价378条/369艘船/5个过境组，3个训练种子。正式模型和论文正文未替换。</p>
<p><strong>完整组合优于本轮普通骨干和匹配位置SE；M2增量有重复支持。M1整体增量较弱，M3配套消融OA持平。</strong></p>
<p><a href="ROUND_FINDINGS_ZH.md">完整报告</a> · <a href="CASE_NOTES_ZH.md">原图观察记录</a> · <a href="PROTOCOL_ZH.md">实验协议</a></p>
<table><thead><tr><th>配置</th><th>平均OA (%)</th><th>平均BA (%)</th></tr></thead><tbody>'''+''.join(body)+'''</tbody></table>
<p class="note">下图保留原尺度裁剪/补零输入，未为展示重新拉伸。实际源参考仅作说明，不是贡献最大的参考。配套重训比较与固定权重M3开关已分别标记；不可用表示本轮分支状态，不等于原始传感器没有该极化。案例用于解释具体结果，不代替总体净收益统计。</p>'''+''.join(sections)+'</html>'
    (ROOT/'index.html').write_text(content,encoding='utf-8')
    print(f'GALLERY_READY {len(ordinary)+len(followup)} panels; all linked panel files exist')


if __name__=='__main__':main()
