"""Compile the user-only report and original-chip plates into one offline HTML."""
import base64
import json
from pathlib import Path
import markdown

ROOT=Path(__file__).resolve().parent

def main():
    findings=(ROOT/'FINDINGS_ZH.md').read_text(encoding='utf-8')
    notes=(ROOT/'RAW_REVIEW_LOCAL.md').read_text(encoding='utf-8')
    def picture(name,caption):
        data=base64.b64encode((ROOT/name).read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{data}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'
    body=markdown.markdown(findings,extensions=['tables','fenced_code'])
    body+=picture('evidence_summary.png','三个层面的证据与扩展边界。图像指标基于原始显示响应，不是分类贡献。')
    body+=picture('port_size_ecdf.png','主队列选定水域对的按船只去重长度分布。')
    body+='<hr>'+markdown.markdown(notes,extensions=['tables'])
    captions={
        'container_ship_medians':'集装箱：靠近两水域中位尺寸的真实船舶',
        'container_ship_size_matched':'集装箱：不同船、同尺寸对照',
        'bulk_carrier_medians':'散货船：两水域总体代表',
        'bulk_carrier_size_matched':'散货船：不同船、同尺寸对照',
        'general_cargo_medians':'普通货船：两水域总体代表',
        'general_cargo_size_matched':'普通货船：不同船、同尺寸对照',
        'product_chemical_tanker_medians':'油化船：两水域总体代表',
        'product_chemical_tanker_size_matched':'油化船：不同船、同尺寸对照',
        'same_vessel_cross_area':'同一AIS身份的跨水域观测',
        'container_ship_baseline_refs':'集装箱：判对、判错与实际参考图',
        'bulk_carrier_baseline_refs':'散货船：判对、判错与实际参考图',
        'general_cargo_baseline_refs':'普通货船：判对、判错与实际参考图',
        'product_chemical_tanker_baseline_refs':'油化船：判对、判错与实际参考图'}
    gallery=json.loads((ROOT/'gallery_local.json').read_text(encoding='utf-8'))
    body+='<h2>全部13组原图图版</h2><p>可在浏览器中放大查看。上行为VV，下行为VH；未缩放原始数据，统一画布坐标，灰色区域不属于海面。raw only表示该图可读取，但当前模型没有使用该极化分支。</p>'
    for g in gallery:body+=picture(g['name']+'_local.png',captions[g['name']])
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>OSS2真实原图研究报告</title><style>
    body{margin:0;background:#f4f6f6;color:#202a2d;font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;line-height:1.8}main{max-width:1160px;margin:32px auto;background:white;padding:40px 52px}h1{font-size:29px;line-height:1.4}h2{margin-top:38px;border-bottom:2px solid #91b8bc;padding-bottom:8px;font-size:21px}p,li{font-size:15px}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}th,td{padding:10px 12px;border-bottom:1px solid #d9e1e2;text-align:left}th{background:#ecf3f3}blockquote{border-left:4px solid #789fa4;padding:8px 20px;background:#f2f6f6}figure{margin:28px 0;padding:10px;border:1px solid #dae3e3}img{width:100%;height:auto}figcaption{font-size:14px;color:#4f6265;text-align:center;padding:10px}code{font-size:13px;background:#f1f3f3;padding:2px 4px}hr{border:0;border-top:2px solid #91b8bc;margin-top:45px}@media(max-width:800px){main{margin:0;padding:20px}table{font-size:12px}th,td{padding:6px}}@media print{main{margin:0;padding:10px}figure,table{break-inside:avoid}body{background:white}}
    </style><main>'''+body+'</main></html>'
    out=ROOT/'OSS2_原图研究报告.html';out.write_text(html,encoding='utf-8')
    assert html.count('<figure>')==15 and html.count('data:image/png;base64,')==15
    print(out);print('bytes',out.stat().st_size,'embedded figures',html.count('<figure>'))

if __name__=='__main__':main()
