"""Record completed human/model visual inspection without changing experiment outputs."""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent


def main():
    path=HERE/'analysis/cases.json'
    cases=json.loads(path.read_text(encoding='utf-8'))
    figures=[]
    for case in cases:
        if case['obs_id'] in (1761,1822):
            case['sampling']='prespecified source-table failure example; local mechanism only'
        figures.append(case['figure'])
        figures.extend(p['figure'] for p in case.get('extra_panels',[]))
    path.write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
    receipt=dict(date='2026-10-25',reviewer='main agent using view_image',
                 reviewed_observations=[c['obs_id'] for c in cases],reviewed_figures=figures,
                 count=9,report='CASE_REVIEW_ZH.md',
                 notes='Native VV/VH and actual true/rival references inspected; third-class harm includes old and new rival panels. No physical-cause naming from appearance.')
    assert len(figures)==9 and all(Path(p).exists() for p in figures)
    (HERE/'CASE_REVIEW_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Recorded actual review of',len(cases),'cases and',len(figures),'panels.')


if __name__=='__main__':main()
