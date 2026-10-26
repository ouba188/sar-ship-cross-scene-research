"""Export small flat tables and mark only actually reviewed panel groups."""
from pathlib import Path
import json
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    pub = pd.read_csv(HERE/'stage1_analysis/main_table.csv',header=[0,1],index_col=[0,1,2])
    pub.columns = ['_'.join(x) for x in pub.columns]
    pub.reset_index().to_csv(HERE/'PUBLIC_MAIN_TABLE.csv',index=False)
    for dirname in ('reference_sensitivity','m3_interventions'):
        f = pd.concat([pd.read_csv(HERE/dirname/s/'metrics.csv') for s in ('oss1','oss2')])
        keys = ['source','cohort','arm']
        cols = ['OA','BA','OA_delta_pp','BA_delta_pp']
        f.groupby(keys)[cols].agg(['mean','std','count']).to_csv(HERE/dirname/'summary.csv')
    for folder, reports in (
        ('stage1_analysis',['CROSS_CASE_REVIEW_ZH.md','WITHIN_CASE_REVIEW_ZH.md']),
        ('regional_24port/analysis',['CASE_REVIEW_ZH.md'])):
        root = HERE/folder
        if not all((root/report).exists() for report in reports):
            raise FileNotFoundError(f'Missing actual review report: {root}')
        receipt = json.loads((root/'receipt.json').read_text(encoding='utf-8'))
        receipt.update(raw_visual_review_pending=False,review_reports=reports)
        (root/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Flat public table, reference/intervention summaries and reviewed receipts written.')


if __name__ == '__main__':
    main()
