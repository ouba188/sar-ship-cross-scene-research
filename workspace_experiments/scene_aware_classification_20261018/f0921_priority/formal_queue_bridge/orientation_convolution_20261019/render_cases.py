"""Native-scale VV/VH and actual support cases, with traceable source records."""
import importlib.util
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent


def main():
    spec = importlib.util.spec_from_file_location('native_review', BRIDGE / 'review_cases.py')
    review = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(review)
    meta = pd.read_csv(BRIDGE / 'learned_geometry_crosspass_20261019/cohort.csv')
    geometry = pd.read_csv(HERE / 'observation_geometry.csv').set_index('row')
    cases = pd.read_csv(HERE / 'selected_cases.csv')
    contrib = pd.read_csv(HERE / 'case_support_contributions.csv')
    folder = HERE / 'case_review'
    folder.mkdir(exist_ok=True)
    native = set()
    for case in cases.itertuples():
        part = contrib[contrib.arm.eq(case.arm) & contrib.comparator.eq(case.comparator) & contrib.row.eq(case.row)
                       & contrib.task.eq(case.task) & contrib.seed.eq(case.seed) & contrib.draw.eq(case.draw)]
        part = part.assign(magnitude=part.total.abs()).sort_values('magnitude', ascending=False).head(2)
        ids = list(dict.fromkeys(int(i) for i in [case.row, case.top_same, case.top_wrong]+part.support_row.tolist() if i >= 0))
        panel = Image.new('RGB', (240*len(ids), 600), 'white')
        d = ImageDraw.Draw(panel)
        d.text((5, 5), f'{case.arm} vs {case.comparator}, {case.category}, {case.task}, seed{case.seed}, draw{case.draw}', fill='black')
        d.text((5, 22), f'query {case.row}: {case.old_prediction}->{case.prediction}, truth {case.truth}; delta margin {case.delta_margin:+.6f}', fill='black')
        for j, i in enumerate(ids):
            r, g = meta.iloc[i], geometry.loc[i]
            d.text((j*240+5, 42), f'row{i} class{r.label} axis{g.axis_deg:.1f}', fill='black')
            d.text((j*240+5, 57), f'relative{g.relative_deg:.1f} radar valid:{g.radar_valid}', fill='black')
            item = part[part.support_row.eq(i)]
            if len(item):
                d.text((j*240+5, 72), f'signed delta {item.iloc[0].total:+.6f}', fill='black')
            for pol, col in enumerate(['png_vv', 'png_vh']):
                panel.paste(Image.open(r[col]).convert('RGB'), (j*240+5, 100+pol*240))
            native.add(i)
        panel.save(folder / f'{case.arm}_{case.comparator}_{case.category}_r{case.row}_s{case.seed}_d{case.draw}.png')
    for i in sorted(native):
        review.native_query(meta.iloc[i], folder / f'row{i}_native.png')
    pd.DataFrame([dict(row=i, **meta.iloc[i][['png_vv', 'png_vh', 'source_tif']].to_dict()) for i in sorted(native)]).to_csv(HERE / 'reviewed_case_sources.csv', index=False)
    print('NATIVE ROWS', sorted(native))


if __name__ == '__main__':
    main()
