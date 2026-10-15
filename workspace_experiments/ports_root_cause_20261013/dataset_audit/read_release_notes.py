"""Extract one bundled ReadMe per public release and inspect selected metadata."""
from pathlib import Path
import fitz
import numpy as np
import pandas as pd
from audit_local import OUT, NATIVE_ROOTS, read, save, write


def main():
    notes = []
    for name, root in NATIVE_ROOTS.items():
        paths = sorted(root.glob('*/ReadMe.pdf'))
        if not paths:
            notes.append(f'# {name}\nNo bundled ReadMe.pdf found.\n')
            continue
        with fitz.open(paths[0]) as doc:
            body = '\n\n'.join(page.get_text() for page in doc)
        notes.append(f'# {name}\nSource: {paths[0]}\n\n{body}\n')
    (OUT / 'BUNDLED_README_TEXT.md').write_text('\n'.join(notes), encoding='utf-8')
    public = read(OUT / 'public_metadata.csv')
    noise = []
    for name, f in public.groupby('dataset'):
        for keys in [['noise_xml', 'multi_ship_xml', 'port_noise_xml'], ['product_type', 'noise_xml']]:
            for values, n in f.groupby(keys).size().items():
                noise.append(dict(dataset=name, dimension='+'.join(keys), value='|'.join(values), rows=n))
    save('noise_details', noise)
    duplicate = read(OUT / 'oss2_same_product_repeated_identity.csv')
    pairs = []
    for (mmsi, product), f in duplicate.groupby(['identity', 'product']):
        a, b = f.iloc[0], f.iloc[1]
        lon_a, lat_a, lon_b, lat_b = np.radians([float(a.sar_lon), float(a.sar_lat), float(b.sar_lon), float(b.sar_lat)])
        h = np.sin((lat_a-lat_b)/2)**2 + np.cos(lat_a)*np.cos(lat_b)*np.sin((lon_a-lon_b)/2)**2
        pairs.append(dict(identity=mmsi, product=product, sample_a=a.sample_id, sample_b=b.sample_id,
                          same_native_xy=a.center_x == b.center_x and a.center_y == b.center_y,
                          same_ais_epoch=a.ais_epoch == b.ais_epoch,
                          distance_m=6371000*2*np.arcsin(np.sqrt(h))))
    save('oss2_same_product_identity_pairs', pairs)
    print('Bundled release notes and noise/repeated-identity tables written.')


if __name__ == '__main__':
    main()
