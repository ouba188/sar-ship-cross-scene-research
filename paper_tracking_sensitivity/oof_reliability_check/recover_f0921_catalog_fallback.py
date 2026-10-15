"""Record authoritative names and access outcomes for public-mirror gaps."""
from datetime import datetime, timedelta
import json
from recover_f0921_incidence_online import OUT, read_requests, session, product_key, write


def main():
    matches = json.loads((OUT/'catalog_matches.json').read_text())
    records = []
    endpoint = 'https://api.daac.asf.alaska.edu/services/search/param'
    for row in read_requests().itertuples():
        key = (row.sensor, row.acquisition_start, row.product_suffix)
        if '|'.join(key) in matches:
            continue
        start = datetime.strptime(row.acquisition_start, '%Y%m%dT%H%M%S')
        params = dict(platform='Sentinel-1', processingLevel='GRD_HD', output='geojson',
            start=(start-timedelta(seconds=1)).isoformat()+'Z',
            end=(start+timedelta(seconds=1)).isoformat()+'Z')
        response = session().get(endpoint, params=params, timeout=40)
        response.raise_for_status()
        exact = [x['properties'] for x in response.json()['features']
                 if product_key(x['properties']['sceneName']) == key]
        record = dict(key='|'.join(key), target_rows=int(row.target_rows),
                      catalog_url=response.url, catalog_exact_matches=len(exact))
        if len(exact) == 1:
            product = exact[0]['sceneName']
            url = f'https://datapool.asf.alaska.edu/METADATA_GRD_HD/SA/{product}.iso.xml'
            metadata = session().get(url, timeout=30)
            record.update(product=product, metadata_url=url, metadata_http_status=metadata.status_code,
                note='ASF identifies exact product; this endpoint requires authorized access if HTTP401. '
                     'No credentials supplied, no substitute scene or mean incidence used.')
        records.append(record)
    write(OUT/'catalog_fallback.json', records)
    print(records, flush=True)


if __name__ == '__main__':
    main()
