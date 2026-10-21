"""Read-only SSH provenance checks using an existing key; no remote files written."""
import argparse
import ast
import json
import logging
from pathlib import Path

import paramiko

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def servers():
    tree = ast.parse((ROOT / 'pull_v5_context_dynamic_to_local.py').read_text(encoding='utf-8-sig'))
    original = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'SERVERS' for t in n.targets))
    later = []
    for p in sorted((ROOT/'safe841_redownload_plan_20260918').glob('*deployment.json')):
        data = json.loads(p.read_text(encoding='utf-8-sig'))
        if 'host' in data and 'port' in data:
            later.append(('later_'+p.stem,data['host'],int(data['port'])))
    return list(original) + later


def connect(host, port):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username='root',
        key_filename=str(Path.home() / '.ssh/codex_remote_zip_validation_ed25519'),
        look_for_keys=False, allow_agent=False, timeout=10, banner_timeout=10, auth_timeout=10)
    return client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['probe','probe-later'])
    args = ap.parse_args()
    logging.getLogger('paramiko').addHandler(logging.NullHandler())
    logging.getLogger('paramiko').propagate = False
    results = json.loads((OUT/'remote_connection_status.json').read_text()) if args.stage == 'probe-later' else []
    for name, host, port in servers():
        if args.stage == 'probe-later' and not name.startswith('later_'):
            continue
        client = None
        row = dict(server=name, host=host, port=port)
        try:
            client = connect(host, port)
            _, stdout, stderr = client.exec_command("/usr/bin/python3 -c 'from osgeo import gdal; print(gdal.VersionInfo())'", timeout=15)
            row.update(status='connected', gdal_version=stdout.read().decode().strip(), error=stderr.read().decode().strip())
        except Exception as exc:
            row.update(status='unavailable', error=type(exc).__name__ + ': ' + str(exc))
        finally:
            if client is not None:
                client.close()
        results.append(row)
        print(name, row['status'], row.get('error',''), flush=True)
    (OUT/'remote_connection_status.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
