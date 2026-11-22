"""Compact local progress; deliberately omit private query/donor logs."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
state=json.loads((ROOT/'state_local.json').read_text(encoding='utf8'))
print('status:',state.get('status'),'current:',state.get('current_step'))
for key,step in state['steps'].items():
    if step['status']=='running':
        path=ROOT/'runs'/key.split('/')[0]/(key.split('/')[1]+'.log')
        if not path.exists():continue
        last=None
        for line in path.read_text(encoding='utf8',errors='replace').splitlines():
            try:value=json.loads(line)
            except json.JSONDecodeError:continue
            if 'epoch' in value:last=value
        if last:
            print(json.dumps({k:last[k] for k in ('stage','epoch','selection_score','controlled_samples',
                'regularization_samples','regularization_class_counts','paired_samples') if k in last}))
            print('losses',last.get('losses'))
    elif step['status']=='failed':print('FAILED',key,step)
print('completed',sum(s['status']=='succeeded' for s in state['steps'].values()))
