"""Matched-arm contracts; all pixels and training steps here are synthetic."""
from argparse import Namespace
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from sar_port.two_stage.cli import (protocol, classification_metrics, selection_score,
    train_stage1, train_base, train_h, evaluate, load_checkpoint)
from sar_port.two_stage.data import prepare_record, collate
from sar_port.two_stage.stage1 import Stage1, Stage1Config
from sar_port.two_stage.training import baseline_step, stage1_optimizers


def make_records(root, controlled=False):
    rows = []
    rng = np.random.default_rng(12)
    Image.fromarray(np.full((64,64),255,np.uint8)).save(root/'q.png')
    for role, port in [('train','a'),('train','b'),('val','c'),('test','d')]:
        for label in ('cargo','tanker'):
            sid=f'{port}_{label}'
            pixels=rng.integers(10,70,(64,64),dtype=np.uint8)
            pixels[28:36,20:44]+=100
            Image.fromarray(pixels).save(root/f'{sid}.png')
            view=dict(image=f'{sid}.png', alignment_valid=True,
                      ship_axis=dict(center=[31.5,31.5],length_px=24,width_px=8,angle=0),
                      background_mask='q.png',background_validated=controlled)
            rows.append(dict(id=sid,label=label,vessel_id=sid,group_id=sid,port=port,
                             development_split=role,vv=dict(view),vh=dict(view),leakage_controlled=controlled))
    return rows


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fixed_split_preserves_roles_and_rejects_identity_leakage(self):
        rows=[dict(id=str(i),label=str(i%2),vessel_id=f'v{i}',group_id=f'g{i}',port='same_port',
                   development_split=role) for i,role in enumerate(['train','train','val','val','test','test',
                                                                    'quarantine_historical_selection'])]
        cfg=dict(protocol='development_split')
        split=protocol(rows,cfg)
        self.assertEqual(split['train'].tolist(),[0,1])
        self.assertEqual(split['validation'].tolist(),[2,3])
        self.assertEqual(split['excluded'].tolist(),[6])
        for key in ('vessel_id','group_id'):
            bad=deepcopy(rows); bad[2][key]=bad[0][key]
            with self.assertRaisesRegex(ValueError,'leaks'):
                protocol(bad,cfg)
        bad=deepcopy(rows); bad[0]['development_split']='selection'
        with self.assertRaisesRegex(ValueError,'Unknown development_split'):
            protocol(bad,cfg)

    def test_metrics_select_balanced_accuracy_without_changing_legacy_default(self):
        metrics=classification_metrics([0]*9+[1],[0]*10,2)
        self.assertAlmostEqual(metrics['accuracy'],.9)
        self.assertAlmostEqual(metrics['balanced_accuracy'],.5)
        self.assertEqual(selection_score(metrics,{}),.9)
        self.assertEqual(selection_score(metrics,dict(selection_metric='balanced_accuracy')),.5)
        with self.assertRaises(ValueError):
            selection_score(metrics,dict(selection_metric='test_accuracy'))

    def test_gate_off_shared_initialization_exact_path_and_single_forward(self):
        torch.manual_seed(93); gated=Stage1(Stage1Config(2))
        reference={k:v.clone() for k,v in gated.state_dict().items()}
        del gated
        torch.manual_seed(93); model=Stage1(Stage1Config(2,gate_enabled=False))
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,reference[key],rtol=0,atol=0)
        del reference
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); rows=make_records(root)
            samples=[prepare_record(r,root) for r in rows[:2]]
            batch=collate(samples)
            batch['valid'][0,1]=0; batch['images'][0,1]=0
            captures={}
            hooks=[]
            for p in range(2):
                for level in range(3):
                    def record(_m,_a,out,p=p,level=level): captures[p,level]=out.detach()
                    hooks.append(model.integrators[p][level].register_forward_hook(record))
            calls=[]; hooks.append(model.register_forward_hook(lambda *_: calls.append(1)))
            output=model(batch['images'],batch['valid'],hard_gate=True)
            for level in range(3):
                for p in range(2):
                    support=(output['valid'][level][:,p]>0)
                    expected=captures[p,level]*support
                    torch.testing.assert_close(output['content'][level][:,p],expected,rtol=0,atol=0)
                self.assertEqual(output['residual'][level].count_nonzero().item(),0)
            calls.clear(); before=model.spatial_readouts[0].project.weight.detach().clone()
            main,disc=stage1_optimizers(model)
            self.assertIsNone(disc)
            active={id(p) for group in main.param_groups for p in group['params']}
            self.assertTrue(all(id(p) not in active for p in model.gates.parameters()))
            with patch('sar_port.two_stage.training.intervene_background',side_effect=AssertionError('must not run')):
                loss=baseline_step(model,batch,torch.tensor([0,1]),main)
            self.assertEqual(len(calls),1)
            self.assertGreater(loss['cls'],0)
            self.assertFalse(torch.equal(before,model.spatial_readouts[0].project.weight))
            self.assertTrue(all(p.grad is None for p in model.gates.parameters()))
            for hook in hooks: hook.remove()

    def test_complete_supervision_still_rejects_uncontrolled_data_but_baseline_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); rows=make_records(root,controlled=False)
            manifest=root/'manifest.json'; manifest.write_text(json.dumps(rows),encoding='utf8')
            cfg=dict(protocol='development_split',stage1_epochs=1,batch_size=4,
                     selection_metric='balanced_accuracy',allow_incomplete_supervision=False)
            split=protocol(rows,cfg); args=Namespace(manifest=manifest,output=root/'s1.pt')
            with self.assertRaisesRegex(ValueError,'audited leakage_controlled'):
                train_stage1(args,cfg,rows,split,'cpu')
            self.assertFalse(args.output.exists())
            cfg['encoder']=dict(gate_enabled=False)
            with contextlib.redirect_stdout(io.StringIO()): train_stage1(args,cfg,rows,split,'cpu')
            saved=load_checkpoint(args.output)
            self.assertEqual(saved['training_objective'],'classification_baseline')
            self.assertFalse(saved['incomplete_supervision'])
            self.assertEqual(saved['history'][0]['losses']['paired_samples'],0)

    def test_base_checkpoint_evaluates_and_h_continuation_preserves_encoder_and_h0(self):
        torch.manual_seed(144)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); rows=make_records(root,controlled=True)
            manifest=root/'manifest.json'; manifest.write_text(json.dumps(rows),encoding='utf8')
            cfg=dict(protocol='development_split',selection_metric='balanced_accuracy',batch_size=4,
                     stage1_epochs=1,base_epochs=1,h_epochs=1,prototypes_per_class=2,
                     h=dict(descriptor_dim=8,sinkhorn_iterations=5),allow_incomplete_supervision=False,
                     loss=dict(hard_gate_after=0,warmup_steps=0,ramp_steps=1))
            split=protocol(rows,cfg)
            s1,base,full=root/'s1.pt',root/'base.pt',root/'h.pt'
            args=Namespace(manifest=manifest,output=s1,checkpoint=None,split='test')
            with contextlib.redirect_stdout(io.StringIO()):
                train_stage1(args,cfg,rows,split,'cpu')
                args.checkpoint,args.output=s1,base; train_base(args,cfg,rows,split,'cpu')
                args.checkpoint,args.output=base,root/'base_eval.json'; evaluate(args,rows,'cpu')
                args.output=full
                with patch('sar_port.two_stage.cli.base_step',side_effect=AssertionError('must not refit h0')):
                    train_h(args,cfg,rows,split,'cpu')
                args.checkpoint,args.output=full,root/'h_eval.json'; evaluate(args,rows,'cpu')
            b,h=load_checkpoint(base),load_checkpoint(full)
            self.assertEqual(b['stage'],'base'); self.assertEqual(h['stage'],'h')
            self.assertEqual(b['reference_ids'],[])
            self.assertTrue(all(rows[i]['development_split']=='train' for i in split['train']))
            for key,value in b['encoder'].items():
                torch.testing.assert_close(value,h['encoder'][key],rtol=0,atol=0)
            for key,value in b['h'].items():
                if key.startswith('h0.'):
                    torch.testing.assert_close(value,h['h'][key],rtol=0,atol=0)
            self.assertTrue(any(not torch.equal(b['h'][k],v) for k,v in h['h'].items() if not k.startswith('h0.')))
            self.assertEqual(len([r for r in h['h_history'] if r['stage']=='base']),1)
            with patch('sar_port.two_stage.cli.load_checkpoint',return_value=dict(b,incomplete_supervision=True)):
                with self.assertRaisesRegex(ValueError,'Incomplete-supervision'):
                    train_h(args,cfg,rows,split,'cpu')
            for name,stage in [('base_eval.json','base'),('h_eval.json','h')]:
                result=json.loads((root/name).read_text(encoding='utf8'))
                self.assertEqual(result['stage'],stage)
                self.assertEqual(result['count'],2)
                self.assertIn('balanced_accuracy',result)
            # Assignment changes cannot bypass the unchanged-manifest check.
            changed=deepcopy(rows); changed[-1]['development_split']='val'
            with self.assertRaisesRegex(ValueError,'unchanged manifest'):
                evaluate(args,changed,'cpu')


if __name__=='__main__': unittest.main()
