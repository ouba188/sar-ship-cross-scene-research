"""Small standard-library tests; no model imports, real training or predictions."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('aux_ablation', HERE/'run_aux_scale_ablation.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class AuxScaleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='aux_check_', dir=HERE)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_only_three_coefficients_change(self):
        original = {'loss': {'scene': 1, 'invariance': 1, 'leakage': .1,
                            'hard_gate_after': 200, 'warmup_steps': 50, 'ramp_steps': 150},
                    'seed': 17, 'encoder': {'gate_enabled': True}, 'nested': {'schedule': [1, 2]}}
        before = copy.deepcopy(original)
        reduced = tool.scaled_config(original, .1)
        self.assertEqual([reduced['loss'][k] for k in ('scene','invariance','leakage')], [.1,.1,.01])
        reduced['loss'].update({k: original['loss'][k] for k in ('scene','invariance','leakage')})
        self.assertEqual(reduced, original)
        self.assertEqual(original, before)
        self.assertTrue(all(tool.scaled_config(original, 0)['loss'][k] == 0 for k in ('scene','invariance','leakage')))

    def test_readiness_and_lock(self):
        tool.write(self.root/'state_local.json', {'steps': {}})
        with self.assertRaisesRegex(RuntimeError, 'must finish'): tool.original_ready(self.root, False)
        steps = {f'seed{s}/{stage}': {'status':'succeeded'} for s in tool.SEEDS
                 for stage in tool.STAGES+('baseline_base_validation','disentangle_base_validation','full_h_validation')}
        tool.write(self.root/'state_local.json', {'steps':steps})
        tool.original_ready(self.root, False)
        (self.root/'.run_comparison.lock').write_text('held')
        with self.assertRaisesRegex(RuntimeError, 'lock'): tool.original_ready(self.root, False)

    def test_no_test_or_h_training_commands(self):
        jobs = [job for n in tool.SCALES for s in tool.SEEDS for job in tool.commands(self.root,self.root,n,s,'cpu')]
        self.assertEqual(len(jobs),18)
        for _,command,_ in jobs:
            self.assertNotIn('test',command)
            self.assertNotIn('train-h',command)
            if 'evaluate' in command: self.assertEqual(command[command.index('--split')+1],'validation')
        with self.assertRaisesRegex(RuntimeError,'only permits'):
            tool.execute(self.root,'bad',['evaluate','--split','test'],self.root/'bad.json',{},self.root)

    def test_success_skip_and_tamper(self):
        output=self.root/'runs_local/result.json'; key='aux_0/seed17/base_validation'
        def fake_run(command, **kwargs):
            kwargs['stdout'].write('synthetic evaluation\n')
            tool.write(output, {'count':227,'balanced_accuracy':.5})
            return subprocess.CompletedProcess(command,0)
        with patch.object(tool.subprocess,'run',side_effect=fake_run) as call:
            tool.execute(self.root,key,['evaluate','--split','validation'],output,{},self.root)
            tool.execute(self.root,key,['evaluate','--split','validation'],output,{},self.root)
            self.assertEqual(call.call_count,1)
        output.write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'changed'):
            tool.execute(self.root,key,[],output,{},self.root)

    def test_failure_not_retried_or_overwritten(self):
        output=self.root/'runs_local/stage1.pt'; key='aux_0/seed17/stage1'
        with patch.object(tool.subprocess,'run',return_value=subprocess.CompletedProcess([],7)) as call:
            with self.assertRaisesRegex(RuntimeError,'exited 7'):
                tool.execute(self.root,key,['train-stage1'],output,{},self.root)
            with self.assertRaisesRegex(RuntimeError,'automatic retry disabled'):
                tool.execute(self.root,key,['train-stage1'],output,{},self.root)
            self.assertEqual(call.call_count,1)
        self.assertTrue(output.with_suffix('.log').exists())

    def test_selection_tie_keeps_one_and_zero_is_ineligible(self):
        values={'original_scale_1':{'mean_h0_ba':.5},'aux_0p1':{'mean_h0_ba':.5},'aux_0':{'mean_h0_ba':1.}}
        self.assertEqual(tool.select_complete_scale(values),1.)
        values['aux_0p1']['mean_h0_ba']=.51
        self.assertEqual(tool.select_complete_scale(values),.1)

    def test_mixed_transition_epoch_can_be_selected(self):
        config={'loss':{'hard_gate_after':200,'warmup_steps':50,'ramp_steps':150}}
        history=[{'epoch':e,'paired_samples':32,'losses':{'paired_samples':32/94},
                  'selection_score':score,'validation_metrics':{'balanced_accuracy':score}}
                 for e,score in enumerate([.9,.8,.7,.6],1)]
        self.assertEqual(tool.selected_stage1(history,config)['epoch'],3)

    def test_aggregate_keeps_stage1_and_excludes_predictions(self):
        root=self.root/'tuning'; experiment=self.root/'original'
        config={'loss':{'hard_gate_after':200,'warmup_steps':50,'ramp_steps':150}}
        metrics={'accuracy':.5,'balanced_accuracy':.5,'macro_f1':.5,
                 'confusion_matrix':[[1,1,0,0],[0,1,1,0],[0,0,1,1],[1,0,0,1]],'count':8}
        history=[{'epoch':e,'paired_samples':32,'losses':{'paired_samples':32/94},
                  'selection_score':.5,'validation_metrics':metrics} for e in (1,2,3)]
        for seed in tool.SEEDS:
            for name in ('baseline','original_scale_1',*tool.SCALES):
                original=name in ('baseline','original_scale_1')
                arm='baseline' if name=='baseline' else 'disentangle'
                directory=experiment/'runs'/f'seed{seed}' if original else root/'runs_local'/name/f'seed{seed}'
                prefix=arm+'_' if original else ''
                cfg=experiment/'configs'/f'{arm}_seed{seed}.json' if original else root/'configs_local'/f'{name}_seed{seed}.json'
                tool.write(cfg,config)
                tool.write(directory/(prefix+'stage1.history.json'),history)
                tool.write(directory/(prefix+'base.history.json'),history[:1])
                tool.write(directory/(prefix+'base_validation.json'),dict(metrics,predictions=[{'id':'private-example'}]))
        result=tool.collect(root,experiment)
        self.assertEqual(len(result['rows']),12)
        self.assertEqual(result['selected_complete_aux_scale'],1.)
        self.assertEqual(result['rows'][0]['h0']['recall_per_class'],[.5]*4)
        self.assertNotIn('private-example',(root/'aggregate_aux_scale_validation.json').read_text())


if __name__=='__main__':
    unittest.main()
