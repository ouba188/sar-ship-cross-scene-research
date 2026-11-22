"""Readiness-only tests: synthetic artifacts, no Torch or training commands."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('seed29_pilot',ROOT/'run_seed29_pilot.py')
pilot=importlib.util.module_from_spec(spec); spec.loader.exec_module(pilot)
STAGES=('baseline_stage1','baseline_base','disentangle_stage1','disentangle_base','full_h')


def read(path):
    return json.loads(path.read_text())


def checked(path,expected):
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise RuntimeError('artifact hash mismatch')


class PilotGateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='pilot_gate_',dir=ROOT)
        self.root=Path(self.temp.name)
        self.state={'status':'yielded_for_seed29_pilot','current_step':None,'steps':{}}
        for seed in (17,29):
            for stage in STAGES+('baseline_base_validation','disentangle_base_validation','full_h_validation'):
                suffix='.json' if stage.endswith('_validation') else '.pt'
                path=self.root/'runs'/f'seed{seed}'/(stage+suffix)
                path.parent.mkdir(parents=True,exist_ok=True); path.write_text(f'synthetic {seed} {stage}')
                self.state['steps'][f'seed{seed}/{stage}']={'status':'succeeded','output_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        self.save()
        self.aux=SimpleNamespace(STAGES=STAGES,read=read,checked=Mock(side_effect=checked),EXPERIMENT=self.root,
                                 original_ready=Mock())

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        (self.root/'state_local.json').write_text(json.dumps(self.state))

    def test_seed43_absent_and_each_step_rehashes_all_sixteen(self):
        pilot.check_ready(self.aux,True)
        pilot.check_ready(self.aux,True,per_step=True)
        self.assertEqual(self.aux.checked.call_count,32)

    def test_active_original_lock_rejected(self):
        (self.root/'.run_comparison.lock').write_text('held')
        with self.assertRaisesRegex(RuntimeError,'lock'): pilot.check_ready(self.aux,True)

    def test_seed29_incomplete_rejected(self):
        del self.state['steps']['seed29/full_h_validation']; self.save()
        with self.assertRaisesRegex(RuntimeError,'seed29/full_h_validation'):pilot.check_ready(self.aux,True)

    def test_failed_or_running_attempt_must_be_archived_removed(self):
        for status in ('running','failed','interrupted'):
            self.state['steps']['seed43/baseline_stage1']={'status':status};self.save()
            with self.assertRaises(RuntimeError):pilot.check_ready(self.aux,True)
        del self.state['steps']['seed43/baseline_stage1'];self.save()
        pilot.check_ready(self.aux,True)

    def test_top_level_failed_or_stale_current_step_rejected(self):
        self.state['status']='failed';self.save()
        with self.assertRaisesRegex(RuntimeError,'running/failed'):pilot.check_ready(self.aux,True)
        self.state['status']='yielded';self.state['current_step']='seed43/baseline_stage1';self.save()
        with self.assertRaisesRegex(RuntimeError,'current step'):pilot.check_ready(self.aux,True)

    def test_tampered_completed_artifact_rejected(self):
        (self.root/'runs/seed29/disentangle_base.pt').write_text('modified')
        with self.assertRaisesRegex(RuntimeError,'hash mismatch'):pilot.check_ready(self.aux,True)

    def test_default_all_ready_behavior_unchanged(self):
        pilot.check_ready(self.aux,False)
        self.aux.original_ready.assert_called_once_with(self.root,verify_artifacts=True)
        self.aux.original_ready.reset_mock()
        pilot.check_ready(self.aux,False,per_step=True)
        self.aux.original_ready.assert_called_once_with(self.root,verify_artifacts=False)


if __name__=='__main__':unittest.main()
