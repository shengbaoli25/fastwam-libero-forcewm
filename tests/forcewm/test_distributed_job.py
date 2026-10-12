"""CPU-only dispatch fixtures; no claims of GPU or checkpoint validation."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/forcewm'))
import distributed_job as job
import bounded_training as guard


@pytest.mark.parametrize('env', [{}, {'TRAIN_JOB_ID': 'x'}, {'TRAIN_JOB_ID': '../escape'},
    {'TRAIN_JOB_ID': 'train-valid', 'PET_NNODES': '2'},
    {'TRAIN_JOB_ID': 'train-valid', 'PET_NODE_RANK': '1'}])
def test_platform_identity_rejects_unbound_or_multinode(env):
    with pytest.raises(ValueError): job.platform_identity(env)


def test_current_job_identity():
    assert job.platform_identity({'TRAIN_JOB_ID': 'train-valid'}) == 'train-valid'


def test_prepare_uses_current_probe_and_exact_native_formal_argv(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    for name in ('src/fastwam/forcewm/model.py', 'scripts/forcewm/train_resolved.py',
                 'scripts/accelerate_configs/accelerate_zero1_ds_4gpu.yaml',
                 'scripts/ds_configs/ds_zero1_config.json'):
        p = source / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('TEST-only')
    base = tmp_path / 'base'; old = base / 'recovery-ab5d6993-v2'; old.mkdir(parents=True)
    ssd = tmp_path / 'ssd'; ssd.mkdir()
    config = old / 'resolved.yaml'
    config.write_text('output_dir: ' + str(ssd / 'fastwam-forcewm-ab5d6993-4rank-30k-v2') + '\nmax_steps: 30000\n')
    inventory = {'files': {'model.py': guard.sha(source / 'src/fastwam/forcewm/model.py')}}
    (old / 'preparation.json').write_text(json.dumps({'source_inventory': inventory}))
    (old / 'environment.json').write_text(json.dumps({'CUDA_VISIBLE_DEVICES': '0,1,2,3',
        'DIFFSYNTH_SKIP_DOWNLOAD': 'true'}))
    monkeypatch.setattr(job, 'BASE', base); monkeypatch.setattr(job, 'SSD', ssd)
    monkeypatch.setattr(job, 'PRIOR_CONFIG_SHA', guard.sha(config))
    monkeypatch.setattr(job, 'hardware_check', lambda: 'TEST-only-four-GPUs')
    monkeypatch.setattr(job.time, 'time', lambda: 1791778000)
    monkeypatch.setenv('TRAIN_JOB_ID', 'train-current-fixture')
    monkeypatch.setenv('RANK', '0'); monkeypatch.setenv('WORLD_SIZE', '1')
    # Undo prepare's environment update on fixture completion.
    for key in ('PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'DIFFSYNTH_SKIP_DOWNLOAD'):
        monkeypatch.setenv(key, 'TEST-only-previous')
    ranks = [{'rank': r, 'world_size': 4, 'global_batch': 64, 'batch_per_rank': 16,
        'actual_engine_updates': 2, 'adam_state_steps': [2], 'accepted': True,
        'all_parameters_finite': True, 'adam_master_and_moments_finite': True,
        'updates': [{'global_loss': 1., 'optimizer_step_was_skipped': False, 'seconds': 3.5}] * 2,
        'teacher_layer_calls': [2] * 30, 'dataset': {'verified': True}, 'startup_seconds': 100,
        'local_adam_master_and_moments_bytes': 12, 'architecture_counts': {'optimizer_parameters': 1}}
        for r in range(4)]
    calls = []
    def fake_execute(argv, cwd, output, seconds, **kwargs):
        calls.append(argv)
        probe = ssd / 'fastwam-forcewm-train-current-fixture-two-adam-v5'; probe.mkdir()
        (probe / 'preflight.json').write_text(json.dumps({'status': 'two_adam_verified', 'ranks': ranks}))
        out = Path(output); out.mkdir()
        receipt = {'status': 'process_exited_zero', 'returncode': 0, 'command': argv,
            'absolute_deadline_unix': job.DEADLINE, 'subreaper_restored': True,
            'ownership': {'closure_verified': True, 'errors': [], 'signals': [], 'live_after_cleanup': []}}
        (out / 'status.json').write_text(json.dumps(receipt))
        return receipt
    monkeypatch.setattr(job, 'execute', fake_execute)
    folder = base / 'distributed-train-current-fixture-v5'; folder.mkdir()
    (folder / 'dispatch.json').write_text(json.dumps({'started_unix': 1791777900}))
    p = job.prepare(source, 'train-current-fixture', folder)
    assert len(calls) == 1
    assert calls[0][0] == str(job.NATIVE)
    assert calls[0][11:13] == ['-m', 'fastwam.forcewm.preflight']
    assert p['argv'][11] == str(source / 'scripts/forcewm/train_resolved.py')
    assert p['argv'][0] == str(job.NATIVE)
    assert p['deadline_unix'] == job.DEADLINE
    assert guard.validate_plan(p, 1791778000) > 0
    assert p['distributed_resume_verified'] is False
    assert 'RANK' not in job.os.environ and 'WORLD_SIZE' not in job.os.environ
