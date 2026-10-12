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


@pytest.mark.parametrize('count', [8])
def test_hardware_check_accepts_an_h200_node_for_eight_rank_execution(monkeypatch, count):
    gpu = '\n'.join(f'{index}, NVIDIA H200, 143771 MiB, GPU-{index}' for index in range(count))
    responses = iter([gpu, ''])

    class Result:
        def __init__(self, stdout):
            self.stdout = stdout

    monkeypatch.setattr(job.subprocess, 'run', lambda *args, **kwargs: Result(next(responses)))
    monkeypatch.setattr(job.os, 'pidfd_open', lambda *args: 77)
    monkeypatch.setattr(job.signal, 'pidfd_send_signal', lambda *args: None)
    monkeypatch.setattr(job.os, 'close', lambda *args: None)
    assert job.hardware_check() == gpu


def test_hardware_check_rejects_an_incomplete_or_wrong_gpu_allocation(monkeypatch):
    class Result:
        stdout = '0, NVIDIA H100, 81559 MiB, GPU-0\n1, NVIDIA H100, 81559 MiB, GPU-1'

    monkeypatch.setattr(job.subprocess, 'run', lambda *args, **kwargs: Result())
    with pytest.raises(ValueError, match='exactly eight H200'):
        job.hardware_check()


def test_prepare_uses_current_probe_and_exact_native_formal_argv(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    for name in ('src/fastwam/forcewm/model.py', 'scripts/forcewm/train_resolved.py',
                 'scripts/accelerate_configs/accelerate_zero1_ds_8gpu.yaml',
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
    monkeypatch.setattr(job, 'hardware_check', lambda: 'TEST-only-eight-GPUs')
    monkeypatch.setattr(job.time, 'time', lambda: 1791778000)
    monkeypatch.setenv('TRAIN_JOB_ID', 'train-current-fixture')
    monkeypatch.setenv('RANK', '0'); monkeypatch.setenv('WORLD_SIZE', '1')
    # Undo prepare's environment update on fixture completion.
    for key in ('PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'DIFFSYNTH_SKIP_DOWNLOAD'):
        monkeypatch.setenv(key, 'TEST-only-previous')
    ranks = [{'rank': r, 'world_size': 8, 'global_batch': 128, 'batch_per_rank': 16,
        'actual_engine_updates': 2, 'adam_state_steps': [2], 'accepted': True,
        'all_parameters_finite': True, 'adam_master_and_moments_finite': True,
        'updates': [{'global_loss': 1., 'optimizer_step_was_skipped': False, 'seconds': 3.5}] * 2,
        'teacher_layer_calls': [2] * 30, 'dataset': {'verified': True}, 'startup_seconds': 100,
        'local_adam_master_and_moments_bytes': 12, 'architecture_counts': {'optimizer_parameters': 1}}
        for r in range(8)]
    calls = []
    def fake_execute(argv, cwd, output, seconds, **kwargs):
        calls.append(argv)
        probe = ssd / 'fastwam-forcewm-train-current-fixture-two-adam-v6'; probe.mkdir()
        (probe / 'preflight.json').write_text(json.dumps({'status': 'two_adam_verified', 'ranks': ranks}))
        out = Path(output); out.mkdir()
        receipt = {'status': 'process_exited_zero', 'returncode': 0, 'command': argv,
            'absolute_deadline_unix': job.DEADLINE, 'subreaper_restored': True,
            'ownership': {'closure_verified': True, 'errors': [], 'signals': [], 'live_after_cleanup': []}}
        (out / 'status.json').write_text(json.dumps(receipt))
        return receipt
    monkeypatch.setattr(job, 'execute', fake_execute)
    folder = base / 'distributed-train-current-fixture-v6'; folder.mkdir()
    (folder / 'dispatch.json').write_text(json.dumps({'started_unix': 1791777900}))
    p = job.prepare(source, 'train-current-fixture', folder)
    assert len(calls) == 1
    assert calls[0][0] == str(job.NATIVE)
    assert calls[0][11:13] == ['-m', 'fastwam.forcewm.preflight']
    assert p['argv'][11] == str(source / 'scripts/forcewm/train_resolved.py')
    assert p['argv'][0] == str(job.NATIVE)
    assert p['deadline_unix'] == job.DEADLINE
    assert guard.validate_plan(p, 1791778000) > 0
    assert p['world_size'] == 8 and p['global_batch'] == 128
    assert p['environment']['CUDA_VISIBLE_DEVICES'] == '0,1,2,3,4,5,6,7'
    assert p['argv'][5:7] == ['--num_processes', '8']
    assert p['argv'][-2:] == ['--expected-world-size', '8']
    assert p['distributed_resume_verified'] is False
    assert 'RANK' not in job.os.environ and 'WORLD_SIZE' not in job.os.environ


def test_eight_rank_probe_budget_does_not_assume_four_rank_speed():
    # This is after the old four-rank estimate cutoff. A bounded fresh probe
    # remains permitted; formal validation still requires measured 8-rank fit.
    job.require_probe_budget(job.DEADLINE - 30 * 3600)
    with pytest.raises(ValueError, match='Insufficient'):
        job.require_probe_budget(job.DEADLINE - 1800)


def test_rank_mismatch_rejected_before_model_initialization():
    spec = importlib.util.spec_from_file_location('rank_probe_contract', ROOT / 'src/fastwam/forcewm/preflight.py')
    probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)
    assert probe.validate_world_size(8, 8) == 128
    assert probe.validate_world_size(4, 4) == 64
    for actual, expected in ((4, 8), (8, 4), (7, 8), (16, 16)):
        with pytest.raises(RuntimeError): probe.validate_world_size(actual, expected)
