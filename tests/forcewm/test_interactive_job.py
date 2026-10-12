"""Explicit notebook identity and CPU process contracts, not GPU verification."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/forcewm'))
import interactive_job as job
from bounded_probe import execute


def test_explicit_notebook_identity_is_distinct_from_distributed_job():
    env = {}
    identity = job.bind_allocation(job.INTERACTIVE_ALLOCATION, env)
    assert identity == {'kind': 'interactive_modeling', 'allocation_id': job.INTERACTIVE_ALLOCATION}
    assert 'TRAIN_JOB_ID' not in env
    assert env['FORCEWM_INTERACTIVE_ALLOCATION_ID'] == job.INTERACTIVE_ALLOCATION


@pytest.mark.parametrize('change', ['other_allocation', 'distributed', 'platform_mismatch', 'previous_binding'])
def test_identity_rejects_other_container_and_job_impersonation(change):
    env = {}; allocation = job.INTERACTIVE_ALLOCATION
    if change == 'other_allocation': allocation = 'another-notebook'
    elif change == 'distributed': env['TRAIN_JOB_ID'] = 'a-distributed-job'
    elif change == 'platform_mismatch': env['NOTEBOOK_ID'] = 'another-notebook'
    else: env['FORCEWM_INTERACTIVE_ALLOCATION_ID'] = 'another-notebook'
    with pytest.raises(ValueError): job.bind_allocation(allocation, env)


def test_interactive_actual_launch_receipt_and_full_tree(tmp_path, monkeypatch):
    monkeypatch.delenv('TRAIN_JOB_ID', raising=False)
    monkeypatch.setenv('FORCEWM_INTERACTIVE_ALLOCATION_ID', job.INTERACTIVE_ALLOCATION)
    identity = {'kind': 'interactive_modeling', 'allocation_id': job.INTERACTIVE_ALLOCATION}
    code = ('import subprocess,sys; p=[subprocess.Popen([sys.executable,"-c",'
            '"import time;time.sleep(.4)"],start_new_session=True) for _ in range(8)];'
            'assert all(c.wait()==0 for c in p)')
    output = tmp_path / 'watch'
    result = execute([sys.executable, '-c', code], str(tmp_path), output, 179940,
        maximum_seconds=180000, duration_from_launch_seconds=180000,
        phase='forcewm-formal30000', execution_identity=identity)
    receipt = json.loads((output / 'formal-launch.json').read_text())
    assert receipt['execution_identity'] == identity
    assert receipt['platform_job_id'] is None
    assert receipt['absolute_deadline_unix'] - receipt['launcher_started_unix'] == 180000
    assert receipt['launcher_start_ticks'] > 0
    assert result['status'] == 'process_exited_zero', result
    assert result['subreaper_restored'] and result['ownership']['closure_verified']
    assert not result['ownership']['signals'] and not result['ownership']['errors']


def test_no_gpu_or_binding_when_plan_only(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / 'scripts/forcewm/interactive_job.py'),
        '--allocation-id', job.INTERACTIVE_ALLOCATION], capture_output=True, text=True, check=True)
    result = json.loads(r.stdout)
    assert result['mode'] == 'plan_only' and result['duration_seconds'] == 180000
