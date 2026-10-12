"""Foreground single-node ForceWM restart with a fresh probe and fixed budget.

No launch on import. Platform task lifetime follows the foreground supervisor;
there is no nohup, daemon, background Popen, or reset of the budget anchor.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from bounded_probe import execute
from bounded_training import sha, validate_plan, verify_completion

HDD = Path('/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340')
SSD = Path(str(HDD).replace('/hdd/', '/ssd/'))
BASE = HDD / 'fastwam-forcewm-ar30'
NATIVE = HDD / 'realman-fastwam/.venv/bin/python'
ANCHOR = 1791734545
DEADLINE = 1791914545
APPROVAL = '同意延长到50小时并重启'
PRIOR_CONFIG_SHA = 'aaf34391200475c2c681d258a82b17bc052d95b9f40c6e4ac913b923b87c0236'


def platform_identity(env):
    job = env.get('TRAIN_JOB_ID', '')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{3,127}', job):
        raise ValueError('Platform TRAIN_JOB_ID required; no notebook fallback')
    if env.get('PET_NNODES', '1') != '1' or env.get('PET_NODE_RANK', '0') != '0':
        raise ValueError('Only the authorized single-node four-rank job is supported')
    return job


def estimate(native):
    ranks = native['ranks']
    warm = max(r['updates'][-1]['seconds'] for r in ranks)
    return warm * 30000 * 1.05 + max(600, max(r['startup_seconds'] for r in ranks)) + 5400 + 3600 + 60


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def hardware_check():
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,uuid',
                          '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout
    if len(gpu.strip().splitlines()) != 4 or not all('H200' in line for line in gpu.strip().splitlines()):
        raise ValueError('Actual four H200 GPUs required for this job')
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name',
                           '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout
    if apps.strip():
        raise ValueError('Current allocation GPUs are already occupied')
    descriptor = os.pidfd_open(os.getpid(), 0)
    try:
        signal.pidfd_send_signal(descriptor, 0)
    finally:
        os.close(descriptor)
    return gpu


def prepare(source, job, folder):
    prior_folder = BASE / 'recovery-ab5d6993-v2'
    old_config = prior_folder / 'resolved.yaml'
    if sha(old_config) != PRIOR_CONFIG_SHA:
        raise ValueError('Frozen GPU-validated recipe changed')
    provenance = json.loads((prior_folder / 'preparation.json').read_text())
    for name, digest in provenance['source_inventory']['files'].items():
        if sha(source / 'src/fastwam/forcewm' / name) != digest:
            raise ValueError('GPU-validated model math changed: ' + name)
    output = SSD / f'fastwam-forcewm-{job}-4rank-30k-v5'
    if output.exists() or Path(str(output) + '_origins').exists():
        raise ValueError('Formal output cannot be reused')
    config = folder / 'resolved.yaml'
    lines = old_config.read_text().splitlines()
    if lines[0] != 'output_dir: ' + str(SSD / 'fastwam-forcewm-ab5d6993-4rank-30k-v2'):
        raise ValueError('Unexpected frozen recipe output')
    lines[0] = 'output_dir: ' + str(output)
    config.write_text('\n'.join(lines) + '\n')
    env = json.loads((prior_folder / 'environment.json').read_text())
    env['PYTHONPATH'] = str(source / 'src') + ':' + str(source)
    if env['CUDA_VISIBLE_DEVICES'] != '0,1,2,3':
        raise ValueError('Original four-rank visibility required')
    # The platform launches this single-node command. Accelerate creates its
    # actual four ML ranks; platform outer-rank variables must not leak inside.
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
                'TORCHELASTIC_RUN_ID', 'TORCHELASTIC_RESTART_COUNT'):
        os.environ.pop(key, None)
    os.environ.update(env)
    probe = SSD / f'fastwam-forcewm-{job}-two-adam-v5'
    watch = SSD / f'fastwam-forcewm-{job}-probe-watch-v5'
    argv = [str(NATIVE), '-m', 'accelerate.commands.launch', '--config_file',
            str(source / 'scripts/accelerate_configs/accelerate_zero1_ds_4gpu.yaml'),
            '--num_processes', '4', '--num_machines', '1', '--machine_rank', '0',
            '-m', 'fastwam.forcewm.preflight', '--config', str(config),
            '--output', str(probe), '--execute']
    hardware = hardware_check()
    provenance.update(allocation_id=job, execution_mode='foreground_distributed_training',
                      observed_at=datetime.now(timezone.utc).isoformat(),
                      resolved_config_sha256=sha(config), hardware=hardware,
                      current_node_probe=True, probe_output=str(probe), probe_argv=argv,
                      argv_sha256=hashlib.sha256(json.dumps(argv, separators=(',', ':')).encode()).hexdigest(),
                      gpu_probe_passed=False, formal_training_started=False,
                      previous_preparation=str(prior_folder / 'preparation.json'),
                      previous_preparation_sha256=sha(prior_folder / 'preparation.json'))
    write_new(folder / 'preparation.json', provenance)
    write_new(folder / 'environment.json', env)
    write_new(folder / 'probe-argv.json', argv)
    print('CURRENT_JOB_PREPARED', json.dumps({'job': job, 'hardware': hardware, 'deadline_unix': DEADLINE}), flush=True)
    result = execute(argv, str(source), str(watch), 1800, deadline_unix=DEADLINE)
    if result['status'] != 'process_exited_zero':
        raise RuntimeError('Fresh current-job probe failed; formal training was not launched')
    native = probe / 'preflight.json'
    plan = dict(schema='forcewm-distributed50h-v5', allocation_id=job,
                allocation_created_unix=json.loads((folder / 'dispatch.json').read_text())['started_unix'],
                original_budget_anchor_unix=ANCHOR, deadline_unix=DEADLINE,
                recovery_of_allocation_id='b2c145b8-a54c-46e0-8580-919ec0549792',
                human_duration_approval=APPROVAL, execution_mode='foreground_distributed_training',
                world_size=4, batch_per_rank=16, global_batch=64, max_steps=30000,
                gradient_accumulation_steps=1, source=str(source), config=str(config),
                native_python=str(NATIVE), training_output=str(output), origins=str(output) + '_origins',
                supervision_output=str(SSD / f'fastwam-forcewm-{job}-formal-watch-v5'),
                preflight=str(native), probe_supervisor=str(watch / 'status.json'),
                preparation=str(folder / 'preparation.json'), environment=env,
                forcewm_source_inventory=provenance['source_inventory'], distributed_resume_verified=False)
    plan['estimated_required_seconds'] = estimate(json.loads(native.read_text()))
    plan['argv'] = argv[:argv.index('-m', 2)] + [str(source / 'scripts/forcewm/train_resolved.py'),
                    '--config', str(config), '--origins', plan['origins']]
    files = [config, native, watch / 'status.json', folder / 'preparation.json',
             folder / 'dispatch.json', source / 'scripts/accelerate_configs/accelerate_zero1_ds_4gpu.yaml',
             source / 'scripts/ds_configs/ds_zero1_config.json']
    files += [f for part in ('src', 'scripts/forcewm') for f in (source / part).rglob('*.py')]
    plan['input_sha256'] = {str(f): sha(f) for f in files}
    write_new(folder / 'plan.json', plan)
    validate_plan(plan, time.time())
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'mode': 'plan_only', 'deadline_unix': DEADLINE,
                          'recipe': '4rank/B16/accum1/global64/30000/save5000'}))
        return
    job = platform_identity(os.environ)
    source = Path(__file__).resolve().parents[2]
    if time.time() >= DEADLINE - 34.06186647181942 * 3600:
        raise ValueError('Prior measured full-run budget no longer fits; queue time did not reset deadline')
    if not NATIVE.is_file() or not os.access(NATIVE, os.X_OK):
        raise ValueError('Original native .venv invocation is unavailable')
    folder = BASE / f'distributed-{job}-v5'
    folder.mkdir(exist_ok=False)
    write_new(folder / 'dispatch.json', {'job': job, 'started_unix': time.time(),
              'deadline_unix': DEADLINE, 'original_budget_anchor_unix': ANCHOR,
              'platform_identity': {k: os.environ.get(k) for k in
                ('TRAIN_JOB_ID', 'RUNNING_ROUND', 'PET_NNODES', 'PET_NODE_RANK', 'PET_NPROC_PER_NODE')},
              'foreground': True, 'formal_training_started': False})
    try:
        plan = prepare(source, job, folder)
        remaining = validate_plan(plan, time.time())
        print('FORMAL_PLAN_VERIFIED', json.dumps({'job': job, 'required_hours': plan['estimated_required_seconds'] / 3600,
                                                 'available_hours': remaining / 3600, 'plan_sha256': sha(folder / 'plan.json')}), flush=True)
        write_new(folder / 'formal-dispatch.json', {'job': job, 'submitted_at': datetime.now(timezone.utc).isoformat(),
                  'plan_sha256': sha(folder / 'plan.json'), 'foreground': True})
        # Remain in the platform foreground throughout full-tree supervision.
        result = execute(plan['argv'], plan['source'], plan['supervision_output'], remaining,
                         maximum_seconds=172800, deadline_unix=DEADLINE,
                         log_name='training.log', phase='forcewm-formal30000')
        final = {'status': 'failed', 'supervisor_status': result['status'],
                 'distributed_resume_verified': False}
        if result['status'] == 'process_exited_zero':
            final.update(verify_completion(plan))
        write_new(folder / 'completion.json', final)
        print('FORMAL_FINAL', json.dumps(final), flush=True)
        raise SystemExit(0 if final['status'] == 'completed_30000' else 2)
    except Exception as error:
        write_new(folder / 'failure.json', {'status': 'failed', 'error': repr(error),
                   'observed_at': datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == '__main__':
    main()
