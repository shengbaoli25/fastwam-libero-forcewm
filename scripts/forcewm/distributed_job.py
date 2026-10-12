"""Foreground eight-rank single-node ForceWM restart with a fresh probe and launch-relative budget.

No launch on import. Platform task lifetime follows the foreground supervisor;
Queue and probe time precede the once-only formal50h clock; no background daemon.
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
from bounded_training import (sha, validate_plan, verify_completion, LAUNCH_SCHEMA,
                              LAUNCH_APPROVAL, LAUNCH_DURATION_SECONDS, INTERACTIVE_SCHEMA, INTERACTIVE_ALLOCATION, INTERACTIVE_RECIPES)

HDD = Path('/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340')
SSD = Path(str(HDD).replace('/hdd/', '/ssd/'))
BASE = HDD / 'fastwam-forcewm-ar30'
NATIVE = HDD / 'realman-fastwam/.venv/bin/python'
APPROVAL = LAUNCH_APPROVAL
PRIOR_CONFIG_SHA = 'aaf34391200475c2c681d258a82b17bc052d95b9f40c6e4ac913b923b87c0236'


def platform_identity(env):
    job = env.get('TRAIN_JOB_ID', '')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{3,127}', job):
        raise ValueError('Platform TRAIN_JOB_ID required; no notebook fallback')
    if env.get('PET_NNODES', '1') != '1' or env.get('PET_NODE_RANK', '0') != '0':
        raise ValueError('Only the authorized single-node eight-rank job is supported')
    return job


def estimate(native):
    ranks = native['ranks']
    warm = max(r['updates'][-1]['seconds'] for r in ranks)
    return warm * 30000 * 1.05 + max(600, max(r['startup_seconds'] for r in ranks)) + 5400 + 3600 + 60


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def hardware_check(expected_model='H200', expected_count=8):
    if expected_count not in (4,8):
        raise ValueError('Only four/eight GPU recipes authorized')
    if expected_model not in ('H100','H200'):
        raise ValueError('Only authorized H100/H200 models are supported')
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,uuid',
                          '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout
    physical_gpus = [line for line in gpu.strip().splitlines() if line]
    if len(physical_gpus) != expected_count or not all(expected_model in line for line in physical_gpus):
        raise ValueError(f'Actual allocation must contain exactly {expected_count} {expected_model} GPUs')
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


def prepare(source, job, folder, *, interactive=False):
    mode='foreground_interactive_modeling' if interactive else 'foreground_distributed_training'
    schema,world,model,version=INTERACTIVE_RECIPES.get(job,(None,None,None,None)) if interactive else (LAUNCH_SCHEMA,8,'H200','v7')
    if interactive and (schema is None or os.environ.get('TRAIN_JOB_ID')
            or os.environ.get('FORCEWM_INTERACTIVE_ALLOCATION_ID')!=job):
        raise ValueError('Current interactive allocation required before any probe')
    prior_folder = BASE / 'recovery-ab5d6993-v2'
    old_config = prior_folder / 'resolved.yaml'
    if sha(old_config) != PRIOR_CONFIG_SHA:
        raise ValueError('Frozen GPU-validated recipe changed')
    provenance = json.loads((prior_folder / 'preparation.json').read_text())
    for name, digest in provenance['source_inventory']['files'].items():
        if name != 'preflight.py' and sha(source / 'src/fastwam/forcewm' / name) != digest:
            raise ValueError('GPU-validated model math changed: ' + name)
    current_files = {p.name: sha(p) for p in sorted((source / 'src/fastwam/forcewm').glob('*.py'))}
    if set(current_files) != set(provenance['source_inventory']['files']):
        raise ValueError('ForceWM model/probe source file set changed unexpectedly')
    provenance['previous_source_inventory'] = provenance['source_inventory']
    provenance['source_inventory'] = {'files': current_files, 'sha256': hashlib.sha256(
        json.dumps(current_files, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()}
    output = SSD / f'fastwam-forcewm-{job}-{world}rank-30k-{version}'
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
    env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str,range(world)))
    # The platform launches this single-node command. Accelerate creates its
    # actual eight ML ranks; platform outer-rank variables must not leak inside.
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
                'TORCHELASTIC_RUN_ID', 'TORCHELASTIC_RESTART_COUNT'):
        os.environ.pop(key, None)
    os.environ.update(env)
    probe = SSD / f'fastwam-forcewm-{job}-two-adam-{version}'
    watch = SSD / f'fastwam-forcewm-{job}-probe-watch-{version}'
    argv = [str(NATIVE), '-m', 'accelerate.commands.launch', '--config_file',
            str(source / f'scripts/accelerate_configs/accelerate_zero1_ds_{world}gpu.yaml'),
            '--num_processes', str(world), '--num_machines', '1', '--machine_rank', '0',
            '-m', 'fastwam.forcewm.preflight', '--config', str(config),
            '--output', str(probe), '--expected-world-size', str(world), '--execute']
    hardware = hardware_check(model,world) if interactive else hardware_check()
    probe_started = time.time()
    probe_deadline = probe_started + 1860
    provenance.update(allocation_id=job, execution_mode=mode,
                      probe_started_unix=probe_started, probe_deadline_unix=probe_deadline,
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
    print('CURRENT_JOB_PREPARED', json.dumps({'job': job, 'hardware': hardware, 'probe_deadline_unix': probe_deadline,
          'formal_duration_seconds': LAUNCH_DURATION_SECONDS}), flush=True)
    result = execute(argv, str(source), str(watch), 1800, deadline_unix=probe_deadline)
    if result['status'] != 'process_exited_zero':
        raise RuntimeError('Fresh current-job probe failed; formal training was not launched')
    native = probe / 'preflight.json'
    plan = dict(schema=schema, allocation_id=job,
                allocation_created_unix=json.loads((folder / 'dispatch.json').read_text())['started_unix'],
                budget_anchor='formal_launcher_start', duration_seconds=LAUNCH_DURATION_SECONDS,
                deadline_unix=None, probe_deadline_unix=probe_deadline,
                recovery_of_allocation_id='b2c145b8-a54c-46e0-8580-919ec0549792',
                human_duration_approval=APPROVAL, execution_mode=mode,
                world_size=world, batch_per_rank=16, global_batch=16*world, max_steps=30000,
                gradient_accumulation_steps=1, source=str(source), config=str(config),
                native_python=str(NATIVE), training_output=str(output), origins=str(output) + '_origins',
                supervision_output=str(SSD / f'fastwam-forcewm-{job}-formal-watch-{version}'),
                preflight=str(native), probe_supervisor=str(watch / 'status.json'),
                preparation=str(folder / 'preparation.json'), environment=env,
                forcewm_source_inventory=provenance['source_inventory'], distributed_resume_verified=False)
    plan['estimated_required_seconds'] = estimate(json.loads(native.read_text()))
    plan['argv'] = argv[:argv.index('-m', 2)] + [str(source / 'scripts/forcewm/train_resolved.py'),
                    '--config', str(config), '--origins', plan['origins']] + (['--expected-world-size', str(world)] if world==8 else [])
    files = [config, native, watch / 'status.json', folder / 'preparation.json',
             folder / 'dispatch.json', source / f'scripts/accelerate_configs/accelerate_zero1_ds_{world}gpu.yaml',
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
        print(json.dumps({'mode': 'plan_only', 'budget_anchor': 'formal_launcher_start',
                          'duration_seconds': LAUNCH_DURATION_SECONDS, 'deadline_unix': None,
                          'recipe': '8rank/B16/accum1/global128/30000/save5000'}))
        return
    job = platform_identity(os.environ)
    source = Path(__file__).resolve().parents[2]
    if not NATIVE.is_file() or not os.access(NATIVE, os.X_OK):
        raise ValueError('Original native .venv invocation is unavailable')
    folder = BASE / f'distributed-{job}-v7'
    folder.mkdir(exist_ok=False)
    write_new(folder / 'dispatch.json', {'job': job, 'started_unix': time.time(),
              'deadline_unix': None, 'budget_anchor': 'formal_launcher_start',
              'duration_seconds': LAUNCH_DURATION_SECONDS, 'human_duration_approval': APPROVAL,
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
                         maximum_seconds=LAUNCH_DURATION_SECONDS,
                         duration_from_launch_seconds=LAUNCH_DURATION_SECONDS,
                         log_name='training.log', phase='forcewm-formal30000')
        final = {'status': 'failed', 'supervisor_status': result['status'],
                 'distributed_resume_verified': False,
                 'actual_deadline_unix': result.get('absolute_deadline_unix'),
                 'launcher_started_unix': result.get('launcher_started_unix')}
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
