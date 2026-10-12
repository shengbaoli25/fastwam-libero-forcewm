"""Eight-H100 ForceWM entrypoint for the user's explicit h100-8 allocation.

Run in that allocation's terminal only. The allocation argument is an explicit
operator binding, not a platform-provided distributed TRAIN_JOB_ID. Actual GPU
inventory, host, boot identity and fresh eight-rank receipts are recorded.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time

from bounded_probe import execute
from bounded_training import (INTERACTIVE_ALLOCATION, INTERACTIVE_RECIPES, LAUNCH_APPROVAL,
                              LAUNCH_DURATION_SECONDS, sha, validate_plan, verify_completion)
from distributed_job import BASE, NATIVE, prepare, write_new


def bind_allocation(allocation, env):
    if allocation not in INTERACTIVE_RECIPES or env.get('TRAIN_JOB_ID'):
        raise ValueError('Only the user-selected H100/H200 notebook is authorized here')
    # Some notebook images expose this field; reject a disagreement if present.
    if env.get('NOTEBOOK_ID') and env['NOTEBOOK_ID'] != allocation:
        raise ValueError('Platform NOTEBOOK_ID disagrees with the explicit allocation')
    if env.get('FORCEWM_INTERACTIVE_ALLOCATION_ID') not in (None, allocation):
        raise ValueError('Existing interactive allocation binding disagrees')
    env['FORCEWM_INTERACTIVE_ALLOCATION_ID'] = allocation
    return {'kind': 'interactive_modeling', 'allocation_id': allocation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allocation-id', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.allocation_id not in INTERACTIVE_RECIPES:
        parser.error('Only the explicitly authorized allocations are accepted')
    schema,world,model,version=INTERACTIVE_RECIPES[args.allocation_id]
    if not args.execute:
        print(json.dumps({'mode': 'plan_only', 'allocation_id': args.allocation_id,
            'budget_anchor': 'formal_launcher_start', 'duration_seconds': LAUNCH_DURATION_SECONDS,
            'recipe': f'{world}rank/B16/accum1/global{world*16}/30000/save5000'}))
        return
    identity = bind_allocation(args.allocation_id, os.environ)
    if not NATIVE.is_file() or not os.access(NATIVE, os.X_OK):
        raise ValueError('Original project .venv invocation is unavailable')
    source = Path(__file__).resolve().parents[2]
    folder = BASE / f'interactive-{args.allocation_id}-{version}'
    folder.mkdir(exist_ok=False)
    write_new(folder / 'dispatch.json', {'allocation_id': args.allocation_id,
        'identity_source': 'explicit_operator_binding_after_platform_UI_verification',
        'platform_notebook_id': os.environ.get('NOTEBOOK_ID'),
        'hostname': socket.gethostname(),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'started_unix': time.time(), 'deadline_unix': None,
        'budget_anchor': 'formal_launcher_start', 'duration_seconds': LAUNCH_DURATION_SECONDS,
        'human_duration_approval': LAUNCH_APPROVAL, 'foreground': True,
        'formal_training_started': False})
    try:
        plan = prepare(source, args.allocation_id, folder, interactive=True)
        remaining = validate_plan(plan, time.time())
        print('FORMAL_PLAN_VERIFIED', json.dumps({'allocation_id': args.allocation_id,
            'required_hours': plan['estimated_required_seconds'] / 3600,
            'available_hours': remaining / 3600,
            'plan_sha256': sha(folder / 'plan.json')}), flush=True)
        write_new(folder / 'formal-dispatch.json', {'allocation_id': args.allocation_id,
            'submitted_at': datetime.now(timezone.utc).isoformat(),
            'plan_sha256': sha(folder / 'plan.json'), 'foreground': True})
        result = execute(plan['argv'], plan['source'], plan['supervision_output'], remaining,
            maximum_seconds=LAUNCH_DURATION_SECONDS,
            duration_from_launch_seconds=LAUNCH_DURATION_SECONDS,
            execution_identity=identity, log_name='training.log', phase='forcewm-formal30000')
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
