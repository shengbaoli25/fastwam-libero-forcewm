"""Launch hash-bound training under an explicitly authorized recipe and budget."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from bounded_probe import execute

LAUNCH_SCHEMA='forcewm-distributed8-launch50h-v7'
INTERACTIVE_SCHEMA='forcewm-interactive8-h100-launch50h-v8'
INTERACTIVE_ALLOCATION='35a848a2-4523-47f0-a0f1-cf2e7ebd8a2b'
INTERACTIVE4_SCHEMA='forcewm-interactive4-h200-launch50h-v9'
INTERACTIVE4_ALLOCATION='ab5d6993-795b-4bf7-9084-e59f089e7788'
INTERACTIVE_RECIPES={INTERACTIVE_ALLOCATION:(INTERACTIVE_SCHEMA,8,'H100','v8'),
                     INTERACTIVE4_ALLOCATION:(INTERACTIVE4_SCHEMA,4,'H200','v9')}
LAUNCH_APPROVAL='截止时间仍为10月14日02:02:25，按照实际开始启动训练的时间来算'
LAUNCH_DURATION_SECONDS=50*3600


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recipe_world_size(plan):
    schemas = ('forcewm-h100-formal-v1','forcewm-recovery48h-v2','forcewm-restart50h-v4','forcewm-distributed50h-v5','forcewm-distributed8-50h-v6',LAUNCH_SCHEMA,INTERACTIVE_SCHEMA,INTERACTIVE4_SCHEMA)
    if plan['schema'] not in schemas:
        raise ValueError('Unknown authorized recipe schema')
    expected = 8 if plan['schema'] in ('forcewm-distributed8-50h-v6',LAUNCH_SCHEMA,INTERACTIVE_SCHEMA) else 4
    if plan['world_size'] != expected:
        raise ValueError('World size differs from explicitly authorized recipe')
    return expected


def validate_plan(plan,now):
    if type(now) not in (int,float) or not math.isfinite(now):
        raise ValueError('Actual finite current time required')
    world=recipe_world_size(plan)
    if (plan['batch_per_rank']!=16 or plan['global_batch']!=16*world or plan['max_steps']!=30000
            or plan['gradient_accumulation_steps']!=1):
        raise ValueError('Exact authorized ForceWM training recipe required')
    launch_relative=plan['schema'] in (LAUNCH_SCHEMA,INTERACTIVE_SCHEMA,INTERACTIVE4_SCHEMA)
    if launch_relative:
        if (plan.get('budget_anchor')!='formal_launcher_start'
                or type(plan.get('duration_seconds')) is not int
                or plan['duration_seconds']!=LAUNCH_DURATION_SECONDS
                or plan.get('deadline_unix') is not None
                or plan.get('formal_launcher_started_unix') is not None
                or plan.get('original_budget_anchor_unix') is not None
                or plan.get('human_duration_approval')!=LAUNCH_APPROVAL
                or type(plan.get('allocation_created_unix')) not in (int,float)
                or not 0<plan['allocation_created_unix']<=now):
            raise ValueError('Authorized50h must start once at actual formal launcher creation')
    elif plan['schema']=='forcewm-h100-formal-v1':
        if plan['deadline_unix']-plan['allocation_created_unix']!=40*3600:
            raise ValueError('Deadline must remain allocation creation plus40h')
    else:
        # Each explicit Oct12 approval extends the ORIGINAL budget anchor.
        # Replacing an allocation or restarting training never resets it.
        deadline=1791914545 if plan['schema'] in ('forcewm-restart50h-v4','forcewm-distributed50h-v5','forcewm-distributed8-50h-v6') else 1791907345
        if (plan.get('original_budget_anchor_unix')!=1791734545
                or plan['deadline_unix']!=deadline
                or plan.get('recovery_of_allocation_id')!='b2c145b8-a54c-46e0-8580-919ec0549792'
                or type(plan['allocation_created_unix']) not in (int,float)
                or not 1791734545<=plan['allocation_created_unix']<=now):
            raise ValueError('Recovery must retain its explicit fixed deadline')
        if plan['schema']=='forcewm-restart50h-v4' and (
                plan.get('allocation_id')!='ab5d6993-795b-4bf7-9084-e59f089e7788'
                or not plan.get('restart_from_failed_plan')
                or plan.get('human_duration_approval')!='同意延长到50小时并重启'):
            raise ValueError('The fixed50h approval applies only to this authorized restart')
    distributed=plan['schema'] in ('forcewm-distributed50h-v5','forcewm-distributed8-50h-v6',LAUNCH_SCHEMA)
    interactive=plan['schema'] in (INTERACTIVE_SCHEMA,INTERACTIVE4_SCHEMA)
    if interactive:
        expected_allocation=INTERACTIVE_ALLOCATION if world==8 else INTERACTIVE4_ALLOCATION
        if (plan.get('allocation_id')!=expected_allocation
                or os.environ.get('FORCEWM_INTERACTIVE_ALLOCATION_ID')!=expected_allocation
                or os.environ.get('TRAIN_JOB_ID')
                or plan.get('execution_mode')!='foreground_interactive_modeling'
                or plan.get('restart_from_failed_plan')):
            raise ValueError('Current h100-8 explicit allocation binding and fresh probe required')
    if distributed:
        approval=LAUNCH_APPROVAL if launch_relative else '同意延长到50小时并重启'
        if (not plan.get('allocation_id') or plan['allocation_id']!=os.environ.get('TRAIN_JOB_ID')
                or plan.get('restart_from_failed_plan')
                or plan.get('execution_mode')!='foreground_distributed_training'
                or plan.get('human_duration_approval')!=approval):
            raise ValueError('Distributed restart requires its current TRAIN_JOB_ID and fresh GPU probe')
    remaining=LAUNCH_DURATION_SECONDS if launch_relative else plan['deadline_unix']-now
    source=Path(plan['source']); config=plan['config']
    expected_argv=[plan['native_python'],'-m','accelerate.commands.launch','--config_file',
        str(source/f'scripts/accelerate_configs/accelerate_zero1_ds_{world}gpu.yaml'),
        '--num_processes',str(world),'--num_machines','1','--machine_rank','0',
        str(source/'scripts/forcewm/train_resolved.py'),'--config',config,'--origins',plan['origins']]
    if world==8:
        expected_argv += ['--expected-world-size','8']
    if plan['argv']!=expected_argv or not plan['native_python'].endswith('/realman-fastwam/.venv/bin/python'):
        raise ValueError('Exact native invocation for the authorized world size required')
    required={config,plan['preflight'],plan['probe_supervisor'],plan['preparation'],
        str(source/f'scripts/accelerate_configs/accelerate_zero1_ds_{world}gpu.yaml'),
        str(source/'scripts/ds_configs/ds_zero1_config.json')}
    required.update(str(p) for folder in ('src','scripts/forcewm') for p in (source/folder).rglob('*.py'))
    if not required.issubset(plan['input_sha256']):
        raise ValueError('All executed source/config/receipts must be hash-bound')
    if (plan['environment']['PYTHONPATH']!=str(source/'src')+':'+str(source)
            or plan['environment']['CUDA_VISIBLE_DEVICES']!=','.join(map(str,range(world)))
            or plan['environment']['DIFFSYNTH_SKIP_DOWNLOAD']!='true'):
        raise ValueError('Bound source and all authorized visible GPUs required')
    for path,digest in plan['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('Bound input changed: '+path)
    native=json.loads(Path(plan['preflight']).read_text())
    outer=json.loads(Path(plan['probe_supervisor']).read_text())
    preparation=json.loads(Path(plan['preparation']).read_text())
    if distributed or interactive:
        if (preparation.get('allocation_id')!=plan['allocation_id']
                or preparation.get('execution_mode')!=plan['execution_mode']
                or preparation.get('current_node_probe') is not True
                or preparation.get('probe_output')!=str(Path(plan['preflight']).parent)
                or outer.get('command')!=preparation.get('probe_argv')
                or outer.get('absolute_deadline_unix')!=(plan.get('probe_deadline_unix') if launch_relative else plan['deadline_unix'])):
            raise ValueError('Actual current-job probe and foreground provenance required')
        if launch_relative and (
                preparation.get('probe_deadline_unix')!=plan['probe_deadline_unix']
                or type(preparation.get('probe_started_unix')) not in (int,float)
                or not plan['allocation_created_unix']<=preparation['probe_started_unix']<=now
                or plan['probe_deadline_unix']-preparation['probe_started_unix']!=1860
                or not 0<outer.get('limit_seconds',0)<=1800):
            raise ValueError('Independent bounded fresh probe required before formal50h')
        if world==8 or interactive:
            expected_probe=expected_argv[:11]+['-m','fastwam.forcewm.preflight',
                '--config',config,'--output',str(Path(plan['preflight']).parent),
                '--expected-world-size',str(world),'--execute']
            if preparation['probe_argv']!=expected_probe:
                raise ValueError('Exact native eight-rank probe invocation required')
    if interactive:
        hardware=preparation.get('hardware','').strip().splitlines()
        if len(hardware)!=world or not all(('H100' if world==8 else 'H200') in row for row in hardware):
            raise ValueError('Current authorized H100/H200 hardware evidence required')
    if plan.get('restart_from_failed_plan'):
        # Reuse the real same-allocation GPU probe only for identical model and
        # training code. A new output directory and repaired outer guard do not
        # justify relabelling a copied receipt as a new GPU test.
        prior_path=plan['restart_from_failed_plan']
        if prior_path not in plan['input_sha256']:
            raise ValueError('Original failed plan must be hash-bound')
        prior=json.loads(Path(prior_path).read_text())
        prior_source=Path(prior['source']);prior_config=Path(prior['config'])
        if (prior['allocation_id']!=plan['allocation_id'] or prior['preflight']!=plan['preflight']
                or prior['probe_supervisor']!=plan['probe_supervisor']
                or prior['preparation']!=plan['preparation']
                or preparation['resolved_config_sha256']!=sha(prior_config)
                or sha(prior_config)!=prior['input_sha256'][str(prior_config)]):
            raise ValueError('Restart must reuse the actual same-allocation probe')
        old_lines=prior_config.read_text().splitlines();new_lines=Path(config).read_text().splitlines()
        if (not old_lines or not new_lines or new_lines[0]!='output_dir: '+plan['training_output']
                or old_lines[0]!='output_dir: '+prior['training_output'] or old_lines[1:]!=new_lines[1:]):
            raise ValueError('Restart may change only the output directory in GPU-validated config')
        for folder in ('src','scripts/forcewm'):
            old_files={str(p.relative_to(prior_source)) for p in (prior_source/folder).rglob('*.py')}
            new_files={str(p.relative_to(source)) for p in (source/folder).rglob('*.py')}
            if old_files!=new_files:raise ValueError('Restart source file set changed')
            for name in old_files-{'scripts/forcewm/process_ownership.py','scripts/forcewm/bounded_training.py'}:
                if sha(source/name)!=prior['input_sha256'][str(prior_source/name)]:
                    raise ValueError('GPU-validated training code changed: '+name)
        for name in ('scripts/accelerate_configs/accelerate_zero1_ds_4gpu.yaml','scripts/ds_configs/ds_zero1_config.json'):
            if sha(source/name)!=prior['input_sha256'][str(prior_source/name)]:
                raise ValueError('GPU-validated distributed config changed')
    elif preparation['resolved_config_sha256']!=sha(config):
        raise ValueError('Resolved config differs from actual GPU probe')
    if plan['forcewm_source_inventory']!=preparation['source_inventory']:
        raise ValueError('GPU-validated ForceWM inventory differs')
    for name,digest in preparation['source_inventory']['files'].items():
        if sha(source/'src/fastwam/forcewm'/name)!=digest:
            raise ValueError('GPU-validated model code changed')
    if (native['status']!='two_adam_verified' or len(native['ranks'])!=world
            or {r['rank'] for r in native['ranks']}!=set(range(world))):
        raise ValueError('All authorized ranks need actual Adam receipts')
    for rank in native['ranks']:
        if (rank['accepted'] is not True or rank['world_size']!=world or rank['global_batch']!=16*world
                or rank['batch_per_rank']!=16 or rank['actual_engine_updates']!=2
                or not rank['adam_state_steps'] or not all(x==2 for x in rank['adam_state_steps'])
                or rank['all_parameters_finite'] is not True or rank['adam_master_and_moments_finite'] is not True
                or len(rank['updates'])!=2 or any(u['optimizer_step_was_skipped'] is not False for u in rank['updates'])
                or any(not math.isfinite(u['global_loss']) for u in rank['updates'])
                or len(rank['teacher_layer_calls'])!=30 or min(rank['teacher_layer_calls'])<2
                or rank['dataset']['verified'] is not True):
            raise ValueError('Actual preflight recipe/update state differs')
    if (outer['status']!='process_exited_zero' or type(outer['returncode']) is not int or outer['returncode']!=0
            or not outer['subreaper_restored'] or not outer['ownership']['closure_verified']
            or any(outer['ownership'][k] for k in ('errors','signals','live_after_cleanup'))):
        raise ValueError('Probe full-tree unforced closure required')
    warm=max(r['updates'][-1]['seconds'] for r in native['ranks'])
    minimum=warm*30000*1.05+max(600,max(r['startup_seconds'] for r in native['ranks']))+5400+3600+60
    estimate=plan['estimated_required_seconds']
    if (type(estimate) not in (int,float) or not math.isfinite(estimate) or estimate<=0
            or not math.isfinite(minimum) or warm<=0 or remaining<max(minimum,estimate)):
        raise ValueError('Measured training plus 5% margin, startup, save/eval and cleanup do not fit')
    required_bytes=6*(sum(r['local_adam_master_and_moments_bytes'] for r in native['ranks'])
        +4*native['ranks'][0]['architecture_counts']['optimizer_parameters'])*1.2+10*2**30
    if shutil.disk_usage(Path(plan['training_output']).parent).free<required_bytes:
        raise ValueError('SSD lacks estimated six complete checkpoints plus headroom')
    if any(Path(plan[k]).exists() for k in ('training_output','origins','supervision_output')):
        raise ValueError('Existing formal output/origins cannot be overwritten or relaunched')
    return remaining-60


def verify_completion(plan):
    world=recipe_world_size(plan)
    output=Path(plan['training_output'])
    for rank in range(world):
        r=json.loads((Path(plan['origins'])/f'rank_{rank:03d}.json').read_text())
        if (r['status']!='complete' or r['rank']!=rank or r['world_size']!=world or r['max_steps']!=30000
                or r['config_sha256']!=plan['input_sha256'][plan['config']]
                or r['forcewm_source_inventory']!=plan['forcewm_source_inventory']
                or r['output']!=plan['training_output']):
            raise ValueError('All authorized rank origins must be complete')
        if world==8 and (r.get('batch_per_rank')!=16 or r.get('global_batch')!=128):
            raise ValueError('All eight origins must record B16/global128')
    state=output/'checkpoints/state/step_030000'
    if json.loads((state/'trainer_state.json').read_text())['global_step']!=30000:
        raise ValueError('Final trainer state missing')
    weight=output/'checkpoints/weights/step_030000.pt'
    if not weight.is_file() or weight.stat().st_size==0:
        raise ValueError('Final weights missing/empty')
    files=[f for f in state.rglob('*') if f.is_file()]
    names={f.name for f in files}
    expected={f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt' for rank in range(world)}
    expected.update(f'random_states_{rank}.pkl' for rank in range(world))
    if (any(f.stat().st_size==0 for f in files) or not expected.issubset(names)
            or not any('model_states' in name for name in names)):
        raise ValueError('Complete nonempty model, optimizer and all rank RNG states required')
    return {'status':'completed_30000','weights_bytes':weight.stat().st_size,
            'state_bytes':sum(f.stat().st_size for f in files),'distributed_resume_verified':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();plan=json.loads(Path(args.plan).read_text())
    remaining=validate_plan(plan,time.time())
    if not args.execute:
        print(json.dumps({'status':'plan_verified','seconds_available':remaining}));return
    os.environ.update(plan['environment'])
    launch_relative=plan['schema'] in (LAUNCH_SCHEMA,INTERACTIVE_SCHEMA,INTERACTIVE4_SCHEMA)
    result=execute(plan['argv'],plan['source'],plan['supervision_output'],remaining,
                   maximum_seconds=LAUNCH_DURATION_SECONDS if launch_relative else
                     (144000 if plan['schema']=='forcewm-h100-formal-v1' else 172800),
                   deadline_unix=None if launch_relative else plan['deadline_unix'],
                   duration_from_launch_seconds=LAUNCH_DURATION_SECONDS if launch_relative else None,
                   execution_identity={'kind':'interactive_modeling','allocation_id':plan['allocation_id']}
                     if plan['schema'] in (INTERACTIVE_SCHEMA,INTERACTIVE4_SCHEMA) else None,
                   log_name='training.log',phase='forcewm-formal30000')
    final={'status':'failed','supervisor_status':result['status'],'formal_training_started':True,
           'distributed_resume_verified':False}
    if result['status']=='process_exited_zero':
        try:final.update(verify_completion(plan))
        except Exception as error:final['error']=repr(error)
    Path(plan['supervision_output'],'completion.json').write_text(json.dumps(final,indent=2))
    print(json.dumps(final));raise SystemExit(0 if final['status']=='completed_30000' else 2)


if __name__=='__main__':main()
