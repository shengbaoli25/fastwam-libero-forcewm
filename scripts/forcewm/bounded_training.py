"""Launch one hash-bound 4-rank training under the user-authorized40h allocation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from bounded_probe import execute


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_plan(plan,now):
    if (plan['schema']!='forcewm-h100-formal-v1' or plan['world_size']!=4
            or plan['batch_per_rank']!=16 or plan['global_batch']!=64 or plan['max_steps']!=30000
            or plan['gradient_accumulation_steps']!=1):
        raise ValueError('Exact authorized ForceWM training recipe required')
    if plan['deadline_unix']-plan['allocation_created_unix']!=40*3600:
        raise ValueError('Deadline must remain allocation creation plus40h')
    remaining=plan['deadline_unix']-now
    source=Path(plan['source']); config=plan['config']
    expected_argv=[plan['native_python'],'-m','accelerate.commands.launch','--config_file',
        str(source/'scripts/accelerate_configs/accelerate_zero1_ds_4gpu.yaml'),
        '--num_processes','4','--num_machines','1','--machine_rank','0',
        str(source/'scripts/forcewm/train_resolved.py'),'--config',config,'--origins',plan['origins']]
    if plan['argv']!=expected_argv or not plan['native_python'].endswith('/realman-fastwam/.venv/bin/python'):
        raise ValueError('Exact native four-rank invocation required')
    required={config,plan['preflight'],plan['probe_supervisor'],plan['preparation'],
        str(source/'scripts/accelerate_configs/accelerate_zero1_ds_4gpu.yaml'),
        str(source/'scripts/ds_configs/ds_zero1_config.json')}
    required.update(str(p) for folder in ('src','scripts/forcewm') for p in (source/folder).rglob('*.py'))
    if not required.issubset(plan['input_sha256']):
        raise ValueError('All executed source/config/receipts must be hash-bound')
    if (plan['environment']['PYTHONPATH']!=str(source/'src')+':'+str(source)
            or plan['environment']['CUDA_VISIBLE_DEVICES']!='0,1,2,3'
            or plan['environment']['DIFFSYNTH_SKIP_DOWNLOAD']!='true'):
        raise ValueError('Bound source and four visible GPUs required')
    for path,digest in plan['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('Bound input changed: '+path)
    native=json.loads(Path(plan['preflight']).read_text())
    outer=json.loads(Path(plan['probe_supervisor']).read_text())
    preparation=json.loads(Path(plan['preparation']).read_text())
    if preparation['resolved_config_sha256']!=sha(config):
        raise ValueError('Resolved config differs from actual GPU probe')
    if plan['forcewm_source_inventory']!=preparation['source_inventory']:
        raise ValueError('GPU-validated ForceWM inventory differs')
    for name,digest in preparation['source_inventory']['files'].items():
        if sha(source/'src/fastwam/forcewm'/name)!=digest:
            raise ValueError('GPU-validated model code changed')
    if (native['status']!='two_adam_verified' or len(native['ranks'])!=4
            or {r['rank'] for r in native['ranks']}!=set(range(4))):
        raise ValueError('All four actual Adam receipts required')
    for rank in native['ranks']:
        if (rank['accepted'] is not True or rank['world_size']!=4 or rank['global_batch']!=64
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
    if not math.isfinite(minimum) or warm<=0 or remaining<max(minimum,plan['estimated_required_seconds']):
        raise ValueError('Measured training plus 5% margin, startup, save/eval and cleanup do not fit')
    required_bytes=6*(sum(r['local_adam_master_and_moments_bytes'] for r in native['ranks'])
        +4*native['ranks'][0]['architecture_counts']['optimizer_parameters'])*1.2+10*2**30
    if shutil.disk_usage(Path(plan['training_output']).parent).free<required_bytes:
        raise ValueError('SSD lacks estimated six complete checkpoints plus headroom')
    if any(Path(plan[k]).exists() for k in ('training_output','origins','supervision_output')):
        raise ValueError('Existing formal output/origins cannot be overwritten or relaunched')
    return remaining-60


def verify_completion(plan):
    output=Path(plan['training_output'])
    for rank in range(4):
        r=json.loads((Path(plan['origins'])/f'rank_{rank:03d}.json').read_text())
        if (r['status']!='complete' or r['rank']!=rank or r['world_size']!=4 or r['max_steps']!=30000
                or r['config_sha256']!=plan['input_sha256'][plan['config']]
                or r['forcewm_source_inventory']!=plan['forcewm_source_inventory']
                or r['output']!=plan['training_output']):
            raise ValueError('All four complete origins required')
    state=output/'checkpoints/state/step_030000'
    if json.loads((state/'trainer_state.json').read_text())['global_step']!=30000:
        raise ValueError('Final trainer state missing')
    weight=output/'checkpoints/weights/step_030000.pt'
    if not weight.is_file() or weight.stat().st_size==0:
        raise ValueError('Final weights missing/empty')
    files=[f for f in state.rglob('*') if f.is_file()]
    names={f.name for f in files}
    expected={f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt' for rank in range(4)}
    expected.update(f'random_states_{rank}.pkl' for rank in range(4))
    if (any(f.stat().st_size==0 for f in files) or not expected.issubset(names)
            or not any('model_states' in name for name in names)):
        raise ValueError('Complete nonempty model, optimizer and four RNG states required')
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
    result=execute(plan['argv'],plan['source'],plan['supervision_output'],remaining,
                   maximum_seconds=144000,deadline_unix=plan['deadline_unix'],
                   log_name='training.log',phase='forcewm-formal30000')
    final={'status':'failed','supervisor_status':result['status'],'formal_training_started':True,
           'distributed_resume_verified':False}
    if result['status']=='process_exited_zero':
        try:final.update(verify_completion(plan))
        except Exception as error:final['error']=repr(error)
    Path(plan['supervision_output'],'completion.json').write_text(json.dumps(final,indent=2))
    print(json.dumps(final));raise SystemExit(0 if final['status']=='completed_30000' else 2)


if __name__=='__main__':main()
