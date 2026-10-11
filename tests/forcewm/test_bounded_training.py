"""TEST-only receipt fixtures; these do not establish GPU or save/load success."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import time
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/forcewm'))
import bounded_training as guard
from bounded_probe import execute

@pytest.fixture
def plan(tmp_path):
    source=tmp_path/'source'
    for name in ('src/fastwam/forcewm/model.py','scripts/forcewm/train_resolved.py',
                 'scripts/accelerate_configs/accelerate_zero1_ds_4gpu.yaml','scripts/ds_configs/ds_zero1_config.json'):
        file=source/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_text('TEST-only')
    config=tmp_path/'resolved.yaml';config.write_text('TEST-only config')
    inventory={'files':{'model.py':guard.sha(source/'src/fastwam/forcewm/model.py')},'sha256':'TEST-only'}
    preparation=tmp_path/'preparation.json';preparation.write_text(json.dumps({
        'resolved_config_sha256':guard.sha(config),'source_inventory':inventory}))
    ranks=[{'rank':r,'world_size':4,'global_batch':64,'batch_per_rank':16,'actual_engine_updates':2,
        'adam_state_steps':[2], 'accepted':True,'all_parameters_finite':True,'adam_master_and_moments_finite':True,
        'updates':[{'global_loss':1.,'optimizer_step_was_skipped':False,'seconds':3.83} for _ in range(2)],
        'teacher_layer_calls':[2]*30,'dataset':{'verified':True},'startup_seconds':170,
        'local_adam_master_and_moments_bytes':12,'architecture_counts':{'optimizer_parameters':1}}
        for r in range(4)]
    native=tmp_path/'preflight.json';native.write_text(json.dumps({'status':'two_adam_verified','ranks':ranks}))
    outer=tmp_path/'status.json';outer.write_text(json.dumps({'status':'process_exited_zero','returncode':0,
        'subreaper_restored':True,'ownership':{'closure_verified':True,'errors':[],'signals':[],'live_after_cleanup':[]}}))
    p={'schema':'forcewm-h100-formal-v1','world_size':4,'batch_per_rank':16,'global_batch':64,
       'max_steps':30000,'gradient_accumulation_steps':1,'source':str(source),'config':str(config),
       'native_python':'/test/realman-fastwam/.venv/bin/python','allocation_created_unix':1000,
       'deadline_unix':145000,'estimated_required_seconds':131000,'training_output':str(tmp_path/'out'),
       'origins':str(tmp_path/'origins'),'supervision_output':str(tmp_path/'watch'),
       'preflight':str(native),'probe_supervisor':str(outer),'preparation':str(preparation),
       'forcewm_source_inventory':inventory,
       'environment':{'PYTHONPATH':str(source/'src')+':'+str(source),'CUDA_VISIBLE_DEVICES':'0,1,2,3',
                      'DIFFSYNTH_SKIP_DOWNLOAD':'true'}}
    p['argv']=[p['native_python'],'-m','accelerate.commands.launch','--config_file',
        str(source/'scripts/accelerate_configs/accelerate_zero1_ds_4gpu.yaml'),'--num_processes','4',
        '--num_machines','1','--machine_rank','0',str(source/'scripts/forcewm/train_resolved.py'),
        '--config',str(config),'--origins',p['origins']]
    p['input_sha256']={str(f):guard.sha(f) for f in tmp_path.rglob('*') if f.is_file()}
    return p

def test_valid_plan(plan):
    assert guard.validate_plan(plan,1100)==143840

@pytest.mark.parametrize('change',['deadline','budget','argv','environment','missing_binding','tamper','relaunch','nan','skipped','forced','bool_rc'])
def test_reject_drift(plan,change):
    if change=='deadline':plan['deadline_unix']+=1
    elif change=='budget':plan['allocation_created_unix']-=20000;plan['deadline_unix']-=20000
    elif change=='argv':plan['argv'][0]='/usr/bin/python3'
    elif change=='environment':plan['environment']['CUDA_VISIBLE_DEVICES']='0,1'
    elif change=='missing_binding':del plan['input_sha256'][plan['config']]
    elif change=='tamper':Path(plan['config']).write_text('changed')
    elif change=='relaunch':Path(plan['supervision_output']).mkdir()
    else:
        key='preflight' if change in ('nan','skipped') else 'probe_supervisor'
        p=Path(plan[key]);value=json.loads(p.read_text())
        if change=='nan':value['ranks'][0]['updates'][0]['global_loss']=float('nan')
        elif change=='skipped':value['ranks'][0]['updates'][0]['optimizer_step_was_skipped']=True
        elif change=='forced':value['ownership']['signals']=[{'signal':15}]
        else:value['returncode']=False
        p.write_text(json.dumps(value));plan['input_sha256'][str(p)]=guard.sha(p)
    with pytest.raises(ValueError):guard.validate_plan(plan,1100)

def complete_fixture(plan):
    origins=Path(plan['origins']);origins.mkdir()
    for r in range(4):
        (origins/f'rank_{r:03d}.json').write_text(json.dumps({'rank':r,'status':'complete',
          'world_size':4,'max_steps':30000,'config_sha256':plan['input_sha256'][plan['config']],
          'forcewm_source_inventory':plan['forcewm_source_inventory'],'output':plan['training_output']}))
    output=Path(plan['training_output']);state=output/'checkpoints/state/step_030000';state.mkdir(parents=True)
    (state/'trainer_state.json').write_text(json.dumps({'global_step':30000}))
    (state/'mp_rank_00_model_states.pt').write_text('TEST-only')
    for r in range(4):
        (state/f'bf16_zero_pp_rank_{r}_mp_rank_00_optim_states.pt').write_text('TEST-only')
        (state/f'random_states_{r}.pkl').write_text('TEST-only')
    weight=output/'checkpoints/weights/step_030000.pt';weight.parent.mkdir();weight.write_text('TEST-only')
    return state

def test_completion_does_not_claim_restore(plan):
    complete_fixture(plan)
    assert guard.verify_completion(plan)['distributed_resume_verified'] is False

@pytest.mark.parametrize('missing',['random_states_3.pkl','bf16_zero_pp_rank_2_mp_rank_00_optim_states.pt','mp_rank_00_model_states.pt'])
def test_completion_rejects_incomplete_state(plan,missing):
    state=complete_fixture(plan);(state/missing).unlink()
    with pytest.raises(ValueError):guard.verify_completion(plan)

def test_expired_deadline_never_launches(tmp_path):
    with pytest.raises(ValueError):
        execute(['/usr/bin/python3','-c','raise SystemExit(99)'],str(tmp_path),tmp_path/'no',100,
                maximum_seconds=144000,deadline_unix=time.time()+30)
    assert not (tmp_path/'no').exists()

def test_absolute_deadline_closes_owned_session(tmp_path):
    code='import subprocess,sys,time;subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"],start_new_session=True);time.sleep(60)'
    # The ML venv lacks pidfd_send_signal; run the production stdlib supervisor
    # with the system Python, just as the actual platform invocation does.
    probe='import sys,time,json;sys.path.insert(0,'+repr(str(ROOT/'scripts/forcewm'))+');from bounded_probe import execute;'
    probe+='r=execute('+repr(['/usr/bin/python3','-c',code])+','+repr(str(tmp_path))+','+repr(str(tmp_path/'watch'))+',5000,maximum_seconds=144000,deadline_unix=time.time()+61.2,log_name="training.log");print(json.dumps(r))'
    result=subprocess.run(['/usr/bin/python3','-c',probe],capture_output=True,text=True,timeout=10,check=True)
    r=json.loads(result.stdout)
    assert r['status']=='failed' and r['ownership']['closure_verified']
    assert r['ownership']['signals'] and not r['ownership']['live_after_cleanup']
    assert r['subreaper_restored']
