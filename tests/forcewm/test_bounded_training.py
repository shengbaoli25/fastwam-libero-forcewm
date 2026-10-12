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

def recovery_plan(plan):
    plan.update(schema='forcewm-recovery48h-v2',
        original_budget_anchor_unix=1791734545,allocation_created_unix=1791767345,
        deadline_unix=1791907345,
        recovery_of_allocation_id='b2c145b8-a54c-46e0-8580-919ec0549792')
    return plan

def test_recovery_preserves_original_fixed_48h_deadline(plan):
    p=recovery_plan(plan)
    assert guard.validate_plan(p,1791767345)==139940

@pytest.mark.parametrize('change',['reset_clock','anchor','allocation','future_created','nan_created','insufficient','eight_ranks'])
def test_recovery_rejects_budget_reset_and_recipe_drift(plan,change):
    p=recovery_plan(plan);now=1791767345
    if change=='reset_clock':p['deadline_unix']=p['allocation_created_unix']+48*3600
    elif change=='anchor':p['original_budget_anchor_unix']+=1
    elif change=='allocation':p['recovery_of_allocation_id']='another-allocation'
    elif change=='future_created':p['allocation_created_unix']=now+1
    elif change=='nan_created':p['allocation_created_unix']=float('nan')
    elif change=='insufficient':now+=14000
    elif change=='eight_ranks':p['world_size']=8;p['global_batch']=128
    with pytest.raises(ValueError):guard.validate_plan(p,now)

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
    world = plan['world_size']
    origins=Path(plan['origins']);origins.mkdir()
    for r in range(world):
        (origins/f'rank_{r:03d}.json').write_text(json.dumps({'rank':r,'status':'complete',
          'world_size':world,'batch_per_rank':16,'global_batch':16*world,'max_steps':30000,'config_sha256':plan['input_sha256'][plan['config']],
          'forcewm_source_inventory':plan['forcewm_source_inventory'],'output':plan['training_output']}))
    output=Path(plan['training_output']);state=output/'checkpoints/state/step_030000';state.mkdir(parents=True)
    (state/'trainer_state.json').write_text(json.dumps({'global_step':30000}))
    (state/'mp_rank_00_model_states.pt').write_text('TEST-only')
    for r in range(world):
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


def restart_fixture(plan,tmp_path):
    import shutil
    config=Path(plan['config']);config.write_text('output_dir: '+plan['training_output']+'\nmax_steps: 30000\nsave_freq: 5000\n')
    prep=Path(plan['preparation']);p=json.loads(prep.read_text());p['resolved_config_sha256']=guard.sha(config);prep.write_text(json.dumps(p))
    for f in (config,prep):plan['input_sha256'][str(f)]=guard.sha(f)
    prior=tmp_path/'prior.json';prior.write_text(json.dumps(plan))
    old_source=Path(plan['source']);source=tmp_path/'new-source';shutil.copytree(old_source,source)
    new=copy.deepcopy(plan);new.update(source=str(source),training_output=str(tmp_path/'new-output'),
        origins=str(tmp_path/'new-origins'),supervision_output=str(tmp_path/'new-watch'),restart_from_failed_plan=str(prior))
    new_config=tmp_path/'new.yaml';new_config.write_text(config.read_text().replace(plan['training_output'],new['training_output']))
    new['config']=str(new_config)
    new['environment']['PYTHONPATH']=str(source/'src')+':'+str(source)
    new['argv']=[s.replace(str(old_source),str(source)).replace(str(config),str(new_config)).replace(plan['origins'],new['origins']) for s in plan['argv']]
    new['input_sha256'].update({str(p):guard.sha(p) for p in source.rglob('*') if p.is_file()})
    for p in (new_config,prior):new['input_sha256'][str(p)]=guard.sha(p)
    return new


def test_restart_reuses_actual_probe_without_claiming_retest(plan,tmp_path):
    plan['allocation_id']='TEST-only'
    p=restart_fixture(plan,tmp_path)
    assert guard.validate_plan(p,1100)==143840

@pytest.mark.parametrize('drift',['recipe','model','distributed','prior_binding','different_allocation','probe'])
def test_restart_rejects_unvalidated_changes(plan,tmp_path,drift):
    plan['allocation_id']='TEST-only';p=restart_fixture(plan,tmp_path)
    if drift in ('recipe','model','distributed'):
        path=Path(p['config'] if drift=='recipe' else str(Path(p['source'])/('src/fastwam/forcewm/model.py' if drift=='model' else 'scripts/ds_configs/ds_zero1_config.json')))
        path.write_text(path.read_text()+'\nchanged');p['input_sha256'][str(path)]=guard.sha(path)
    elif drift=='prior_binding':del p['input_sha256'][p['restart_from_failed_plan']]
    elif drift=='different_allocation':p['allocation_id']='another'
    else:p['preflight']=p['preparation']
    with pytest.raises((ValueError,KeyError)):guard.validate_plan(p,1100)


def approved_50h_plan(plan,tmp_path):
    plan['allocation_id']='ab5d6993-795b-4bf7-9084-e59f089e7788'
    recovery_plan(plan)
    p=restart_fixture(plan,tmp_path)
    p.update(schema='forcewm-restart50h-v4',deadline_unix=1791914545,
             human_duration_approval='同意延长到50小时并重启')
    return p


def test_explicit_50h_restart_retains_original_budget_anchor(plan,tmp_path):
    p=approved_50h_plan(plan,tmp_path)
    assert guard.validate_plan(p,1791778000)==136485
    assert p['deadline_unix']-p['original_budget_anchor_unix']==50*3600

@pytest.mark.parametrize('change',['reset_clock','unapproved','allocation','anchor','insufficient','recipe'])
def test_50h_restart_rejects_unauthorized_drift(plan,tmp_path,change):
    p=approved_50h_plan(plan,tmp_path);now=1791778000
    if change=='reset_clock':p['deadline_unix']=now+50*3600
    elif change=='unapproved':p['human_duration_approval']=''
    elif change=='allocation':p['allocation_id']='different'
    elif change=='anchor':p['original_budget_anchor_unix']+=1
    elif change=='insufficient':now=p['deadline_unix']-1000
    else:p['global_batch']=128
    with pytest.raises(ValueError):guard.validate_plan(p,now)


def distributed_fixture(plan, monkeypatch):
    job = 'train-test-current-node'
    recovery_plan(plan)
    plan.update(schema='forcewm-distributed50h-v5', allocation_id=job,
                deadline_unix=1791914545, human_duration_approval='同意延长到50小时并重启',
                execution_mode='foreground_distributed_training')
    monkeypatch.setenv('TRAIN_JOB_ID', job)
    prep = Path(plan['preparation']); value = json.loads(prep.read_text())
    value.update(allocation_id=job, execution_mode=plan['execution_mode'], current_node_probe=True,
                 probe_output=str(Path(plan['preflight']).parent), probe_argv=['TEST-only-current-job'])
    prep.write_text(json.dumps(value)); plan['input_sha256'][str(prep)] = guard.sha(prep)
    outer = Path(plan['probe_supervisor']); value = json.loads(outer.read_text())
    value.update(command=['TEST-only-current-job'], absolute_deadline_unix=1791914545)
    outer.write_text(json.dumps(value)); plan['input_sha256'][str(outer)] = guard.sha(outer)
    return plan


def test_distributed_job_requires_fresh_probe_and_preserves_deadline(plan, monkeypatch):
    p = distributed_fixture(plan, monkeypatch)
    assert guard.validate_plan(p, 1791778000) == 136485


@pytest.mark.parametrize('drift', ['job_identity', 'old_probe', 'probe_path', 'probe_command',
                                 'deadline', 'background', 'copied_restart', 'missing_job', 'budget'])
def test_distributed_rejects_other_job_receipts_and_queue_budget_drift(plan, monkeypatch, drift):
    p = distributed_fixture(plan, monkeypatch); now = 1791778000
    if drift == 'job_identity': monkeypatch.setenv('TRAIN_JOB_ID', 'different-job')
    elif drift == 'missing_job': monkeypatch.delenv('TRAIN_JOB_ID')
    elif drift == 'background': p['execution_mode'] = 'background'
    elif drift == 'copied_restart': p['restart_from_failed_plan'] = 'old-notebook-plan'
    elif drift == 'budget': now = p['deadline_unix'] - 1000
    else:
        path = Path(p['preparation'] if drift in ('old_probe', 'probe_path') else p['probe_supervisor'])
        value = json.loads(path.read_text())
        if drift == 'old_probe': value['allocation_id'] = 'old-notebook'
        elif drift == 'probe_path': value['probe_output'] += '-different'
        elif drift == 'probe_command': value['command'] = ['another-probe']
        elif drift == 'deadline': value['absolute_deadline_unix'] += 1
        path.write_text(json.dumps(value)); p['input_sha256'][str(path)] = guard.sha(path)
    with pytest.raises(ValueError): guard.validate_plan(p, now)


def distributed8_fixture(plan, monkeypatch):
    p = distributed_fixture(plan, monkeypatch)
    p.update(schema='forcewm-distributed8-50h-v6', world_size=8, global_batch=128)
    source = Path(p['source'])
    accel = source / 'scripts/accelerate_configs/accelerate_zero1_ds_8gpu.yaml'
    accel.write_text('TEST-only-eight-ranks')
    p['input_sha256'][str(accel)] = guard.sha(accel)
    p['argv'][4] = str(accel); p['argv'][6] = '8'
    p['argv'] += ['--expected-world-size', '8']
    p['environment']['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    native = Path(p['preflight']); value = json.loads(native.read_text())
    rows = value['ranks']
    value['ranks'] = [dict(copy.deepcopy(rows[r % 4]), rank=r, world_size=8, global_batch=128) for r in range(8)]
    native.write_text(json.dumps(value)); p['input_sha256'][str(native)] = guard.sha(native)
    probe_argv = p['argv'][:11] + ['-m', 'fastwam.forcewm.preflight', '--config', p['config'],
        '--output', str(native.parent), '--expected-world-size', '8', '--execute']
    prep = Path(p['preparation']); value = json.loads(prep.read_text()); value['probe_argv'] = probe_argv
    prep.write_text(json.dumps(value)); p['input_sha256'][str(prep)] = guard.sha(prep)
    outer = Path(p['probe_supervisor']); value = json.loads(outer.read_text()); value['command'] = probe_argv
    outer.write_text(json.dumps(value)); p['input_sha256'][str(outer)] = guard.sha(outer)
    return p


def test_eight_rank_plan_requires_current_eight_rank_evidence(plan, monkeypatch):
    p = distributed8_fixture(plan, monkeypatch)
    assert guard.validate_plan(p, 1791778000) == 136485
    state = complete_fixture(p)
    assert guard.verify_completion(p)['status'] == 'completed_30000'
    assert guard.verify_completion(p)['distributed_resume_verified'] is False
    (state / 'bf16_zero_pp_rank_7_mp_rank_00_optim_states.pt').unlink()
    with pytest.raises(ValueError): guard.verify_completion(p)


@pytest.mark.parametrize('drift', ['four_receipts', 'duplicate_rank', 'rank_world', 'rank_global',
    'four_visible', 'four_formal_argv', 'four_probe_argv', 'unverified_schema', 'budget'])
def test_eight_rank_plan_rejects_partial_or_old_evidence(plan, monkeypatch, drift):
    p = distributed8_fixture(plan, monkeypatch); now = 1791778000
    if drift in ('four_receipts', 'duplicate_rank', 'rank_world', 'rank_global'):
        file = Path(p['preflight']); value = json.loads(file.read_text())
        if drift == 'four_receipts': value['ranks'] = value['ranks'][:4]
        elif drift == 'duplicate_rank': value['ranks'][7]['rank'] = 6
        elif drift == 'rank_world': value['ranks'][7]['world_size'] = 4
        else: value['ranks'][7]['global_batch'] = 64
        file.write_text(json.dumps(value)); p['input_sha256'][str(file)] = guard.sha(file)
    elif drift == 'four_visible': p['environment']['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    elif drift == 'four_formal_argv': p['argv'][6] = '4'
    elif drift == 'four_probe_argv':
        for key in ('preparation', 'probe_supervisor'):
            file = Path(p[key]); value = json.loads(file.read_text())
            value['probe_argv' if key == 'preparation' else 'command'][6] = '4'
            file.write_text(json.dumps(value)); p['input_sha256'][str(file)] = guard.sha(file)
    elif drift == 'unverified_schema': p['schema'] = 'forcewm-distributed50h-v5'
    else: now = p['deadline_unix'] - 1000
    with pytest.raises(ValueError): guard.validate_plan(p, now)


@pytest.mark.parametrize('missing', ['origin7', 'rng7'])
def test_eight_rank_completion_rejects_missing_upper_rank(plan, monkeypatch, missing):
    p = distributed8_fixture(plan, monkeypatch); state = complete_fixture(p)
    file = Path(p['origins']) / 'rank_007.json' if missing == 'origin7' else state / 'random_states_7.pkl'
    file.unlink()
    with pytest.raises((FileNotFoundError, ValueError)): guard.verify_completion(p)


def launch_relative_fixture(plan, monkeypatch):
    p=distributed8_fixture(plan,monkeypatch)
    p.update(schema=guard.LAUNCH_SCHEMA,budget_anchor='formal_launcher_start',duration_seconds=180000,
        deadline_unix=None,allocation_created_unix=1792000000,
        human_duration_approval=guard.LAUNCH_APPROVAL,probe_deadline_unix=1792001960)
    del p['original_budget_anchor_unix']
    prep=Path(p['preparation']);value=json.loads(prep.read_text())
    value.update(probe_started_unix=1792000100,probe_deadline_unix=p['probe_deadline_unix'])
    prep.write_text(json.dumps(value));p['input_sha256'][str(prep)]=guard.sha(prep)
    outer=Path(p['probe_supervisor']);value=json.loads(outer.read_text())
    value.update(absolute_deadline_unix=p['probe_deadline_unix'],limit_seconds=1799)
    outer.write_text(json.dumps(value));p['input_sha256'][str(outer)]=guard.sha(outer)
    return p


def test_launch_relative50h_survives_queue_past_old_absolute_deadline(plan,monkeypatch):
    p=launch_relative_fixture(plan,monkeypatch)
    assert guard.validate_plan(p,1792000200)==179940
    assert guard.validate_plan(p,1792001000)==179940


@pytest.mark.parametrize('drift',['duration','anchor','prior_anchor','preset_deadline','preset_start',
    'future_created','nan_created','nan_now','approval','wrong_job','copied_restart',
    'probe_deadline','prep_probe_deadline','probe_start','unbounded_probe','estimated_nan',
    'too_slow','relaunch_output','relaunch_watch'])
def test_launch_relative_rejects_reset_and_unbound_evidence(plan,monkeypatch,drift):
    p=launch_relative_fixture(plan,monkeypatch);now=1792000200
    if drift=='duration':p['duration_seconds']+=3600
    elif drift=='anchor':p['budget_anchor']='allocation_created'
    elif drift=='prior_anchor':p['original_budget_anchor_unix']=1791734545
    elif drift=='preset_deadline':p['deadline_unix']=now+180000
    elif drift=='preset_start':p['formal_launcher_started_unix']=now
    elif drift=='future_created':p['allocation_created_unix']=now+1
    elif drift=='nan_created':p['allocation_created_unix']=float('nan')
    elif drift=='nan_now':now=float('nan')
    elif drift=='approval':p['human_duration_approval']='old fixed50h approval'
    elif drift=='wrong_job':monkeypatch.setenv('TRAIN_JOB_ID','different-job')
    elif drift=='copied_restart':p['restart_from_failed_plan']='old-plan'
    elif drift=='estimated_nan':p['estimated_required_seconds']=float('nan')
    elif drift.startswith('relaunch_'):
        Path(p['training_output' if drift=='relaunch_output' else 'supervision_output']).mkdir()
    else:
        key='preflight' if drift=='too_slow' else 'preparation' if drift in ('prep_probe_deadline','probe_start') else 'probe_supervisor'
        path=Path(p[key]);v=json.loads(path.read_text())
        if drift=='too_slow':v['ranks'][7]['updates'][-1]['seconds']=6.
        elif drift in ('probe_deadline','prep_probe_deadline'):
            v['absolute_deadline_unix' if drift=='probe_deadline' else 'probe_deadline_unix']+=1
        elif drift=='probe_start':v['probe_started_unix']=now+1
        else:v['limit_seconds']=3600
        path.write_text(json.dumps(v));p['input_sha256'][str(path)]=guard.sha(path)
    with pytest.raises(ValueError):guard.validate_plan(p,now)


def interactive_fixture(plan, monkeypatch):
    p=launch_relative_fixture(plan,monkeypatch)
    p.update(schema=guard.INTERACTIVE_SCHEMA,allocation_id=guard.INTERACTIVE_ALLOCATION,
        execution_mode='foreground_interactive_modeling')
    monkeypatch.delenv('TRAIN_JOB_ID')
    monkeypatch.setenv('FORCEWM_INTERACTIVE_ALLOCATION_ID',guard.INTERACTIVE_ALLOCATION)
    f=Path(p['preparation']);v=json.loads(f.read_text())
    v.update(allocation_id=p['allocation_id'],execution_mode=p['execution_mode'],
        hardware='\n'.join(f'{r}, H100, TEST-only' for r in range(8)))
    f.write_text(json.dumps(v));p['input_sha256'][str(f)]=guard.sha(f)
    return p


def test_interactive_current_eight_h100_plan(plan,monkeypatch):
    p=interactive_fixture(plan,monkeypatch)
    assert guard.validate_plan(p,1792000200)==179940


@pytest.mark.parametrize('drift',['identity','mode','missing_binding','distributed_impersonation',
    'old_receipt','old_gpu','four_gpu','old_probe_deadline'])
def test_interactive_rejects_old_or_mismatched_allocation(plan,monkeypatch,drift):
    p=interactive_fixture(plan,monkeypatch)
    if drift=='identity':p['allocation_id']='other-notebook'
    elif drift=='mode':p['execution_mode']='foreground_distributed_training'
    elif drift=='missing_binding':monkeypatch.delenv('FORCEWM_INTERACTIVE_ALLOCATION_ID')
    elif drift=='distributed_impersonation':monkeypatch.setenv('TRAIN_JOB_ID','spoofed-job')
    else:
        f=Path(p['preparation']);v=json.loads(f.read_text())
        if drift=='old_receipt':v['allocation_id']='old-notebook'
        elif drift=='old_gpu':v['hardware']=v['hardware'].replace('H100','H200')
        elif drift=='four_gpu':v['hardware']='\n'.join(v['hardware'].splitlines()[:4])
        else:v['probe_deadline_unix']-=1
        f.write_text(json.dumps(v));p['input_sha256'][str(f)]=guard.sha(f)
    with pytest.raises(ValueError):guard.validate_plan(p,1792000200)


def interactive4_fixture(plan,monkeypatch):
    p=interactive_fixture(plan,monkeypatch)
    p.update(schema=guard.INTERACTIVE4_SCHEMA,allocation_id=guard.INTERACTIVE4_ALLOCATION,
        world_size=4,global_batch=64)
    monkeypatch.setenv('FORCEWM_INTERACTIVE_ALLOCATION_ID',guard.INTERACTIVE4_ALLOCATION)
    p['environment']['CUDA_VISIBLE_DEVICES']='0,1,2,3'
    p['argv'][4]=p['argv'][4].replace('_8gpu','_4gpu');p['argv'][6]='4';p['argv']=p['argv'][:-2]
    f=Path(p['preflight']);v=json.loads(f.read_text());v['ranks']=v['ranks'][:4]
    for r in v['ranks']:r.update(world_size=4,global_batch=64)
    f.write_text(json.dumps(v));p['input_sha256'][str(f)]=guard.sha(f)
    probe=p['argv'][:11]+['-m','fastwam.forcewm.preflight','--config',p['config'],
        '--output',str(f.parent),'--expected-world-size','4','--execute']
    f=Path(p['preparation']);v=json.loads(f.read_text())
    v.update(allocation_id=p['allocation_id'],hardware='\n'.join(f'{r}, H200, TEST-only' for r in range(4)),probe_argv=probe)
    f.write_text(json.dumps(v));p['input_sha256'][str(f)]=guard.sha(f)
    f=Path(p['probe_supervisor']);v=json.loads(f.read_text());v['command']=probe
    f.write_text(json.dumps(v));p['input_sha256'][str(f)]=guard.sha(f)
    return p


def test_current_four_h200_launch_relative_contract(plan,monkeypatch):
    p=interactive4_fixture(plan,monkeypatch)
    assert guard.validate_plan(p,1792000200)==179940
    p['world_size']=8;p['global_batch']=128
    with pytest.raises(ValueError):guard.validate_plan(p,1792000200)
