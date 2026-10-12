"""Real stdlib process tests, independent sessions; no GPU or production data."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('mode',['success','failure','deadline'])
@pytest.mark.parametrize('sessions',[4,8])
def test_all_sessions_closed(tmp_path,mode,sessions):
    worker='import time;time.sleep('+('0.4' if mode=='success' else '60')+')'
    code='import subprocess,sys,time,json,pathlib\n'
    code+=f'workers=[subprocess.Popen([sys.executable,"-c",{worker!r}],start_new_session=True) for _ in range({sessions})]\n'
    code+=f'pathlib.Path({str(tmp_path/"pids.json")!r}).write_text(json.dumps([p.pid for p in workers]))\n'
    code+='time.sleep(.2)\n'
    code+=('for p in workers:assert p.wait()==0\n' if mode=='success' else 'raise SystemExit(7)\n' if mode=='failure' else 'time.sleep(60)\n')
    launcher=tmp_path/'launcher.py';launcher.write_text(code)
    argv=tmp_path/'argv.json';argv.write_text(json.dumps(['/usr/bin/python3',str(launcher)]))
    out=tmp_path/'out'
    result=subprocess.run(['/usr/bin/python3',str(ROOT/'scripts/forcewm/bounded_probe.py'),
        '--command-json',str(argv),'--cwd',str(tmp_path),'--output',str(out),
        '--seconds','1.2' if mode=='deadline' else '10','--execute'],capture_output=True,text=True,timeout=20)
    status=json.loads((out/'status.json').read_text())
    assert status['subreaper_restored'] and status['ownership']['closure_verified'],result.stderr
    assert not status['ownership']['live_after_cleanup']
    assert all(not Path(f'/proc/{pid}').exists() for pid in json.loads((tmp_path/'pids.json').read_text()))
    if mode=='success':assert result.returncode==0 and not status['ownership']['signals']
    else:assert result.returncode!=0 and status['ownership']['signals']


@pytest.mark.parametrize('alteration',['duration','phase','preset_deadline','seconds','maximum'])
def test_relative_formal_clock_rejects_inconsistent_allowances(tmp_path,alteration):
    sys.path.insert(0,str(ROOT/'scripts/forcewm'))
    from bounded_probe import execute
    args=dict(maximum_seconds=180000,duration_from_launch_seconds=180000,phase='forcewm-formal30000')
    seconds=179940
    if alteration=='duration':args['duration_from_launch_seconds']=180001
    elif alteration=='phase':args['phase']='forcewm-two-adam'
    elif alteration=='preset_deadline':args['deadline_unix']=1
    elif alteration=='seconds':seconds-=1
    else:args['maximum_seconds']=172800
    with pytest.raises(ValueError):execute([],str(tmp_path),tmp_path/'never-launched',seconds,**args)
    assert not (tmp_path/'never-launched').exists()


def test_relative_formal_clock_is_actual_launch_bound_and_cannot_reset(tmp_path):
    # Real owned eight-session success, not a GPU/model result. Long allowance
    # does not make the short fixture wait; the child exits normally.
    runner=tmp_path/'runner.py'
    runner.write_text('import sys,json,time,pathlib\n'
        +f'sys.path.insert(0,{str(ROOT/"scripts/forcewm")!r})\n'
        +'from bounded_probe import execute\n'
        +'code="import subprocess,sys;children=[subprocess.Popen([sys.executable,\\"-c\\",\\"import time;time.sleep(.4)\\"],start_new_session=True) for _ in range(8)];assert all(p.wait()==0 for p in children)"\n'
        +f'out={str(tmp_path/"watch")!r}\n'
        +'before=time.time();result=execute([sys.executable,"-c",code],".",out,179940,maximum_seconds=180000,duration_from_launch_seconds=180000,phase="forcewm-formal30000")\n'
        +'receipt=json.loads(pathlib.Path(out,"formal-launch.json").read_text())\n'
        +'assert before<=receipt["launcher_started_unix"]<=time.time()\n'
        +'assert receipt["absolute_deadline_unix"]-receipt["launcher_started_unix"]==180000\n'
        +'assert receipt["launcher_start_ticks"]>0 and receipt["platform_job_id"]=="TEST-only-job"\n'
        +'assert result["status"]=="process_exited_zero",result\n'
        +'assert result["subreaper_restored"] and result["ownership"]["closure_verified"] and not result["ownership"]["signals"]\n'
        +'saved=pathlib.Path(out,"formal-launch.json").read_bytes()\n'
        +'try:execute([sys.executable,"-c",code],".",out,179940,maximum_seconds=180000,duration_from_launch_seconds=180000,phase="forcewm-formal30000")\n'
        +'except FileExistsError:pass\n'
        +'else:raise AssertionError("clock reset admitted")\n'
        +'assert pathlib.Path(out,"formal-launch.json").read_bytes()==saved\n')
    import os
    env=dict(os.environ,TRAIN_JOB_ID='TEST-only-job')
    r=subprocess.run(['/usr/bin/python3',str(runner)],capture_output=True,text=True,env=env,timeout=20)
    assert r.returncode==0,r.stdout+r.stderr


def test_legacy_guard_cannot_acquire_launch_relative50h_by_changing_only_maximum(tmp_path):
    sys.path.insert(0,str(ROOT/'scripts/forcewm'))
    from bounded_probe import execute
    with pytest.raises(ValueError):
        execute([],str(tmp_path),tmp_path/'never-launched',179940,maximum_seconds=180000)
    assert not (tmp_path/'never-launched').exists()
