"""TEST-only injected /proc reuse; never GPU validation."""
import sys
from pathlib import Path
from types import SimpleNamespace
import signal
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/forcewm'))
import process_ownership as m

def row(pid,start,ppid=1):
 return dict(pid=pid,start_ticks=start,ppid=ppid,pgid=pid,session=pid,state='S')
def record(r):
 return dict(r,initial_ppid=r['ppid'],initial_pgid=r['pgid'],initial_session=r['session'],ancestry=[],discovered_by='observed_descendant',adopted=False)
@pytest.fixture
def setup(monkeypatch):
 t=m.ProcessOwnership();t.enabled=True
 owner=dict(t.owner,state='S',ppid=1);start=owner['start_ticks']+10
 old=row(2147483001,start,owner['pid']);t.records[old['pid']]=record(old);t.current[old['pid']]=old
 scan={owner['pid']:owner}
 monkeypatch.setattr(m,'scan_processes',lambda:dict(scan))
 monkeypatch.setattr(m,'read_proc',lambda pid:scan.get(pid))
 return t,old,scan

def test_unrelated_replacement_does_not_stop_or_inherit(setup):
 t,old,scan=setup;scan[old['pid']]=row(old['pid'],old['start_ticks']+1)
 assert t.observe();assert not t.current and not t.records and not t.errors
 e=t.evidence();assert e['retired_identity_count']==1
 assert e['tracked_processes'][0]['start_ticks']==old['start_ticks']
 assert e['tracked_processes'][0]['replacement_identity']['start_ticks']==old['start_ticks']+1
 t.observe();assert not t.current and not t.errors

@pytest.mark.parametrize('adopted',[False,True])
def test_owned_replacement_gets_new_ancestry(setup,adopted):
 t,old,scan=setup
 parent=row(2147483002,old['start_ticks'],t.owner['pid']);t.records[parent['pid']]=record(parent);scan[parent['pid']]=parent
 if adopted:t.launcher=SimpleNamespace(pid=parent['pid'],returncode=None,poll=lambda:None)
 new=row(old['pid'],old['start_ticks']+1,t.owner['pid'] if adopted else parent['pid']);scan[new['pid']]=new
 t.observe();assert t.current[new['pid']]['start_ticks']==new['start_ticks']
 assert t.records[new['pid']]['ancestry'][-1]['pid']==new['ppid']
 assert len([r for r in t.evidence()['tracked_processes'] if r['pid']==new['pid']])==2
 assert not t.errors

def test_multiple_generations_preserved(setup):
 t,old,scan=setup;t.launcher=SimpleNamespace(pid=999,returncode=0,poll=lambda:0)
 for offset in range(1,5):
  scan[old['pid']]=row(old['pid'],old['start_ticks']+offset,t.owner['pid']);t.observe()
 assert t.evidence()['retired_identity_count']==4
 assert len(t.evidence()['tracked_processes'])==5
 assert not t.errors

def test_backwards_start_is_corruption(setup):
 t,old,scan=setup;scan[old['pid']]=row(old['pid'],old['start_ticks']-1)
 with pytest.raises(m.OwnershipError):t.observe()
 assert t.errors

def test_unreaped_launcher_cannot_reuse(setup):
 t,old,scan=setup;t.records[old['pid']]['discovered_by']='registered_launcher'
 t.launcher=SimpleNamespace(pid=old['pid'],returncode=None,poll=lambda:None)
 scan[old['pid']]=row(old['pid'],old['start_ticks']+1)
 with pytest.raises(m.OwnershipError,match='Unreaped launcher'):t.observe()

@pytest.mark.parametrize('operation',['signal','reap'])
def test_reuse_between_scan_and_action_never_touches_replacement(setup,monkeypatch,operation):
 t,old,scan=setup;scan[old['pid']]=row(old['pid'],old['start_ticks']+1)
 calls=[];monkeypatch.setattr(m.os,'pidfd_open',lambda *a:123)
 monkeypatch.setattr(m.os,'close',lambda fd:calls.append(('close',fd)))
 monkeypatch.setattr(m.signal,'pidfd_send_signal',lambda *a:calls.append(('signal',a)))
 monkeypatch.setattr(m.os,'waitpid',lambda *a:calls.append(('reap',a)))
 if operation=='signal':t._signal_pid(old,signal.SIGTERM,'TEST-only')
 else:t._reap_adopted()
 assert not any(c[0] in ['signal','reap'] for c in calls)
 assert not t.errors and not t.current
 assert t.evidence()['retired_identity_count']==1

def test_reused_parent_cannot_donate_ancestry(setup,monkeypatch):
 t,old,scan=setup;scan[old['pid']]=old
 child=row(2147483003,old['start_ticks']+2,old['pid']);scan[child['pid']]=child
 monkeypatch.setattr(m,'read_proc',lambda pid:row(pid,old['start_ticks']+1) if pid==old['pid'] else scan.get(pid))
 t.observe();assert child['pid'] not in t.records

def test_stale_action_cannot_retire_new_owned_identity(setup,monkeypatch):
 t,old,scan=setup;t.launcher=SimpleNamespace(pid=999,returncode=0,poll=lambda:0)
 new=row(old['pid'],old['start_ticks']+1,t.owner['pid']);scan[new['pid']]=new;t.observe()
 monkeypatch.setattr(m.os,'pidfd_open',lambda *a:123);monkeypatch.setattr(m.os,'close',lambda fd:None)
 monkeypatch.setattr(m.signal,'pidfd_send_signal',lambda *a:pytest.fail('must not signal replacement'))
 t._signal_pid(old,signal.SIGTERM,'TEST-only')
 assert t.current[new['pid']]['start_ticks']==new['start_ticks']
 assert not t.errors

@pytest.mark.parametrize('mode',['success','failure','deadline'])
def test_reuse_during_four_real_sessions(tmp_path,mode):
 """Inject a prior generation at an actual worker PID; use real signal/reap."""
 import json,subprocess
 worker='import time;time.sleep('+('0.6' if mode=='success' else '60')+')'
 launcher=tmp_path/'launcher.py'
 launcher.write_text('import subprocess,sys,json,pathlib,time\n'+
  f'ps=[subprocess.Popen([sys.executable,"-c",{worker!r}],start_new_session=True) for _ in range(4)]\n'+
  f'pathlib.Path({str(tmp_path/"workers.json")!r}).write_text(json.dumps([p.pid for p in ps]))\n'+
  ('for p in ps: assert p.wait()==0\n' if mode=='success' else 'time.sleep(.3);raise SystemExit(7)\n' if mode=='failure' else 'time.sleep(60)\n'))
 harness=tmp_path/'harness.py'
 harness.write_text('import sys,json,pathlib\n'+f'sys.path.insert(0,{str(ROOT/"scripts/forcewm")!r})\n'+
  'import process_ownership as m\nfrom bounded_probe import execute\noriginal=m.ProcessOwnership.observe\ninjected=False\n'+
  'def observe(self):\n global injected\n'+
  f' path=pathlib.Path({str(tmp_path/"workers.json")!r})\n'+
  ' if not injected and path.exists():\n'+
  '  for pid in json.loads(path.read_text()):\n'+
  '   row=m.read_proc(pid)\n'+
  '   if row is not None:\n'+
  '    old=dict(row);old["start_ticks"]-=1\n'+
  '    self.records[pid]=dict(old,initial_ppid=row["ppid"],initial_pgid=pid,initial_session=pid,ancestry=[],discovered_by="observed_descendant",adopted=False)\n'+
  '  injected=True\n return original(self)\n'+
  'm.ProcessOwnership.observe=observe\n'+
  f'r=execute(["/usr/bin/python3",{str(launcher)!r}],{str(tmp_path)!r},{str(tmp_path/"out")!r},{1.2 if mode=="deadline" else 10})\n'+
  f'pathlib.Path({str(tmp_path/"result.json")!r}).write_text(json.dumps(r))\n')
 run=subprocess.run(['/usr/bin/python3',str(harness)],capture_output=True,text=True,timeout=20)
 assert run.returncode==0,run.stderr
 q=json.loads((tmp_path/'result.json').read_text());e=q['ownership']
 assert e['retired_identity_count']==4
 assert q['subreaper_restored'] and e['closure_verified']
 assert not e['errors'] and not e['live_after_cleanup']
 pids=json.loads((tmp_path/'workers.json').read_text())
 assert all(not Path(f'/proc/{pid}').exists() for pid in pids)
 assert all(len([r for r in e['tracked_processes'] if r['pid']==pid])==2 for pid in pids)
 if mode=='success':assert q['returncode']==0 and not e['signals']
 else:assert q['status']=='failed' and e['signals']
