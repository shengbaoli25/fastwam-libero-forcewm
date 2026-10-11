"""Bound a disposable probe and all descendants with the reviewed pidfd tracker."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime,timezone
from process_ownership import ProcessOwnership


def execute(command,cwd,output,seconds):
    if not 1<=seconds<=3600:raise ValueError('Probe must be bounded to at most one hour')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    end=time.monotonic()+seconds;tracker=ProcessOwnership();child=None;pending=[]
    handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    result={'status':'starting','started_at':datetime.now(timezone.utc).isoformat(),'command':command,'limit_seconds':seconds}
    def save():
        temp=output/'status.tmp';temp.write_text(json.dumps(result,indent=2));temp.replace(output/'status.json')
    try:
        for s in handlers:signal.signal(s,lambda s,f:pending.append(s))
        tracker.enable_subreaper()
        with (output/'probe.log').open('xb') as log:
            child=subprocess.Popen(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            tracker.attach_launcher(child,phase='forcewm-two-adam')
        tracker.register_launcher(child,phase='forcewm-two-adam');result['status']='running';save()
        while child.poll() is None:
            tracker.observe()
            if pending:raise RuntimeError('Probe interrupted')
            if time.monotonic()>=end:raise TimeoutError('Probe time limit reached')
            try:child.wait(timeout=.2)
            except subprocess.TimeoutExpired:pass
        result['returncode']=child.returncode
        closure=tracker.cleanup(end+60)
        if child.returncode!=0:raise RuntimeError('Native probe exited nonzero')
        if not closure['closure_verified'] or closure['signals'] or closure['errors']:
            raise RuntimeError('Successful probe requires unforced full descendant closure')
        result['status']='process_exited_zero'
    except BaseException as error:
        result.update(status='failed',error=str(error))
    finally:
        if tracker.enabled:
            result['ownership']=tracker.cleanup(end+60)
            try:tracker.restore_subreaper();result['subreaper_restored']=True
            except Exception as e:result.update(status='failed',cleanup_error=str(e))
        if child is not None:result['returncode']=child.poll()
        for s,h in handlers.items():signal.signal(s,h)
        result['finished_at']=datetime.now(timezone.utc).isoformat();save()
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--command-json',required=True);p.add_argument('--cwd',required=True)
    p.add_argument('--output',required=True);p.add_argument('--seconds',type=float,default=1800)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    command=json.loads(Path(a.command_json).read_text())
    if not isinstance(command,list) or not command or not all(isinstance(x,str) for x in command):p.error('JSON argv list required')
    if not a.execute:print(json.dumps({'mode':'plan_only','argv':command,'seconds':a.seconds}))
    else:
        result=execute(command,a.cwd,a.output,a.seconds)
        print(json.dumps(result));raise SystemExit(0 if result['status']=='process_exited_zero' else 2)
