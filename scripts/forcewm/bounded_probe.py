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


def execute(command,cwd,output,seconds,*,maximum_seconds=3600,deadline_unix=None,log_name='probe.log',phase='forcewm-two-adam',duration_from_launch_seconds=None,execution_identity=None):
    if execution_identity is not None:
        if (duration_from_launch_seconds is None or not isinstance(execution_identity,dict)
                or execution_identity.get('kind')!='interactive_modeling'
                or execution_identity.get('allocation_id')!=os.environ.get('FORCEWM_INTERACTIVE_ALLOCATION_ID')
                or not execution_identity.get('allocation_id') or os.environ.get('TRAIN_JOB_ID')
                or set(execution_identity)!={'kind','allocation_id'}):
            raise ValueError('Explicit current interactive identity required; no distributed impersonation')
    cap=180000 if duration_from_launch_seconds is not None else 172800
    if not 1<=seconds<=maximum_seconds<=cap:raise ValueError('Execution exceeds its bounded time allowance')
    if duration_from_launch_seconds is not None:
        if (type(duration_from_launch_seconds) is not int
                or duration_from_launch_seconds != 180000 or maximum_seconds != 180000
                or seconds != 179940 or deadline_unix is not None
                or phase != 'forcewm-formal30000'):
            raise ValueError('Launch-relative allowance requires the authorized50h formal phase')
    if deadline_unix is not None:
        seconds=min(seconds,deadline_unix-time.time()-60)
        if seconds<1:raise ValueError('Absolute deadline has no execution and cleanup time remaining')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    end=time.monotonic()+seconds;tracker=ProcessOwnership();child=None;pending=[]
    handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    result={'status':'starting','started_at':datetime.now(timezone.utc).isoformat(),'command':command,'limit_seconds':seconds}
    def save():
        temp=output/'status.tmp';temp.write_text(json.dumps(result,indent=2));temp.replace(output/'status.json')
    def cleanup_deadline():
        return end+60 if deadline_unix is None else min(end+60,time.monotonic()+max(0,deadline_unix-time.time()))
    result['absolute_deadline_unix']=deadline_unix
    try:
        for s in handlers:signal.signal(s,lambda s,f:pending.append(s))
        tracker.enable_subreaper()
        with (output/log_name).open('xb') as log:
            launched_unix=time.time();launched_monotonic=time.monotonic()
            child=subprocess.Popen(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            if duration_from_launch_seconds is not None:
                deadline_unix=launched_unix+duration_from_launch_seconds
                end=launched_monotonic+seconds
                result.update(launcher_started_unix=launched_unix,
                    absolute_deadline_unix=deadline_unix,budget_anchor='formal_launcher_start',
                    duration_seconds=duration_from_launch_seconds)
            tracker.attach_launcher(child,phase=phase)
        launcher=tracker.register_launcher(child,phase=phase)
        if duration_from_launch_seconds is not None:
            # Exclusive output and receipt: this run cannot reset its clock.
            receipt={k:result[k] for k in ('command','launcher_started_unix',
                'absolute_deadline_unix','budget_anchor','duration_seconds')}
            receipt.update(launcher_pid=launcher['pid'],launcher_start_ticks=launcher['start_ticks'],
                           platform_job_id=os.environ.get('TRAIN_JOB_ID'),
                           execution_identity=execution_identity)
            with (output/'formal-launch.json').open('x') as stream:
                json.dump(receipt,stream,indent=2,allow_nan=False)
            print('FORMAL_ACTUAL_LAUNCH',json.dumps(receipt),flush=True)
        result['status']='running';save()
        while child.poll() is None:
            tracker.observe()
            if pending:raise RuntimeError('Probe interrupted')
            if time.monotonic()>=end or (deadline_unix is not None and time.time()>=deadline_unix-60):
                raise TimeoutError('Execution time limit reached')
            try:child.wait(timeout=.2)
            except subprocess.TimeoutExpired:pass
        result['returncode']=child.returncode
        closure=tracker.cleanup(cleanup_deadline())
        if child.returncode!=0:raise RuntimeError('Native probe exited nonzero')
        if not closure['closure_verified'] or closure['signals'] or closure['errors']:
            raise RuntimeError('Successful probe requires unforced full descendant closure')
        result['status']='process_exited_zero'
    except BaseException as error:
        result.update(status='failed',error=str(error))
    finally:
        if tracker.enabled:
            result['ownership']=tracker.cleanup(cleanup_deadline())
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
