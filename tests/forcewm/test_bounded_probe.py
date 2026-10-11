"""Real stdlib process tests, independent sessions; no GPU or production data."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('mode',['success','failure','deadline'])
def test_all_four_sessions_closed(tmp_path,mode):
    worker='import time;time.sleep('+('0.4' if mode=='success' else '60')+')'
    code='import subprocess,sys,time,json,pathlib\n'
    code+=f'workers=[subprocess.Popen([sys.executable,"-c",{worker!r}],start_new_session=True) for _ in range(4)]\n'
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
