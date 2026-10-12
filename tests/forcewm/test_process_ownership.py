"""Exit-race regressions from the H100 incident; CPU/process tests only."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/forcewm"))
import process_ownership as ownership


def stat_row(*, pid=66897, state="Z", pgid=-1, session=-1, start=1906410121):
    # /proc fields from the actual error, with an intentionally tricky comm.
    fields = [state, "0", str(pgid), str(session), *(["0"] * 15), str(start)]
    return f"{pid} (TEST-only worker (exit)) " + " ".join(fields)


@pytest.mark.parametrize("state", ["Z", "X"])
@pytest.mark.parametrize("disappeared", [FileNotFoundError, ProcessLookupError])
def test_terminal_invalid_group_requires_observed_disappearance(monkeypatch, state, disappeared):
    reads = iter([stat_row(state=state), disappeared()])

    def read(path):
        assert str(path) == "/proc/66897/stat"
        value = next(reads)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(Path, "read_text", read)
    assert ownership.read_proc(66897) is None
    assert list(reads) == []


@pytest.mark.parametrize("state", ["Z", "X", "R", "S"])
def test_persistent_invalid_identity_still_fails_closed(monkeypatch, state):
    monkeypatch.setattr(Path, "read_text", lambda path: stat_row(state=state))
    with pytest.raises(ownership.OwnershipError, match="Invalid Linux process identity"):
        ownership.read_proc(66897)


@pytest.mark.parametrize("change", [{"pid": 66898}, {"start": -1}])
def test_terminal_state_does_not_excuse_corrupt_identity(monkeypatch, change):
    calls = []

    def read(path):
        calls.append(path)
        return stat_row(**change)

    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(ownership.OwnershipError, match="Invalid Linux process identity"):
        ownership.read_proc(66897)
    assert len(calls) == 1


def test_valid_zombie_is_preserved_for_owned_reaping(monkeypatch):
    monkeypatch.setattr(Path, "read_text", lambda path: stat_row(pgid=66897, session=66897))
    assert ownership.read_proc(66897) == {
        "pid": 66897, "state": "Z", "ppid": 0, "pgid": 66897,
        "session": 66897, "start_ticks": 1906410121,
    }


def test_valid_later_identity_after_terminal_race_is_returned_without_old_ownership(monkeypatch):
    reads = iter([stat_row(), stat_row(state="R", pgid=66897, session=66897, start=1906410122)])
    monkeypatch.setattr(Path, "read_text", lambda path: next(reads))
    actual = ownership.read_proc(66897)
    assert actual['start_ticks'] == 1906410122 and actual['state'] == 'R'


def test_permission_denied_on_confirmation_is_not_disappearance(monkeypatch):
    calls = []

    def read(path):
        calls.append(path)
        if len(calls) == 1:
            return stat_row()
        raise PermissionError("TEST-only")

    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(PermissionError):
        ownership.read_proc(66897)


def test_zombie_scan_race_does_not_terminate_four_healthy_sessions(tmp_path):
    """Inject only the /proc race while using real Popen/pidfd/subreaper work."""
    worker = "import time;time.sleep(.6)"
    launcher = tmp_path / "launcher.py"
    launcher.write_text(
        "import subprocess,sys,json,pathlib\n"
        f"workers=[subprocess.Popen([sys.executable,'-c',{worker!r}],start_new_session=True) for _ in range(4)]\n"
        f"pathlib.Path({str(tmp_path / 'workers.json')!r}).write_text(json.dumps([p.pid for p in workers]))\n"
        "for p in workers: assert p.wait()==0\n"
    )
    harness = tmp_path / "harness.py"
    harness.write_text(
        "import sys,json,pathlib\n"
        f"sys.path.insert(0,{str(ROOT / 'scripts/forcewm')!r})\n"
        "from bounded_probe import execute\n"
        "original_iter=pathlib.Path.iterdir;original_read=pathlib.Path.read_text\n"
        "injected_reads=0\n"
        "def iterdir(path):\n"
        "    yield from original_iter(path)\n"
        f"    if str(path)=='/proc' and pathlib.Path({str(tmp_path / 'workers.json')!r}).exists(): yield pathlib.Path('/proc/2147483647')\n"
        "def read(path,*args,**kwargs):\n"
        "    global injected_reads\n"
        "    if str(path)=='/proc/2147483647/stat':\n"
        "        injected_reads+=1\n"
        "        if injected_reads%2==0: raise FileNotFoundError('TEST-only exited process')\n"
        f"        return {stat_row(pid=2147483647)!r}\n"
        "    return original_read(path,*args,**kwargs)\n"
        "pathlib.Path.iterdir=iterdir;pathlib.Path.read_text=read\n"
        f"result=execute(['/usr/bin/python3',{str(launcher)!r}],{str(tmp_path)!r},{str(tmp_path / 'out')!r},10)\n"
        "result['test_only_injected_reads']=injected_reads\n"
        f"pathlib.Path({str(tmp_path / 'result.json')!r}).write_text(json.dumps(result))\n"
    )
    run = subprocess.run(["/usr/bin/python3", str(harness)], capture_output=True, text=True, timeout=20)
    assert run.returncode == 0, run.stderr
    status = json.loads((tmp_path / "result.json").read_text())
    assert status["status"] == "process_exited_zero", status.get("error")
    assert status["test_only_injected_reads"] >= 8
    assert status["returncode"] == 0 and status["subreaper_restored"]
    tree = status["ownership"]
    assert tree["closure_verified"] and not tree["signals"] and not tree["errors"]
    assert not tree["live_after_cleanup"]
    workers = json.loads((tmp_path / "workers.json").read_text())
    assert len(workers) == 4
    assert all(not Path(f"/proc/{pid}").exists() for pid in workers)
    recorded = {row["pid"]: row for row in tree["tracked_processes"]}
    assert all(recorded[pid]["initial_session"] == pid for pid in workers)
    assert 2147483647 not in recorded
