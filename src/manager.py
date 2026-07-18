"""Process manager for local LLM runtime and server.

Provides start/stop/status helpers for the uvicorn FastAPI server and for
an optional llama.cpp native process. Uses subprocess and simple pidfiles.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional
import time

PID_DIR = Path('.cache') / 'pids'
PID_DIR.mkdir(parents=True, exist_ok=True)


def _write_pid(name: str, pid: int):
    (PID_DIR / f"{name}.pid").write_text(str(pid))


def _read_pid(name: str) -> Optional[int]:
    p = PID_DIR / f"{name}.pid"
    if not p.exists():
        return None
    try:
        return int(p.read_text().strip())
    except Exception:
        return None


def start_uvicorn(app_module: str = 'src.llm_server:app', host: str = '127.0.0.1', port: int = 5100) -> int:
    """Start the uvicorn server as a background process and record its PID."""
    cmd = [sys.executable, '-m', 'uvicorn', f'{app_module}', '--host', host, '--port', str(port), '--log-level', 'info']
    proc = subprocess.Popen(cmd)
    _write_pid('uvicorn', proc.pid)
    # give it a moment
    time.sleep(0.5)
    return proc.pid


def stop_process(name: str) -> bool:
    pid = _read_pid(name)
    if not pid:
        return False
    try:
        import psutil
        p = psutil.Process(pid)
        p.terminate()
        p.wait(timeout=5)
        (PID_DIR / f"{name}.pid").unlink(missing_ok=True)
        return True
    except Exception:
        return False


def status(name: str) -> dict:
    pid = _read_pid(name)
    alive = False
    if pid:
        try:
            import psutil
            alive = psutil.pid_exists(pid)
        except Exception:
            alive = False
    return {'name': name, 'pid': pid, 'alive': alive}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['start', 'stop', 'status'])
    args = p.parse_args()
    if args.cmd == 'start':
        pid = start_uvicorn()
        print('started uvicorn pid', pid)
    elif args.cmd == 'stop':
        ok = stop_process('uvicorn')
        print('stopped' if ok else 'not running')
    else:
        print(status('uvicorn'))
