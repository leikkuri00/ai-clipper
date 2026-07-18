"""Safe Git utilities used by the agent.

Provides non-destructive wrappers around common git operations and requires
explicit confirmation for destructive actions like force pushes or resets.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List


def git_cmd(args: List[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    cmd = ['git'] + args
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)


def status(cwd: Path | None = None) -> str:
    r = git_cmd(['status', '--porcelain', '--branch'], cwd)
    return r.stdout + r.stderr


def diff(cwd: Path | None = None) -> str:
    r = git_cmd(['--no-pager', 'diff', '--staged', '--patch'], cwd)
    return r.stdout + r.stderr


def commit(message: str, cwd: Path | None = None) -> dict:
    add = git_cmd(['add', '--all'], cwd)
    c = git_cmd(['commit', '-m', message], cwd)
    return {'add': add.returncode, 'commit': c.returncode, 'out': c.stdout + c.stderr}


def current_branch(cwd: Path | None = None) -> str:
    r = git_cmd(['rev-parse', '--abbrev-ref', 'HEAD'], cwd)
    return r.stdout.strip()


def create_branch(name: str, cwd: Path | None = None) -> dict:
    r = git_cmd(['checkout', '-b', name], cwd)
    return {'rc': r.returncode, 'out': r.stdout + r.stderr}


def safe_reset_hard(cwd: Path | None = None, confirm: bool = False) -> dict:
    if not confirm:
        return {'error': 'confirmation required'}
    r = git_cmd(['reset', '--hard'], cwd)
    return {'rc': r.returncode, 'out': r.stdout + r.stderr}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--status', action='store_true')
    args = p.parse_args()
    if args.status:
        print(status())
