"""Agent core: planning, safe execution, and integration glue.

This module provides a minimal planner that queries the LLM adapter and indexer
to propose actions. It contains a safe executor that requires explicit user
confirmation before running any shell or git-destructive commands.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

import requests

from src.indexer import discover_repo_files, Indexer
from src.git_utils import status as git_status, commit as git_commit


class Agent:
    def __init__(self, llm_url: str = 'http://127.0.0.1:5100'):
        self.llm_url = llm_url
        self.indexer = Indexer()

    def plan(self, prompt: str) -> str:
        # Ask the LLM server for a plan (simple generate call)
        r = requests.post(f"{self.llm_url}/generate", json={"prompt": prompt, "max_tokens": 256})
        if r.status_code != 200:
            return f"LLM error: {r.status_code} {r.text}"
        return r.json().get('output', '')

    def index_repo(self, root: Path | str = '.') -> int:
        files = discover_repo_files(Path(root))
        self.indexer.index_paths(files)
        return len(files)

    def show_status(self) -> str:
        return git_status()

    def safe_run(self, cmd: List[str], confirm: bool = False) -> dict:
        """Run a shell command only if confirm==True. Returns output dict."""
        if not confirm:
            return {'error': 'confirmation required', 'cmd': cmd}
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return {'rc': proc.returncode, 'out': proc.stdout, 'err': proc.stderr}

    def commit_changes(self, message: str, confirm: bool = False) -> dict:
        if not confirm:
            return {'error': 'confirmation required'}
        return git_commit(message)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=str, help='Prompt to ask the LLM for a plan')
    p.add_argument('--index', action='store_true')
    p.add_argument('--status', action='store_true')
    p.add_argument('--commit', type=str, help='Commit message (requires --confirm)')
    p.add_argument('--confirm', action='store_true')
    args = p.parse_args()
    agent = Agent()
    if args.index:
        n = agent.index_repo('.')
        print('Indexed', n)
    if args.plan:
        print(agent.plan(args.plan))
    if args.status:
        print(agent.show_status())
    if args.commit:
        print(agent.commit_changes(args.commit, confirm=args.confirm))
