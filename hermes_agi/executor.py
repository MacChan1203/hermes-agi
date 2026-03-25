from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from .memory import remember_successful_command, set_environment_info
from .agent_state import AgentState


class Executor:
    def __init__(self, repo_root: Path | str) -> None:
        self.repo_root = Path(repo_root)

    def execute(self, step: str, state: AgentState) -> Dict[str, Any]:
        python_bin = shlex.quote(sys.executable)

        if step == "Inspect project structure":
            cmd = f'pwd && {python_bin} --version && ls -la && find . -maxdepth 2 | sort | head -120'
        elif step == "Read README":
            cmd = 'if [ -f README.md ]; then sed -n "1,220p" README.md; else echo "README.md not found"; fi'
        elif step == "Read pyproject config":
            cmd = 'if [ -f pyproject.toml ]; then sed -n "1,240p" pyproject.toml; else echo "pyproject.toml not found"; fi'
        elif step == "Read requirements":
            cmd = 'if [ -f requirements.txt ]; then sed -n "1,220p" requirements.txt; else echo "requirements.txt not found"; fi'
        elif step == "Read core config files":
            cmd = (
                'for f in pyproject.toml requirements.txt README.md .env .env.example; do '
                'if [ -f "$f" ]; then echo "\\n===== $f ====="; sed -n "1,160p" "$f"; fi; '
                "done"
            )
        elif step == "Read main entry point":
            cmd = (
                'for f in run_agent.py main.py agent_runner.py cli.py hermes_agi/agent_runner.py; do '
                'if [ -f "$f" ]; then echo "\\n===== $f ====="; sed -n "1,240p" "$f"; fi; '
                "done"
            )
        elif step == "Inspect CLI entry point":
            cmd = (
                'for f in cli.py hermes_agi/cli.py; do '
                'if [ -f "$f" ]; then echo "\\n===== $f ====="; sed -n "1,220p" "$f"; fi; '
                "done"
            )
        elif step == "Inspect tests":
            cmd = 'if [ -d tests ]; then find tests -maxdepth 2 | sort | head -120; else echo "tests directory not found"; fi'
        elif step == "Inspect state store":
            cmd = 'if [ -f hermes_agi/state_store.py ]; then sed -n "1,260p" hermes_agi/state_store.py; else echo "state_store.py not found"; fi'
        elif step == "Inspect toolsets":
            cmd = 'if [ -f hermes_agi/toolsets.py ]; then sed -n "1,260p" hermes_agi/toolsets.py; else echo "toolsets.py not found"; fi'
        elif step == "Inspect tool distributions":
            cmd = 'if [ -f hermes_agi/toolset_distributions.py ]; then sed -n "1,260p" hermes_agi/toolset_distributions.py; else echo "toolset_distributions.py not found"; fi'
        elif step == "Inspect model tools":
            cmd = 'if [ -f hermes_agi/model_tools.py ]; then sed -n "1,260p" hermes_agi/model_tools.py; else echo "model_tools.py not found"; fi'
        elif step == "Inspect time handling":
            cmd = 'if [ -f hermes_agi/hermes_time.py ]; then sed -n "1,240p" hermes_agi/hermes_time.py; else echo "hermes_time.py not found"; fi'
        elif step == "Inspect constants":
            cmd = 'if [ -f hermes_agi/hermes_constants.py ]; then sed -n "1,220p" hermes_agi/hermes_constants.py; else echo "hermes_constants.py not found"; fi'
        elif step == "Inspect mini-swe-agent path support":
            cmd = 'if [ -f hermes_agi/minisweagent_path.py ]; then sed -n "1,240p" hermes_agi/minisweagent_path.py; else echo "minisweagent_path.py not found"; fi'
        elif step == "Summarize findings and propose next upgrade":
            return {
                "ok": True,
                "stdout": "Summary step is logical-only; no shell execution needed.",
                "stderr": "",
                "returncode": 0,
                "command": None,
            }
        elif step == "Check installed commands and PATH":
            cmd = 'echo "$PATH" && which python || true && which python3 || true && which pip || true'
        elif step == "Inspect file permissions":
            cmd = "pwd && ls -la"
        elif step == "Check Python environment and pip packages":
            cmd = f'{python_bin} --version && {python_bin} -m pip list | head -60'
        elif step.startswith("CMD:"):
            # LLM プランナーが生成した任意のシェルコマンド (最初の行のみ使用)
            cmd = step[len("CMD:"):].strip().splitlines()[0].strip()
        else:
            return {
                "ok": False,
                "stdout": "",
                "stderr": f"Unknown step: {step}",
                "returncode": 1,
                "command": None,
            }

        proc = subprocess.run(
            ["bash", "-lc", cmd],
            capture_output=True,
            text=True,
            cwd=str(self.repo_root),
        )

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        if proc.returncode == 0:
            remember_successful_command(state, cmd)

        lines = stdout.splitlines()

        cwd = None
        pyver = None

        if step == "Inspect project structure":
            cwd = lines[0].strip() if lines else None
            for line in lines[:8]:
                if line.lower().startswith("python "):
                    pyver = line.strip()
                    break
        elif step == "Check Python environment and pip packages":
            for line in lines[:8]:
                if line.lower().startswith("python "):
                    pyver = line.strip()
                    break

        set_environment_info(
            state,
            cwd=cwd,
            python_version=pyver,
            python_executable=sys.executable,
        )

        if step == "Inspect project structure":
            state.working_memory["project_structure_text"] = stdout

        return {
            "ok": proc.returncode == 0,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": proc.returncode,
            "command": cmd,
        }
