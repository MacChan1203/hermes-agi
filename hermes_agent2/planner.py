from __future__ import annotations

from pathlib import Path

from .agent_state import AgentState


class Planner:
    MAX_INSPECTION_STEPS = 6

    def bootstrap_plan(self, state: AgentState) -> None:
        if state.current_plan:
            return

        state.current_plan = [
            "Inspect project structure",
            "Plan next inspection steps",
        ]

    def _insert_if_missing(self, state: AgentState, step: str, planned_steps: list[str]) -> None:
        if step in planned_steps:
            return
        if step in state.completed_steps:
            return
        planned_steps.append(step)

    def _plan_from_structure(self, state: AgentState) -> None:
        structure_text = state.working_memory.get("project_structure_text", "") or ""

        planned_steps: list[str] = []

        # 優先度順
        if "README.md" in structure_text:
            self._insert_if_missing(state, "Read README", planned_steps)
        if "requirements.txt" in structure_text:
            self._insert_if_missing(state, "Read requirements", planned_steps)
        if "pyproject.toml" in structure_text:
            self._insert_if_missing(state, "Read pyproject config", planned_steps)

        if "run_agent.py" in structure_text or "main.py" in structure_text:
            self._insert_if_missing(state, "Read main entry point", planned_steps)
        if "cli.py" in structure_text:
            self._insert_if_missing(state, "Inspect CLI entry point", planned_steps)

        if "state_store.py" in structure_text:
            self._insert_if_missing(state, "Inspect state store", planned_steps)

        if "toolsets.py" in structure_text:
            self._insert_if_missing(state, "Inspect toolsets", planned_steps)
        if "toolset_distributions.py" in structure_text:
            self._insert_if_missing(state, "Inspect tool distributions", planned_steps)
        if "model_tools.py" in structure_text:
            self._insert_if_missing(state, "Inspect model tools", planned_steps)

        if "hermes_time.py" in structure_text:
            self._insert_if_missing(state, "Inspect time handling", planned_steps)
        if "hermes_constants.py" in structure_text:
            self._insert_if_missing(state, "Inspect constants", planned_steps)

        if "minisweagent_path.py" in structure_text:
            self._insert_if_missing(state, "Inspect mini-swe-agent path support", planned_steps)

        if "/tests" in structure_text or "tests/" in structure_text:
            self._insert_if_missing(state, "Inspect tests", planned_steps)

        # 多すぎる場合は上位だけに絞る
        planned_steps = planned_steps[: self.MAX_INSPECTION_STEPS]

        # 実際の current_plan に反映
        for step in planned_steps:
            if step not in state.current_plan and step not in state.completed_steps:
                state.current_plan.append(step)

        # 最後に必ず総括
        if "Summarize findings and propose next upgrade" not in state.current_plan:
            state.current_plan.append("Summarize findings and propose next upgrade")

    def next_step(self, state: AgentState, repo_root: Path | str | None = None) -> str | None:
        _ = repo_root

        if not state.current_plan:
            self.bootstrap_plan(state)

        if state.current_plan:
            step = state.current_plan.pop(0)

            if step == "Plan next inspection steps":
                self._plan_from_structure(state)
                if state.current_plan:
                    return state.current_plan.pop(0)
                return "Summarize findings and propose next upgrade"

            return step

        return None
