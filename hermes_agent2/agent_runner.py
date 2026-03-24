from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent_state import AgentState
from .executor import Executor
from .memory import initialize_working_memory
from .planner import Planner
from .reviewer import Reviewer
from .state_store import SessionDB


class HermesAgentV9:
    """旧 Hermes の運用感と v9 の plan-act-review を合わせた軽量版。"""

    def __init__(
        self,
        repo_root: str | Path = ".",
        model: str = "local/mock-model",
        max_iterations: int = 8,
        session_db: SessionDB | None = None,
        source: str = "cli",
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.model = model
        self.max_iterations = max_iterations
        self.session_db = session_db or SessionDB()
        self.source = source
        self.planner = Planner()
        self.executor = Executor(self.repo_root)
        self.reviewer = Reviewer()

    def run(self, state: AgentState) -> AgentState:
        initialize_working_memory(state)
        state.max_iterations = state.max_iterations or self.max_iterations
        if not state.session_id:
            state.session_id = str(uuid.uuid4())
        self.session_db.create_session(state.session_id, source=self.source, model=self.model, title=state.user_goal)
        self.session_db.append_message(state.session_id, "user", state.user_goal)

        while not state.is_done and state.iteration_count < state.max_iterations:
            state.iteration_count += 1
            step = self.planner.next_step(state, self.repo_root)
            if not step:
                state.is_done = True
                state.last_status = "finished"
                break

            state.last_step = step
            self.session_db.append_message(state.session_id, "assistant", f"次の一手: {step}")
            result = self.executor.execute(step, state)


            review = self.reviewer.evaluate(step, result, state)

            state.observations.append(review["summary"])
            state.last_status = review["status"]

            state.working_memory["last_improvement_hints"] = review.get("improvement_hints", [])

            if review.get("priority_upgrades"):
                state.working_memory["priority_upgrades"] = review["priority_upgrades"]


            self.session_db.append_message(state.session_id, "tool", result.get("stdout", "") or result.get("stderr", ""), tool_name="terminal")
            self.session_db.append_message(state.session_id, "assistant", review["summary"])

            if review["status"] == "success":
                state.completed_steps.append(step)
            else:
                state.failed_steps.append(step)
                recovery_action = review.get("recovery_action")
                if recovery_action:
                    state.current_plan.insert(0, recovery_action)

            if review.get("goal_achieved", False):
                state.is_done = True

        self.session_db.end_session(state.session_id, "completed" if state.is_done else "stopped")
        return state

    def chat(self, message: str) -> str:
        state = AgentState(
            user_goal=message,
            success_criteria=["次の一手を出せる", "失敗時に立て直せる", "進捗を日本語で説明できる"],
            constraints=["破壊的操作はしない", "まず読んで把握する"],
            max_iterations=self.max_iterations,
        )
        final_state = self.run(state)
        return self.render_progress(final_state)

    def render_progress(self, state: AgentState) -> str:
        lines: List[str] = []
        lines.append("=== Hermes Agent 2 / v9 進捗 ===")
        lines.append(f"目的: {state.user_goal}")
        lines.append(f"反復回数: {state.iteration_count}/{state.max_iterations}")
        lines.append(f"最後のステップ: {state.last_step}")
        lines.append(f"最後の状態: {state.last_status}")
        lines.append("")
        lines.append("[完了したステップ]")
        lines.extend([f"- {step}" for step in state.completed_steps] or ["- なし"])
        lines.append("")
        lines.append("[失敗したステップ]")
        lines.extend([f"- {step}" for step in state.failed_steps] or ["- なし"])
        lines.append("")
        lines.append("[観測メモ]")
        lines.extend([f"- {obs}" for obs in state.observations] or ["- なし"])
        env = state.working_memory.get("environment", {})
        lines.append("")
        lines.append("[作業メモ]")
        lines.append(f"- cwd: {env.get('cwd')}")
        lines.append(f"- python_version: {env.get('python_version')}")
        lines.append(f"- python_executable: {env.get('python_executable')}")
        lines.append(f"- session_id: {state.session_id}")
        lines.append("")
        lines.append("[直近の改善ヒント]")
        hints = state.working_memory.get("last_improvement_hints", [])
        if hints:
            for hint in hints:
                lines.append(f"- {hint}")
        else:
            lines.append("- なし")

        lines.append("")
        lines.append("[優先改善案]")
        upgrades = state.working_memory.get("priority_upgrades", [])
        if upgrades:
            for upgrade in upgrades:
                lines.append(f"- {upgrade}")
        else:
            lines.append("- なし")

        return "\n".join(lines)
