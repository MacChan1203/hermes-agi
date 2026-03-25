from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .agent_state import AgentState

if TYPE_CHECKING:
    from .mistral_client import MistralClient

_SYSTEM_TEMPLATE = """\
あなたはローカルマシン上で動作する {role} エージェントのプランナーです。
これは Python プロジェクトです。Node.js / npm / package.json は存在しません。
目標: {goal}

【絶対的な制約】
- インターネット・Web検索・URLアクセス・Google CLI は存在しない
- ブラウザ・GUIアプリケーションは使用できない
- tree コマンドは存在しない → 代わりに find を使う
- 使えるコマンドは bash 標準コマンドのみ: ls, find, cat, grep, head, python など

【使えるコマンド例 (必ずこの形式を使う)】
- CMD: find . -maxdepth 3 -not -path '*/__pycache__/*' | sort | head -60
- CMD: cat requirements.txt
- CMD: ls -la
- CMD: python --version
- CMD: grep -r "keyword" . --include="*.py" | head -20

【禁止事項】
- replace, sed -i, awk などのファイル書き換えコマンド禁止
- find -exec ... \; など複雑な -exec オプション禁止 (grep -r や ls を使うこと)
- locate コマンド禁止 (macOS では使えない。find を使うこと)
- xargs 禁止 (代わりに直接 grep -r や find を使うこと)
- 失敗済みのコマンドと同じコマンドを再実行禁止

失敗済みのステップ一覧: {failed}
→ 上記と同じコマンドは絶対に出力しない。別のアプローチを取る。

次に実行すべき「単一のステップ」のみを返してください。
形式のルール:
- シェルコマンドは必ず「CMD: 」プレフィックスを付けて1行で書く
- タスクが完了したと判断したら「DONE」とだけ返す

現在の状態:
- 完了済み: {completed}
- 観測メモ: {observations}
- 制約: {constraints}

返答はステップ1行のみ。説明不要。\
"""


class Planner:
    MAX_INSPECTION_STEPS = 6

    def __init__(
        self,
        llm: Optional[MistralClient] = None,
        role: str = "worker",
    ) -> None:
        self.llm = llm
        self.role = role

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def next_step(self, state: AgentState, repo_root: Path | str | None = None) -> str | None:
        if self.llm:
            return self._llm_next_step(state)
        return self._static_next_step(state, repo_root)

    # ------------------------------------------------------------------
    # LLM ベースのプランニング
    # ------------------------------------------------------------------

    def _llm_next_step(self, state: AgentState) -> str | None:
        if state.is_done:
            return None

        # recovery_action などで挿入されたステップを優先する
        if state.current_plan:
            return state.current_plan.pop(0)

        prompt = _SYSTEM_TEMPLATE.format(
            role=self.role,
            goal=state.user_goal,
            completed=", ".join(state.completed_steps[-5:]) or "なし",
            failed=", ".join(state.failed_steps[-3:]) or "なし",
            observations="; ".join(state.observations[-3:]) or "なし",
            constraints=", ".join(state.constraints) or "なし",
        )
        assert self.llm is not None
        response = self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=256,
        )
        # 最初の1行のみ使用 (LLMが説明を追記することがある)
        step = response.strip().splitlines()[0].strip()
        if not step or step.upper() == "DONE":
            return None
        return step

    # ------------------------------------------------------------------
    # 静的プランニング (LLM なし fallback)
    # ------------------------------------------------------------------

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

        planned_steps = planned_steps[: self.MAX_INSPECTION_STEPS]

        for step in planned_steps:
            if step not in state.current_plan and step not in state.completed_steps:
                state.current_plan.append(step)

        if "Summarize findings and propose next upgrade" not in state.current_plan:
            state.current_plan.append("Summarize findings and propose next upgrade")

    def _static_next_step(self, state: AgentState, repo_root: Path | str | None = None) -> str | None:
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
