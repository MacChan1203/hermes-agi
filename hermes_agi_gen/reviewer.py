from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .agent_state import AgentState
from .errors import classify_error, should_retry_error_type, should_retry_step
from .memory import remember_failure

if TYPE_CHECKING:
    from .mistral_client import MistralClient

_REVIEW_SYSTEM = """\
あなたは汎用 AGI エージェントの実行結果を評価するレビュアーです。

ドメイン: {domain}
目標: {goal}
実行したステップ: {step}
標準出力: {stdout}
標準エラー: {stderr}
終了コード: {returncode}

【評価基準】
- 終了コード 0 かつ stdout に有用な情報があれば "success"
- エラーがあっても目標に対して十分な情報が得られていれば "success" でよい
- "goal_achieved" は目標全体が達成されたと判断できる場合のみ true
- recovery_action はツール形式で返す: ANSWER: / SEARCH: / CMD: / READ: / PYTHON: / DONE: のいずれか
- ブラウザ・GUI は使えない。ウェブ情報が必要なら SEARCH: を使う
- tree コマンドは使えない → 代わりに CMD: find . -maxdepth 3 | sort | head -60

以下の JSON のみを返してください (説明不要):
{{
  "status": "success" または "failed",
  "goal_achieved": true または false,
  "summary": "日本語の簡潔な要約 (60文字以内)",
  "learned_fact": "このステップで学んだ重要な事実 (なければ null)",
  "recovery_action": "失敗時のみ: ツール形式の1行。成功時: null"
}}\
"""


class Reviewer:
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

    def evaluate(self, step: str, result: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        if self.llm:
            return self._llm_evaluate(step, result, state)
        return self._static_evaluate(step, result, state)

    # ------------------------------------------------------------------
    # LLM ベースの評価
    # ------------------------------------------------------------------

    def _llm_evaluate(self, step: str, result: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        # PLAN: は計画立案ステップ。サブステップが current_plan に入っており未実行なので
        # goal_achieved=False を強制し、next_step でサブステップを実行させる
        if step.upper().startswith("PLAN:"):
            return {
                "status": "success",
                "summary": result.get("stdout", "計画を立案しました"),
                "goal_achieved": False,
                "recovery_action": None,
                "evidence": result.get("stdout", "")[:400],
                "improvement_hints": [],
            }
        stdout = (result.get("stdout", "") or "")[:800]
        stderr = (result.get("stderr", "") or "")[:400]
        returncode = result.get("returncode", -1)

        domain = getattr(state, "domain", "general")

        assert self.llm is not None
        data = self.llm.chat_json(
            [
                {
                    "role": "user",
                    "content": _REVIEW_SYSTEM.format(
                        step=step,
                        stdout=stdout,
                        stderr=stderr,
                        returncode=returncode,
                        goal=state.user_goal,
                        domain=domain,
                    ),
                }
            ],
            temperature=0.1,
            max_tokens=768,
        )

        if not isinstance(data, dict):
            return self._fallback_review(step, result, state)

        status = data.get("status", "failed" if not result.get("ok") else "success")
        goal_achieved = bool(data.get("goal_achieved", False))
        summary = str(data.get("summary", f"{step} を実行しました"))
        recovery_action = data.get("recovery_action") or None
        learned_fact = data.get("learned_fact") or None

        review: Dict[str, Any] = {
            "status": status,
            "summary": summary,
            "goal_achieved": goal_achieved,
            "recovery_action": recovery_action,
            "evidence": stdout[:400],
            "improvement_hints": [],
            "learned_fact": learned_fact,
        }

        if goal_achieved:
            review["priority_upgrades"] = self._priority_upgrades(state)

        return review

    def _fallback_review(self, step: str, result: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        if result.get("ok"):
            return {
                "status": "success",
                "summary": f"{step} を完了しました",
                "goal_achieved": False,
                "recovery_action": None,
                "evidence": (result.get("stdout", "") or "")[:400],
                "improvement_hints": [],
            }
        return self._static_failure_review(step, result, state)

    # ------------------------------------------------------------------
    # 静的評価 (LLM なし fallback)
    # ------------------------------------------------------------------

    def _static_evaluate(self, step: str, result: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        if result["ok"]:
            # ANSWER: ステップは直接回答なので即座に goal_achieved=True
            is_answer = step.upper().startswith("ANSWER:")
            # PLAN: は計画立案のみ。goal_achieved=False で次のステップへ
            is_plan = step.upper().startswith("PLAN:")
            is_final = step == "Summarize findings and propose next upgrade"

            summary = result.get("stdout", "") if (is_answer or is_plan) else self._success_summary(step)
            hints = [] if (is_answer or is_plan) else self._improvement_hints(step)

            review: Dict[str, Any] = {
                "status": "success",
                "summary": summary,
                "goal_achieved": is_answer or is_final,  # PLAN: は goal_achieved=False
                "recovery_action": None,
                "evidence": (result.get("stdout", "") or "")[:400],
                "improvement_hints": hints,
            }

            if is_final:
                review["priority_upgrades"] = self._priority_upgrades(state)

            return review

        return self._static_failure_review(step, result, state)

    def _static_failure_review(self, step: str, result: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        stderr = result.get("stderr", "")
        error_type = classify_error(stderr)

        remember_failure(state, step, error_type, stderr)
        error_history = state.working_memory.get("error_history", [])

        recovery_map = {
            "missing_command": "Check installed commands and PATH",
            "permission_error": "Inspect file permissions",
            "missing_python_module": "Check Python environment and pip packages",
            "connection_error": "Check running services and ports",
            "missing_file": "Inspect project structure",
            "syntax_error": "Read main entry point",
            "unknown_error": "Inspect project structure",
        }

        recovery_action = recovery_map.get(error_type, "Inspect project structure")

        can_retry_same_step = should_retry_step(step, state.failed_steps)
        can_retry_same_error = should_retry_error_type(error_type, error_history)

        if not can_retry_same_step or not can_retry_same_error:
            recovery_action = "Summarize findings and propose next upgrade"

        return {
            "status": "failed",
            "summary": f"{step} で失敗しました: {error_type}",
            "goal_achieved": False,
            "recovery_action": recovery_action,
            "evidence": stderr[:400],
            "error_type": error_type,
            "improvement_hints": [
                f"失敗分類: {error_type}",
                f"次は {recovery_action} を試す",
            ],
        }

    # ------------------------------------------------------------------
    # ヘルパー (静的)
    # ------------------------------------------------------------------

    def _success_summary(self, step: str) -> str:
        summaries = {
            "Inspect project structure": "プロジェクト構造を確認しました",
            "Read README": "README を確認しました",
            "Read requirements": "requirements.txt を確認しました",
            "Read pyproject config": "pyproject.toml を確認しました",
            "Read main entry point": "メインの入口処理を確認しました",
            "Inspect CLI entry point": "CLI の入口処理を確認しました",
            "Inspect tests": "テスト構成を確認しました",
            "Inspect state store": "状態保存の仕組みを確認しました",
            "Inspect toolsets": "toolset 定義を確認しました",
            "Inspect tool distributions": "tool distribution 定義を確認しました",
            "Inspect model tools": "model tool 定義を確認しました",
            "Inspect time handling": "時刻処理を確認しました",
            "Inspect constants": "定数定義を確認しました",
            "Inspect mini-swe-agent path support": "mini-swe-agent 連携用パス処理を確認しました",
            "Summarize findings and propose next upgrade": "全体を総括し、次の改善候補を整理しました",
        }
        return summaries.get(step, f"{step} を確認しました")

    def _improvement_hints(self, step: str) -> List[str]:
        hints_map: Dict[str, List[str]] = {
            "Inspect project structure": ["重要ファイルの優先順位づけをさらに明確化する"],
            "Read README": ["README に起動手順と設計概要が不足していないか確認する"],
            "Summarize findings and propose next upgrade": ["Reviewer に改善優先順位づけを持たせる"],
        }
        return hints_map.get(step, ["改善候補を整理する"])

    def _priority_upgrades(self, state: AgentState) -> List[str]:
        completed = set(state.completed_steps)
        suggestions: List[str] = []

        if "Inspect CLI entry point" in completed:
            suggestions.append("CLI 引数名を --max-turns / --repo-root のようなハイフン形式へ統一する")
        if "Inspect tests" in completed:
            suggestions.append("Planner / Executor / Reviewer の単体テストを追加する")
        if "Inspect toolsets" in completed:
            suggestions.append("toolset 選択を固定定義から動的選択へ発展させる")
        if "Inspect state store" in completed:
            suggestions.append("総括結果を session に保存して次回再利用できるようにする")
        if "Read main entry point" in completed:
            suggestions.append("run_agent.py を薄くして内部 API と CLI の責務を分離する")

        if not suggestions:
            suggestions.append("Planner と Reviewer の連携を強化する")

        return suggestions[:3]
