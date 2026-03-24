from __future__ import annotations

from typing import Any, Dict, List

from .agent_state import AgentState
from .errors import classify_error, should_retry_error_type, should_retry_step
from .memory import remember_failure


class Reviewer:
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
        hints_map = {
            "Inspect project structure": [
                "重要ファイルの優先順位づけをさらに明確化する",
                "inspection 対象の上限数を設定値化する",
            ],
            "Read README": [
                "README に起動手順と設計概要が不足していないか確認する",
                "CLI 例や開発者向け手順を追記できるようにする",
            ],
            "Read requirements": [
                "requirements.txt と pyproject.toml の二重管理を整理する",
                "依存関係の役割ごとに分類できるようにする",
            ],
            "Read pyproject config": [
                "依存管理の正本を pyproject.toml に寄せる",
                "開発用依存と本番依存を分離する",
            ],
            "Read main entry point": [
                "run_agent.py の責務を薄くし、ロジックを内部モジュールへ寄せる",
                "CLI 引数名を統一する",
            ],
            "Inspect CLI entry point": [
                "CLI オプションをハイフン形式に統一する",
                "help 表示をわかりやすく整える",
            ],
            "Inspect tests": [
                "Planner / Reviewer / Executor 単体テストを増やす",
                "CLI のスモークテストを追加する",
            ],
            "Inspect state store": [
                "session ごとの要約保存を強化する",
                "再開時の復元戦略を整理する",
            ],
            "Inspect toolsets": [
                "toolset 選択を静的定義から動的選択へ近づける",
                "利用頻度の低い toolset を整理する",
            ],
            "Inspect tool distributions": [
                "distribution の意味を README に説明する",
                "重みの調整根拠を記録する",
            ],
            "Inspect model tools": [
                "tool registry と executor の責務分離を明確化する",
                "将来の LLM 接続点をここに集約する",
            ],
            "Inspect time handling": [
                "timezone 設定の診断表示を追加する",
                "設定変更後の cache reset 動線を整える",
            ],
            "Inspect constants": [
                "プロバイダ別定数を設定ファイル側へ寄せる",
                "未使用定数を整理する",
            ],
            "Inspect mini-swe-agent path support": [
                "worktree 判定時のログを増やす",
                "見つからない場合の診断メッセージを強化する",
            ],
            "Summarize findings and propose next upgrade": [
                "Reviewer に改善優先順位づけを持たせる",
                "総括結果を次回 session に引き継げるようにする",
            ],
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

    def evaluate(self, step: str, result: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        if result["ok"]:
            summary = self._success_summary(step)
            hints = self._improvement_hints(step)

            review: Dict[str, Any] = {
                "status": "success",
                "summary": summary,
                "goal_achieved": step == "Summarize findings and propose next upgrade",
                "recovery_action": None,
                "evidence": (result.get("stdout", "") or "")[:400],
                "improvement_hints": hints,
            }

            if step == "Summarize findings and propose next upgrade":
                review["priority_upgrades"] = self._priority_upgrades(state)

            return review

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
