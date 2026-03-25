"""メタ認知エンジン。エージェント自身の状態を観察し、戦略を自己調整する。"""
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Optional

from .agent_state import AgentState

if TYPE_CHECKING:
    from .long_term_memory import LongTermMemory


class MetaCognition:
    """エージェントの実行状態を監視し、行き詰まりの検出・戦略転換・次ゴール提案を行う。"""

    STUCK_FAILURE_THRESHOLD = 3   # 連続失敗がこの数に達したら行き詰まりとみなす
    REPEATED_ERROR_THRESHOLD = 2  # 同一エラーがこの回数続いたら行き詰まりとみなす

    # ------------------------------------------------------------------
    # 行き詰まり検出
    # ------------------------------------------------------------------

    def is_stuck(self, state: AgentState) -> bool:
        """エージェントが行き詰まっているか判定する。"""
        # 完了ゼロで失敗が連続している
        if (
            len(state.failed_steps) >= self.STUCK_FAILURE_THRESHOLD
            and len(state.completed_steps) == 0
        ):
            return True

        # 同じエラータイプが連続している
        error_history = state.working_memory.get("error_history", [])
        if len(error_history) >= self.REPEATED_ERROR_THRESHOLD:
            recent = error_history[-self.REPEATED_ERROR_THRESHOLD:]
            if len(set(recent)) == 1:
                return True

        return False

    # ------------------------------------------------------------------
    # パフォーマンス評価
    # ------------------------------------------------------------------

    def performance_score(self, state: AgentState) -> float:
        """成功率を 0.0〜1.0 で返す。"""
        total = len(state.completed_steps) + len(state.failed_steps)
        if total == 0:
            return 0.5
        return len(state.completed_steps) / total

    # ------------------------------------------------------------------
    # 戦略転換
    # ------------------------------------------------------------------

    def suggest_pivot(self, state: AgentState, memory: LongTermMemory) -> Optional[str]:
        """行き詰まり時に、別のアプローチを提案する。"""
        # 過去の成功戦略から未試行のものを探す
        successful = memory.get_successful_strategies(limit=5)
        for s in successful:
            strategy = s.get("strategy", "")
            if strategy and strategy not in state.failed_steps and strategy not in state.completed_steps:
                return f"[メタ認知] 過去の成功戦略を参考: {strategy}"

        # エラータイプに応じたリカバリ
        error_history = state.working_memory.get("error_history", [])
        if error_history:
            top_error = Counter(error_history).most_common(1)[0][0]
            pivot_map = {
                "missing_command": "CMD: which python3 && python3 --version && ls -la",
                "missing_file": "CMD: find . -maxdepth 3 -not -path '*/__pycache__/*' | sort | head -60",
                "permission_error": "CMD: ls -la && id",
                "missing_python_module": "CMD: python3 -m pip list | head -30",
                "connection_error": "CMD: ls -la && cat requirements.txt 2>/dev/null | head -20",
            }
            if top_error in pivot_map:
                return pivot_map[top_error]

        return "CMD: ls -la && find . -maxdepth 2 -not -path '*/__pycache__/*' | sort | head -40"

    # ------------------------------------------------------------------
    # 次ゴールの自律提案
    # ------------------------------------------------------------------

    def generate_next_goal(self, state: AgentState, memory: LongTermMemory) -> Optional[str]:
        """現在のゴール達成後、次に取り組むべきゴールを自律的に提案する。"""
        if not state.is_done:
            return None

        # 今セッションの priority_upgrades を最優先
        priority_upgrades = state.working_memory.get("priority_upgrades", [])
        if priority_upgrades:
            return priority_upgrades[0]

        # 長期記憶に蓄積された改善候補を参照
        recent = memory.recall_recent(limit=10)
        for item in recent:
            if item["key"].startswith("priority_upgrade_"):
                return item["value"]

        return None

    # ------------------------------------------------------------------
    # 自己振り返り
    # ------------------------------------------------------------------

    def reflection_summary(self, state: AgentState) -> str:
        """セッション終了時の自己振り返りサマリを生成する。"""
        score = self.performance_score(state)
        lines = [
            "[メタ認知レポート]",
            f"パフォーマンス: {score:.0%}  ({len(state.completed_steps)} 成功 / {len(state.failed_steps)} 失敗)",
        ]

        error_history = state.working_memory.get("error_history", [])
        if error_history:
            top_errors = Counter(error_history).most_common(2)
            lines.append("主要エラー: " + ", ".join(f"{e}({c}回)" for e, c in top_errors))

        next_goal = state.working_memory.get("suggested_next_goal")
        if next_goal:
            lines.append(f"次の推奨ゴール: {next_goal}")
        elif state.working_memory.get("priority_upgrades"):
            lines.append(f"次の改善候補: {state.working_memory['priority_upgrades'][0]}")

        return "\n".join(lines)
