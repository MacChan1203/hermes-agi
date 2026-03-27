"""汎用プランナー。ドメインを問わずあらゆるタスクに対応する。"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .agent_state import AgentState
from .tools import TOOL_CONSTRAINTS, TOOL_DESCRIPTIONS, TOOL_EXAMPLES

if TYPE_CHECKING:
    from .mistral_client import MistralClient

# ------------------------------------------------------------------
# ドメイン別ヒント（LLM プロンプトに注入）
# ------------------------------------------------------------------
_DOMAIN_HINTS: dict[str, str] = {
    "general": "質問・相談には ANSWER: で直接回答する。ファイル操作が必要なら CMD:/READ:、外部情報が必要なら SEARCH: を使う。",
    "coding": "コードを読んで理解してから変更を提案する。テストを確認する。外部ライブラリ情報は SEARCH: で調べる。",
    "research": "SEARCH: でウェブ情報を収集し、複数ソースを比較して整理する。根拠のある結論を出す。",
    "writing": "既存の文章・構造を確認してから追記・編集する。参考資料が必要なら SEARCH: を活用する。",
    "data": "データファイルの形式・サイズを確認してから処理する。簡単な集計は CALC: で行う。手法に迷ったら SEARCH: で調べる。",
    "ops": "現在の状態を確認してから操作する。破壊的操作は避ける。エラー解決策は SEARCH: で調べる。",
}

# ------------------------------------------------------------------
# LLM プランナー プロンプト
# ------------------------------------------------------------------
_SYSTEM_TEMPLATE = """\
あなたはローカルマシン上で動作する汎用 AGI エージェントのプランナーです。

役割: {role}
ドメイン: {domain}
目標: {goal}
{context_section}
ドメインヒント: {domain_hint}

{tool_descriptions}

{tool_examples}

{tool_constraints}

【長期記憶 (過去のセッションで学んだこと)】
{long_term_memories}

【既知の失敗パターン (これらは避ける)】
{known_failures}

現在の状態:
- 完了済み: {completed}
- 失敗済み: {failed}
- 観測メモ: {observations}
- ユーザー制約: {constraints}

次に実行すべきアクションを1行だけ返してください。
- 単純なタスク: ツール形式1行 (ANSWER: / SEARCH: / CALC: / CMD: / READ: / PYTHON: など)
- 複雑なタスク: PLAN: step1 || step2 || step3 で全ステップを一括計画する
- DONE: は目標が完全に達成されたときだけ使う
- 説明・コメント不要。アクション1行のみ。\
"""

# 会話的なクエリを検出するキーワード
_CONVERSATIONAL_KEYWORDS: frozenset[str] = frozenset({
    "答えられますか", "できますか", "何ができ", "何を知", "教えて", "説明して",
    "どう思", "どう考え", "とは何", "とはなん", "とは？", "とは?",
    "あなたは", "あなたに", "あなたの", "できること", "機能", "使い方",
    "こんにちは", "はじめまして", "よろしく", "ありがとう", "お願い",
    "can you", "what can", "are you", "what is", "how do", "help me",
    "hello", "hi ", "please", "thank",
})

_STATIC_ANSWER_GENERAL = (
    "ANSWER: はい、幅広いトピックに対応できます。"
    "コーディング、調査・リサーチ、文章作成、データ分析、一般的な質問など"
    "さまざまなタスクをサポートします。"
    "具体的に何を手伝いましょうか？"
)


def _is_conversational(goal: str) -> bool:
    """ファイル操作やコマンド実行を必要としない会話的クエリかどうかを判定する。"""
    g = goal.lower()
    return any(kw in g for kw in _CONVERSATIONAL_KEYWORDS)


# ------------------------------------------------------------------
# 静的フォールバック: ドメイン別初期プラン
# ------------------------------------------------------------------
_STATIC_BOOTSTRAP: dict[str, list[str]] = {
    "general": [
        "CMD: pwd && ls -la && find . -maxdepth 2 -not -path '*/__pycache__/*' | sort | head -60",
        "DONE: 現状を把握しました。",
    ],
    "coding": [
        "CMD: pwd && ls -la && find . -maxdepth 2 -not -path '*/__pycache__/*' | sort | head -60",
        "CMD: cat README.md 2>/dev/null | head -60 || echo 'README なし'",
        "CMD: cat requirements.txt 2>/dev/null || cat pyproject.toml 2>/dev/null | head -40 || echo '依存ファイルなし'",
        "DONE: プロジェクト構造と依存関係を確認しました。",
    ],
    "research": [
        "SEARCH: {goal}",
        "DONE: 調査結果をまとめました。",
    ],
    "writing": [
        "CMD: pwd && ls -la",
        "CMD: find . -name '*.md' -o -name '*.txt' -o -name '*.rst' | head -20",
        "DONE: 文書ファイルを確認しました。",
    ],
    "data": [
        "CMD: pwd && ls -la",
        "CMD: find . -name '*.csv' -o -name '*.json' -o -name '*.parquet' | head -20",
        "DONE: データファイルを確認しました。",
    ],
    "ops": [
        "CMD: pwd && ls -la && env | grep -v SECRET | head -20",
        "DONE: 環境を確認しました。",
    ],
}


class Planner:
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
        return self._static_next_step(state)

    # ------------------------------------------------------------------
    # LLM ベースのプランニング
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # サマリー生成プロンプト
    # ------------------------------------------------------------------

    _SUMMARIZE_PROMPT = """\
以下の情報をもとに、ユーザーの目標に対する最終的なまとめを日本語で作成してください。

目標: {goal}

収集した情報:
{evidence}

簡潔かつ網羅的にまとめてください（200〜400字程度）。
"""

    def _generate_summary_answer(self, state: AgentState) -> str:
        """ワーキングメモリの検索結果や観測を使って最終まとめを ANSWER: 形式で生成する。"""
        assert self.llm is not None
        parts: list[str] = []
        results = state.working_memory.get("last_search_results", [])
        for r in results[:5]:
            if r.get("snippet"):
                parts.append(f"- {r['title']}: {r['snippet']}")
        if state.observations:
            parts.extend(f"- {o}" for o in state.observations[-3:])
        evidence = "\n".join(parts) or "（情報なし）"

        prompt = self._SUMMARIZE_PROMPT.format(goal=state.user_goal, evidence=evidence)
        summary = self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512,
        )
        return f"ANSWER: {summary}" if summary else "DONE: まとめを生成できませんでした"

    def _llm_next_step(self, state: AgentState) -> str | None:
        if state.is_done:
            return None

        # current_plan にステップが積まれている場合はそれを優先
        if state.current_plan:
            next_step = state.current_plan.pop(0)
            # DONE: まとめ系トリガーなら実際のまとめを LLM で生成
            if next_step.upper().startswith("DONE:") and any(
                kw in next_step for kw in ("まとめ", "結論", "要約", "summarize", "summary")
            ):
                return self._generate_summary_answer(state)
            return next_step

        # 長期記憶をプロンプトに注入
        ltm_strategies = state.working_memory.get("ltm_strategies", [])
        ltm_failures = state.working_memory.get("ltm_known_failures", [])

        ltm_mem_text = "\n".join(
            f"- [{s['outcome']}] {s['strategy']}" for s in ltm_strategies[:3]
        ) or "なし"

        ltm_fail_text = "\n".join(
            f"- {f['command_pattern'][:60]} ({f['error_type']}, {f['count']}回失敗)"
            for f in ltm_failures[:5]
        ) or "なし"

        domain = getattr(state, "domain", "general")
        context = getattr(state, "context", "")
        context_section = f"コンテキスト: {context}\n" if context else ""
        domain_hint = _DOMAIN_HINTS.get(domain, _DOMAIN_HINTS["general"])

        prompt = _SYSTEM_TEMPLATE.format(
            role=self.role,
            domain=domain,
            goal=state.user_goal,
            context_section=context_section,
            domain_hint=domain_hint,
            tool_descriptions=TOOL_DESCRIPTIONS,
            tool_examples=TOOL_EXAMPLES,
            tool_constraints=TOOL_CONSTRAINTS,
            completed=", ".join(state.completed_steps[-5:]) or "なし",
            failed=", ".join(state.failed_steps[-3:]) or "なし",
            observations="; ".join(state.observations[-3:]) or "なし",
            constraints=", ".join(state.constraints) or "なし",
            long_term_memories=ltm_mem_text,
            known_failures=ltm_fail_text,
        )

        assert self.llm is not None
        response = self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
        )

        # 最初の意味ある行を取得
        for line in response.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # DONE 系は None を返してループ終了させる
            if line.upper().startswith("DONE"):
                state.working_memory["completion_summary"] = line[5:].strip() if ":" in line else ""
                return None
            # PLAN: は executor に渡してサブステップを展開させる
            if line.upper().startswith("PLAN:"):
                return line
            return line

        return None

    # ------------------------------------------------------------------
    # 静的プランニング (LLM なし fallback)
    # ------------------------------------------------------------------

    def _static_next_step(self, state: AgentState) -> str | None:
        if not state.current_plan:
            domain = getattr(state, "domain", "general")
            # 会話的なクエリには直接回答する（ファイル探索は不要）
            if domain == "general" and _is_conversational(state.user_goal):
                state.current_plan = [_STATIC_ANSWER_GENERAL, "DONE: 回答しました。"]
            elif domain == "research":
                # research はゴールをそのまま検索クエリにする
                state.current_plan = [
                    f"SEARCH: {state.user_goal}",
                    "DONE: 調査結果をまとめました。",
                ]
            else:
                plan = _STATIC_BOOTSTRAP.get(domain, _STATIC_BOOTSTRAP["general"])
                state.current_plan = [s for s in plan if s not in state.completed_steps]

        if state.current_plan:
            return state.current_plan.pop(0)

        return None
