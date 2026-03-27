#!/usr/bin/env python3
"""Hermes AI - 汎用インタラクティブ CLI。チャットとエージェントを統合。"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

# .env を先にロードしてから hermes モジュールを import
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule

from hermes_agi_gen import AgentState, HermesAgentV9
from hermes_agi_gen.code_agents import CodeGeneratorAgent, CodeReviewerAgent
from hermes_agi_gen.hermes_constants import DOMAIN_CONFIG
from hermes_agi_gen.mistral_client import MistralClient
from hermes_agi_gen.state_store import SessionDB

console = Console()

# ---------------------------------------------------------------------------
# システムプロンプト
# ---------------------------------------------------------------------------

_GENERAL_SYSTEM = """\
あなたは Hermes AI、汎用の対話型アシスタントです。
ユーザーの質問・相談・依頼に対して、丁寧かつ簡潔に日本語で回答してください。
コーディング、調査、文章作成、データ分析、一般的な質問など、あらゆるトピックに対応します。
"""

_INTENT_SYSTEM = """\
ユーザーのメッセージが「タスク実行」か「直接回答」かを判定してください。

タスク実行 (type=task):
  ファイル操作・コード実行・ディレクトリ調査・データ処理・システム操作など、
  ローカル環境で実際に何かを行う必要があるもの。

直接回答 (type=chat):
  挨拶・雑談・自己紹介・能力質問・説明・意見・アドバイスなど、
  テキストのみで応答できるもの。
  「何ができますか」「あなたは〜ですか」「〜とは何ですか」は必ず chat。

ドメイン (domain): general / coding / research / writing / data / ops

例:
  「あなたは何ができますか？」→ {"type": "chat", "domain": "general"}
  「こんにちは」           → {"type": "chat", "domain": "general"}
  「Pythonとは何ですか？」  → {"type": "chat", "domain": "general"}
  「このディレクトリを調べて」→ {"type": "task", "domain": "ops"}
  「テストを実行して」      → {"type": "task", "domain": "coding"}

以下の JSON のみを返してください（説明不要）:
{"type": "chat", "domain": "general"}
"""

# ---------------------------------------------------------------------------
# ヘルプテキスト
# ---------------------------------------------------------------------------

_HELP = """\
## 使えるコマンド

| コマンド | 説明 |
|---|---|
| `<メッセージ>` | 自由に話しかける（チャット or 自動でエージェント起動） |
| `/run <目標>` | エージェントモードで明示的にタスクを実行 |
| `/reflect [対象]` | 自己診断: 自分のソースを読んで改善提案を生成 |
| `/apply` | 最後の `/reflect` 提案をコードに適用 |
| `/generate <説明>` | 自然言語からコードを生成 |
| `/review` | コードのレビュー（貼り付けモード） |
| `/clear` | 会話履歴をリセット |
| `/provider` | 現在の LLM プロバイダーを表示 |
| `/help` | このヘルプを表示 |
| `/quit` | 終了 |

**ヒント**: `/reflect` の対象は `planner` `executor` `reviewer` `runner` `search` `cli` など。
省略するとコアファイル全体を診断します。
"""

# 短縮名 → 実ファイルパスのマッピング
_REFLECT_TARGETS: dict[str, str] = {
    "planner":  "hermes_agi_gen/planner.py",
    "executor": "hermes_agi_gen/executor.py",
    "reviewer": "hermes_agi_gen/reviewer.py",
    "runner":   "hermes_agi_gen/agent_runner.py",
    "memory":   "hermes_agi_gen/long_term_memory.py",
    "meta":     "hermes_agi_gen/meta_cognition.py",
    "tools":    "hermes_agi_gen/tools.py",
    "search":   "hermes_agi_gen/web_search.py",
    "state":    "hermes_agi_gen/agent_state.py",
    "cli":      "cli.py",
}

_REFLECT_CORE_FILES = [
    "hermes_agi_gen/planner.py",
    "hermes_agi_gen/executor.py",
    "hermes_agi_gen/reviewer.py",
    "hermes_agi_gen/agent_runner.py",
]

_SELF_REFLECT_CONTEXT = """\
あなたは Hermes AI 自身です。これはあなた自身のローカルのソースコードです。
ファイルは READ: ツールで読み込んでください（SEARCH: は不要）。
以下の観点で自己診断を行い、具体的な改善提案をしてください:
1. バグ・エラーになりうる箇所 (コード行を具体的に指摘)
2. パフォーマンス・効率の改善機会
3. 設計・保守性の改善案
4. 未実装・TODO・将来拡張のアイデア
各項目に修正案のコードスニペットを含めてください。
"""

_SELF_APPLY_CONTEXT = """\
あなたは Hermes AI 自身のコードを改善するエンジニアです。
提示された改善提案を実際のコードに適用してください。
手順:
1. READ: で対象ファイルを読む
2. 改善内容を特定する
3. WRITE: で修正済みファイルを書き込む
破壊的な変更は避け、既存の動作を維持しながら改善してください。
"""

# 明らかに会話的なクエリを LLM より前に判定するキーワード集
# 「〜して教えて」系のタスクを検出するアクション動詞
_TASK_ACTION_VERBS: frozenset[str] = frozenset({
    "実行して", "起動して", "インストールして", "ビルドして", "デプロイして",
    "テストして", "チェックして", "確認して", "調べて", "列挙して", "探して",
    "読んで", "開いて", "作って", "書いて", "修正して", "変更して", "削除して",
    "移動して", "コピーして", "検索して", "分析して", "比較して",
    "run ", "execute ", "check ", "find ", "search ", "read ",
})


_CHAT_KEYWORDS: frozenset[str] = frozenset({
    # 自己紹介・能力質問（これらは必ず chat）
    "あなたは何", "あなたに何", "何ができますか", "できますか", "できること",
    "何者ですか", "どんなことができ",
    # 挨拶・雑談（これらは必ず chat）
    "こんにちは", "こんばんは", "おはようございます", "はじめまして", "よろしく",
    "ありがとう", "お疲れ様", "お疲れさま",
    # 純粋な説明要求（動詞なし）
    "とは何ですか", "とはなんですか", "とは？", "とは?",
    "どう思いますか", "どう考えますか", "ご意見",
    # 英語
    "what can you do", "can you help", "are you", "what are you",
    "hello", "hi there",
})


def _is_likely_chat(message: str) -> bool:
    """LLM を呼ばずにキーワードで会話クエリを判定する。"""
    m = message.lower()
    # アクション動詞が含まれていれば task 寄り → chat と判定しない
    if any(v in m for v in _TASK_ACTION_VERBS):
        return False
    return any(kw in m for kw in _CHAT_KEYWORDS)


# ---------------------------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------------------------

def _provider_label(llm: MistralClient) -> str:
    url = llm.base_url
    if "groq" in url:
        return f"[bold yellow]Groq[/bold yellow] ({llm.model})"
    if "mistral" in url:
        return f"[bold blue]Mistral[/bold blue] ({llm.model})"
    if "openrouter" in url:
        return f"[bold magenta]OpenRouter[/bold magenta] ({llm.model})"
    return f"[bold white]Ollama[/bold white] ({llm.model})"


def _collect_code() -> str:
    """複数行コードの貼り付けを受け取る。END で終了。"""
    console.print("[dim]コードを貼り付けてください。終わったら新しい行に [bold]END[/bold] と入力してください。[/dim]")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def _display(result: str, title: str, border: str = "green") -> None:
    console.print(Panel(Markdown(result), title=title, border_style=border))


def _has_action_verb(message: str) -> bool:
    """タスク系アクション動詞が含まれているか判定する。"""
    return any(v in message.lower() for v in _TASK_ACTION_VERBS)


def _classify_intent(llm: MistralClient, message: str) -> Tuple[str, str]:
    """メッセージが 'task' か 'chat' かを判定し (type, domain) を返す。

    優先順位:
      1. キーワード事前フィルター (chat 確定)
      2. LLM 判定
      3. アクション動詞フォールバック (LLM が chat と言っても task に補正)
    """
    # 1. 明らかな会話クエリはLLM不要
    if _is_likely_chat(message):
        return "chat", "general"

    # 2. LLM で判定
    result = llm.chat_json(
        [
            {"role": "system", "content": _INTENT_SYSTEM},
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        max_tokens=64,
    )
    if isinstance(result, dict):
        intent_type = result.get("type", "chat")
        domain = result.get("domain", "general")
        valid_domains = set(DOMAIN_CONFIG.keys())
        if domain not in valid_domains:
            domain = "general"
        # 3. LLM が chat と言ってもアクション動詞があれば task に補正
        if intent_type == "chat" and _has_action_verb(message):
            intent_type = "task"
        if intent_type in {"task", "chat"}:
            return intent_type, domain

    # 最終フォールバック: アクション動詞があれば task、なければ chat
    return ("task", "general") if _has_action_verb(message) else ("chat", "general")


def _run_agent(
    llm: MistralClient,
    goal: str,
    domain: str,
    context: str = "",
    max_iterations: int = 8,
) -> str:
    """HermesAgentV9 を起動してタスクを実行し、完了サマリーを返す。"""
    console.print(Rule(
        f"[bold yellow]エージェントモード[/bold yellow]  domain=[cyan]{domain}[/cyan]",
        style="yellow",
    ))

    agent = HermesAgentV9(
        repo_root=Path("."),
        model=llm.model,
        max_iterations=max_iterations,
        llm=llm,
    )

    cfg = DOMAIN_CONFIG.get(domain, DOMAIN_CONFIG["general"])
    state = AgentState(
        user_goal=goal,
        domain=domain,
        context=context,
        success_criteria=cfg["success_criteria"],
        constraints=cfg["constraints"],
        max_iterations=max_iterations,
    )

    final_state = agent.run(state)

    console.print(Rule(style="yellow"))

    summary = final_state.working_memory.get("completion_summary", "")
    if not summary and final_state.observations:
        summary = final_state.observations[-1]

    if summary:
        _display(summary, title="エージェント完了", border="yellow")

    if final_state.suggested_next_goal:
        console.print(
            f"[dim]次の推奨ゴール: {final_state.suggested_next_goal}[/dim]"
        )

    return summary


# ---------------------------------------------------------------------------
# メインループ
# ---------------------------------------------------------------------------

def main() -> None:
    llm = MistralClient()            # スマートモデル: プランニング・回答生成
    fast_llm = MistralClient.fast()  # 高速モデル: インテント分類
    db = SessionDB()
    generator = CodeGeneratorAgent(llm=llm, session_db=db)
    reviewer = CodeReviewerAgent(llm=llm, session_db=db)

    history: List[Dict[str, str]] = []
    last_reflection: Dict[str, str] = {}  # {"file": ..., "suggestion": ...}

    fast_label = (
        f" / fast=[dim]{fast_llm.model}[/dim]"
        if fast_llm.model != llm.model else ""
    )
    console.print(Panel(
        f"[bold cyan]Hermes AI[/bold cyan]\n"
        f"LLM: {_provider_label(llm)}{fast_label}\n"
        f"[dim]メッセージを入力 → チャット or 自動エージェント / /help でコマンド一覧 / /quit で終了[/dim]",
        border_style="cyan",
    ))

    while True:
        try:
            raw = Prompt.ask("[bold green]hermes[/bold green]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]終了します。[/dim]")
            break

        raw = raw.strip()
        if not raw:
            continue

        # --- 終了 ---
        if raw in {"/quit", "/exit", "quit", "exit"}:
            console.print("[dim]終了します。[/dim]")
            break

        # --- ヘルプ ---
        elif raw == "/help":
            console.print(Markdown(_HELP))

        # --- 会話履歴リセット ---
        elif raw == "/clear":
            history.clear()
            console.print("[dim]会話履歴をリセットしました。[/dim]")

        # --- プロバイダー表示 ---
        elif raw == "/provider":
            console.print(f"スマート: {_provider_label(llm)}")
            console.print(f"高速:     {_provider_label(fast_llm)}")
            console.print(f"[dim]DB  : {db.db_path}[/dim]")

        # --- 明示的なエージェント起動 ---
        elif raw.startswith("/run"):
            goal = raw[4:].strip()
            if not goal:
                console.print("[red]使い方: /run <実行したい目標・タスク>[/red]")
                continue
            with console.status("[yellow]ドメインを判定中...[/yellow]"):
                _, domain = _classify_intent(fast_llm, goal)  # 高速モデルで分類
            _run_agent(llm, goal, domain)  # スマートモデルで実行

        # --- 自己診断・改善提案 ---
        elif raw.startswith("/reflect"):
            target = raw[8:].strip().lower()
            if target in _REFLECT_TARGETS:
                file_path = _REFLECT_TARGETS[target]
                files_desc = file_path
                goal = (
                    f"READ: {file_path} でファイルを読み込み、自己診断してください。"
                    f"バグ・パフォーマンス問題・設計改善・未実装機能の観点で "
                    f"具体的なコードスニペット付きの改善提案を日本語でまとめてください。"
                    f"SEARCH: は使わず、READ: でローカルファイルを直接読んでください。"
                )
            else:
                if target and target not in _REFLECT_TARGETS:
                    console.print(
                        f"[yellow]対象 '{target}' は不明です。"
                        f"コアファイル全体を診断します。[/yellow]"
                    )
                files_desc = ", ".join(_REFLECT_CORE_FILES)
                read_steps = " || ".join(f"READ: {f}" for f in _REFLECT_CORE_FILES)
                goal = (
                    f"PLAN: {read_steps} || ANSWER: まとめ を使って "
                    f"コアファイルを順に読んで自己診断してください。\n"
                    f"バグ・パフォーマンス問題・設計改善・未実装機能の観点で "
                    f"優先度付きの改善提案を日本語でまとめてください。"
                    f"SEARCH: は使わず READ: でローカルファイルを直接読んでください。"
                )
            console.print(
                f"[magenta]自己診断モード[/magenta] — 対象: [bold]{files_desc}[/bold]"
            )
            suggestion = _run_agent(
                llm, goal, "coding",
                context=_SELF_REFLECT_CONTEXT,
                max_iterations=12,
            )
            if suggestion:
                last_reflection["file"] = files_desc
                last_reflection["suggestion"] = suggestion

        # --- 改善提案を適用 ---
        elif raw.startswith("/apply"):
            if not last_reflection:
                console.print("[yellow]まず /reflect を実行してください。[/yellow]")
                continue
            detail = raw[6:].strip()
            files = last_reflection.get("file", "（不明）")
            suggestion = last_reflection.get("suggestion", "")
            apply_goal = (
                f"以下の改善提案を {files} に適用してください。\n\n"
                f"改善提案:\n{suggestion[:800]}\n\n"
                + (f"特に適用したい改善: {detail}\n" if detail else "")
                + "既存の動作を維持しながら改善を適用し、WRITE: でファイルを更新してください。"
            )
            console.print(
                f"[magenta]自己改善モード[/magenta] — 対象: [bold]{files}[/bold]"
            )
            _run_agent(
                llm, apply_goal, "coding",
                context=_SELF_APPLY_CONTEXT,
                max_iterations=12,
            )

        # --- コード生成 ---
        elif raw.startswith("/generate"):
            description = raw[len("/generate"):].strip()
            if not description:
                console.print("[red]使い方: /generate <コードの説明>[/red]")
                continue
            with console.status("[cyan]コードを生成中...[/cyan]"):
                result = generator.generate(description)
            _display(result, title="生成されたコード")

        # --- コードレビュー ---
        elif raw == "/review":
            code = _collect_code()
            if not code.strip():
                console.print("[yellow]コードが入力されていません。[/yellow]")
                continue
            with console.status("[cyan]レビュー中...[/cyan]"):
                result = reviewer.review(code)
            _display(result, title="コードレビュー")

        # --- フリーテキスト: チャット or エージェントを自動判断 ---
        else:
            with console.status("[cyan]判断中...[/cyan]"):
                intent_type, domain = _classify_intent(fast_llm, raw)  # 高速モデルで分類

            if intent_type == "task":
                console.print(
                    f"[yellow]タスクを検出しました（domain=[cyan]{domain}[/cyan]）。"
                    f"エージェントを起動します...[/yellow]"
                )
                _run_agent(llm, raw, domain)  # スマートモデルで実行
            else:
                # 通常チャット（スマートモデル + 会話履歴付き）
                history.append({"role": "user", "content": raw})
                messages = [{"role": "system", "content": _GENERAL_SYSTEM}] + history
                with console.status("[cyan]考え中...[/cyan]"):
                    reply = llm.chat(messages, temperature=0.7, max_tokens=2048)
                if not reply:
                    console.print("[red]応答を取得できませんでした。[/red]")
                    history.pop()
                else:
                    history.append({"role": "assistant", "content": reply})
                    _display(reply, title="Hermes AI")


if __name__ == "__main__":
    main()
