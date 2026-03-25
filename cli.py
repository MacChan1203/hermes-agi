#!/usr/bin/env python3
"""Hermes Code Platform - インタラクティブ CLI。"""
from __future__ import annotations

from pathlib import Path

# .env を先にロードしてから hermes モジュールを import
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from hermes_agi.code_agents import CodeGeneratorAgent, CodeReviewerAgent
from hermes_agi.mistral_client import MistralClient
from hermes_agi.state_store import SessionDB

console = Console()

_HELP = """\
## 使えるコマンド

| コマンド | 説明 |
|---|---|
| `/generate <説明>` | 自然言語からコードを生成 |
| `/review` | コードのレビュー (貼り付けモード) |
| `/provider` | 現在のLLMプロバイダーを表示 |
| `/help` | このヘルプを表示 |
| `/quit` | 終了 |

**ヒント**: `/generate` の後の説明は日本語でも英語でも OK。
"""


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


def _display(result: str, title: str) -> None:
    console.print(Panel(Markdown(result), title=title, border_style="green"))


def main() -> None:
    llm = MistralClient()
    db = SessionDB()
    generator = CodeGeneratorAgent(llm=llm, session_db=db)
    reviewer = CodeReviewerAgent(llm=llm, session_db=db)

    console.print(Panel(
        f"[bold cyan]Hermes Code Platform[/bold cyan]\n"
        f"LLM: {_provider_label(llm)}\n"
        f"[dim]/help でコマンド一覧 / /quit で終了[/dim]",
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

        # 終了
        if raw in {"/quit", "/exit", "quit", "exit"}:
            console.print("[dim]終了します。[/dim]")
            break

        # ヘルプ
        elif raw == "/help":
            console.print(Markdown(_HELP))

        # プロバイダー表示
        elif raw == "/provider":
            console.print(f"LLM: {_provider_label(llm)}")
            console.print(f"[dim]DB : {db.db_path}[/dim]")

        # コード生成
        elif raw.startswith("/generate"):
            description = raw[len("/generate"):].strip()
            if not description:
                console.print("[red]使い方: /generate <コードの説明>[/red]")
                continue
            with console.status("[cyan]コードを生成中...[/cyan]"):
                result = generator.generate(description)
            _display(result, title="生成されたコード")

        # コードレビュー
        elif raw == "/review":
            code = _collect_code()
            if not code.strip():
                console.print("[yellow]コードが入力されていません。[/yellow]")
                continue
            with console.status("[cyan]レビュー中...[/cyan]"):
                result = reviewer.review(code)
            _display(result, title="コードレビュー")

        else:
            console.print(f"[yellow]不明なコマンド: {raw!r} — /help で一覧を確認してください[/yellow]")


if __name__ == "__main__":
    main()
