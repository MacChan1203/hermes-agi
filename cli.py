#!/usr/bin/env python3
"""Hermes Agent 2 の簡易 CLI。"""
from __future__ import annotations

import argparse

from run_agent import AIAgent


def main(query: str | None = None, repo_root: str = ".", model: str = "local/mock-model", max_turns: int = 8):
    agent = AIAgent(repo_root=repo_root, model=model, max_iterations=max_turns)
    if query:
        print(agent.chat(query))
        return

    print("Hermes Agent 2 へようこそ。終了するには exit と入力してください。")
    while True:
        try:
            text = input("Hermes> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("
終了します。")
            break
        if text.lower() in {"exit", "quit"}:
            print("終了します。")
            break
        if not text:
            continue
        print(agent.chat(text))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', default=None)
    parser.add_argument('--repo_root', default='.')
    parser.add_argument('--model', default='local/mock-model')
    parser.add_argument('--max_turns', type=int, default=8)
    args = parser.parse_args()
    main(args.query, args.repo_root, args.model, args.max_turns)
