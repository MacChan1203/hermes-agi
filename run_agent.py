#!/usr/bin/env python3
"""Hermes AGI の最小ランナー。出力はできるだけ日本語。"""
from __future__ import annotations

from pathlib import Path
import argparse

from hermes_agi import HermesAgentV9


class AIAgent(HermesAgentV9):
    def run_conversation(self, message: str, task_id: str | None = None, **_: object) -> dict:
        rendered = self.chat(message)
        return {
            "completed": True,
            "api_calls": 0,
            "messages": [{"role": "assistant", "content": rendered}],
            "final_response": rendered,
            "partial": False,
        }


def main(query: str = "Hermes AGI の状態を見て、次の改善案を提案してください。", repo_root: str = ".", model: str = "local/mock-model", max_turns: int = 8):
    print("🤖 Hermes AGI")
    print("=" * 50)
    agent = AIAgent(repo_root=Path(repo_root), model=model, max_iterations=max_turns)
    result = agent.run_conversation(query)
    print(result["final_response"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', default="Hermes AGI の状態を見て、次の改善案を提案してください。")
    parser.add_argument('--repo_root', default='.')
    parser.add_argument('--model', default='local/mock-model')
    parser.add_argument('--max_turns', type=int, default=8)
    args = parser.parse_args()
    main(args.query, args.repo_root, args.model, args.max_turns)
