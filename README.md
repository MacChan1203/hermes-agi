# hermes-agent2

Hermes Agent の既存設計を踏まえつつ、v9 の **Plan → Act → Review** を中核にして再構成した軽量版です。

## この版で残したもの

- `hermes_time.py` のタイムゾーン解決と安全なフォールバック
- `minisweagent_path.py` の worktree / submodule 探索
- `utils.py` の atomic write
- `hermes_constants.py` の API エンドポイント定数
- `hermes_state.py` の方向性を受けた SQLite + FTS5 の `SessionDB`
- `toolsets.py`, `toolset_distributions.py` の toolset 発想
- `run_agent.py`, `cli.py`, `batch_runner.py`, `mini_swe_runner.py`, `trajectory_compressor.py` の入口

## この版で強化したもの

- `AgentState` による状態管理
- `Planner` による小さな計画生成
- `Executor` による観測中心の実行
- `Reviewer` による失敗分類と回復アクション
- セッション保存を `SessionDB` に統合
- 出力メッセージを極力日本語化

## すぐ試す

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_agent.py --query "Hermes Agent 2 の状態を見てください"
```

## CLI

```bash
python cli.py
python cli.py --query "このプロジェクトの次の改善案を出してください"
```

## 主要ファイル

- `hermes_agent2/agent_runner.py` : v9 の本体
- `hermes_agent2/state_store.py` : SQLite セッション保存
- `hermes_agent2/planner.py` : 動的 plan
- `hermes_agent2/executor.py` : shell ベース観測
- `hermes_agent2/reviewer.py` : review と recovery

## 設計上の意見

この版では、旧 Hermes の巨大な機能面をそのまま全部移すより、
**自律的に前進する骨格** を先に固める方が正しいと判断しました。

つまり、まずは

1. 現状把握
2. 小さく実行
3. 結果検証
4. 失敗時に立て直し

を確実に回せることを優先しています。

## 次にやると強いこと

- 既存 `tools/registry` と本格接続する
- `run_agent.py` に OpenAI/OpenRouter 呼び出しを戻す
- `cli.py` に旧 TUI を段階的に戻す
- `toolset` の requirement チェックを本格化する
- `SessionDB` にセッションタイトル自動生成と親子チェーンを入れる

