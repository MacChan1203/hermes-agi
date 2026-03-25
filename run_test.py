from hermes_agi import AgentOrchestrator, MistralClient
llm = MistralClient()
orch = AgentOrchestrator(llm=llm)
print(orch.run('このプロジェクトの構造を調べてください'))
