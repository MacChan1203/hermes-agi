from hermes_agent2 import AgentOrchestrator, MistralClient
llm = MistralClient()
orch = AgentOrchestrator(llm=llm)
print(orch.run('このプロジェクトの構造を調べてください'))
