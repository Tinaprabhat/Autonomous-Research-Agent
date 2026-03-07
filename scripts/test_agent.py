import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from backend.app.agents.research_agent import ResearchAgent

agent = ResearchAgent()

query = "What are the main approaches to meta-learning?"

result = agent.run(query)

print("\nResearch Agent Output:\n")
print(result)