class Planner:

    def plan(self, query):

        plan = {
            "task": "research_query",
            "query": query,
            "steps": [
                "retrieve_documents",
                "analyze_documents",
                "generate_summary"
            ]
        }

        return plan