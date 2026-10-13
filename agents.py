import json
import re
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from config import config, get_llm
from tools import researcher_tools, web_tools, analyst_tools

RESEARCHER_PROMPT = """You are the Lead Researcher agent in the AgentIQ system.
Your responsibility is searching internal company documents, policies, and files using your knowledge base tool.
IMPORTANT: Whenever the user mentions 'uploaded file', 'my resume', 'the job description', or references attached documents, 
you MUST IMMEDIATELY call `search_knowledge_base` with keywords like 'resume', 'experience', 'skills', 'job description', or the document name.
Never say you cannot see the files without searching `search_knowledge_base` first."""

WEB_SEARCHER_PROMPT = """You are the Web Search Specialist in the AgentIQ system.
Your responsibility is searching the live internet for recent public facts, news, and external documentation.
Always include titles and source URLs for findings."""

ANALYST_PROMPT = """You are the Quantitative Analyst in the AgentIQ system.
Your responsibility is performing mathematical calculations, evaluating metrics, and synthesizing data.
Always use the calculate tool for arithmetic calculations to ensure 100% precision."""

SUPERVISOR_PROMPT = """You are the Executive Supervisor orchestrating a multi-agent AI team:
- 'Researcher': Searches internal company documentation and knowledge base.
- 'WebSearcher': Searches the live internet for external public information.
- 'Analyst': Solves math problems, performs calculations, and evaluates data.

Review the user's objective and past messages.
Choose the best specialist to act next, or 'FINISH' if the request has been fully answered.

You MUST respond strictly in valid JSON format with this exact structure:
{{
  "next": "Researcher" | "WebSearcher" | "Analyst" | "FINISH",
  "reasoning": "brief explanation"
}}"""


class AgentRoute(BaseModel):
    next: Literal["Researcher", "WebSearcher", "Analyst", "FINISH"] = "FINISH"
    reasoning: str = ""


def create_specialist_agent(llm, tools, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    if tools:
        return prompt | llm.bind_tools(tools)
    return prompt | llm


def parse_router_json(text: str) -> AgentRoute:
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            next_agent = data.get("next", "FINISH")
            if next_agent not in ["Researcher", "WebSearcher", "Analyst", "FINISH"]:
                next_agent = "FINISH"
            return AgentRoute(next=next_agent, reasoning=data.get("reasoning", ""))
        except Exception:
            pass
    return AgentRoute(next="FINISH", reasoning="Could not parse router output, completing.")


class SupervisorRouter:
    def __init__(self, llm):
        prompt = ChatPromptTemplate.from_messages([
            ("system", SUPERVISOR_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ])
        self.chain = prompt | llm | StrOutputParser()

    def invoke(self, inputs: dict) -> AgentRoute:
        raw_text = self.chain.invoke(inputs)
        return parse_router_json(raw_text)


def init_agents():
    llm = get_llm()
    return {
        "researcher": create_specialist_agent(llm, researcher_tools, RESEARCHER_PROMPT),
        "web_searcher": create_specialist_agent(llm, web_tools, WEB_SEARCHER_PROMPT),
        "analyst": create_specialist_agent(llm, analyst_tools, ANALYST_PROMPT),
        "supervisor": SupervisorRouter(llm),
    }


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    print("--- Initializing Specialist Agents ---")
    agents = init_agents()
    print("Agents initialized: Researcher, WebSearcher, Analyst, Supervisor")

    print("\n--- Testing Supervisor Decision Routing ---")
    test_queries = [
        "What is our company's remote work VPN policy?",
        "What is the current stock price of Google?",
        "If our revenue was $1,500,000 with 18% tax, what is the net revenue?",
    ]

    for q in test_queries:
        messages = [HumanMessage(content=q)]
        decision = agents["supervisor"].invoke({"messages": messages})
        print(f"Query: '{q}'\n  -> Routed to: {decision.next}\n  -> Reason: {decision.reasoning}\n")
