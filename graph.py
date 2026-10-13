from typing import TypedDict, Annotated, Sequence, Dict, Any, Generator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from config import config, get_llm
from tools import researcher_tools, web_tools, analyst_tools
from agents import (
    RESEARCHER_PROMPT,
    WEB_SEARCHER_PROMPT,
    ANALYST_PROMPT,
    SupervisorRouter,
)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next: str
    steps_count: int


llm = get_llm()
researcher_agent = create_react_agent(llm, researcher_tools, prompt=RESEARCHER_PROMPT)
web_searcher_agent = create_react_agent(llm, web_tools, prompt=WEB_SEARCHER_PROMPT)
analyst_agent = create_react_agent(llm, analyst_tools, prompt=ANALYST_PROMPT)
supervisor_router = SupervisorRouter(llm)


def supervisor_node(state: AgentState) -> Dict[str, Any]:
    if state.get("steps_count", 0) >= config.agent.max_agent_steps:
        return {"next": "FINISH"}
    decision = supervisor_router.invoke({"messages": state["messages"]})
    return {"next": decision.next}


def researcher_node(state: AgentState) -> Dict[str, Any]:
    res = researcher_agent.invoke({"messages": state["messages"]})
    last_msg = res["messages"][-1].content
    return {
        "messages": [AIMessage(content=f"[Researcher Findings]:\n{last_msg}", name="Researcher")],
        "steps_count": state.get("steps_count", 0) + 1,
    }


def web_searcher_node(state: AgentState) -> Dict[str, Any]:
    res = web_searcher_agent.invoke({"messages": state["messages"]})
    last_msg = res["messages"][-1].content
    return {
        "messages": [AIMessage(content=f"[WebSearcher Findings]:\n{last_msg}", name="WebSearcher")],
        "steps_count": state.get("steps_count", 0) + 1,
    }


def analyst_node(state: AgentState) -> Dict[str, Any]:
    res = analyst_agent.invoke({"messages": state["messages"]})
    last_msg = res["messages"][-1].content
    return {
        "messages": [AIMessage(content=f"[Analyst Findings]:\n{last_msg}", name="Analyst")],
        "steps_count": state.get("steps_count", 0) + 1,
    }


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    specialist_messages = [
        m for m in state["messages"]
        if isinstance(m, AIMessage) and getattr(m, "name", "") in ["Researcher", "WebSearcher", "Analyst"]
    ]

    if not specialist_messages:
        last_msg = state["messages"][-1].content
        return {"messages": [AIMessage(content=last_msg, name="AgentIQ")]}

    combined_findings = "\n\n".join([m.content for m in specialist_messages])
    user_query = state["messages"][0].content

    synthesis_prompt = [
        SystemMessage(
            content="You are AgentIQ, an elite AI research assistant. Your task is to provide the final comprehensive, "
            "beautifully formatted answer to the user. Present all the specialist findings clearly using clean markdown. "
            "Include source citations. NEVER output JSON, code tags, or action commands."
        ),
        HumanMessage(
            content=f"User Question: {user_query}\n\nSpecialist Findings:\n{combined_findings}\n\nFinal Answer:"
        ),
    ]

    response = llm.invoke(synthesis_prompt)
    final_text = response.content.strip()

    if ("FINISH" in final_text and len(final_text) < 100) or final_text.startswith("{"):
        clean_text = combined_findings
        for prefix in ["[WebSearcher Findings]:", "[Researcher Findings]:", "[Analyst Findings]:"]:
            clean_text = clean_text.replace(prefix, "")
        final_text = clean_text.strip()

    return {"messages": [AIMessage(content=final_text, name="AgentIQ")]}


def route_supervisor(state: AgentState) -> str:
    target = state.get("next", "FINISH")
    if target == "Researcher":
        return "researcher"
    elif target == "WebSearcher":
        return "web_searcher"
    elif target == "Analyst":
        return "analyst"
    return "synthesizer"


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("web_searcher", web_searcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("synthesizer", synthesizer_node)

    workflow.add_edge(START, "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "researcher": "researcher",
            "web_searcher": "web_searcher",
            "analyst": "analyst",
            "synthesizer": "synthesizer",
        },
    )

    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("web_searcher", "supervisor")
    workflow.add_edge("analyst", "supervisor")
    workflow.add_edge("synthesizer", END)

    return workflow.compile()


agent_app = build_graph()


def run_workflow(user_query: str) -> Dict[str, Any]:
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "next": "",
        "steps_count": 0,
    }
    final_state = agent_app.invoke(initial_state)
    return {
        "query": user_query,
        "final_answer": final_state["messages"][-1].content,
        "history": final_state["messages"],
    }


def stream_workflow(user_query: str) -> Generator[Dict[str, Any], None, None]:
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "next": "",
        "steps_count": 0,
    }
    for event in agent_app.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            if "messages" in node_output and node_output["messages"]:
                msg = node_output["messages"][-1]
                yield {
                    "node": node_name,
                    "sender": getattr(msg, "name", node_name),
                    "content": msg.content,
                }
            elif "next" in node_output:
                yield {
                    "node": node_name,
                    "next": node_output["next"],
                }


if __name__ == "__main__":
    print("--- Testing AgentIQ LangGraph End-to-End Workflow ---")
    test_query = "What is the policy for VPN usage and core working hours according to our company rules?"

    print(f"User Query: {test_query}\n")
    print("Streaming agent execution trace:")
    for step in stream_workflow(test_query):
        if "next" in step:
            print(f"  [Supervisor Decision] -> Next Specialist: {step['next']}")
        else:
            print(f"  [{step['sender']}] completed step ({len(step['content'])} characters)")

    print("\n--- Workflow Execution Completed ---")
