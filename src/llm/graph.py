from langchain_core.messages import HumanMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.mongodb.saver import MongoDBSaver
from langgraph.types import Send

from src.llm.state import AgentState
from src.llm.nodes import (
    router_node,
    qna_node,
    update_brief_node,
    validation_node,
    clarifying_node,
    brief_format_node,
    nonsense_node,
    summarize_node,
    estimation_node,
)


def _route_from_start(state: AgentState) -> str:
    last = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if last and last.content == "__ESTIMATE__":
        return "estimation_node"
    if len(state["messages"]) >= 9:
        return "summarize_node"
    return "router_node"


def _route_from_router(state: AgentState):
    if state.get("is_nonsense"):
        return "nonsense_node"
    sends = []
    if state.get("has_question"):
        sends.append(Send("qna_node", state))
    if state.get("has_project_info"):
        sends.append(Send("update_brief_node", state))
    return sends if sends else "nonsense_node"


def _route_from_validation(state: AgentState) -> str:
    if state.get("brief_status") == "complete":
        return "brief_format_node"
    return "clarifying_node"


def build_graph(checkpointer: MongoDBSaver):
    builder = StateGraph(AgentState)

    builder.add_node("summarize_node", summarize_node)
    builder.add_node("router_node", router_node)
    builder.add_node("qna_node", qna_node)
    builder.add_node("update_brief_node", update_brief_node)
    builder.add_node("validation_node", validation_node)
    builder.add_node("clarifying_node", clarifying_node)
    builder.add_node("brief_format_node", brief_format_node)
    builder.add_node("nonsense_node", nonsense_node)
    builder.add_node("estimation_node", estimation_node)

    # Entry point — summarize if needed, or go directly to router/estimation
    builder.add_conditional_edges(
        START,
        _route_from_start,
        {
            "estimation_node": "estimation_node",
            "summarize_node": "summarize_node",
            "router_node": "router_node",
        },
    )

    # Summarize always continues to router
    builder.add_edge("summarize_node", "router_node")

    # Fan-out: router → parallel qna/extraction (via Send) or nonsense
    builder.add_conditional_edges(
        "router_node",
        _route_from_router,
        ["qna_node", "update_brief_node", "nonsense_node"],
    )

    # Fan-in: both parallel nodes converge on validation
    builder.add_edge("qna_node", "validation_node")
    builder.add_edge("update_brief_node", "validation_node")

    # Validation → format or clarify
    builder.add_conditional_edges(
        "validation_node",
        _route_from_validation,
        {"brief_format_node": "brief_format_node", "clarifying_node": "clarifying_node"},
    )

    # Terminal nodes → END
    for terminal in ["nonsense_node", "clarifying_node", "brief_format_node", "estimation_node"]:
        builder.add_edge(terminal, END)

    return builder.compile(checkpointer=checkpointer)


async def reset_state_async(graph, thread_id: str) -> None:
    cfg = {"configurable": {"thread_id": thread_id}}
    current = await graph.aget_state(cfg)
    if current and current.values:
        messages = current.values.get("messages", [])
        await graph.aupdate_state(
            cfg,
            {
                "messages": [RemoveMessage(id=m.id) for m in messages],
                "summary": None,
                "brief": {
                    "project_type": None,
                    "project_description": None,
                    "goals": [],
                    "key_features": [],
                    "additional_features": [],
                    "integrations": [],
                    "client_materials": [],
                },
                "rejected_options": {},
                "estimation": None,
                "brief_status": "in_progress",
                "response_type": "brief_clarifying",
                "qna_response": None,
                "empty_fields": [],
            },
        )


async def ask_bot_async(graph, question: str, thread_id: str) -> dict:
    result = await graph.ainvoke(
        {"messages": [HumanMessage(question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return {
        "response_text": result["messages"][-1].content,
        "response_type": result.get("response_type", "brief_clarifying"),
    }