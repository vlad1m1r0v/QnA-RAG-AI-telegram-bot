from langchain_core.messages import HumanMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.mongodb.saver import MongoDBSaver
from langgraph.types import Send

from src.config import config
from src.llm.state import AgentState
from src.llm.nodes import (
    router_node,
    qna_node,
    extraction_node,
    validation_node,
    clarifying_node,
    brief_format_node,
    nonsense_node,
    summarize_node,
)


def _should_summarize(state: AgentState) -> str:
    if len(state["messages"]) > config.memory.window_size:
        return "summarize"
    return END


def _route_from_router(state: AgentState):
    if state.get("is_nonsense"):
        return "nonsense_node"
    sends = []
    if state.get("has_question"):
        sends.append(Send("qna_node", state))
    if state.get("has_project_info"):
        sends.append(Send("extraction_node", state))
    # Fallback: if neither flag is set treat as nonsense
    return sends if sends else "nonsense_node"


def _route_from_validation(state: AgentState) -> str:
    if state.get("brief_status") == "complete":
        return "brief_format_node"
    return "clarifying_node"


def build_graph(checkpointer: MongoDBSaver):
    builder = StateGraph(AgentState)

    builder.add_node("router_node", router_node)
    builder.add_node("qna_node", qna_node)
    builder.add_node("extraction_node", extraction_node)
    builder.add_node("validation_node", validation_node)
    builder.add_node("clarifying_node", clarifying_node)
    builder.add_node("brief_format_node", brief_format_node)
    builder.add_node("nonsense_node", nonsense_node)
    builder.add_node("summarize", summarize_node)

    # Entry point
    builder.add_edge(START, "router_node")

    # Fan-out: router → parallel qna/extraction (via Send) or nonsense
    builder.add_conditional_edges("router_node", _route_from_router)

    # Fan-in: both parallel nodes converge on validation
    builder.add_edge("qna_node", "validation_node")
    builder.add_edge("extraction_node", "validation_node")

    # Validation → format or clarify
    builder.add_conditional_edges(
        "validation_node",
        _route_from_validation,
        {"brief_format_node": "brief_format_node", "clarifying_node": "clarifying_node"},
    )

    # Terminal nodes → summarize or END
    for terminal in ["nonsense_node", "clarifying_node", "brief_format_node"]:
        builder.add_conditional_edges(
            terminal,
            _should_summarize,
            {"summarize": "summarize", END: END},
        )
    builder.add_edge("summarize", END)

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
                "summary": "",
                "project_type": None,
                "project_description": None,
                "goals": [],
                "key_features": [],
                "additional_features": [],
                "integrations": [],
                "client_materials": [],
                "brief_status": "in_progress",
                "response_type": "brief_clarifying",
                "qna_response": None,
                "empty_fields": [],
                "weak_fields": [],
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