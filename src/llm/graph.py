from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.mongodb.saver import MongoDBSaver

from src.config import config
from src.llm.state import State
from src.llm.nodes import rag_answer_node, summarize_node


def _should_summarize(state: State) -> str:
    if len(state["messages"]) > config.memory.window_size:
        return "summarize"
    return END


def build_graph(checkpointer: MongoDBSaver):
    builder = StateGraph(State)

    builder.add_node("rag_answer", rag_answer_node)
    builder.add_node("summarize", summarize_node)

    builder.add_edge(START, "rag_answer")
    builder.add_conditional_edges(
        "rag_answer",
        _should_summarize,
        {"summarize": "summarize", END: END},
    )
    builder.add_edge("summarize", END)

    return builder.compile(checkpointer=checkpointer)


async def ask_bot_async(graph, question: str, thread_id: str) -> str:
    result = await graph.ainvoke(
        {"messages": [HumanMessage(question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content