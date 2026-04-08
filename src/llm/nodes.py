from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langchain_groq import ChatGroq

from src.config import config
from src.secrets import secrets
from src.llm.state import State
from src.llm.retriever import get_retriever
from src.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _get_llm() -> ChatGroq:
    return ChatGroq(
        temperature=config.llm.temperature,
        groq_api_key=secrets.groq_api_key,
        model_name=config.llm.model,
    )


def _format_docs(docs) -> str:
    logger.info(f"Retrieved {len(docs)} chunks from Qdrant")
    chunks = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        snippet = doc.page_content[:150].replace("\n", " ")
        logger.info(f"Chunk {i+1} | Source: {source} | Content: {snippet}...")
        chunks.append(f"Source: {source}\nContent: {doc.page_content}")
    return "\n\n".join(chunks)


async def rag_answer_node(state: State) -> dict:
    last_human = next(
        m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)
    )

    retriever = get_retriever()
    docs = await retriever.ainvoke(f"query: {last_human.content}")
    context = _format_docs(docs)

    summary = state.get("summary", "")
    system_text = (
        "Ви — професійний консультант компанії.\n"
        "Використовуйте надані фрагменти контексту, щоб відповісти на запитання користувача.\n"
        "Якщо ви не знаєте відповіді, просто скажіть, що ви не знаєте.\n"
        "Відповідайте українською мовою, структуруйте текст.\n\n"
        f"КОНТЕКСТ:\n{context}"
    )
    if summary:
        system_text += f"\n\nСУМАРИЗАЦІЯ ПОПЕРЕДНЬОЇ РОЗМОВИ:\n{summary}"

    response = await _get_llm().ainvoke(
        [SystemMessage(system_text)] + list(state["messages"])
    )
    return {"messages": [response]}


async def summarize_node(state: State) -> dict:
    messages = state["messages"]
    old_messages = messages[:-config.memory.window_size]
    existing_summary = state.get("summary", "")

    formatted = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in old_messages
    )
    summary_prompt = (
        f"Створіть стислу сумаризацію розмови не більше {config.memory.summary_max_sentences} речень.\n"
        + (f"Існуюча сумаризація: {existing_summary}\n\n" if existing_summary else "")
        + f"Повідомлення для сумаризації:\n{formatted}\n\nСумаризація:"
    )

    response = await _get_llm().ainvoke([HumanMessage(summary_prompt)])
    logger.info(f"Summarized {len(old_messages)} old messages, trimming state")

    return {
        "messages": [RemoveMessage(id=m.id) for m in old_messages],
        "summary": response.content,
    }