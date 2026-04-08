from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.config import config
from src.secrets import secrets
from src.utils.logger import get_logger

logger = get_logger(__name__)

def get_rag_chain():
    llm = ChatGroq(
        temperature=config.llm.temperature,
        groq_api_key=secrets.groq_api_key,
        model_name=config.llm.model,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=config.embeddings.model,
        model_kwargs={'device': config.embeddings.device},
    )

    client = QdrantClient(url=secrets.qdrant_url)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=secrets.qdrant_collection_name,
        embedding=embeddings,
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": config.qdrant.top_k}
    )

    template = """Ви — професійний консультант компанії.
Використовуйте надані фрагменти контексту, щоб відповісти на запитання користувача.
Якщо ви не знаєте відповіді, просто скажіть, що ви не знаєте, не намагайтеся вигадувати відповідь.

КОНТЕКСТ:
{context}

ПИТАННЯ: {question}

ВІДПОВІДЬ (відповідайте українською мовою, структуруйте текст):"""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        logger.info(f"Retrieved {len(docs)} chunks from Qdrant")

        formatted_chunks = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown')
            content_snippet = doc.page_content[:150].replace('\n', ' ')

            logger.info(f"Chunk {i+1} | Source: {source} | Content: {content_snippet}...")

            formatted_chunks.append(f"Source: {source}\nContent: {doc.page_content}")

        return "\n\n".join(formatted_chunks)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_bot(question: str) -> str:
    return get_rag_chain().invoke(f"query: {question}")


async def ask_bot_async(question: str) -> str:
    return await get_rag_chain().ainvoke(f"query: {question}")
