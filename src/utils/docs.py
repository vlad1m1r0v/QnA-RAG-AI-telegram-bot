from src.utils.logger import get_logger

logger = get_logger(__name__)


def format_docs(docs) -> str:
    logger.info(f"Retrieved {len(docs)} chunks from Qdrant")
    chunks = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        snippet = doc.page_content[:150].replace("\n", " ")
        logger.info(f"Chunk {i + 1} | Source: {source} | Content: {snippet}...")
        chunks.append(f"Source: {source}\nContent: {doc.page_content}")
    return "\n\n".join(chunks)