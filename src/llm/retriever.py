from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.config import config
from src.secrets import secrets
from src.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_retriever():
    logger.info(f"Loading embedding model: {config.embeddings.model}")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.embeddings.model,
        model_kwargs={"device": config.embeddings.device},
    )
    client = QdrantClient(url=secrets.qdrant_url)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=secrets.qdrant_collection_name,
        embedding=embeddings,
    )
    return vector_store.as_retriever(search_kwargs={"k": config.qdrant.top_k})