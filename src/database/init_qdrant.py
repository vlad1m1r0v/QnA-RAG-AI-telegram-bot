import os
import asyncio

from src.logger import get_logger
from src.config import settings

os.environ["USER_AGENT"] = settings.user_agent
os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.hf_token

import nest_asyncio
from langchain_community.document_loaders import SitemapLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = get_logger(__name__)

nest_asyncio.apply()

SITEMAPS = [
    "https://avada-media.com/sitemap.xml",
    "https://avadacrm.com/sitemap.xml",
    "https://cryptonislabs.com/sitemap.xml",
    "https://arionisgames.com/sitemap.xml",
    "https://cortexintellect.com/sitemap.xml",
]


QDRANT_COLLECTION_NAME = settings.collection_name
QDRANT_URL = settings.qdrant_url

def parse_content(soup):
    for s in soup(['nav', 'header', 'footer', 'aside', 'script', 'style']):
        s.decompose()

    return "passage: " + soup.get_text(separator=" ", strip=True)

async def main():
    logger.info("Starting database initialization process")
    
    logger.info(f"Loading embedding model: {settings.embedding_model}")
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={'device': 'cpu'}
    )

    all_docs = []
    for sitemap_url in SITEMAPS:
        logger.info(f"Parsing sitemap: {sitemap_url}")
        try:
            loader = SitemapLoader(
                web_path=sitemap_url,
                parsing_function=parse_content,
            )
            loader.requests_per_second = 2
            
            docs = loader.load() 
            
            all_docs.extend(docs)
            logger.info(f"Successfully loaded {len(docs)} pages from {sitemap_url}")
        except Exception as e:
            logger.error(f"Failed to process {sitemap_url}: {str(e)}", exc_info=True)

    logger.info(f"Total documents loaded: {len(all_docs)}")

    logger.info("Splitting text into chunks")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, 
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(all_docs)
    logger.info(f"Created {len(chunks)} chunks from documents")

    logger.info(f"Connecting to Qdrant at {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL)
    
    if client.collection_exists(QDRANT_COLLECTION_NAME):
        logger.warning(f"Collection '{QDRANT_COLLECTION_NAME}' already exists. Recreating...")
        client.delete_collection(QDRANT_COLLECTION_NAME)
        
    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
    )

    logger.info("Uploading vectors to Qdrant (this may take a few minutes)...")
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION_NAME,
    )

    logger.info("Database initialization process completed successfully!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
    except Exception as e:
        logger.critical(f"Unexpected error during initialization: {str(e)}", exc_info=True)