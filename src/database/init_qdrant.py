import os
import asyncio
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import config
from src.secrets import secrets
from src.utils.logger import get_logger

os.environ["USER_AGENT"] = config.scraper.user_agent
os.environ["HUGGING_FACE_HUB_TOKEN"] = secrets.hf_token

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


def parse_content(soup):
    for s in soup(['nav', 'header', 'footer', 'aside', 'script', 'style']):
        s.decompose()

    return "passage: " + soup.get_text(separator=" ", strip=True)


def _load_sitemap(sitemap_url: str) -> tuple[str, list]:
    loader = SitemapLoader(web_path=sitemap_url, parsing_function=parse_content)
    loader.requests_per_second = config.scraper.requests_per_second
    return sitemap_url, loader.load()


async def main():
    logger.info("Starting database initialization process")

    logger.info(f"Loading embedding model: {config.embeddings.model}")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.embeddings.model,
        model_kwargs={'device': config.embeddings.device},
        encode_kwargs={'batch_size': config.embeddings.embed_batch_size},
    )

    logger.info(f"Scraping {len(SITEMAPS)} sitemaps in parallel...")
    all_docs = []
    with ThreadPoolExecutor(max_workers=len(SITEMAPS)) as executor:
        futures = {executor.submit(_load_sitemap, url): url for url in SITEMAPS}
        for future in as_completed(futures):
            url = futures[future]
            try:
                _, docs = future.result()
                all_docs.extend(docs)
                logger.info(f"Loaded {len(docs)} pages from {url}")
            except Exception as e:
                logger.error(f"Failed to process {url}: {str(e)}", exc_info=True)

    logger.info(f"Total documents loaded: {len(all_docs)}")

    logger.info("Splitting text into chunks")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.text_splitter.chunk_size,
        chunk_overlap=config.text_splitter.chunk_overlap,
        separators=config.text_splitter.separators,
    )
    chunks = text_splitter.split_documents(all_docs)
    logger.info(f"Created {len(chunks)} chunks from documents")

    logger.info(f"Connecting to Qdrant at {secrets.qdrant_url}")
    client = QdrantClient(url=secrets.qdrant_url)

    if client.collection_exists(secrets.qdrant_collection_name):
        logger.warning(f"Collection '{secrets.qdrant_collection_name}' already exists. Recreating...")
        client.delete_collection(secrets.qdrant_collection_name)

    client.create_collection(
        collection_name=secrets.qdrant_collection_name,
        vectors_config=models.VectorParams(
            size=config.embeddings.vector_size,
            distance=models.Distance[config.qdrant.distance],
        ),
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=secrets.qdrant_collection_name,
        embedding=embeddings,
    )

    batch_size = config.embeddings.embed_batch_size
    total_batches = math.ceil(len(chunks) / batch_size)
    logger.info(f"Uploading {len(chunks)} chunks in {total_batches} batches of {batch_size}...")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vector_store.add_documents(batch)
        logger.info(f"Batch {i // batch_size + 1}/{total_batches} uploaded")

    logger.info("Database initialization process completed successfully!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
    except Exception as e:
        logger.critical(f"Unexpected error during initialization: {str(e)}", exc_info=True)
