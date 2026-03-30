from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.config import settings

def get_rag_chain():
    llm = ChatGroq(
        temperature=0.2,
        groq_api_key=settings.groq_api_key,
        model_name=settings.llm_model
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={'device': 'cpu'}
    )
    
    client = QdrantClient(url=settings.qdrant_url)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=embeddings,
    )
    
    retriever = vector_store.as_retriever(
        search_kwargs={"k": settings.top_k}
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
        return "\n\n".join(f"Source: {d.metadata.get('source')}\Content: {d.page_content}" for d in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_bot(question: str):
    formatted_query = f"query: {question}"
    chain = get_rag_chain()
    return chain.invoke(formatted_query)