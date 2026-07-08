"""Utilities for loading and querying the persisted Chroma vector store."""

from typing import Any

from src.embeddings.vector_store import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PERSIST_DIRECTORY,
)
from src.retrievers.retrieval_result import RetrievalResult

DEFAULT_RELEVANCE_THRESHOLD = 0.7


def load_vector_store() -> Any:
    """Load the persisted Chroma vector store.

    Returns:
        A Chroma vector store instance connected to the persisted collection.
    """
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )


def retrieve_documents(query: str, k: int = 3) -> list[RetrievalResult]:
    """Retrieve the most relevant documents for a query with similarity scores.

    Args:
        query: User query for semantic search.
        k: Number of documents to retrieve.

    Returns:
        A list of retrieval results containing the document and similarity score.

    Raises:
        ValueError: If the query is empty or k is less than 1.
    """
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    vector_store = load_vector_store()
    results_with_score = vector_store.similarity_search_with_score(query=query, k=k)

    relevant_results = [
        RetrievalResult(document=document, score=score)
        for document, score in results_with_score
        if score <= DEFAULT_RELEVANCE_THRESHOLD
    ]

    return relevant_results