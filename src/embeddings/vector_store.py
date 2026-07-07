"""Utilities for creating a persistent Chroma vector store."""

from pathlib import Path
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def create_vector_store(documents: list[Document]) -> Chroma:
    """Create and persist a Chroma vector store.

    Args:
        documents: Chunked LangChain documents to embed and store.

    Returns:
        A persistent Chroma vector store.

    Raises:
        ValueError: If no documents are provided.
        RuntimeError: If the vector store cannot be created.
    """
    if not documents:
        raise ValueError("No documents were provided to create the vector store.")

    persist_directory = Path(PERSIST_DIRECTORY)

    if persist_directory.exists() and persist_directory.is_dir():
        shutil.rmtree(persist_directory)

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
        )

        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(persist_directory),
        )

        return vector_store

    except Exception as exc:
        raise RuntimeError("Failed to create the vector store.") from exc