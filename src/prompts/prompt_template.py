from langchain_core.documents import Document


def build_prompt(question: str, documents: list[Document]) -> str:
    """Build a grounded prompt from retrieved documents.

    Args:
        question: The user's question.
        documents: Retrieved LangChain documents.

    Returns:
        A formatted prompt string.

    Raises:
        ValueError: If the question is empty or no documents are provided.
    """
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    if not documents:
        raise ValueError("Documents must not be empty.")

    context = "\n--------------------\n".join(
        document.page_content.strip()
        for document in documents
    )

    return f"""
You are an AI assistant that answers questions using only the provided context.

Instructions:
- Answer ONLY using the provided context.
- Do not use outside knowledge.
- If the answer cannot be found in the context, reply:
  "I don't have enough information from the provided documents."
- Do not hallucinate or invent facts.

Context:
{context}

Question:
{question}

Answer:
"""