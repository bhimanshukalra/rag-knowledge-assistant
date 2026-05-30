import re

from chromadb import Collection
from flashrank import Ranker
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from config import RERANK_CANDIDATE_K
from permissions import user_can_access
from reranking import rerank_documents_with_model
from retrieval import hybrid_retrieve


def build_context(documents: list[Document]):
    return "\n\n".join(
        (
            f"[{index}] Source: {doc.metadata['source']} | "
            f"Chunk: {doc.metadata['chunk']}\n"
            f"{doc.page_content}"
        )
        for index, doc in enumerate(documents, start=1)
    )


def build_sources(documents: list[Document]):
    return "\n".join(
        (f"[{index}] {doc.metadata['source']} | " f"Chunk: {doc.metadata['chunk']}")
        for index, doc in enumerate(documents, start=1)
    )


# Check for citation markers like [1] or [2], not arbitrary bracketed text.
def answer_has_citation(answer: str):
    return bool(re.search(r"\[\d+\]", answer))


def safe_invoke_llm(llm: ChatGoogleGenerativeAI, prompt: str):
    try:
        return llm.invoke(prompt).content, None
    except Exception as error:
        return None, error


def ask(
    question: str,
    documents: list[Document],
    collection: Collection,
    embedding_model: GoogleGenerativeAIEmbeddings,
    reranker: Ranker,
    llm: ChatGoogleGenerativeAI,
    user_groups: list[str],
):
    accessible_documents = [
        document for document in documents if user_can_access(document, user_groups)
    ]

    if not accessible_documents:
        return "I do not know based on the available sources."

    candidate_docs = hybrid_retrieve(
        question,
        accessible_documents,
        collection,
        embedding_model,
        user_groups,
        top_k=RERANK_CANDIDATE_K,
    )
    relevant_docs = rerank_documents_with_model(question, candidate_docs, reranker)
    context = build_context(relevant_docs)
    sources = build_sources(relevant_docs)
    prompt = f"""
Answer the question using only the context below.

Citation rules:
- Every factual claim must include a citation like [1].
- Only cite sources listed in the context.
- If the context does not contain the answer, say: I do not know based on the available sources.
- Do not cite a source unless it directly supports the sentence.

Context:
{context}
Question:
{question}

Answer:
"""
    response, error = safe_invoke_llm(llm, prompt)

    if error:
        return (
            "The assistant could not generate an answer because the model API failed. "
            "Please try again later.\n\n"
            f"Sources: \n{sources}"
        )

    if not answer_has_citation(response):
        return (
            "I do not know based on the available sources.\n\n" f"Sources: \n{sources}"
        )

    return f"{response}\n\nSources:\n{sources}"
