from typing import Literal

from chromadb.api.models.Collection import Collection
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import RRF_K, TOP_K
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from vector_store import retrieve_from_vector_store


def retrieve_with_keywords(
    question: str, documents: list[Document], top_k: int = TOP_K
):
    vectorizer = TfidfVectorizer()
    document_vectors = vectorizer.fit_transform(
        [document.page_content for document in documents]
    )
    question_vector = vectorizer.transform([question])

    similarities = cosine_similarity(question_vector, document_vectors).flatten()
    ranked_indexes = similarities.argsort()[::-1][:top_k]

    return [documents[index] for index in ranked_indexes]


def get_document_key(document: Document):
    return (document.metadata["source"], document.metadata["chunk"])


def add_rrf_scores(
    scores: dict,
    documents: list[Document],
    source_name: Literal["vector", "keyword"],
    rrf_k: int = RRF_K,
):
    for rank, document in enumerate(documents, start=1):
        key = get_document_key(document)

        if key not in scores:
            scores[key] = {
                "document": document,
                "score": 0,
                "sources": [],
            }

        scores[key]["score"] += 1 / (rrf_k + rank)
        scores[key]["sources"].append(source_name)


def hybrid_retrieve(
    question: str,
    accessible_documents: list[Document],
    collection: Collection,
    embedding_model: GoogleGenerativeAIEmbeddings,
    user_groups: list[str],
    top_k: int = TOP_K,
):
    keyword_docs = retrieve_with_keywords(question, accessible_documents, top_k=top_k)
    vector_docs = retrieve_from_vector_store(
        question, collection, embedding_model, user_groups, top_k=top_k
    )

    scores: dict = {}

    add_rrf_scores(scores, vector_docs, source_name="vector")
    add_rrf_scores(scores, keyword_docs, source_name="keyword")

    ranked_results = sorted(
        scores.values(), key=lambda item: item["score"], reverse=True
    )

    return [item["document"] for item in ranked_results[:top_k]]
