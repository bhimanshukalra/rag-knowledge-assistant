from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document

from config import RERANK_TOP_K


def rerank_documents_with_model(
    question: str,
    documents: list[Document],
    reranker: Ranker,
    top_k: int = RERANK_TOP_K,
):
    if not documents:
        return []

    passages = [
        {
            "id": index,
            "text": document.page_content,
            "metadata": document.metadata,
        }
        for index, document in enumerate(documents)
    ]

    request = RerankRequest(query=question, passages=passages)
    results = reranker.rerank(request)

    reranked_documents = []

    for result in results[:top_k]:
        document_index = result["id"]
        reranked_documents.append(documents[document_index])

    return reranked_documents


def parse_rerank_response(response_text: str):
    indexes = []
    for part in response_text.split(","):
        part = part.strip()

        if not part.isdigit():
            continue

        indexes.append(int(part))

    return indexes
