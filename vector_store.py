import chromadb
from chromadb.api.models.Collection import Collection
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import (
    VECTOR_DB_DIR,
    COLLECTION_NAME,
    TOP_K,
)
from documents import get_document_hash, get_document_id
from permissions import (
    build_access_filter,
    build_access_metadata,
    metadata_matches_access_groups,
    normalize_access_groups,
    user_can_access,
)


def get_vector_collection():
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

    return client.get_or_create_collection(name=COLLECTION_NAME)


def index_documents(
    collection: Collection,
    documents: list[Document],
    embedding_model: GoogleGenerativeAIEmbeddings,
):
    existing = collection.get(include=["metadatas"])
    existing_metadata_by_id = dict(zip(existing["ids"], existing["metadatas"]))

    documents_to_index = []
    ids_to_index = []
    metadatas_to_index = []

    for document in documents:
        document_id = get_document_id(document)
        content_hash = get_document_hash(document)
        access_groups = document.metadata["access_groups"]
        existing_metadata = existing_metadata_by_id.get(document_id) or {}

        if existing_metadata.get(
            "content_hash"
        ) == content_hash and metadata_matches_access_groups(
            existing_metadata, access_groups
        ):
            continue

        documents_to_index.append(document.page_content)
        ids_to_index.append(document_id)
        metadatas_to_index.append(
            {
                "source": document.metadata["source"],
                "chunk": document.metadata["chunk"],
                "content_hash": content_hash,
                **build_access_metadata(access_groups),
            }
        )

    if not documents_to_index:
        return 0

    embeddings = embedding_model.embed_documents(documents_to_index)

    collection.upsert(
        ids=ids_to_index,
        documents=documents_to_index,
        metadatas=metadatas_to_index,
        embeddings=embeddings,
    )

    return len(documents_to_index)


def retrieve_from_vector_store(
    question: str,
    collection: Collection,
    embedding_model: GoogleGenerativeAIEmbeddings,
    user_groups: list[str],
    top_k: int = TOP_K,
):
    question_embedding = embedding_model.embed_query(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        where=build_access_filter(user_groups),
    )

    retrieved_documents: list[Document] = []

    for content, metadata in zip(results["documents"][0], results["metadatas"][0]):
        metadata["access_groups"] = normalize_access_groups(metadata)
        document = Document(page_content=content, metadata=metadata)
        if user_can_access(document, user_groups):
            retrieved_documents.append(document)
        if len(retrieved_documents) == top_k:
            break

    return retrieved_documents
