import hashlib
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from chromadb.api.models.Collection import Collection
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw"
SUPPORTED_EXTENSIONS = {".txt", ".md"}
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 2
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"
VECTOR_DB_DIR = BASE_DIR / "data" / "vector_db"
COLLECTION_NAME = "company_knowledge"


def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    if chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap

    return chunks


def load_documents(data_dir=DATA_DIR):
    documents: list[Document] = []

    for path in sorted(data_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        content = path.read_text(encoding="utf-8").strip()

        if not content:
            continue

        for chunk_index, chunk in enumerate(chunk_text(content), start=1):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"source": path.name, "chunk": chunk_index},
                )
            )

    return documents


def build_context(documents):
    return "\n\n".join(
        (
            f"Source: {doc.metadata['source']} | "
            f"Chunk: {doc.metadata['chunk']}\n"
            f"{doc.page_content}"
        )
        for doc in documents
    )


def ask(
    question: str,
    collection: Collection,
    embedding_model: GoogleGenerativeAIEmbeddings,
    llm: ChatGoogleGenerativeAI,
):
    relevant_docs = retrieve_from_vector_store(question, collection, embedding_model)
    context = build_context(relevant_docs)
    prompt = f"""
Answer the question using only the context below.
Include the source filename and chunk number for any information you use.
If the context does not contain the answer, say you do not know.

Context:
{context}
Question:
{question}

Answer:
"""
    response = llm.invoke(prompt)

    return response.content


def get_document_id(document):
    source = document.metadata["source"]
    chunk = document.metadata["chunk"]

    return f"{source}: chunk-{chunk}"


def get_document_hash(document):
    return hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()


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
        existing_metadata = existing_metadata_by_id.get(document_id) or {}

        if existing_metadata.get("content_hash") == content_hash:
            continue

        documents_to_index.append(document.page_content)
        ids_to_index.append(document_id)
        metadatas_to_index.append(
            {
                "source": document.metadata["source"],
                "chunk": document.metadata["chunk"],
                "content_hash": content_hash,
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
    top_k: int = TOP_K,
):
    question_embedding = embedding_model.embed_query(question)

    results = collection.query(query_embeddings=[question_embedding], n_results=top_k)

    retrieved_documents = []

    for content, metadata in zip(results["documents"][0], results["metadatas"][0]):
        retrieved_documents.append(Document(page_content=content, metadata=metadata))

    return retrieved_documents


def main():
    load_dotenv()

    documents = load_documents()
    print(f"Loaded {len(documents)} chunks from {DATA_DIR}")

    if not documents:
        raise ValueError(f"No supported files found in {DATA_DIR}")

    embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    collection = get_vector_collection()
    indexed_count = index_documents(collection, documents, embedding_model)

    print(f"Indexed {indexed_count} new or changed chunks in {VECTOR_DB_DIR}")

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)

    question = input("Ask a question: ")
    answer = ask(question, collection, embedding_model, llm)

    print("\nAnswer: ")
    print(answer)


if __name__ == "__main__":
    main()
