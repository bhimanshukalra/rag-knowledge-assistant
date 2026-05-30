import hashlib

from langchain_core.documents import Document

from company_knowledge_assistant.config import (
    DATA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    SUPPORTED_EXTENSIONS,
    SOURCE_ACCESS,
)


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
                    metadata={
                        "source": path.name,
                        "chunk": chunk_index,
                        "access_groups": SOURCE_ACCESS.get(path.name, ["general"]),
                    },
                )
            )

    return documents


def get_document_id(document):
    source = document.metadata["source"]
    chunk = document.metadata["chunk"]

    return f"{source}: chunk-{chunk}"


def get_document_hash(document):
    return hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()
