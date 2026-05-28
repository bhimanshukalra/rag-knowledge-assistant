from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw"
SUPPORTED_EXTENSIONS = {".txt", ".md"}
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 2
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"


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
    documents = []

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


def retrieve_with_embeddings(
    question, documents, document_embeddings, embedding_model, top_k=TOP_K
):
    question_embedding = embedding_model.embed_query(question)

    similarities = cosine_similarity(
        [question_embedding], document_embeddings
    ).flatten()

    ranked_indexes = similarities.argsort()[::-1][:top_k]

    return [documents[index] for index in ranked_indexes]


def build_context(documents):
    return "\n\n".join(
        (
            f"Source: {doc.metadata['source']} | "
            f"Chunk: {doc.metadata['chunk']}\n"
            f"{doc.page_content}"
        )
        for doc in documents
    )


def ask(question, documents, document_embeddings, embedding_model, llm):
    relevant_docs = retrieve_with_embeddings(
        question, documents, document_embeddings, embedding_model
    )
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


def main():
    load_dotenv()

    documents = load_documents()
    print(f"Loaded {len(documents)} chunks from {DATA_DIR}")

    if not documents:
        raise ValueError(f"No supported files found in {DATA_DIR}")

    embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    document_embeddings = embedding_model.embed_documents(
        [doc.page_content for doc in documents]
    )

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)

    question = input("Ask a question: ")
    answer = ask(question, documents, document_embeddings, embedding_model, llm)

    print("\nAnswer: ")
    print(answer)


if __name__ == "__main__":
    main()
