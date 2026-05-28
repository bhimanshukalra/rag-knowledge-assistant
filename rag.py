from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.retrievers import TFIDFRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
DATA_DIR = Path("data/raw")
SUPPORTED_EXTENSIONS = {".txt", ".md"}
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def chunk_text(text, chunk_size = CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    if chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if(chunk):
            chunks.append(chunk)

        start = end - chunk_overlap
    
    return chunks

def load_documents():
    documents = []

    for path in sorted(DATA_DIR.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        content = path.read_text(encoding='utf-8').strip()

        if not content:
            continue

        for chunk_index, chunk in enumerate(chunk_text(content), start=1):
            documents.append(Document(page_content=chunk, metadata = {"source": path.name, "chunk": chunk_index}))

    return documents

documents = load_documents()

print(f"Loaded {len(documents)} chunks from {DATA_DIR}")

if not documents:
    raise ValueError(f"No supported files found in {DATA_DIR}")

retriever = TFIDFRetriever.from_documents(documents)
retriever.k = 2

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)

def ask(question):
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join(f"Source: {doc.metadata['source']} | Chunk: {doc.metadata['chunk']}\n{doc.page_content}" for doc in relevant_docs)
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

question = input("Ask a question: ")
answer = ask(question)

print("\nAnswer: ")
print(answer)