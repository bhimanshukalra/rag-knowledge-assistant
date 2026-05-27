from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.retrievers import TFIDFRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
DATA_DIR = Path("data/raw")
SUPPORTED_EXTENSIONS = {".txt", ".md"}

def load_documents():
    documents = []

    for path in sorted(DATA_DIR.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        content = path.read_text(encoding='utf-8').strip()

        if not content:
            continue

        documents.append(Document(page_content=content, metadata = {"source": path.name}))

    return documents

documents = load_documents()

if not documents:
    raise ValueError(f"No supported files found in {DATA_DIR}")

retriever = TFIDFRetriever.from_documents(documents)
retriever.k = 2

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)

def ask(question):
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join(f"Source: {doc.metadata['source']}\n{doc.page_content}" for doc in relevant_docs)
    prompt = f"""
Answer the question using only the context below.
Include the source filename for any information you use.
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