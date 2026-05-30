from dotenv import load_dotenv
from flashrank import Ranker
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from config import (
    CHAT_MODEL,
    DATA_DIR,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    USERS,
    VECTOR_DB_DIR,
)
from documents import load_documents
from generation import ask
from vector_store import get_vector_collection, index_documents


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

    username = input("What's your username? ").strip().lower()
    if username not in USERS:
        raise ValueError(f"Unknown user: {username}")

    user_groups = USERS[username]["groups"]

    question = input("Ask a question: ")
    reranker = Ranker(model_name=RERANKER_MODEL)

    answer = ask(
        question, documents, collection, embedding_model, reranker, llm, user_groups
    )

    print("\nAnswer: ")
    print(answer)


if __name__ == "__main__":
    main()
