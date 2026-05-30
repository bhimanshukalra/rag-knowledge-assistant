import csv
from pathlib import Path

from dotenv import load_dotenv
from flashrank import Ranker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from rag import (
    EMBEDDING_MODEL,
    RERANK_CANDIDATE_K,
    RERANKER_MODEL,
    USERS,
    get_vector_collection,
    hybrid_retrieve,
    index_documents,
    load_documents,
    rerank_documents_with_model,
    user_can_access,
)

EVAL_FILE = Path("evals/questions.csv")


def load_eval_questions(eval_file=EVAL_FILE):
    with eval_file.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def get_sources(documents):
    return {document.metadata["source"] for document in documents}


def evaluate():
    load_dotenv()

    documents = load_documents()
    embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    collection = get_vector_collection()
    index_documents(collection, documents, embedding_model)
    reranker = Ranker(model_name=RERANKER_MODEL)

    rows = load_eval_questions()

    passed = 0

    for row in rows:
        question = row["question"]
        username = row["username"]
        expected_source = row["expected_source"].strip()
        forbidden_source = row["forbidden_source"].strip()

        if username not in USERS:
            raise ValueError(f"Unknown username in eval row: {username}")

        user_groups = USERS[username]["groups"]

        accessible_documents = [
            document for document in documents if user_can_access(document, user_groups)
        ]

        candidate_docs = hybrid_retrieve(
            question,
            accessible_documents,
            collection,
            embedding_model,
            user_groups,
            top_k=RERANK_CANDIDATE_K,
        )

        retrieved_docs = rerank_documents_with_model(question, candidate_docs, reranker)

        retrieved_sources = get_sources(retrieved_docs)

        success = True

        if expected_source and expected_source not in retrieved_sources:
            success = False

        if forbidden_source and forbidden_source in retrieved_sources:
            success = False

        if success:
            passed += 1

        status = "PASS" if success else "FAIL"

        print(f"{status}: {question}")
        print(f"expected: {expected_source or '(no source)'}")
        print(f"forbidden: {forbidden_source or '(none)'}")
        print(f"retrieved: {sorted(retrieved_sources) or ['(no source)']}")

    total = len(rows)
    accuracy = (passed / total) * 100 if total else 0

    print(
        f"Total questions: {total}\n"
        f"Passed: {passed}\n"
        f"Failed: {total-passed}\n"
        f"Retrieval accuracy: {accuracy:.1f}%"
    )


if __name__ == "__main__":
    evaluate()
