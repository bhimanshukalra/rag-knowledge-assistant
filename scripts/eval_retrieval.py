import csv
import time
from pathlib import Path

from dotenv import load_dotenv
from flashrank import Ranker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from company_knowledge_assistant.config import (
    EMBEDDING_MODEL,
    EVAL_FILE,
    RERANK_CANDIDATE_K,
    RERANKER_MODEL,
    USERS,
)
from company_knowledge_assistant.documents import load_documents
from company_knowledge_assistant.permissions import user_can_access
from company_knowledge_assistant.reranking import rerank_documents_with_model
from company_knowledge_assistant.retrieval import hybrid_retrieve
from company_knowledge_assistant.vector_store import (
    get_vector_collection,
    index_documents,
)


def load_eval_questions(eval_file=EVAL_FILE):
    with eval_file.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def get_sources(documents):
    return {document.metadata["source"] for document in documents}


def print_report(results):
    total_questions = len(results)
    passed_count = sum(1 for result in results if result["success"])
    failed_count = total_questions - passed_count
    accuracy = (passed_count / total_questions) * 100 if total_questions else 0
    average_latency = (
        sum(result["latency"] for result in results) / total_questions
        if total_questions
        else 0
    )

    print("\nEvaluation Report")
    print("=================")
    print(f"Total questions: {total_questions}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Retrieval accuracy: {accuracy:.1f}%")
    print(f"Average latency: {average_latency:.2f}s")

    failed_results = [result for result in results if not result["success"]]

    print("\nFailed questions:")

    if not failed_results:
        print("None")
        return

    for result in failed_results:
        print()
        print(f"Question: {result['question']}")
        print(f"User: {result['username']}")
        print(f"Expected source: {result['expected_source'] or '(no source)'}")
        print(f"Forbidden source: {result['forbidden_source'] or '(none)'}")
        print(f"Retrieved sources: {', '.join(result['retrieved_sources'])}")
        print(f"Reason: {result['reason']}")


def evaluate():
    results = run_evaluation()
    print_report(results)


def run_evaluation():
    load_dotenv()

    documents = load_documents()
    embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    collection = get_vector_collection()
    indexed_count = index_documents(collection, documents, embedding_model)
    print(f"Indexed {indexed_count} new or changed chunks")
    reranker = Ranker(model_name=RERANKER_MODEL)

    rows = load_eval_questions()
    results = []

    for row in rows:
        start_time = time.perf_counter()
        question = row["question"]
        username = row["username"]
        expected_source = row["expected_source"].strip()
        forbidden_source = row["forbidden_source"].strip()

        if not expected_source and not forbidden_source:
            raise ValueError(
                "Eval row must include expected_source or forbidden_source: "
                f"{question}"
            )

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
        failure_reasons = []

        if expected_source and expected_source not in retrieved_sources:
            success = False
            failure_reasons.append("Expected source was not retrieved")

        if forbidden_source and forbidden_source in retrieved_sources:
            success = False
            failure_reasons.append("Forbidden source was retrieved")

        status = "PASS" if success else "FAIL"
        latency = time.perf_counter() - start_time
        sorted_sources = sorted(retrieved_sources)

        results.append(
            {
                "question": question,
                "username": username,
                "success": success,
                "expected_source": expected_source,
                "forbidden_source": forbidden_source,
                "retrieved_sources": sorted_sources or ["(no source)"],
                "latency": latency,
                "reason": "; ".join(failure_reasons) if failure_reasons else "Passed",
            }
        )

        print(f"{status} ({latency:.2f}s): {question}")
        print(f"expected: {expected_source or '(no source)'}")
        print(f"forbidden: {forbidden_source or '(none)'}")
        print(f"retrieved: {sorted_sources or ['(no source)']}")

    return results


if __name__ == "__main__":
    evaluate()
