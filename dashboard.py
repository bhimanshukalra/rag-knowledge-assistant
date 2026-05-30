import streamlit as st

from eval_retrieval import run_evaluation

st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")


def calculate_metrics(results):
    total = len(results)
    passed = sum(1 for result in results if result["success"])
    failed = total - passed
    accuracy = (passed / total) * 100 if total else 0
    average_latency = (
        sum(result["latency"] for result in results) / total if total else 0
    )

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "accuracy": accuracy,
        "average_latency": average_latency,
    }


def format_sources(sources):
    return ", ".join(sources) if sources else "(no source)"


def build_table_rows(results):
    return [
        {
            "status": "PASS" if result["success"] else "FAIL",
            "question": result["question"],
            "username": result["username"],
            "expected_source": result["expected_source"] or "(no source)",
            "forbidden_source": result["forbidden_source"] or "(none)",
            "retrieved_sources": format_sources(result["retrieved_sources"]),
            "latency_seconds": round(result["latency"], 2),
            "reason": result["reason"],
        }
        for result in results
    ]


st.title("RAG Evaluation Dashboard")

if st.button("Run Evaluation"):
    with st.spinner("Running retrieval evaluation..."):
        st.session_state["results"] = run_evaluation()

results = st.session_state.get("results", [])

if results:
    metrics = calculate_metrics(results)

    total_column, passed_column, failed_column, accuracy_column = st.columns(4)
    total_column.metric("Total Questions", metrics["total"])
    passed_column.metric("Passed", metrics["passed"])
    failed_column.metric("Failed", metrics["failed"])
    accuracy_column.metric("Retrieval Accuracy", f"{metrics['accuracy']:.1f}%")

    latency_column, cost_column = st.columns(2)
    latency_column.metric(
        "Average Retrieval Latency", f"{metrics['average_latency']:.2f}s"
    )
    cost_column.metric("Estimated Cost", "Not tracked yet")

    st.info(
        "Cost is not tracked yet because the app does not capture token usage or "
        "per-request API pricing. We will add this once the evaluation stores usage data."
    )

    st.subheader("Results")

    table_rows = build_table_rows(results)
    usernames = sorted({row["username"] for row in table_rows})

    status_filter, user_filter = st.columns(2)
    selected_status = status_filter.selectbox("Status", ["All", "PASS", "FAIL"])
    selected_user = user_filter.selectbox("User", ["All", *usernames])

    filtered_rows = table_rows

    if selected_status != "All":
        filtered_rows = [
            row for row in filtered_rows if row["status"] == selected_status
        ]

    if selected_user != "All":
        filtered_rows = [
            row for row in filtered_rows if row["username"] == selected_user
        ]

    st.dataframe(filtered_rows, use_container_width=True, hide_index=True)

    st.subheader("Failed Questions")

    failed_rows = [row for row in table_rows if row["status"] == "FAIL"]

    if not failed_rows:
        st.success("No failed questions.")
    else:
        for row in failed_rows:
            with st.expander(row["question"]):
                st.write(f"User: {row['username']}")
                st.write(f"Expected source: {row['expected_source']}")
                st.write(f"Forbidden source: {row['forbidden_source']}")
                st.write(f"Retrieved sources: {row['retrieved_sources']}")
                st.write(f"Reason: {row['reason']}")
else:
    st.write("Run the evaluation to see retrieval metrics and failed questions.")
