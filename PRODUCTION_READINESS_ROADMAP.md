# Production Readiness Roadmap

This project is already a strong learning MVP. It has the right foundation:
hybrid search, vector database persistence, reranking, citations, permissions,
evaluation, a dashboard, and a clean package structure.

The goal now is to make it more robust, measurable, maintainable, and closer to
the expectations of a production RAG system.

## Product Positioning

Use this product story:

```text
Company Knowledge Assistant answers questions over internal documents, respects
user permissions, cites sources, and includes evaluation tooling to measure
retrieval quality.
```

The strongest product angle is:

```text
This is not just a chatbot over documents. It is a measurable RAG system with
permissions, hybrid retrieval, reranking, citations, and evaluation.
```

## What A Strong RAG Project Should Have

### 1. Clear Product Story

The README and demo should make the use case obvious:

- The assistant answers company knowledge questions.
- It retrieves from realistic internal documents.
- It refuses or avoids information the user is not allowed to access.
- It cites the sources used for every answer.
- It includes evaluation so quality can be measured.

### 2. Clean Architecture

The current package structure is a good base:

```text
company_knowledge_assistant/
scripts/
dashboard.py
data/
evals/
README.md
```

Keep improving it with:

- `tests/` for focused automated tests.
- `docs/` for screenshots, architecture notes, or demo assets.
- Clear separation between reusable app code and runnable scripts.

### 3. Realistic Sample Data

Add realistic company-style documents:

- company overview
- engineering handbook
- security policy
- HR or benefits policy
- product FAQ
- support or customer policy

The data should feel like something a real workplace assistant would search.

### 4. PDF Support

Add PDF parsing because real RAG systems often need it.

Minimum target:

- Load PDFs from `data/raw/`.
- Extract text.
- Preserve page numbers in metadata.
- Return citations like `security-guidelines.pdf, page 3`.
- Add eval questions that expect PDF sources.

### 5. Strong Evaluation Set

Grow `evals/questions.csv` to at least 50 questions.

Include:

- easy factual questions
- paraphrased questions
- exact keyword questions
- questions that require permission filtering
- questions where the answer does not exist
- questions that should retrieve PDF sources
- questions that test similar-looking documents

Track:

- retrieval accuracy
- forbidden-source leakage count
- average latency
- failed questions

### 6. Useful Dashboard

The dashboard should help explain system quality quickly.

Show:

- total eval questions
- pass rate
- retrieval accuracy
- average latency
- failed questions
- permission failures
- estimated cost
- answer quality scores when available

Later, add saved runs so the dashboard can show trends over time.

### 7. Focused Tests

Add a small but meaningful test suite.

Prioritize tests for:

- chunking behavior
- permission filtering
- citation detection
- eval row validation
- retrieval result formatting

You do not need huge test coverage. A few high-value tests are enough to show
engineering maturity.

### 8. Cost And Latency Tracking

Add simple usage metrics:

- embedding calls
- answer generation calls
- reranking calls
- latency per evaluation question
- estimated cost per evaluation run

Then surface those metrics in the dashboard.

### 9. Demo Flow

Prepare a short product demo:

1. Ask as Alice: an engineering question should retrieve engineering content.
2. Ask as Bob: the same engineering question should not expose restricted content.
3. Ask a question with no answer: the assistant should say it does not know.
4. Run the evaluation script.
5. Open the dashboard and show pass rate, latency, and failed cases.

This demo proves product behavior, security awareness, and measurement.

### 10. Strong README

The README should include:

- project overview
- architecture diagram
- feature list
- setup instructions
- run commands
- example questions
- sample answers with citations
- dashboard screenshot or screenshot placeholder
- evaluation summary
- known limitations
- future improvements

Many readers will skim the README before reading code, so the README should tell
the project story clearly.

### 11. Known Limitations

Keep an honest limitations section.

Mention things like:

- local ChromaDB instead of a managed vector database
- sample data instead of real company data
- simple local user model instead of real authentication
- basic chunking strategy
- cost tracking still estimated
- no async ingestion workers yet

This shows good engineering judgment.

## Minimum Path To A Strong Portfolio Project

If you want the shortest strong path, do these first:

1. Add PDF parsing with page citations.
2. Add 20 to 50 evaluation questions.
3. Add tests for permissions, chunking, and citations.
4. Add dashboard screenshots to the README.
5. Add one architecture diagram.
6. Add cost and latency metrics to eval/dashboard.
7. Add a demo script section to the README.
8. Add a known limitations section.

After these improvements, this can be presented as a serious portfolio-quality
RAG project, not just a tutorial implementation.

## Technical Talking Points

Be ready to explain:

- why hybrid search is useful
- why vector search alone is not enough
- how permissions are enforced before answering
- why reranking improves context quality
- how citations reduce hallucination risk
- what evaluation checks measure
- what the dashboard tells you
- what you would change for production

## Final Target

The final project should demonstrate that you understand both sides of RAG:

- the machine learning/product side: retrieval quality, citations, reranking,
  evaluation, and failure modes
- the engineering side: package structure, tests, permissions, observability,
  error handling, and maintainability
