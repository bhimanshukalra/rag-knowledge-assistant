# Company Knowledge Assistant

A project-based learning repository for understanding Python and Retrieval-Augmented
Generation (RAG) by building a serious, production-shaped knowledge assistant.

The goal is not to jump straight into a complex system. The goal is to grow from a
small Python script into a realistic RAG application one milestone at a time, while
learning the important concepts behind each layer.

## Project Goal

Build a company knowledge assistant that can answer questions using internal
company content, while showing citations and respecting user access permissions.

Example questions:

- "What is our refund policy for enterprise customers?"
- "Where is the onboarding checklist for new engineers?"
- "What did the product team decide about pricing in the Q3 notes?"
- "Which documents support this answer?"

## Target Features

By the end of this project, the assistant should support:

- Uploading PDFs, documents, webpages, and Slack-like text exports.
- Chunking source content into searchable pieces.
- Creating embeddings for chunks.
- Storing chunks and embeddings in a vector database.
- Hybrid search using both keyword search and vector search.
- Reranking retrieved results before answering.
- Citations in every answer.
- User permissions so different users can access different documents.
- An evaluation set with 50 realistic questions and expected sources.
- A dashboard showing retrieval accuracy, answer quality, latency, and cost.

## What We Will Learn

This project is meant to teach:

- Python fundamentals through practical code.
- How RAG systems work.
- Embeddings and semantic search.
- Vector databases.
- Chunking strategies.
- Keyword search vs vector search vs hybrid search.
- Reranking.
- Prompting with retrieved context.
- Source citations.
- RAG failure modes.
- Evaluation-driven development.
- Basic observability for latency and cost.

## High-Level Architecture

The target system can be thought of as two main flows.

### Ingestion Flow

```text
Source files / webpages / exports
        |
        v
Document loaders
        |
        v
Text extraction and cleanup
        |
        v
Chunking
        |
        v
Embedding
        |
        v
Vector database + metadata store
```

### Question Answering Flow

```text
User question
        |
        v
Permission filter
        |
        v
Hybrid retrieval
        |
        v
Reranking
        |
        v
Prompt with context
        |
        v
LLM answer with citations
        |
        v
Metrics and evaluation logs
```

## Suggested Milestones

### Milestone 1: Python and Minimal RAG

- Keep a small list of documents in Python.
- Ask a question from the terminal.
- Retrieve relevant text.
- Send retrieved text to an LLM.
- Print an answer.

Learning focus:

- Python functions
- Lists and strings
- Environment variables
- Basic LangChain concepts
- The core RAG loop

### Milestone 2: Load Real Files

- Add a `data/` folder.
- Load `.txt` and `.md` files.
- Preserve source metadata such as filename and page/section.
- Return citations from source metadata.

Learning focus:

- File handling in Python
- Data modeling
- Metadata
- Why citations need to be planned early

### Milestone 3: Chunking

- Split long documents into smaller chunks.
- Experiment with chunk size and overlap.
- Store chunk IDs and source references.

Learning focus:

- Chunking strategies
- Recall vs precision
- Context window limits
- How bad chunking causes bad answers

### Milestone 4A: In-Memory Embeddings and Vector Search

- Generate embeddings for each chunk.
- Store embeddings in memory while the script is running.
- Retrieve chunks by semantic similarity.
- Compare semantic retrieval with the earlier TF-IDF retrieval behavior.

Learning focus:

- Embeddings
- Vector similarity
- Semantic search
- Why semantic search is different from keyword search

### Milestone 4B: Persist Embeddings in a Vector Database

- Choose a beginner-friendly vector database.
- Store chunks, embeddings, and metadata in the vector database.
- Load existing indexed chunks instead of rebuilding everything every run.
- Retrieve chunks from the vector database by semantic similarity.

Learning focus:

- Vector databases
- Persistence
- Indexing
- Metadata storage and filtering

### Milestone 5A: Simple Hybrid Search

- Add keyword search.
- Combine keyword results with vector results.
- Compare retrieval quality across search methods.

Learning focus:

- Keyword matching
- Semantic matching
- Hybrid retrieval
- Ranking tradeoffs

### Milestone 5B: Reciprocal Rank Fusion

- Retrieve ranked candidates from keyword search and vector search.
- Combine the two ranked lists with Reciprocal Rank Fusion.
- Prefer chunks that rank well across multiple retrieval methods.
- Compare RRF results with the simple hybrid merge.

Learning focus:

- Rank fusion
- Why raw keyword and vector scores are hard to compare directly
- Balancing exact-match and semantic retrieval
- Preparing better candidates for reranking

### Milestone 6A: LLM-Based Reranking

- Retrieve more candidate chunks than needed.
- Ask the LLM to choose the most relevant chunks.
- Send only selected chunks to the answer prompt.
- Add fallback behavior if the reranking output is invalid.

Learning focus:

- LLM-based relevance judgment
- Candidate retrieval vs final context selection
- Reducing irrelevant context
- Reranking failure modes

### Milestone 6B: Dedicated Reranking Model

- Use a reranker model instead of the answer LLM.
- Compare dedicated reranking with LLM-based reranking.
- Measure latency, cost, and quality tradeoffs.
- Decide which reranking path is more appropriate for the project.

Learning focus:

- Dedicated reranker models
- Cross-encoder style relevance scoring
- Cost and latency tradeoffs
- Production reranking patterns

### Milestone 7: Citations

- Require every answer to include citations.
- Link each cited claim back to source chunks.
- Make the assistant say when the sources do not contain enough information.

Learning focus:

- Grounded answers
- Source attribution
- Hallucination reduction
- Trustworthy UX

### Milestone 8: Permissions

- Add users.
- Assign documents to users or groups.
- Filter retrieval results by permissions before answering.

Learning focus:

- Access control
- Metadata filtering
- Security risks in RAG systems
- Why retrieval must respect permissions before generation

### Milestone 9: Evaluation

- Create 50 realistic test questions.
- Record expected source documents for each question.
- Measure whether the retriever finds the right sources.
- Track answer quality manually at first, then with helper metrics.

Learning focus:

- Retrieval accuracy
- Answer evaluation
- Regression testing
- Building confidence through measurements

### Milestone 10A: CLI Evaluation Report

- Extend the evaluation script with a clearer terminal report.
- Show total questions, passed checks, failed checks, and retrieval accuracy.
- List failed questions with expected, forbidden, and retrieved sources.
- Add simple latency measurements per question.
- Keep the report easy to run from the command line.

Learning focus:

- Observability
- Retrieval metrics
- Debugging failed cases
- Latency measurement

### Milestone 10B: Lightweight Dashboard UI

- Build a small dashboard using the evaluation results.
- Show retrieval accuracy.
- Show answer quality scores when available.
- Show latency.
- Show estimated cost.
- Track changes over time.

Learning focus:

- Product feedback loops
- Cost awareness
- Measuring RAG system health
- Turning evaluation data into a useful interface

## Current Repository Structure

The reusable RAG code lives in the `company_knowledge_assistant` package. Runnable
entrypoints live in `scripts/`, and the Streamlit dashboard stays at the project
root.

```text
.
├── README.md
├── pyproject.toml
├── dashboard.py
├── company_knowledge_assistant/
│   ├── __init__.py
│   ├── config.py
│   ├── documents.py
│   ├── generation.py
│   ├── permissions.py
│   ├── reranking.py
│   ├── retrieval.py
│   └── vector_store.py
├── scripts/
│   ├── eval_retrieval.py
│   └── rag.py
├── data/
│   └── raw/
│       ├── company-overview.txt
│       ├── engineering-handbook.md
│       └── security-guidelines.pdf
└── evals/
    └── questions.csv
```

Main modules:

- `config.py`: constants, model names, paths, users, and source access rules.
- `documents.py`: file loading, chunking, document IDs, and content hashes.
- `permissions.py`: user group checks and access metadata helpers.
- `vector_store.py`: ChromaDB collection setup, indexing, and vector retrieval.
- `retrieval.py`: TF-IDF keyword retrieval, vector retrieval, and hybrid search.
- `reranking.py`: FlashRank reranking helpers.
- `generation.py`: prompt construction, citation checks, safe LLM calls, and answers.
- `scripts/rag.py`: terminal app for asking questions.
- `scripts/eval_retrieval.py`: retrieval evaluation CLI report.
- `dashboard.py`: Streamlit evaluation dashboard.

## Key Concepts

### RAG

Retrieval-Augmented Generation means the system retrieves relevant information
first, then gives that information to a language model so it can answer with
context.

### Embeddings

Embeddings are numeric representations of text. Text with similar meaning should
have similar embeddings, which makes semantic search possible.

### Vector Database

A vector database stores embeddings and lets us search for nearby vectors. In a
RAG system, nearby vectors usually represent chunks of text that are semantically
related to the user's question.

### Chunking

Chunking is the process of splitting large documents into smaller pieces. Good
chunking helps retrieval. Poor chunking can hide the answer, remove important
context, or retrieve irrelevant text.

### Hybrid Search

Hybrid search combines keyword search and vector search. Keyword search is good
for exact terms, names, IDs, and acronyms. Vector search is good for meaning and
paraphrases.

### Reranking

Reranking takes the initial search results and sorts them again using a more
precise model or scoring method. This helps choose the best context before asking
the LLM to answer.

### Evaluation

Evaluation tells us whether the system is improving. For RAG, we care about
whether the system retrieved the right sources and whether the final answer is
faithful to those sources.

## RAG Failure Modes To Watch For

- The right document is not ingested.
- The document is ingested but chunked poorly.
- The retriever finds the wrong chunks.
- The retriever finds the right chunks but ranks them too low.
- The LLM ignores the context.
- The LLM invents facts not present in the sources.
- Citations point to sources that do not actually support the answer.
- Permission filters are applied too late or incorrectly.
- Evaluation questions are too easy and hide real problems.

## Development Philosophy

We will build this in small, working steps.

Each milestone should leave the project in a runnable state. When we add a new
feature, we should also add a small way to test or evaluate it. This keeps the
project beginner-friendly while still moving toward a serious RAG system.

## Current Run Commands

Run the terminal assistant:

```bash
python scripts/rag.py
```

Run the retrieval evaluation report:

```bash
python scripts/eval_retrieval.py
```

Run the dashboard:

```bash
streamlit run dashboard.py
```

## Environment Notes

This project targets Python 3.11 or newer. The current dependencies are listed in
`pyproject.toml`.

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

The scripts use environment variables through `python-dotenv`, so API keys
should live in a local `.env` file. The `.env` file should not be committed.

Example:

```text
GOOGLE_API_KEY=your_api_key_here
```

## Next Improvement Steps

The current project is a strong learning MVP. To keep improving it, work in small
focused steps:

1. Add tests for core behavior.
   - Test chunking.
   - Test permission filtering.
   - Test citation detection.
   - Test retrieval evaluation logic.

2. Add PDF support.
   - Extract text from PDFs in `data/raw/`.
   - Preserve page numbers in metadata.
   - Show page numbers in citations.

3. Improve chunking.
   - Move from character-based chunking to token-aware chunking.
   - Try splitting by headings and paragraphs.
   - Compare retrieval results across chunk sizes.

4. Improve evaluation.
   - Grow `evals/questions.csv` toward 50 realistic questions.
   - Add more permission-related eval cases.
   - Track retrieval accuracy over time.

5. Add cost and usage tracking.
   - Count embedding calls.
   - Count answer-generation calls.
   - Estimate cost per evaluation run.
   - Show cost in the dashboard.

6. Improve error handling.
   - Add clearer handling for API failures.
   - Add timeouts and retry behavior.
   - Make dashboard failures easier to understand.

7. Add a real ingestion workflow.
   - Track which files have been indexed.
   - Support reindexing changed files.
   - Support deleting removed documents from the vector database.

8. Add a user-facing app.
   - Build a simple web chat interface.
   - Add document upload.
   - Show citations as clickable source references.

9. Move toward production readiness.
   - Add authentication.
   - Store users and document permissions in a database.
   - Add logging, tracing, and monitoring.
   - Run evaluation checks in CI before merging changes.

## High-ROI Showcase TODO

These are the most worthwhile next improvements if the goal is to showcase this
as a focused AI engineering project without turning it into a full SaaS:

- [x] Fix `DATA_DIR` and `VECTOR_DB_DIR` pathing.
      After moving code into `company_knowledge_assistant/`, paths should resolve
      from the repository root, not from inside the package directory.

- [x] Add PDF parsing with page-level citations.
      This makes ingestion more realistic and improves trust in source attribution.

- [ ] Expand evals to 25 to 50 strong questions.
      Cover normal questions, restricted-access questions, no-answer questions,
      exact-match questions, semantic questions, and PDF-backed questions.

- [ ] Add a small test suite.
      Focus on chunking, permissions, retrieval, citation detection, and evaluation
      row validation.

- [ ] Add README screenshots and a short architecture diagram.
      Make it easy to understand the system quickly before reading the code.

- [ ] Add a demo flow section.
      Show how to run ingestion, ask a question as Alice, ask a restricted question
      as Bob, and view the evaluation dashboard.

- [ ] Add a known limitations section.
      Clearly mention local ChromaDB, simple local users, basic chunking, no real
      upload workflow, and no production deployment yet.

## RAG-Specific Improvement Areas

These are the main areas to improve if focusing only on the RAG system itself,
separate from UI, tests, packaging, or deployment:

- Parsing: support PDFs, DOCX, HTML, and other source formats while preserving
  source metadata.
- Chunking: move beyond character-based chunks toward token-aware and
  structure-aware splitting.
- Metadata: store page numbers, headings, document type, access groups, chunk
  IDs, and document versions.
- Embeddings: avoid embedding noisy chunks, re-embed only changed content, and
  compare embedding model quality.
- Vector store lifecycle: support changed files, deleted files, rebuilds, and
  reliable metadata filtering.
- Keyword search: improve exact-match retrieval for names, IDs, acronyms, and
  policy terms.
- Hybrid retrieval: tune vector and keyword result counts, remove duplicates,
  and handle low-confidence retrieval.
- Reranking: compare reranker models, tune candidate counts, and measure whether
  reranking improves final context quality.
- Context construction: order chunks by relevance, avoid redundancy, preserve
  source details, and stay within context limits.
- Citations: cite exact source, page, and chunk, then verify citations support
  the answer.
- Unknown-answer behavior: say "I do not know" when retrieval finds no strong
  supporting context.
- Permissions: enforce access filters during retrieval and again before
  generating answers or showing citations.
- Query handling: support query rewriting, acronym expansion, metadata filters,
  and follow-up questions later.
- Failure handling: handle embedding, vector DB, reranker, and LLM failures
  gracefully.
- RAG metrics: track retrieved sources, latency, context chunks, citation count,
  retrieval method contribution, and estimated cost.

## Long-Term Definition of Done

Current status against the long-term target:

- Partially done: A user can upload or ingest multiple knowledge sources.
  The project can ingest multiple local `.txt` and `.md` files from `data/raw/`.
  There is not yet an upload UI, and PDF parsing is not implemented yet.

- Done for current scope: The assistant answers questions using only permitted
  sources.
  User groups and document access groups are checked before retrieval and
  answering.

- Mostly done: Every answer includes citations.
  The answer prompt requires citations, and answers without citation markers are
  rejected. The project does not yet verify that every citation truly supports
  each claim.

- Done: Retrieval combines keyword and vector search.
  The project uses TF-IDF keyword retrieval, ChromaDB vector retrieval, and
  hybrid retrieval with Reciprocal Rank Fusion.

- Done as an MVP: Reranking improves final context quality.
  FlashRank reranking is implemented. The project still needs deeper measurement
  of how much reranking improves retrieval quality.

- Partially done: Evaluation results are tracked over time.
  The project has an evaluation script and dashboard, but evaluation runs are not
  saved historically yet.

- Partially done: A dashboard shows quality, latency, and cost.
  The dashboard shows retrieval accuracy and latency. Cost and answer quality
  scoring are not tracked yet.

- Done: The codebase remains understandable to someone learning Python and RAG.
  The code is split into a package with focused modules and runnable scripts.

Overall, this project is complete as a serious learning MVP. It is not yet a
production-ready RAG system.

## Biggest Remaining Gaps

- PDF parsing with page-level citations.
- Upload or ingestion workflow for adding new documents.
- Saved evaluation history across runs.
- Actual cost tracking for embedding, reranking, and answer generation calls.
- Answer quality scoring beyond retrieval correctness.
- Stronger citation verification to check whether cited sources truly support
  each claim.
