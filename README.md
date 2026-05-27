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

## Current Starting Point

The repository currently contains a small script, `rag.py`, that demonstrates the
first version of RAG:

1. Define a few in-memory documents.
2. Retrieve relevant documents with a TF-IDF retriever.
3. Pass retrieved context to a Gemini model.
4. Print an answer to the user's question.

This is intentionally simple. It gives us a working mental model before we add
file ingestion, embeddings, databases, evaluation, permissions, and a user
interface.

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

### Milestone 4: Embeddings and Vector Search

- Generate embeddings for each chunk.
- Store chunks in a vector database.
- Retrieve chunks by semantic similarity.

Learning focus:

- Embeddings
- Vector similarity
- Vector databases
- Why semantic search is different from keyword search

### Milestone 5: Hybrid Search

- Add keyword search.
- Combine keyword results with vector results.
- Compare retrieval quality across search methods.

Learning focus:

- Keyword matching
- Semantic matching
- Hybrid retrieval
- Ranking tradeoffs

### Milestone 6: Reranking

- Retrieve more candidate chunks than needed.
- Rerank them before sending context to the LLM.
- Measure whether answers improve.

Learning focus:

- Rerankers
- Candidate retrieval vs final context selection
- Reducing irrelevant context

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

### Milestone 10: Dashboard

- Show retrieval accuracy.
- Show answer quality scores.
- Show latency.
- Show estimated cost.
- Track changes over time.

Learning focus:

- Observability
- Product feedback loops
- Cost awareness
- Measuring RAG system health

## Proposed Repository Structure

This structure can evolve as we build:

```text
.
├── README.md
├── rag.py
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── ingestion/
│   ├── retrieval/
│   ├── generation/
│   ├── evaluation/
│   └── app/
├── tests/
├── evals/
│   └── questions.csv
└── docs/
```

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

## First Next Steps

1. Make the current `rag.py` easy to run.
2. Add a few sample text files under `data/raw/`.
3. Replace the hardcoded documents with loaded files.
4. Add basic citations using filenames.
5. Create the first 5 evaluation questions before the system becomes complex.

## Environment Notes

The current script uses environment variables through `python-dotenv`, so API keys
should live in a local `.env` file. The `.env` file should not be committed.

Example:

```text
GOOGLE_API_KEY=your_api_key_here
```

## Long-Term Definition of Done

This project is complete when:

- A user can upload or ingest multiple knowledge sources.
- The assistant answers questions using only permitted sources.
- Every answer includes citations.
- Retrieval combines keyword and vector search.
- Reranking improves final context quality.
- Evaluation results are tracked over time.
- A dashboard shows quality, latency, and cost.
- The codebase remains understandable to someone learning Python and RAG.
