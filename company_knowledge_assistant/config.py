from pathlib import Path

PACKAGE_DIR = Path(__file__).parent
PROJECT_ROOT = PACKAGE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
SUPPORTED_EXTENSIONS = {".txt", ".md"}
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"
VECTOR_DB_DIR = PROJECT_ROOT / "data" / "vector_db"
COLLECTION_NAME = "company_knowledge"
RRF_K = 60
TOP_K = 2
CANDIDATE_K = 5
RERANK_CANDIDATE_K = 5
RERANK_TOP_K = 2
RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"
USERS = {
    "alice": {"groups": ["general", "engineering"]},
    "bob": {"groups": ["general"]},
}
SOURCE_ACCESS = {
    "company-overview.txt": ["general"],
    "engineering-handbook.md": ["engineering"],
}
EVAL_FILE = Path("evals/questions.csv")
