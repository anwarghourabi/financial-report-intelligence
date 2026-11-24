from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "financial_reports"

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"
RERANKING_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
LLM_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

DATA_PATH = ROOT_PATH / "data"
PDF_PATH = DATA_PATH / "pdfs"
FINANCEBENCH_PATH = DATA_PATH / "financebench"
EVAL_PATH = ROOT_PATH / "eval"

COMPANIES = [
    "PepsiCo", "Amcor", "Johnson & Johnson", "3M", "Best Buy", "MGM Resorts",
    "Adobe", "JPMorgan", "Nike", "General Mills", "Corning", "Boeing",
]