import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import QDRANT_URL, COLLECTION_NAME, DENSE_MODEL, SPARSE_MODEL, EVAL_PATH

dense_embeddings = HuggingFaceEmbeddings(model_name=DENSE_MODEL)
sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL)

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=dense_embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
)

questions = pd.read_json(EVAL_PATH / "financebench_subset.jsonl", lines=True)

# shift = 0  -> FinanceBench pages start at 0 (like PyPDFLoader)
# shift = -1 -> FinanceBench pages start at 1
found = {0: 0, -1: 0, 1: 0}

for _, q in questions.head(20).iterrows():
    evidence_page = q["evidence"][0]["evidence_page_num"]
    flt = Filter(must=[FieldCondition(key="metadata.doc_name", match=MatchValue(value=q["doc_name"]))])
    docs = vector_store.similarity_search(q["question"], k=5, filter=flt)
    pages = [d.metadata["page"] for d in docs]

    for shift in found:
        if evidence_page + shift in pages:
            found[shift] += 1

for shift, score in found.items():
    print(f"shift {shift:+d} : evidence page found in top 5 for {score}/20 questions")