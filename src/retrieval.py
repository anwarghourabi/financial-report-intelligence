
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import CrossEncoder

from config import (
    QDRANT_URL, COLLECTION_NAME, DENSE_MODEL, SPARSE_MODEL, RERANKING_MODEL,
)


# 1. Models are loaded ONCE (when the module is imported)

dense_embeddings = HuggingFaceEmbeddings(model_name=DENSE_MODEL)
sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL)
reranker = CrossEncoder(RERANKING_MODEL)


def get_vector_store(retrieval_mode=RetrievalMode.HYBRID):
    """Connect to the Qdrant collection. Mode can be DENSE, SPARSE or HYBRID."""
    return QdrantVectorStore.from_existing_collection(
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=retrieval_mode,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )


vector_store = get_vector_store()


# 2. Metadata filter (company / year / doc_name)

def build_filter(company=None, year=None, doc_name=None):
    """Build a Qdrant filter from the given metadata. Returns None if no filter."""
    conditions = list()

    if company:
        conditions.append(FieldCondition(key="metadata.company", match=MatchValue(value=company)))
    if year:
        conditions.append(FieldCondition(key="metadata.year", match=MatchValue(value=int(year))))
    if doc_name:
        conditions.append(FieldCondition(key="metadata.doc_name", match=MatchValue(value=doc_name)))

    return Filter(must=conditions) if conditions else None


# 3. Citation helpers

DOC_TYPE_LABELS = {"10k": "10-K", "10q": "10-Q", "8k": "8-K", "earnings": "Earnings"}


def format_citation(document):
    """Ex: '3M 10-K 2018, p. 15' (page + 1 because PyPDFLoader starts at 0)."""
    m = document.metadata
    doc_type = DOC_TYPE_LABELS.get(str(m["doc_type"]).lower(), m["doc_type"])
    return f"{m['company']} {doc_type} {m['year']}, p. {m['page'] + 1}"


def format_docs_with_citations(documents):
    """Context given to the LLM : each passage is preceded by its citation."""
    return "\n\n".join(
        f"[{format_citation(doc)}]\n{doc.page_content}" for doc in documents
    )


# 4. The search function

def search(question, company=None, year=None, doc_name=None,
           k=5, n_candidates=20, use_reranker=True, store=None):
    """
    Hybrid search in the annual reports.
    1. retrieve n_candidates chunks (hybrid dense + BM25, filtered by metadata)
    2. rerank them with the CrossEncoder (optional)
    3. return the k best Documents (with metadata["citation"] and metadata["score"])
    """
    store = store or vector_store

    documents = store.similarity_search(
        question,
        k=n_candidates if use_reranker else k,
        filter=build_filter(company, year, doc_name),
    )

    if use_reranker and documents:
        pairs = [(question, doc.page_content) for doc in documents]
        scores = reranker.predict(pairs)

        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        documents = list()
        for doc, score in ranked[:k]:
            doc.metadata["score"] = float(score)
            documents.append(doc)

    for doc in documents:
        doc.metadata["citation"] = format_citation(doc)

    return documents[:k]


if __name__ == "__main__":
    # Quick manual test
    docs = search("What were 3M's net sales in 2018?", company="3M", year=2018)
    for i, doc in enumerate(docs, start=1):
        print(f"\n--- {i} : {doc.metadata['citation']} (score {doc.metadata.get('score', 0):.2f}) ---")
        print(doc.page_content[:300])