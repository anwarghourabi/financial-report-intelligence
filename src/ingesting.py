import os
import hashlib

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType
import re
from config import (
    QDRANT_URL, COLLECTION_NAME, DENSE_MODEL, SPARSE_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, PDF_PATH, FINANCEBENCH_PATH,
)


# 1. Helpers (same as in the course)

def calculate_file_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()


def hash_exists_in_qdrant(vector_store, file_hash):

    if vector_store is None:
        return False

    points, _ = vector_store.client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter={
            "must": [
                {"key": "metadata.file_hash", "match": {"value": file_hash}}
            ]
        },
        limit=1,
    )

    return len(points) > 0

def clean_text(text):
    """Collapse the many spaces / line breaks produced by PyPDF."""
    return re.sub(r"\s+", " ", text).strip()


# 2. Load the documents information from FinanceBench (company, year, type)

docs_info = pd.read_json(
    FINANCEBENCH_PATH / "data/financebench_document_information.jsonl", lines=True
)
docs_info = docs_info.set_index("doc_name")


def get_doc_metadata(doc_name):
    """Return company / year / doc_type of a report, using its FinanceBench name."""
    if doc_name not in docs_info.index:
        return None

    row = docs_info.loc[doc_name]
    return {
        "doc_name": doc_name,
        "company": row["company"],
        "year": int(row["doc_period"]),
        "doc_type": row["doc_type"],
    }


# 3. Embedding models (dense + sparse BM25) and splitter

dense_embeddings = HuggingFaceEmbeddings(model_name=DENSE_MODEL)
sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


# 4. Connect to the existing collection (or None if it doesn't exist yet)

client = QdrantClient(url=QDRANT_URL)

if client.collection_exists(collection_name=COLLECTION_NAME):
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )
else:
    vector_store = None


# 5. Ingest PDF by PDF

pdf_files = sorted(PDF_PATH.glob("*.pdf"))
print(f"Found {len(pdf_files)} PDF in {PDF_PATH}")

total_chunks = 0

for i, pdf_file in enumerate(pdf_files, start=1):

    doc_name = pdf_file.stem                      # "3M_2018_10K.pdf" -> "3M_2018_10K"
    print(f"[{i}/{len(pdf_files)}] {doc_name}")

    doc_metadata = get_doc_metadata(doc_name)
    if doc_metadata is None:
        print("   -> not found in FinanceBench, skipped")
        continue

    file_hash = calculate_file_hash(pdf_file)
    if hash_exists_in_qdrant(vector_store, file_hash):
        print("   -> already indexed, skipped")
        continue

    # 5.1 Load the PDF : one Document per page (metadata.page starts at 0)
    loader = PyPDFLoader(str(pdf_file))
    pages = loader.load()
    for page in pages:
        page.page_content = clean_text(page.page_content)   
    pages = [page for page in pages if page.page_content.strip()]   # drop empty pages

    # 5.2 Add our metadata to every page
    for page in pages:
        page.metadata.update({
            "file_hash": file_hash,
            "source_file": str(pdf_file.relative_to(PDF_PATH.parent.parent)),
            **doc_metadata,
        })

    # 5.3 Split into chunks (each chunk keeps the metadata of its page)
    chunks = splitter.split_documents(pages)

    # 5.4 Store in Qdrant (create the collection the first time)
    if vector_store is None:
        vector_store = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            url=QDRANT_URL,
            collection_name=COLLECTION_NAME,
        )
    else:
        vector_store.add_documents(chunks)

    total_chunks += len(chunks)
    print(f"   -> {len(pages)} pages, {len(chunks)} chunks")


# 6. Indexes on the metadata used for filtering (faster filters)

for field, schema in [
    ("metadata.company", PayloadSchemaType.KEYWORD),
    ("metadata.doc_name", PayloadSchemaType.KEYWORD),
    ("metadata.file_hash", PayloadSchemaType.KEYWORD),
    ("metadata.year", PayloadSchemaType.INTEGER),
]:
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name=field,
        field_schema=schema,
    )

print(f"Done. {total_chunks} new chunks stored in '{COLLECTION_NAME}'.")