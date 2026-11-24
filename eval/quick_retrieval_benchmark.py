import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from langchain_qdrant import RetrievalMode

from config import EVAL_PATH
from retrieval import search, get_vector_store

K = 5

questions = pd.read_json(EVAL_PATH / "financebench_subset.jsonl", lines=True)

retrieval_modes = [RetrievalMode.DENSE, RetrievalMode.SPARSE, RetrievalMode.HYBRID]
reranking_strategies = [False, True]

results = list()

for mode in retrieval_modes:
    store = get_vector_store(mode)

    for use_reranker in reranking_strategies:
        found = 0
        reciprocal_ranks = list()

        for _, q in questions.iterrows():
            evidence_page = q["evidence"][0]["evidence_page_num"]
            docs = search(q["question"], doc_name=q["doc_name"], k=K,
                          use_reranker=use_reranker, store=store)
            pages = [d.metadata["page"] for d in docs]

            if evidence_page in pages:
                found += 1
                reciprocal_ranks.append(1 / (pages.index(evidence_page) + 1))
            else:
                reciprocal_ranks.append(0)

        results.append({
            "mode": mode.value,
            "reranker": use_reranker,
            f"page_recall@{K}": round(found / len(questions), 3),
            "MRR": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 3),
        })
        print(results[-1])

df = pd.DataFrame(results)
print("\n", df.to_string(index=False))
df.to_csv(EVAL_PATH / "quick_retrieval_results.csv", index=False)