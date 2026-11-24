import shutil
import pandas as pd

from config import COMPANIES, FINANCEBENCH_PATH, PDF_PATH, EVAL_PATH

PDF_PATH.mkdir(parents=True, exist_ok=True)

# 1. Load FinanceBench questions and keep only our companies
questions = pd.read_json(FINANCEBENCH_PATH / "data/financebench_open_source.jsonl", lines=True)
questions = questions[questions["company"].isin(COMPANIES)]

# 2. Copy the PDF of each report used by these questions
doc_names = questions["doc_name"].unique()
missing = list()

for doc_name in doc_names:
    source = FINANCEBENCH_PATH / "pdfs" / f"{doc_name}.pdf"
    if source.exists():
        shutil.copy(source, PDF_PATH / source.name)
    else:
        missing.append(doc_name)

print(f"Copied {len(doc_names) - len(missing)} PDF, kept {len(questions)} questions.")
if missing:
    print("Missing PDF:", missing)

# 3. Save the questions subset for the evaluation (Ghada)
questions.to_json(EVAL_PATH / "financebench_subset.jsonl", orient="records", lines=True)

# 4. Also copy every 10-K of our companies (needed for compare_years)
docs = pd.read_json(FINANCEBENCH_PATH / "data/financebench_document_information.jsonl", lines=True)
docs_10k = docs[docs["company"].isin(COMPANIES) & (docs["doc_type"] == "10k")]

for doc_name in docs_10k["doc_name"]:
    source = FINANCEBENCH_PATH / "pdfs" / f"{doc_name}.pdf"
    if source.exists():
        shutil.copy(source, PDF_PATH / source.name)

print(f"Total PDF in data/pdfs: {len(list(PDF_PATH.glob('*.pdf')))}")