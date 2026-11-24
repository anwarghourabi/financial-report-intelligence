import pandas as pd

questions = pd.read_json("data/financebench/data/financebench_open_source.jsonl", lines=True)

print(questions.columns.tolist())

# 1. Companies with the most questions
print(questions["company"].value_counts().head(20))

# 2. Number of different reports per company (we need several years for compare_years)
print(questions.groupby("company")["doc_name"].nunique().sort_values(ascending=False).head(20))

