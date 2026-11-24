import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from config import COMPANIES, LLM_MODEL
from retrieval import search, format_docs_with_citations

load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=LLM_MODEL,
    temperature=0,
)


# 1. Helper : accept "3m", "pepsico", "J&J"... and return the exact name stored in Qdrant

def normalize_company(company):
    if not company:
        return None
    for name in COMPANIES:
        if company.strip().lower() == name.lower():
            return name
    for name in COMPANIES:
        if company.strip().lower() in name.lower() or name.lower() in company.strip().lower():
            return name
    return company


# 2. Tool 1 : search_filings

def search_filings(query: str, company: str = None, year: int = None) -> str:
    """
    Search passages in the annual reports (10-K) of the companies.
    Use it for open questions (risks, strategy, explanations...).
    - query : a precise search query in English, with financial keywords
      (ex: "net income attributable to PepsiCo" rather than "profit of the company")
    - company : exact company name, ex: "3M", "PepsiCo", "Johnson & Johnson"
    - year : fiscal year of the report, ex: 2018
    Returns the passages, each one with its citation (company, document, year, page).
    """
    documents = search(query, company=normalize_company(company), year=year, k=5)
    if not documents:
        return "No passage found."
    return format_docs_with_citations(documents)


# 3. Tool 2 : get_financial_metric

class MetricValue(BaseModel):
    value: float | None = Field(description="The figure as a number (ex: 32765.0), null if not found")
    unit: str = Field(description="Unit, use one of : 'USD millions', 'USD billions', 'USD per share', '%'")
    citation: str = Field(description="Citation of the passage used, exactly as written between brackets, '' if not found")
    explanation: str = Field(description="One short sentence : which line / column of which table the figure comes from")


metric_prompt = ChatPromptTemplate.from_template(
    '''
    You are a financial analyst. Extract ONE figure from the context.

    Company : {company}
    Fiscal year : {year}
    Metric : {metric}

    Rules :
    - Use ONLY the context. Each passage starts with its citation between brackets.
    - Take the figure of fiscal year {year} (reports show several years side by side : pick the right column).
    - Prefer the consolidated financial statements (Statement of Income, Balance Sheet, Cash Flows),
      full year figures, not quarterly or segment figures.
    - If the figure is not in the context, return value = null.

    context :
    {context}
    '''
)

metric_chain = (
    metric_prompt | llm.with_structured_output(MetricValue, method="json_schema")
).with_retry(stop_after_attempt=3)


def extract_metric(company, year, metric):
    """Search + LLM extraction of one figure. Returns a MetricValue."""
    company = normalize_company(company)
    query = f"{metric} {company} consolidated statement {year}"
    documents = search(query, company=company, year=year, k=8)

    if not documents:
        return MetricValue(value=None, unit="", citation="", explanation="No document found")

    result = metric_chain.invoke({
        "company": company,
        "year": year,
        "metric": metric,
        "context": format_docs_with_citations(documents),
    })
    result.citation = result.citation.strip("[]")
    return result


def get_financial_metric(company: str, year: int, metric: str) -> str:
    """
    Get ONE financial figure of a company for a fiscal year, with its source.
    Use it for questions like "What were 3M's net sales in 2018?".
    - company : exact company name, ex: "3M", "PepsiCo"
    - year : fiscal year, ex: 2018
    - metric : the financial line in English, as written in a 10-K,
      ex: "net sales", "net income attributable to the company", "total assets",
      "capital expenditures", "cash and cash equivalents", "diluted EPS"
    """
    r = extract_metric(company, year, metric)
    if r.value is None:
        return f"{metric} of {company} in {year} : not found in the reports."
    return f"{metric} of {company} in {year} : {r.value:,.2f} {r.unit} (source : {r.citation}). {r.explanation}"


# 4. Tool 3 : compare_years

def compare_years(company: str, metric: str, years: list[int]) -> str:
    """
    Compare a financial figure of a company over several fiscal years,
    with the change in % between consecutive years.
    Use it for questions like "How did Nike's revenue evolve from 2019 to 2022?".
    - company : exact company name, ex: "Nike"
    - metric : the financial line in English, ex: "revenues", "net income", "operating income"
    - years : list of fiscal years, ex: [2019, 2020, 2021, 2022]
    """
    lines = list()
    previous, previous_year = None, None

    for year in sorted(int(y) for y in years):
        r = extract_metric(company, year, metric)

        if r.value is None:
            lines.append(f"{year} : not found")
            previous = None
            continue

        line = f"{year} : {r.value:,.2f} {r.unit} (source : {r.citation})"
        if previous:   # previous year found and not 0
            change = (r.value - previous) / abs(previous) * 100
            line += f" -> {change:+.1f} % vs {previous_year}"
        lines.append(line)
        previous, previous_year = r.value, year

    return f"{metric} of {normalize_company(company)} :\n" + "\n".join(lines)


if __name__ == "__main__":
    # Quick manual tests (without the agent)
    print(search_filings("main risk factors", company="Boeing", year=2022)[:1000], "\n")
    print(get_financial_metric("3M", 2018, "net sales"), "\n")
    print(compare_years("Nike", "revenues", [2020, 2021, 2022]))