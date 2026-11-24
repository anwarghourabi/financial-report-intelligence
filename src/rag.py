import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from config import LLM_MODEL
from retrieval import search, format_docs_with_citations

load_dotenv()


llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=LLM_MODEL,
    temperature=0,
)


class RagAnswer(BaseModel):
    answer: str = Field(description="The answer to the question. 'I don't know' if the context does not contain it.")
    sources: list[str] = Field(description="Citations used, exactly as written in the context, ex: '3M 10-K 2018, p. 56'")


prompt = ChatPromptTemplate.from_template(
    '''
    You are a financial analyst assistant.
    Answer the question using ONLY the context extracted from annual reports.
    Each passage of the context starts with its citation between brackets.

    Rules:
    - Give precise figures with their unit (ex: $ 32,765 million).
    - Cite the passages you used, exactly as written between brackets.
    - If the answer is not in the context, answer "I don't know" with no sources.
    - Prefer figures for the full fiscal year from the consolidated statements, not quarterly or segment figures.

    question :
    {question}

    context :
    {context}
    '''
)

rag_chain = (
    prompt | llm.with_structured_output(RagAnswer, method="json_schema")
).with_retry(stop_after_attempt=3)


def answer_question(question, company=None, year=None, k=8):
    """Full RAG : search the reports, then answer with citations."""
    documents = search(question, company=company, year=year, k=k)

    if not documents:
        return RagAnswer(answer="I don't know", sources=[]), documents

    result = rag_chain.invoke({
        "question": question,
        "context": format_docs_with_citations(documents),
    })
    result.sources = [s.strip("[]") for s in result.sources]
    return result, documents


if __name__ == "__main__":
    print("Financial Report RAG - /bye to quit")
    print("Format : question | company | year   (company and year are optional)\n")

    while True:
        user_input = input("Question : ")
        if user_input.lower() in ["/bye", "/quit"]:
            break

        parts = [p.strip() for p in user_input.split("|")]
        question = parts[0]
        company = parts[1] if len(parts) > 1 and parts[1] else None
        year = int(parts[2]) if len(parts) > 2 and parts[2] else None

        try:
            result, _ = answer_question(question, company, year)
        except Exception as e:
            print("Error :", e, "\n")
            continue
        print("AI :", result.answer)
        print("Sources :", result.sources, "\n")