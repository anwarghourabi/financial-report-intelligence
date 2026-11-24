import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from config import COMPANIES, LLM_MODEL
from tools import search_filings, get_financial_metric, compare_years

load_dotenv()


# 1. Tools : the plain functions of tools.py become LangChain tools

TOOLS = [tool(search_filings), tool(get_financial_metric), tool(compare_years)]

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=LLM_MODEL,
    temperature=0,
)
llm_with_tools = llm.bind_tools(TOOLS)


# 2. System prompt

SYSTEM_PROMPT = SystemMessage(f'''
    You are a financial analyst assistant. You answer questions about the annual reports (10-K)
    of these companies only : {", ".join(COMPANIES)}.
    Available fiscal years : mostly 2015 to 2022.

    Rules :
    - ALWAYS use the tools to get information. Never answer from memory.
    - One precise figure -> get_financial_metric. Evolution over years -> compare_years.
      Open question -> search_filings.
    - Rewrite the user question into precise financial keywords in English before calling a tool
      (ex: "profit" -> "net income attributable to <company>", "sales" -> "net sales" or "revenues").
    - Use the exact company names listed above.
    - Every figure in your answer MUST come with its source, ex: (3M 10-K 2018, p. 56).
    - If the tools do not find the information, say you don't know. Never invent a figure.
    - Answer in the language of the user.
''')


# 3. The graph : agent <-> tools

def call_model(state: MessagesState):
    response = llm_with_tools.invoke([SYSTEM_PROMPT] + state["messages"])
    return {"messages": [response]}


graph = StateGraph(MessagesState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(TOOLS))

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)   # tool call ? -> "tools", else -> END
graph.add_edge("tools", "agent")                        # after the tool, back to the LLM

agent = graph.compile(checkpointer=MemorySaver())       # MemorySaver = conversation memory


def ask(question, thread_id="default"):
    """Ask the agent a question. Returns the final answer and the tool calls of THIS question."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 12}

    # number of messages already in the conversation memory
    n_before = len(agent.get_state(config).values.get("messages", []))

    result = agent.invoke({"messages": [HumanMessage(question)]}, config=config)
    new_messages = result["messages"][n_before:]

    tool_calls = [
        call for message in new_messages
        for call in getattr(message, "tool_calls", []) or []
    ]
    return result["messages"][-1].content, tool_calls


if __name__ == "__main__":
    print("Financial Report Agent - /bye to quit\n")

    while True:
        question = input("You : ")
        if question.lower() in ["/bye", "/quit"]:
            break

        try:
            answer, tool_calls = ask(question)
        except Exception as e:
            print("Error :", e, "\n")
            continue

        for call in tool_calls:
            print(f"  -> {call['name']}({call['args']})")
        print("AI :", answer, "\n")