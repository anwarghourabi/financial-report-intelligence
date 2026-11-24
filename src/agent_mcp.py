import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import Client
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from agent import SYSTEM_PROMPT
from config import LLM_MODEL

load_dotenv()

SERVER_PATH = str(Path(__file__).resolve().parent / "mcp_server.py")


def mcp_to_langchain_tool(client, mcp_tool):
    """Wrap an MCP tool (name, description, JSON schema) into a LangChain tool."""
    async def call_tool(**kwargs):
        result = await client.call_tool(mcp_tool.name, kwargs)
        return result.data if result.data is not None else str(result.content)

    return StructuredTool.from_function(
        coroutine=call_tool,
        name=mcp_tool.name,
        description=mcp_tool.description,
        args_schema=mcp_tool.inputSchema,
    )


async def main():
    # 1. Launch our MCP server (subprocess, stdio) and discover its tools
    async with Client(SERVER_PATH) as client:
        mcp_tools = await client.list_tools()
        tools = [mcp_to_langchain_tool(client, t) for t in mcp_tools]
        print("Tools loaded from MCP :", [t.name for t in tools], "\n")

        # 2. Same LangGraph agent as agent.py, with the MCP tools
        llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name=LLM_MODEL, temperature=0)
        llm_with_tools = llm.bind_tools(tools)

        async def call_model(state: MessagesState):
            response = await llm_with_tools.ainvoke([SYSTEM_PROMPT] + state["messages"])
            return {"messages": [response]}

        graph = StateGraph(MessagesState)
        graph.add_node("agent", call_model)
        graph.add_node("tools", ToolNode(tools))
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")
        agent = graph.compile(checkpointer=MemorySaver())

        config = {"configurable": {"thread_id": "mcp"}, "recursion_limit": 12}

        while True:
            question = input("You : ")
            if question.lower() in ["/bye", "/quit"]:
                break
            try:
                result = await agent.ainvoke({"messages": [HumanMessage(question)]}, config=config)
                print("AI :", result["messages"][-1].content, "\n")
            except Exception as e:
                print("Error :", e, "\n")


asyncio.run(main())