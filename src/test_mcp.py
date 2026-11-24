import asyncio

from fastmcp import Client

from mcp_server import mcp


async def main():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("Tools exposed by the MCP server :", [t.name for t in tools], "\n")

        result = await client.call_tool(
            "get_financial_metric",
            {"company": "3M", "year": 2018, "metric": "net sales"},
        )
        print(result.data)


asyncio.run(main())