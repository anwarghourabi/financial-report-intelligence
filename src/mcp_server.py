from fastmcp import FastMCP

from tools import search_filings, get_financial_metric, compare_years

mcp = FastMCP("financial-reports")

mcp.tool()(search_filings)
mcp.tool()(get_financial_metric)
mcp.tool()(compare_years)

if __name__ == "__main__":
    mcp.run()          # transport stdio by default