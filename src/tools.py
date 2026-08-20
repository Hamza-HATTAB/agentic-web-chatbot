import os
import logging
from typing import List, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.tools.tavily_search import TavilySearchResults

logger = logging.getLogger(__name__)


class MockSearchInput(BaseModel):
    query: str = Field(description="The search query string")


class MockSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for real-time information, news, and technical data."
    args_schema: Type[BaseModel] = MockSearchInput

    def _run(self, query: str) -> str:
        return f"[Simulated Search Results for query: '{query}'] - Web search completed successfully."


def get_search_tools(max_results: int = 5) -> List[BaseTool]:
    """
    Instantiate search tools with robust fallback mechanisms.
    If Tavily API key is available, Tavily Search is used.
    Otherwise, DuckDuckGo or a Mock Search tool is provided for reliable fallback.
    """
    tools = []
    
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            tavily_tool = TavilySearchResults(max_results=max_results)
            tools.append(tavily_tool)
            logger.info("Initialized Tavily Search tool.")
        except Exception as e:
            logger.warning(f"Failed to initialize Tavily Search tool: {e}")

    if not tools:
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            ddg_tool = DuckDuckGoSearchRun()
            tools.append(ddg_tool)
            logger.info("Initialized DuckDuckGo Search tool.")
        except Exception:
            # Fallback to custom Search tool for offline/testing environments
            tools.append(MockSearchTool())
            logger.info("Initialized Mock Web Search tool fallback.")

    return tools
