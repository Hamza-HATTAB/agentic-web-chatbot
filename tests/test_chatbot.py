import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

from src.config import Settings
from src.state import AgentState
from src.tools import get_search_tools
from src.graph import ChatbotGraphBuilder


def test_settings_validation():
    settings = Settings(groq_api_key="mock_groq_key")
    assert settings.validate_keys(model_provider="groq") is True


def test_search_tools_fallback(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    tools = get_search_tools(max_results=3)
    assert len(tools) > 0
    assert tools[0].name in ["duckduckgo_search", "tavily_search_results_json", "web_search"]


def test_memory_checkpointer_thread_persistence():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Persistent response")

    checkpointer = MemorySaver()
    builder = ChatbotGraphBuilder(llm=mock_llm, checkpointer=checkpointer)
    graph = builder.build_basic_graph()

    config = {"configurable": {"thread_id": "test_thread_123"}}
    initial_state = {"messages": [HumanMessage(content="Hello thread")]}

    result = graph.invoke(initial_state, config=config)
    assert len(result["messages"]) == 2

    # Query graph state directly via checkpointer
    state_snapshot = graph.get_state(config)
    assert len(state_snapshot.values["messages"]) == 2


def test_hitl_interrupt_graph_compilation():
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm

    builder = ChatbotGraphBuilder(llm=mock_llm)
    graph = builder.build_agentic_search_graph(hitl_interrupt=True)
    
    assert graph is not None
