import logging
from typing import Literal, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from src.state import AgentState
from src.tools import get_search_tools

logger = logging.getLogger(__name__)


class ChatbotGraphBuilder:
    """
    Advanced Graph Builder featuring MemorySaver persistent state checkpointers (Ch. 11),
    conditional tool routing (Ch. 11), and optional Human-In-The-Loop HITL approval interrupts (Ch. 14).
    """
    def __init__(self, llm: BaseChatModel, max_search_results: int = 5, checkpointer: Optional[MemorySaver] = None):
        self.llm = llm
        self.max_search_results = max_search_results
        self.checkpointer = checkpointer or MemorySaver()

    def _chatbot_node(self, state: AgentState) -> dict:
        """Process messages and invoke bound LLM."""
        response = self.llm.invoke(state["messages"])
        return {"messages": [response]}

    def build_basic_graph(self):
        """Construct stateful chatbot graph with MemorySaver checkpointer."""
        builder = StateGraph(AgentState)
        builder.add_node("chatbot", self._chatbot_node)
        builder.add_edge(START, "chatbot")
        builder.add_edge("chatbot", END)
        return builder.compile(checkpointer=self.checkpointer)

    def build_agentic_search_graph(self, hitl_interrupt: bool = False):
        """
        Construct tool-augmented agentic chatbot graph with Tavily/DDG search,
        MemorySaver checkpointer, and optional HITL interrupt_before tool execution.
        """
        tools = get_search_tools(max_results=self.max_search_results)
        
        if tools:
            llm_with_tools = self.llm.bind_tools(tools)
            tool_node = ToolNode(tools=tools)

            def agent_node(state: AgentState) -> dict:
                response = llm_with_tools.invoke(state["messages"])
                return {"messages": [response]}

            builder = StateGraph(AgentState)
            builder.add_node("agent", agent_node)
            builder.add_node("tools", tool_node)

            builder.add_edge(START, "agent")
            builder.add_conditional_edges("agent", tools_condition)
            builder.add_edge("tools", "agent")

            # Enable Human-In-The-Loop breakpoint before tools execution if requested
            interrupt_nodes = ["tools"] if hitl_interrupt else []

            return builder.compile(
                checkpointer=self.checkpointer,
                interrupt_before=interrupt_nodes
            )
        else:
            logger.warning("No search tools available. Falling back to basic chatbot graph.")
            return self.build_basic_graph()
