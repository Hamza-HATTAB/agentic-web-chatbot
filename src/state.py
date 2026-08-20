from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    LangGraph state schema maintaining the message trajectory.
    Uses add_messages reducer for immutable message append operations.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
