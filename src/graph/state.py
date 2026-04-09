"""LangGraph state definition for the hedge fund agent system."""

import operator
from typing import Annotated, Any, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from typing_extensions import TypedDict


def merge_dicts(a: dict, b: dict) -> dict:
    """Merge two dicts, with b overriding a."""
    merged = a.copy()
    merged.update(b)
    return merged


class AgentState(TypedDict):
    """State shared across all agents in the graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[dict[str, Any], merge_dicts]
    metadata: Annotated[dict[str, Any], merge_dicts]
