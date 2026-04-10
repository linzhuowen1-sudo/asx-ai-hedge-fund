"""LangGraph orchestration — defines the agent execution graph."""

from __future__ import annotations


from langgraph.graph import END, StateGraph

from src.agents.fundamentals import fundamentals_agent
from src.agents.valuation import valuation_agent
from src.agents.technicals import technicals_agent
from src.agents.sentiment import sentiment_agent
from src.agents.risk_manager import risk_manager_agent
from src.agents.portfolio_manager import portfolio_manager_agent
from src.graph.state import AgentState


def build_graph(
    analysts: list[str] | None = None,
) -> StateGraph:
    """Build the hedge fund agent graph.

    Args:
        analysts: List of analyst names to include. If None, uses all.
                  Options: fundamentals, valuation, technicals, sentiment

    Returns:
        Compiled LangGraph StateGraph
    """
    available_analysts = {
        "fundamentals": fundamentals_agent,
        "valuation": valuation_agent,
        "technicals": technicals_agent,
        "sentiment": sentiment_agent,
    }

    if analysts is None:
        analysts = list(available_analysts.keys())

    selected = {name: fn for name, fn in available_analysts.items() if name in analysts}

    graph = StateGraph(AgentState)

    # Add analyst nodes
    for name, agent_fn in selected.items():
        graph.add_node(name, agent_fn)

    # Add risk manager and portfolio manager
    graph.add_node("risk_manager", risk_manager_agent)
    graph.add_node("portfolio_manager", portfolio_manager_agent)

    # Analysts run in parallel from START
    analyst_names = list(selected.keys())
    for name in analyst_names:
        graph.add_edge("__start__", name)

    # Risk manager also runs in parallel with analysts
    graph.add_edge("__start__", "risk_manager")

    # All analysts + risk manager feed into portfolio manager
    for name in analyst_names:
        graph.add_edge(name, "portfolio_manager")
    graph.add_edge("risk_manager", "portfolio_manager")

    # Portfolio manager is the final node
    graph.add_edge("portfolio_manager", END)

    return graph.compile()


def run_hedge_fund(
    tickers: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    portfolio: dict | None = None,
    analysts: list[str] | None = None,
) -> dict:
    """Run the hedge fund analysis.

    Args:
        tickers: List of ASX tickers (e.g., ["BHP.AX", "CBA.AX"])
        start_date: Analysis start date (YYYY-MM-DD)
        end_date: Analysis end date (YYYY-MM-DD)
        portfolio: Current portfolio state
        analysts: Which analysts to use

    Returns:
        Final state including trade decisions
    """
    from datetime import datetime, timedelta
    from src.tools.asx_data import ensure_asx_ticker

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    tickers = [ensure_asx_ticker(t) for t in tickers]

    if portfolio is None:
        portfolio = {"cash": 100_000, "positions": {}, "total_value": 100_000}

    graph = build_graph(analysts=analysts)

    initial_state = {
        "messages": [],
        "data": {},
        "metadata": {
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
            "portfolio": portfolio,
        },
    }

    result = graph.invoke(initial_state)
    return result
