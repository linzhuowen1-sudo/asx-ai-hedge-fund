"""Portfolio manager agent — aggregates signals and makes final trade decisions."""

import json

from langchain_core.messages import HumanMessage

from src.data.models import Action, AnalystSignal, PortfolioDecisions, Signal, TradeDecision
from src.graph.state import AgentState
from src.llm.models import get_llm


PORTFOLIO_PROMPT = """You are a portfolio manager for an AI hedge fund that trades on the Australian Securities Exchange (ASX).
All values are in AUD. The ASX benchmark is the S&P/ASX 200 index.

You must make trading decisions based on analyst signals, risk constraints, and current portfolio state.

## Current Portfolio
Cash: ${cash:,.2f} AUD
Positions: {positions}

## Risk Constraints
{risk_constraints}

## Analyst Signals
{signals_text}

## Instructions
For each ticker, decide on an action: buy, sell, short, cover, or hold.
Consider:
1. Signal consensus across analysts
2. Confidence-weighted signal strength
3. Risk manager warnings
4. Position sizing (never exceed max_position_pct of portfolio)
5. Maintain minimum cash reserve
6. ASX-specific factors: franking credits make dividend stocks attractive for hold, mining stocks can be volatile

Respond with EXACTLY this JSON format:
{{
    "decisions": [
        {{
            "ticker": "<ticker>",
            "action": "buy|sell|short|cover|hold",
            "quantity": <integer>,
            "confidence": <number 10-100>,
            "reasoning": "<brief explanation>"
        }}
    ]
}}
"""


def _format_signals(data: dict, tickers: list[str]) -> str:
    """Format all analyst signals into readable text."""
    signal_keys = [
        ("fundamentals_signals", "Fundamentals"),
        ("valuation_signals", "Valuation"),
        ("technicals_signals", "Technicals"),
        ("sentiment_signals", "Sentiment"),
        ("risk_signals", "Risk Manager"),
    ]

    lines = []
    for ticker in tickers:
        lines.append(f"\n### {ticker}")
        for key, name in signal_keys:
            signals = data.get(key, {})
            if ticker in signals:
                s = signals[ticker]
                lines.append(f"  {name}: {s['signal']} (confidence: {s['confidence']}%) — {s['reasoning']}")
    return "\n".join(lines)


def portfolio_manager_agent(state: AgentState) -> dict:
    """Make final trade decisions based on all analyst signals."""
    tickers = state["metadata"]["tickers"]
    data = state["data"]
    portfolio = state["metadata"].get("portfolio", {"cash": 100_000, "positions": {}})

    risk_constraints = data.get("risk_constraints", {})
    signals_text = _format_signals(data, tickers)

    positions_text = json.dumps(portfolio.get("positions", {}), indent=2) or "None"
    risk_text = json.dumps(risk_constraints, indent=2)

    prompt = PORTFOLIO_PROMPT.format(
        cash=portfolio.get("cash", 100_000),
        positions=positions_text,
        risk_constraints=risk_text,
        signals_text=signals_text,
    )

    llm = get_llm()

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Parse JSON
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        result = json.loads(content)
        decisions = PortfolioDecisions(**result)
    except Exception as e:
        # Fallback: hold everything
        decisions = PortfolioDecisions(
            decisions=[
                TradeDecision(
                    ticker=t,
                    action=Action.HOLD,
                    quantity=0,
                    confidence=10,
                    reasoning=f"Portfolio manager error: {str(e)[:80]}",
                )
                for t in tickers
            ]
        )

    return {
        "messages": [HumanMessage(content="Portfolio decisions complete.", name="portfolio_manager")],
        "data": {"decisions": [d.model_dump() for d in decisions.decisions]},
    }
