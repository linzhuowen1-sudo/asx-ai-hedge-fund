"""Fundamentals analyst agent — evaluates profitability, growth, financial health."""

from langchain_core.messages import HumanMessage

from src.data.models import AnalystSignal, Signal
from src.graph.state import AgentState
from src.llm.models import get_llm
from src.tools.asx_data import get_financial_metrics, get_income_statement, get_balance_sheet


def fundamentals_agent(state: AgentState) -> dict:
    """Analyze fundamental financial data for each ticker."""
    tickers = state["metadata"]["tickers"]
    signals = {}

    for ticker in tickers:
        metrics = get_financial_metrics(ticker)
        if not metrics:
            signals[ticker] = AnalystSignal(
                agent_name="fundamentals_analyst",
                ticker=ticker,
                signal=Signal.NEUTRAL,
                confidence=10,
                reasoning="Unable to fetch financial metrics.",
            )
            continue

        # Score-based analysis
        total_score = 0
        max_score = 0
        details = []

        # Profitability
        if metrics.roe is not None:
            max_score += 1
            if metrics.roe > 0.15:
                total_score += 1
                details.append(f"Strong ROE: {metrics.roe:.1%}")
            elif metrics.roe < 0.05:
                total_score -= 0.5
                details.append(f"Weak ROE: {metrics.roe:.1%}")

        if metrics.net_margin is not None:
            max_score += 1
            if metrics.net_margin > 0.15:
                total_score += 1
                details.append(f"Healthy net margin: {metrics.net_margin:.1%}")
            elif metrics.net_margin < 0:
                total_score -= 1
                details.append(f"Negative net margin: {metrics.net_margin:.1%}")

        # Growth
        if metrics.revenue_growth is not None:
            max_score += 1
            if metrics.revenue_growth > 0.10:
                total_score += 1
                details.append(f"Strong revenue growth: {metrics.revenue_growth:.1%}")
            elif metrics.revenue_growth < 0:
                total_score -= 0.5
                details.append(f"Revenue declining: {metrics.revenue_growth:.1%}")

        if metrics.earnings_growth is not None:
            max_score += 1
            if metrics.earnings_growth > 0.10:
                total_score += 1
                details.append(f"Strong earnings growth: {metrics.earnings_growth:.1%}")
            elif metrics.earnings_growth < -0.10:
                total_score -= 1
                details.append(f"Earnings declining: {metrics.earnings_growth:.1%}")

        # Financial health
        if metrics.debt_to_equity is not None:
            max_score += 1
            if metrics.debt_to_equity < 50:
                total_score += 1
                details.append(f"Low leverage: D/E {metrics.debt_to_equity:.1f}")
            elif metrics.debt_to_equity > 200:
                total_score -= 1
                details.append(f"High leverage: D/E {metrics.debt_to_equity:.1f}")

        if metrics.current_ratio is not None:
            max_score += 1
            if metrics.current_ratio > 1.5:
                total_score += 1
                details.append(f"Good liquidity: CR {metrics.current_ratio:.2f}")
            elif metrics.current_ratio < 1.0:
                total_score -= 0.5
                details.append(f"Liquidity concern: CR {metrics.current_ratio:.2f}")

        # Dividend (important for ASX investors)
        if metrics.dividend_yield is not None and metrics.dividend_yield > 0.03:
            total_score += 0.5
            details.append(f"Attractive dividend: {metrics.dividend_yield:.1%}")

        # Determine signal
        if max_score == 0:
            signal = Signal.NEUTRAL
            confidence = 20.0
        else:
            ratio = total_score / max_score
            if ratio > 0.3:
                signal = Signal.BULLISH
                confidence = min(40 + ratio * 60, 95)
            elif ratio < -0.1:
                signal = Signal.BEARISH
                confidence = min(40 + abs(ratio) * 60, 95)
            else:
                signal = Signal.NEUTRAL
                confidence = 40.0

        signals[ticker] = AnalystSignal(
            agent_name="fundamentals_analyst",
            ticker=ticker,
            signal=signal,
            confidence=round(confidence, 1),
            reasoning="; ".join(details) if details else "Insufficient data for detailed analysis.",
        )

    return {
        "messages": [HumanMessage(content="Fundamentals analysis complete.", name="fundamentals_analyst")],
        "data": {"fundamentals_signals": {t: s.model_dump() for t, s in signals.items()}},
    }
