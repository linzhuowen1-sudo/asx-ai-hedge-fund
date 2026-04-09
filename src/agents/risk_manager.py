"""Risk manager agent — evaluates portfolio risk and sets constraints."""

from langchain_core.messages import HumanMessage

from src.data.models import AnalystSignal, Signal
from src.graph.state import AgentState
from src.tools.asx_data import get_financial_metrics, get_price_history

# Risk parameters
MAX_POSITION_PCT = 0.20       # Max 20% of portfolio in one stock
MAX_SECTOR_PCT = 0.40         # Max 40% in one sector
MAX_TOTAL_EXPOSURE = 1.5      # Max 150% gross exposure (with leverage)
MIN_CASH_PCT = 0.05           # Keep at least 5% cash


def risk_manager_agent(state: AgentState) -> dict:
    """Evaluate risk constraints and provide risk-adjusted signals."""
    tickers = state["metadata"]["tickers"]
    portfolio = state["metadata"].get("portfolio", {})
    signals = {}

    total_value = portfolio.get("total_value", 100_000)

    for ticker in tickers:
        metrics = get_financial_metrics(ticker)
        details = []
        risk_score = 0  # Higher = more risky

        # Beta risk
        if metrics and metrics.beta is not None:
            if metrics.beta > 1.5:
                risk_score += 2
                details.append(f"High beta: {metrics.beta:.2f} — volatile")
            elif metrics.beta > 1.0:
                risk_score += 1
                details.append(f"Above-market beta: {metrics.beta:.2f}")
            elif metrics.beta < 0.5:
                details.append(f"Low beta: {metrics.beta:.2f} — defensive")

        # Leverage risk
        if metrics and metrics.debt_to_equity is not None:
            if metrics.debt_to_equity > 200:
                risk_score += 2
                details.append(f"High leverage: D/E {metrics.debt_to_equity:.0f}")
            elif metrics.debt_to_equity > 100:
                risk_score += 1
                details.append(f"Moderate leverage: D/E {metrics.debt_to_equity:.0f}")

        # Liquidity risk (market cap)
        if metrics and metrics.market_cap is not None:
            if metrics.market_cap < 500_000_000:  # < $500M AUD
                risk_score += 2
                details.append(f"Small cap: ${metrics.market_cap/1e9:.2f}B — liquidity risk")
            elif metrics.market_cap < 2_000_000_000:
                risk_score += 1
                details.append(f"Mid cap: ${metrics.market_cap/1e9:.2f}B")
            else:
                details.append(f"Large cap: ${metrics.market_cap/1e9:.1f}B")

        # Position concentration check
        existing_position = portfolio.get("positions", {}).get(ticker, {})
        if existing_position:
            position_value = existing_position.get("market_value", 0)
            position_pct = position_value / total_value if total_value > 0 else 0
            if position_pct > MAX_POSITION_PCT:
                risk_score += 2
                details.append(f"Position concentration: {position_pct:.1%} > {MAX_POSITION_PCT:.0%} limit")

        # Determine risk signal
        if risk_score >= 4:
            signal = Signal.BEARISH
            confidence = min(50 + risk_score * 8, 95)
            details.append("HIGH RISK — reduce exposure recommended")
        elif risk_score >= 2:
            signal = Signal.NEUTRAL
            confidence = 50.0
            details.append("MODERATE RISK — standard position sizing")
        else:
            signal = Signal.BULLISH
            confidence = 60.0
            details.append("LOW RISK — full position sizing permitted")

        signals[ticker] = AnalystSignal(
            agent_name="risk_manager",
            ticker=ticker,
            signal=signal,
            confidence=round(confidence, 1),
            reasoning="; ".join(details),
        )

    # Portfolio-level risk constraints
    risk_constraints = {
        "max_position_pct": MAX_POSITION_PCT,
        "max_sector_pct": MAX_SECTOR_PCT,
        "max_total_exposure": MAX_TOTAL_EXPOSURE,
        "min_cash_pct": MIN_CASH_PCT,
    }

    return {
        "messages": [HumanMessage(content="Risk analysis complete.", name="risk_manager")],
        "data": {
            "risk_signals": {t: s.model_dump() for t, s in signals.items()},
            "risk_constraints": risk_constraints,
        },
    }
