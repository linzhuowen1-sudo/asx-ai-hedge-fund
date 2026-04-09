"""Valuation analyst agent — evaluates P/E, P/B, P/S relative to sector."""

from langchain_core.messages import HumanMessage

from src.data.models import AnalystSignal, Signal
from src.graph.state import AgentState
from src.tools.asx_data import get_financial_metrics


# ASX sector median P/E ratios (approximate)
ASX_SECTOR_PE = {
    "Financial Services": 14,
    "Basic Materials": 12,
    "Healthcare": 30,
    "Technology": 35,
    "Energy": 10,
    "Consumer Cyclical": 18,
    "Consumer Defensive": 22,
    "Industrials": 20,
    "Real Estate": 16,
    "Communication Services": 20,
    "Utilities": 18,
}
DEFAULT_PE = 18


def valuation_agent(state: AgentState) -> dict:
    """Evaluate valuation metrics relative to ASX sector benchmarks."""
    tickers = state["metadata"]["tickers"]
    signals = {}

    for ticker in tickers:
        metrics = get_financial_metrics(ticker)
        if not metrics:
            signals[ticker] = AnalystSignal(
                agent_name="valuation_analyst",
                ticker=ticker,
                signal=Signal.NEUTRAL,
                confidence=10,
                reasoning="Unable to fetch metrics for valuation.",
            )
            continue

        details = []
        score = 0
        checks = 0

        # P/E analysis
        if metrics.pe_ratio is not None and metrics.pe_ratio > 0:
            checks += 1
            # Compare to sector or default
            from src.tools.asx_data import get_company_info
            info = get_company_info(ticker)
            sector_pe = ASX_SECTOR_PE.get(info.sector, DEFAULT_PE) if info and info.sector else DEFAULT_PE

            if metrics.pe_ratio < sector_pe * 0.7:
                score += 1
                details.append(f"P/E {metrics.pe_ratio:.1f} well below sector median {sector_pe}")
            elif metrics.pe_ratio > sector_pe * 1.5:
                score -= 1
                details.append(f"P/E {metrics.pe_ratio:.1f} well above sector median {sector_pe}")
            else:
                details.append(f"P/E {metrics.pe_ratio:.1f} near sector median {sector_pe}")

        # P/B analysis
        if metrics.pb_ratio is not None and metrics.pb_ratio > 0:
            checks += 1
            if metrics.pb_ratio < 1.0:
                score += 1
                details.append(f"P/B {metrics.pb_ratio:.2f} — trading below book value")
            elif metrics.pb_ratio > 5.0:
                score -= 0.5
                details.append(f"P/B {metrics.pb_ratio:.2f} — premium valuation")
            else:
                details.append(f"P/B {metrics.pb_ratio:.2f}")

        # P/S analysis
        if metrics.ps_ratio is not None and metrics.ps_ratio > 0:
            checks += 1
            if metrics.ps_ratio < 1.5:
                score += 0.5
                details.append(f"P/S {metrics.ps_ratio:.2f} — attractive on revenue basis")
            elif metrics.ps_ratio > 10:
                score -= 0.5
                details.append(f"P/S {metrics.ps_ratio:.2f} — expensive on revenue basis")

        # Dividend yield (ASX investors value dividends highly)
        if metrics.dividend_yield is not None:
            checks += 1
            if metrics.dividend_yield > 0.05:
                score += 1
                details.append(f"High yield: {metrics.dividend_yield:.1%}")
            elif metrics.dividend_yield > 0.03:
                score += 0.5
                details.append(f"Decent yield: {metrics.dividend_yield:.1%}")

        if checks == 0:
            signal = Signal.NEUTRAL
            confidence = 15.0
        else:
            ratio = score / checks
            if ratio > 0.25:
                signal = Signal.BULLISH
                confidence = min(40 + ratio * 60, 90)
            elif ratio < -0.25:
                signal = Signal.BEARISH
                confidence = min(40 + abs(ratio) * 60, 90)
            else:
                signal = Signal.NEUTRAL
                confidence = 35.0

        signals[ticker] = AnalystSignal(
            agent_name="valuation_analyst",
            ticker=ticker,
            signal=signal,
            confidence=round(confidence, 1),
            reasoning="; ".join(details) if details else "Insufficient valuation data.",
        )

    return {
        "messages": [HumanMessage(content="Valuation analysis complete.", name="valuation_analyst")],
        "data": {"valuation_signals": {t: s.model_dump() for t, s in signals.items()}},
    }
