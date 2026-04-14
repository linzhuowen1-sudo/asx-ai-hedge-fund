"""Multi-timeframe alignment agent — checks trend consistency across Weekly/Daily/4H.

Uses TradingView data for multi-timeframe analysis. Falls back to yfinance daily-only
if TradingView is unavailable.

Key rules:
  - Never trade against combined Weekly + Daily direction
  - Strong alignment (all agree) → high confidence
  - Mixed signals → reduce confidence, suggest caution
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.data.models import AnalystSignal, Signal
from src.graph.state import AgentState
from src.tools.tradingview_data import get_tv_multi_timeframe


# Timeframes to analyze (longest → shortest)
TIMEFRAMES = ["1W", "1d", "4h"]
TIMEFRAME_LABELS = {"1W": "Weekly", "1d": "Daily", "4h": "4-Hour"}
TIMEFRAME_WEIGHTS = {"1W": 3.0, "1d": 2.0, "4h": 1.0}


def _classify_recommendation(rec: str) -> int:
    """Map TradingView recommendation to directional score.

    Returns: 2 (strong buy), 1 (buy), 0 (neutral), -1 (sell), -2 (strong sell)
    """
    rec = rec.upper()
    mapping = {
        "STRONG_BUY": 2,
        "BUY": 1,
        "NEUTRAL": 0,
        "SELL": -1,
        "STRONG_SELL": -2,
    }
    return mapping.get(rec, 0)


def timeframe_agent(state: AgentState) -> dict:
    """Analyze trend alignment across multiple timeframes for each ticker."""
    tickers = state["metadata"]["tickers"]
    signals = {}

    for ticker in tickers:
        tf_data = get_tv_multi_timeframe(ticker, TIMEFRAMES)

        if not any(tf_data.values()):
            signals[ticker] = AnalystSignal(
                agent_name="timeframe_analyst",
                ticker=ticker,
                signal=Signal.NEUTRAL,
                confidence=10,
                reasoning="TradingView data unavailable for multi-timeframe analysis.",
            )
            continue

        details = []
        weighted_score = 0.0
        total_weight = 0.0
        directions = {}

        for tf in TIMEFRAMES:
            analysis = tf_data.get(tf)
            label = TIMEFRAME_LABELS[tf]
            weight = TIMEFRAME_WEIGHTS[tf]

            if analysis is None:
                details.append(f"{label}: N/A")
                continue

            summary_rec = analysis["summary"]["recommendation"]
            osc_rec = analysis["oscillators"]["recommendation"]
            ma_rec = analysis["moving_averages"]["recommendation"]

            dir_score = _classify_recommendation(summary_rec)
            directions[tf] = dir_score
            weighted_score += dir_score * weight
            total_weight += weight

            buy_count = analysis["summary"]["buy"]
            sell_count = analysis["summary"]["sell"]
            neutral_count = analysis["summary"]["neutral"]

            details.append(
                f"{label}: {summary_rec} (B:{buy_count} S:{sell_count} N:{neutral_count}) "
                f"| Osc:{osc_rec} MA:{ma_rec}"
            )

        # Alignment analysis
        if total_weight == 0:
            signal = Signal.NEUTRAL
            confidence = 10.0
            details.append("No timeframe data available")
        else:
            avg_score = weighted_score / total_weight

            # Check alignment between weekly and daily
            weekly_dir = directions.get("1W", 0)
            daily_dir = directions.get("1d", 0)
            h4_dir = directions.get("4h", 0)

            all_dirs = [d for d in directions.values()]
            all_bullish = all(d > 0 for d in all_dirs) if all_dirs else False
            all_bearish = all(d < 0 for d in all_dirs) if all_dirs else False

            if all_bullish:
                signal = Signal.BULLISH
                confidence = min(70 + abs(avg_score) * 15, 95)
                details.append("ALIGNED BULLISH across all timeframes")
            elif all_bearish:
                signal = Signal.BEARISH
                confidence = min(70 + abs(avg_score) * 15, 95)
                details.append("ALIGNED BEARISH across all timeframes")
            elif weekly_dir * daily_dir > 0:
                # Weekly and daily agree
                if weekly_dir > 0:
                    signal = Signal.BULLISH
                else:
                    signal = Signal.BEARISH
                confidence = 50 + abs(avg_score) * 10
                if h4_dir != 0 and h4_dir * weekly_dir < 0:
                    details.append(f"W+D agree ({['bearish','','bullish'][weekly_dir+1]}), 4H diverges — short-term pullback")
                    confidence -= 10
            elif weekly_dir != 0 and daily_dir != 0 and weekly_dir * daily_dir < 0:
                # Weekly and daily disagree — conflicting
                signal = Signal.NEUTRAL
                confidence = 30
                details.append("WARNING: Weekly and Daily CONFLICT — avoid new positions")
            else:
                # Some neutral
                if avg_score > 0.3:
                    signal = Signal.BULLISH
                elif avg_score < -0.3:
                    signal = Signal.BEARISH
                else:
                    signal = Signal.NEUTRAL
                confidence = 35 + abs(avg_score) * 15

        signals[ticker] = AnalystSignal(
            agent_name="timeframe_analyst",
            ticker=ticker,
            signal=signal,
            confidence=round(confidence, 1),
            reasoning="; ".join(details),
        )

    return {
        "messages": [HumanMessage(content="Multi-timeframe analysis complete.", name="timeframe_analyst")],
        "data": {"timeframe_signals": {t: s.model_dump() for t, s in signals.items()}},
    }
