"""TradingView data source — real-time technical analysis via tradingview-ta.

Provides an alternative/supplementary data source to yfinance.
ASX tickers use the "ASX" exchange in TradingView.
"""

from __future__ import annotations

from typing import Optional

from src.data.cache import get_cache, set_cache


def _to_tv_symbol(ticker: str) -> tuple[str, str]:
    """Convert ASX ticker to TradingView format.

    'BHP.AX' -> ('ASX', 'BHP')
    'CBA.AX' -> ('ASX', 'CBA')
    """
    symbol = ticker.replace(".AX", "").upper()
    return "ASX", symbol


def get_tv_analysis(
    ticker: str,
    interval: str = "1d",
) -> Optional[dict]:
    """Get TradingView technical analysis for an ASX ticker.

    Args:
        ticker: ASX ticker (e.g., 'BHP.AX')
        interval: Timeframe — '1m', '5m', '15m', '1h', '4h', '1d', '1W', '1M'

    Returns:
        Dict with summary, oscillators, moving_averages, and raw indicators.
    """
    cache_key = f"tv_analysis:{ticker}:{interval}"
    cached = get_cache(cache_key, ttl=300)  # 5 min cache
    if cached:
        return cached

    try:
        from tradingview_ta import TA_Handler, Interval

        interval_map = {
            "1m": Interval.INTERVAL_1_MINUTE,
            "5m": Interval.INTERVAL_5_MINUTES,
            "15m": Interval.INTERVAL_15_MINUTES,
            "1h": Interval.INTERVAL_1_HOUR,
            "4h": Interval.INTERVAL_4_HOURS,
            "1d": Interval.INTERVAL_1_DAY,
            "1W": Interval.INTERVAL_1_WEEK,
            "1M": Interval.INTERVAL_1_MONTH,
        }

        tv_interval = interval_map.get(interval, Interval.INTERVAL_1_DAY)
        exchange, symbol = _to_tv_symbol(ticker)

        handler = TA_Handler(
            symbol=symbol,
            screener="australia",
            exchange=exchange,
            interval=tv_interval,
        )

        analysis = handler.get_analysis()

        result = {
            "summary": {
                "recommendation": analysis.summary.get("RECOMMENDATION", "NEUTRAL"),
                "buy": analysis.summary.get("BUY", 0),
                "sell": analysis.summary.get("SELL", 0),
                "neutral": analysis.summary.get("NEUTRAL", 0),
            },
            "oscillators": {
                "recommendation": analysis.oscillators.get("RECOMMENDATION", "NEUTRAL"),
                "buy": analysis.oscillators.get("BUY", 0),
                "sell": analysis.oscillators.get("SELL", 0),
                "neutral": analysis.oscillators.get("NEUTRAL", 0),
                "compute": analysis.oscillators.get("COMPUTE", {}),
            },
            "moving_averages": {
                "recommendation": analysis.moving_averages.get("RECOMMENDATION", "NEUTRAL"),
                "buy": analysis.moving_averages.get("BUY", 0),
                "sell": analysis.moving_averages.get("SELL", 0),
                "neutral": analysis.moving_averages.get("NEUTRAL", 0),
                "compute": analysis.moving_averages.get("COMPUTE", {}),
            },
            "indicators": {k: v for k, v in analysis.indicators.items()},
        }

        set_cache(cache_key, result)
        return result

    except Exception:
        return None


def get_tv_multi_timeframe(
    ticker: str,
    intervals: list[str] | None = None,
) -> dict[str, Optional[dict]]:
    """Get TradingView analysis across multiple timeframes.

    Args:
        ticker: ASX ticker
        intervals: List of intervals. Defaults to ['1d', '4h', '1W']

    Returns:
        Dict mapping interval -> analysis result
    """
    if intervals is None:
        intervals = ["1W", "1d", "4h"]

    results = {}
    for interval in intervals:
        results[interval] = get_tv_analysis(ticker, interval)
    return results
