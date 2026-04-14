"""Trade plan generator — computes buy/sell zones, stop loss, and multi-horizon targets.

Uses raw indicator data from technicals agent + TradingView daily/weekly data
to produce actionable price levels at three horizons:

  Short-term (1-5 days):   Daily Pivot Points, Bollinger Bands, Keltner, Donchian
  Medium-term (1-4 weeks): Daily SMA/EMA, Fibonacci Pivots, Supertrend
  Long-term (1-3 months):  Weekly SMA/EMA, Weekly Pivot Points, Weekly Bollinger

Pure computation, no LLM needed.
"""

from __future__ import annotations

from src.tools.tradingview_data import get_tv_analysis


def compute_trade_plan(
    ticker: str,
    indicators: dict,
    timeframe_signal: dict | None = None,
    action: str = "hold",
) -> dict:
    """Generate a trade plan with price zones and multi-horizon targets.

    Args:
        ticker: ASX ticker
        indicators: Raw indicator snapshot from technicals agent
        timeframe_signal: Timeframe agent output (signal, confidence, reasoning)
        action: Portfolio manager decision (buy/sell/short/cover/hold)

    Returns:
        Dict with buy_zone, sell_zone, stop_loss, short/mid/long targets, validity
    """
    price = indicators.get("current_price")
    if not price or price <= 0:
        return _empty_plan(ticker)

    atr = indicators.get("atr") or 0
    is_bearish = action in ("sell", "short")

    # Collect support/resistance from technicals indicators (short-term)
    supports, resistances = _collect_levels_from_indicators(indicators, price)

    # Get TradingView daily and weekly data for mid/long-term levels
    tv_daily = get_tv_analysis(ticker, "1d")
    tv_weekly = get_tv_analysis(ticker, "1W")

    mid_targets = _compute_mid_targets(tv_daily, price, is_bearish)
    long_targets = _compute_long_targets(tv_weekly, price, is_bearish)

    # Short-term targets from local indicators
    short_targets = _compute_short_targets(resistances, supports, price, action, atr)

    # Zones and stop loss
    buy_zone = _compute_zone(supports, price, "support")
    sell_zone = _compute_zone(resistances, price, "resistance")
    stop_loss = _compute_stop_loss(supports, price, atr, action)

    # Validity
    validity = _compute_validity(timeframe_signal)

    # Risk/reward based on mid-term target
    primary_targets = mid_targets if mid_targets else short_targets
    rr = _risk_reward(price, stop_loss, primary_targets)

    # Reasoning
    reasoning_parts = []
    if buy_zone["low"]:
        buy_refs = [n for n, v in supports[:3]]
        reasoning_parts.append(f"Buy zone: {', '.join(buy_refs)}")
    if sell_zone["low"]:
        sell_refs = [n for n, v in resistances[:3]]
        reasoning_parts.append(f"Sell zone: {', '.join(sell_refs)}")
    if stop_loss:
        reasoning_parts.append(f"Stop loss at {stop_loss:.2f} ({abs(price - stop_loss) / price * 100:.1f}% risk)")

    return {
        "ticker": ticker,
        "current_price": round(price, 2),
        "buy_zone": {"low": _r(buy_zone["low"]), "high": _r(buy_zone["high"])},
        "sell_zone": {"low": _r(sell_zone["low"]), "high": _r(sell_zone["high"])},
        "stop_loss": _r(stop_loss),
        "short_term_targets": [{"price": _r(t[1]), "label": t[0]} for t in short_targets[:2]],
        "mid_term_targets": [{"price": _r(t[1]), "label": t[0]} for t in mid_targets[:2]],
        "long_term_targets": [{"price": _r(t[1]), "label": t[0]} for t in long_targets[:2]],
        "risk_reward": rr,
        "validity_days": validity["days"],
        "validity_label": validity["label"],
        "reasoning": "; ".join(reasoning_parts),
    }


# ──────────────────────── Level Collection ────────────────────────


def _collect_levels_from_indicators(indicators: dict, price: float) -> tuple[list, list]:
    """Collect support/resistance from local technicals indicators."""
    supports = []
    resistances = []

    # Supertrend
    st_val = indicators.get("supertrend_val")
    st_dir = indicators.get("supertrend_dir")
    if st_val:
        if st_dir == 1:
            supports.append(("Supertrend", st_val))
        else:
            resistances.append(("Supertrend", st_val))

    # Bollinger Bands
    for name, key in [("BB Lower", "bb_lower"), ("BB Upper", "bb_upper"), ("BB Middle", "bb_middle")]:
        val = indicators.get(key)
        if val:
            if val < price:
                supports.append((name, val))
            else:
                resistances.append((name, val))

    # Donchian
    for name, key in [("Donchian Low", "donchian_lower"), ("Donchian High", "donchian_upper")]:
        val = indicators.get(key)
        if val:
            if val < price:
                supports.append((name, val))
            else:
                resistances.append((name, val))

    # Keltner
    for name, key in [("Keltner Low", "keltner_lower"), ("Keltner High", "keltner_upper")]:
        val = indicators.get(key)
        if val:
            if val < price:
                supports.append((name, val))
            else:
                resistances.append((name, val))

    # Fibonacci
    fib = indicators.get("fib_levels")
    if fib:
        for level_name, level_val in fib.items():
            if level_val < price:
                supports.append((f"Fib {level_name}", level_val))
            elif level_val > price:
                resistances.append((f"Fib {level_name}", level_val))

    # Pivot Points
    pivots = indicators.get("pivot_points")
    if pivots:
        for name, val in pivots.items():
            if val < price:
                supports.append((f"Pivot {name.upper()}", val))
            elif val > price:
                resistances.append((f"Pivot {name.upper()}", val))

    # Moving Averages
    for ma_name, ma_key in [("SMA20", "sma20"), ("SMA50", "sma50"), ("SMA200", "sma200")]:
        ma_val = indicators.get(ma_key)
        if ma_val:
            if ma_val < price:
                supports.append((ma_name, ma_val))
            elif ma_val > price:
                resistances.append((ma_name, ma_val))

    # VWAP
    vwap = indicators.get("vwap")
    if vwap:
        if vwap < price:
            supports.append(("VWAP", vwap))
        else:
            resistances.append(("VWAP", vwap))

    supports.sort(key=lambda x: price - x[1])
    resistances.sort(key=lambda x: x[1] - price)
    supports = [(n, v) for n, v in supports if v > price * 0.8]
    resistances = [(n, v) for n, v in resistances if v < price * 1.2]

    return supports, resistances


# ──────────────────────── Multi-Horizon Targets ────────────────────────


def _compute_short_targets(
    resistances: list, supports: list, price: float, action: str, atr: float,
) -> list[tuple[str, float]]:
    """Short-term targets (1-5 days) from local indicators."""
    if action in ("sell", "short"):
        candidates = [(n, v) for n, v in supports if v < price * 0.99]
    else:
        candidates = [(n, v) for n, v in resistances if v > price * 1.005]

    if len(candidates) < 2 and atr > 0:
        if action in ("sell", "short"):
            candidates.append(("ATR x1.5", price - 1.5 * atr))
            candidates.append(("ATR x2.5", price - 2.5 * atr))
        else:
            candidates.append(("ATR x1.5", price + 1.5 * atr))
            candidates.append(("ATR x2.5", price + 2.5 * atr))

    return _dedupe(candidates, price)[:2]


def _compute_mid_targets(
    tv_daily: dict | None, price: float, is_bearish: bool,
) -> list[tuple[str, float]]:
    """Medium-term targets (1-4 weeks) from daily TradingView data.

    Uses daily Fibonacci Pivots + key EMAs as medium-term reference levels.
    """
    if not tv_daily:
        return []

    ind = tv_daily.get("indicators", {})
    candidates = []

    # Daily Fibonacci Pivot levels
    fib_keys = [
        ("Pivot.M.Fibonacci.R1", "D-Fib R1"),
        ("Pivot.M.Fibonacci.R2", "D-Fib R2"),
        ("Pivot.M.Fibonacci.R3", "D-Fib R3"),
        ("Pivot.M.Fibonacci.S1", "D-Fib S1"),
        ("Pivot.M.Fibonacci.S2", "D-Fib S2"),
        ("Pivot.M.Fibonacci.S3", "D-Fib S3"),
        ("Pivot.M.Fibonacci.Middle", "D-Fib Pivot"),
    ]

    for tv_key, label in fib_keys:
        val = ind.get(tv_key)
        if val is not None:
            candidates.append((label, val))

    # Daily key EMAs
    for period in [50, 100, 200]:
        val = ind.get(f"EMA{period}")
        if val is not None:
            candidates.append((f"D-EMA{period}", val))

    # Daily SMA50/200
    for period in [50, 200]:
        val = ind.get(f"SMA{period}")
        if val is not None:
            candidates.append((f"D-SMA{period}", val))

    # Filter direction
    if is_bearish:
        candidates = [(n, v) for n, v in candidates if v < price * 0.99]
        candidates.sort(key=lambda x: price - x[1])
    else:
        candidates = [(n, v) for n, v in candidates if v > price * 1.01]
        candidates.sort(key=lambda x: x[1] - price)

    return _dedupe(candidates, price)[:2]


def _compute_long_targets(
    tv_weekly: dict | None, price: float, is_bearish: bool,
) -> list[tuple[str, float]]:
    """Long-term targets (1-3 months) from weekly TradingView data.

    Uses weekly Fibonacci Pivots, weekly EMAs, weekly Bollinger Bands.
    """
    if not tv_weekly:
        return []

    ind = tv_weekly.get("indicators", {})
    candidates = []

    # Weekly Fibonacci Pivot levels
    fib_keys = [
        ("Pivot.M.Fibonacci.R1", "W-Fib R1"),
        ("Pivot.M.Fibonacci.R2", "W-Fib R2"),
        ("Pivot.M.Fibonacci.R3", "W-Fib R3"),
        ("Pivot.M.Fibonacci.S1", "W-Fib S1"),
        ("Pivot.M.Fibonacci.S2", "W-Fib S2"),
        ("Pivot.M.Fibonacci.S3", "W-Fib S3"),
    ]

    for tv_key, label in fib_keys:
        val = ind.get(tv_key)
        if val is not None:
            candidates.append((label, val))

    # Weekly key EMAs
    for period in [20, 50, 100, 200]:
        val = ind.get(f"EMA{period}")
        if val is not None:
            candidates.append((f"W-EMA{period}", val))

    # Weekly Bollinger
    bb_upper = ind.get("BB.upper")
    bb_lower = ind.get("BB.lower")
    if bb_upper is not None:
        candidates.append(("W-BB Upper", bb_upper))
    if bb_lower is not None:
        candidates.append(("W-BB Lower", bb_lower))

    # Filter direction
    if is_bearish:
        candidates = [(n, v) for n, v in candidates if v < price * 0.98]
        candidates.sort(key=lambda x: price - x[1])
    else:
        candidates = [(n, v) for n, v in candidates if v > price * 1.02]
        candidates.sort(key=lambda x: x[1] - price)

    return _dedupe(candidates, price)[:2]


# ──────────────────────── Zones & Helpers ────────────────────────


def _compute_zone(levels: list[tuple[str, float]], price: float, zone_type: str) -> dict:
    """Find a price zone from clustered indicator levels."""
    if not levels:
        return {"low": None, "high": None}

    close_levels = [v for _, v in levels[:4]]

    if zone_type == "support":
        zone_low = min(close_levels)
        zone_high = max(close_levels[:2]) if len(close_levels) >= 2 else close_levels[0]
        if zone_high > price:
            zone_high = price * 0.99
        if zone_low > zone_high:
            zone_low = zone_high * 0.98
    else:
        zone_low = min(close_levels[:2]) if len(close_levels) >= 2 else close_levels[0]
        zone_high = max(close_levels[:3]) if len(close_levels) >= 3 else max(close_levels)
        if zone_low < price:
            zone_low = price * 1.01

    return {"low": zone_low, "high": zone_high}


def _compute_stop_loss(
    supports: list[tuple[str, float]], price: float, atr: float, action: str,
) -> float | None:
    """Compute stop loss level."""
    if action in ("sell", "short"):
        return round(price + 2 * atr, 2) if atr > 0 else round(price * 1.05, 2)

    if supports:
        lowest_support = min(v for _, v in supports[:3])
        stop = lowest_support - atr if atr > 0 else lowest_support * 0.97
        if stop < price * 0.85:
            stop = price * 0.85
        return stop
    elif atr > 0:
        return price - 2 * atr
    else:
        return price * 0.95


def _risk_reward(
    price: float, stop_loss: float | None, targets: list[tuple[str, float]],
) -> float | None:
    """Compute risk/reward ratio based on first target."""
    if not stop_loss or not targets:
        return None
    risk = abs(price - stop_loss)
    if risk == 0:
        return None
    reward = abs(targets[0][1] - price)
    return round(reward / risk, 2)


def _compute_validity(timeframe_signal: dict | None) -> dict:
    """Determine signal validity period from timeframe analysis."""
    if not timeframe_signal:
        return {"days": 5, "label": "5 trading days (default)"}

    reasoning = timeframe_signal.get("reasoning", "")
    confidence = timeframe_signal.get("confidence", 50)

    has_weekly = "Weekly:" in reasoning
    has_daily = "Daily:" in reasoning
    has_4h = "4-Hour:" in reasoning or "4h:" in reasoning.lower()
    aligned = "ALIGNED" in reasoning

    if aligned and has_weekly:
        return {"days": 20, "label": "3-4 weeks (all timeframes aligned)"}
    elif "CONFLICT" in reasoning:
        return {"days": 2, "label": "1-2 trading days (timeframe conflict)"}
    elif has_weekly and confidence > 70:
        return {"days": 15, "label": "2-3 weeks (weekly dominant)"}
    elif has_daily and confidence > 60:
        return {"days": 8, "label": "5-10 trading days (daily dominant)"}
    elif has_4h:
        return {"days": 3, "label": "1-3 trading days (4H signal)"}
    else:
        return {"days": 5, "label": "5 trading days (default)"}


def _dedupe(candidates: list[tuple[str, float]], price: float) -> list[tuple[str, float]]:
    """Remove targets too close together (within 1% of each other)."""
    if not candidates:
        return []
    filtered = [candidates[0]]
    for n, v in candidates[1:]:
        if abs(v - filtered[-1][1]) / price > 0.01:
            filtered.append((n, v))
    return filtered


def _r(v: float | None) -> float | None:
    """Round to 2 decimal places."""
    return round(v, 2) if v is not None else None


def _empty_plan(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "current_price": None,
        "buy_zone": {"low": None, "high": None},
        "sell_zone": {"low": None, "high": None},
        "stop_loss": None,
        "short_term_targets": [],
        "mid_term_targets": [],
        "long_term_targets": [],
        "risk_reward": None,
        "validity_days": 5,
        "validity_label": "N/A",
        "reasoning": "Insufficient data for trade plan",
    }
