"""Trade plan generator — computes buy/sell zones, stop loss, and validity period.

Uses raw indicator data from technicals agent and timeframe alignment to produce
actionable price levels. Pure computation, no LLM needed.

Price zones are derived from convergence of multiple indicators:
  - Fibonacci retracement levels
  - Bollinger Bands
  - Supertrend support/resistance
  - Donchian Channel boundaries
  - Pivot Points
  - Key moving averages (SMA20/50/200)

Validity period is derived from multi-timeframe alignment:
  - Weekly signal dominant → 2-4 weeks
  - Daily signal dominant → 5-10 trading days
  - 4H signal dominant → 1-3 trading days
  - Conflicting signals → shortest validity
"""

from __future__ import annotations


def compute_trade_plan(
    ticker: str,
    indicators: dict,
    timeframe_signal: dict | None = None,
    action: str = "hold",
) -> dict:
    """Generate a trade plan with price zones and validity.

    Args:
        ticker: ASX ticker
        indicators: Raw indicator snapshot from technicals agent
        timeframe_signal: Timeframe agent output (signal, confidence, reasoning)
        action: Portfolio manager decision (buy/sell/short/cover/hold)

    Returns:
        Dict with buy_zone, sell_zone, stop_loss, targets, validity_days, reasoning
    """
    price = indicators.get("current_price")
    if not price or price <= 0:
        return _empty_plan(ticker)

    atr = indicators.get("atr") or 0
    atr_pct = atr / price if price > 0 else 0.02

    # Collect support levels (potential buy zones)
    supports = []
    # Collect resistance levels (potential sell/target zones)
    resistances = []

    # ── Supertrend ──
    st_val = indicators.get("supertrend_val")
    st_dir = indicators.get("supertrend_dir")
    if st_val:
        if st_dir == 1:  # bullish — supertrend is support
            supports.append(("Supertrend", st_val))
        else:  # bearish — supertrend is resistance
            resistances.append(("Supertrend", st_val))

    # ── Bollinger Bands ──
    bb_upper = indicators.get("bb_upper")
    bb_lower = indicators.get("bb_lower")
    bb_middle = indicators.get("bb_middle")
    if bb_lower:
        supports.append(("BB Lower", bb_lower))
    if bb_upper:
        resistances.append(("BB Upper", bb_upper))
    if bb_middle:
        if price > bb_middle:
            supports.append(("BB Middle", bb_middle))
        else:
            resistances.append(("BB Middle", bb_middle))

    # ── Donchian Channel ──
    dc_upper = indicators.get("donchian_upper")
    dc_lower = indicators.get("donchian_lower")
    if dc_lower:
        supports.append(("Donchian Low", dc_lower))
    if dc_upper:
        resistances.append(("Donchian High", dc_upper))

    # ── Keltner Channel ──
    kc_upper = indicators.get("keltner_upper")
    kc_lower = indicators.get("keltner_lower")
    if kc_lower:
        supports.append(("Keltner Low", kc_lower))
    if kc_upper:
        resistances.append(("Keltner High", kc_upper))

    # ── Fibonacci levels ──
    fib = indicators.get("fib_levels")
    if fib:
        for level_name, level_val in fib.items():
            if level_val < price:
                supports.append((f"Fib {level_name}", level_val))
            elif level_val > price:
                resistances.append((f"Fib {level_name}", level_val))

    # ── Pivot Points ──
    pivots = indicators.get("pivot_points")
    if pivots:
        for name, val in pivots.items():
            if val < price:
                supports.append((f"Pivot {name.upper()}", val))
            elif val > price:
                resistances.append((f"Pivot {name.upper()}", val))

    # ── Moving Averages ──
    for ma_name, ma_key in [("SMA20", "sma20"), ("SMA50", "sma50"), ("SMA200", "sma200")]:
        ma_val = indicators.get(ma_key)
        if ma_val:
            if ma_val < price:
                supports.append((ma_name, ma_val))
            elif ma_val > price:
                resistances.append((ma_name, ma_val))

    # ── VWAP ──
    vwap = indicators.get("vwap")
    if vwap:
        if vwap < price:
            supports.append(("VWAP", vwap))
        else:
            resistances.append(("VWAP", vwap))

    # Sort by proximity to current price
    supports.sort(key=lambda x: price - x[1])  # closest first
    resistances.sort(key=lambda x: x[1] - price)  # closest first

    # Filter to only levels within reasonable range (within 20% of price)
    supports = [(n, v) for n, v in supports if v > price * 0.8]
    resistances = [(n, v) for n, v in resistances if v < price * 1.2]

    # ── Compute zones ──
    buy_zone = _compute_zone(supports, price, "support")
    sell_zone = _compute_zone(resistances, price, "resistance")
    stop_loss = _compute_stop_loss(supports, price, atr, action)
    targets = _compute_targets(resistances, supports, price, action, atr)

    # ── Validity period ──
    validity = _compute_validity(timeframe_signal)

    # ── Build reasoning ──
    reasoning_parts = []
    if buy_zone["low"]:
        buy_refs = [n for n, v in supports[:3]]
        reasoning_parts.append(f"Buy zone based on: {', '.join(buy_refs)}")
    if sell_zone["low"]:
        sell_refs = [n for n, v in resistances[:3]]
        reasoning_parts.append(f"Sell zone based on: {', '.join(sell_refs)}")
    if stop_loss:
        reasoning_parts.append(f"Stop loss at {stop_loss:.2f} ({(price - stop_loss) / price * 100:.1f}% risk)")

    return {
        "ticker": ticker,
        "current_price": round(price, 2),
        "buy_zone": {"low": _r(buy_zone["low"]), "high": _r(buy_zone["high"])},
        "sell_zone": {"low": _r(sell_zone["low"]), "high": _r(sell_zone["high"])},
        "stop_loss": _r(stop_loss),
        "targets": [{"price": _r(t[1]), "label": t[0]} for t in targets[:3]],
        "risk_reward": _risk_reward(price, stop_loss, targets),
        "validity_days": validity["days"],
        "validity_label": validity["label"],
        "reasoning": "; ".join(reasoning_parts),
    }


def _compute_zone(levels: list[tuple[str, float]], price: float, zone_type: str) -> dict:
    """Find a price zone from clustered indicator levels."""
    if not levels:
        return {"low": None, "high": None}

    # Take the 2-3 closest levels and form a zone
    close_levels = [v for _, v in levels[:4]]

    if zone_type == "support":
        # Buy zone: cluster of support levels below price
        zone_low = min(close_levels)
        zone_high = max(close_levels[:2]) if len(close_levels) >= 2 else close_levels[0]
        # Ensure zone makes sense
        if zone_high > price:
            zone_high = price * 0.99
        if zone_low > zone_high:
            zone_low = zone_high * 0.98
    else:
        # Sell zone: cluster of resistance levels above price
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
        # For short: stop loss above current price
        return round(price + 2 * atr, 2) if atr > 0 else round(price * 1.05, 2)

    # For buy/hold: stop loss below support
    if supports:
        # Below the strongest support cluster, minus 1 ATR buffer
        lowest_support = min(v for _, v in supports[:3])
        stop = lowest_support - atr if atr > 0 else lowest_support * 0.97
        # Don't set stop more than 15% below price
        if stop < price * 0.85:
            stop = price * 0.85
        return stop
    elif atr > 0:
        return price - 2 * atr
    else:
        return price * 0.95


def _compute_targets(
    resistances: list[tuple[str, float]],
    supports: list[tuple[str, float]],
    price: float,
    action: str,
    atr: float = 0,
) -> list[tuple[str, float]]:
    """Compute 1-3 price targets, ensuring meaningful spread."""
    if action in ("sell", "short"):
        candidates = [(n, v) for n, v in supports if v < price * 0.99]
    else:
        candidates = [(n, v) for n, v in resistances if v > price * 1.01]

    # If not enough targets from indicators, add ATR-based projections
    if len(candidates) < 3 and atr > 0:
        if action in ("sell", "short"):
            for i, mult in enumerate([1.5, 3.0, 5.0], 1):
                candidates.append((f"ATR x{mult:.1f}", price - mult * atr))
        else:
            for i, mult in enumerate([1.5, 3.0, 5.0], 1):
                candidates.append((f"ATR x{mult:.1f}", price + mult * atr))

    # Remove duplicates close together (within 0.5%)
    filtered = []
    for n, v in candidates:
        if not filtered or abs(v - filtered[-1][1]) / price > 0.005:
            filtered.append((n, v))

    return filtered[:3]


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

    # Parse which timeframes are aligned
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
        "targets": [],
        "risk_reward": None,
        "validity_days": 5,
        "validity_label": "N/A",
        "reasoning": "Insufficient data for trade plan",
    }
