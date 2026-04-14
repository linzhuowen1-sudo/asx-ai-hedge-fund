"""Technical analyst agent — comprehensive indicator suite for ASX stocks.

Indicators:
  Trend:       SMA(20/50/200), EMA(12/26), MACD, ADX, Supertrend
  Momentum:    RSI(14), Stochastic(14,3), Williams %R, CCI
  Volatility:  Bollinger Bands(20,2), ATR(14), Keltner Channels, Donchian Channel
  Volume:      OBV, VWAP, Volume Trend, Accumulation/Distribution
  Support:     Fibonacci Retracement Levels, Pivot Points
"""

from __future__ import annotations


from datetime import datetime, timedelta

import numpy as np
from langchain_core.messages import HumanMessage

from src.data.models import AnalystSignal, Signal
from src.graph.state import AgentState
from src.tools.asx_data import get_price_history


# ──────────────────────────── Helper functions ────────────────────────────


def _ema(data: list[float], period: int) -> float | None:
    """Exponential Moving Average."""
    if len(data) < period:
        return None
    multiplier = 2 / (period + 1)
    ema = np.mean(data[:period])
    for price in data[period:]:
        ema = (price - ema) * multiplier + ema
    return ema


def _ema_series(data: list[float], period: int) -> list[float]:
    """Return full EMA series (useful for MACD signal line, etc.)."""
    if len(data) < period:
        return []
    multiplier = 2 / (period + 1)
    result = [np.mean(data[:period])]
    for price in data[period:]:
        result.append((price - result[-1]) * multiplier + result[-1])
    return result


def _sma(data: list[float], period: int) -> float | None:
    if len(data) < period:
        return None
    return float(np.mean(data[-period:]))


# ───── Trend Indicators ─────


def compute_sma_crossovers(closes: list[float]) -> dict:
    """SMA 20/50/200 crossover signals."""
    sma20 = _sma(closes, 20)
    sma50 = _sma(closes, 50)
    sma200 = _sma(closes, 200)
    return {"sma20": sma20, "sma50": sma50, "sma200": sma200}


def compute_macd(closes: list[float]) -> dict:
    """MACD with proper signal line and histogram."""
    if len(closes) < 35:
        return {"macd_line": None, "signal_line": None, "histogram": None}

    ema12_series = _ema_series(closes, 12)
    ema26_series = _ema_series(closes, 26)

    # Align lengths — EMA26 is shorter
    offset = len(ema12_series) - len(ema26_series)
    macd_series = [
        ema12_series[offset + i] - ema26_series[i]
        for i in range(len(ema26_series))
    ]

    if len(macd_series) < 9:
        return {"macd_line": macd_series[-1] if macd_series else None, "signal_line": None, "histogram": None}

    signal_series = _ema_series(macd_series, 9)
    macd_line = macd_series[-1]
    signal_line = signal_series[-1] if signal_series else None
    histogram = (macd_line - signal_line) if signal_line is not None else None

    return {"macd_line": macd_line, "signal_line": signal_line, "histogram": histogram}


def compute_adx(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float | None:
    """Average Directional Index — trend strength (0-100)."""
    n = len(closes)
    if n < period + 1:
        return None

    plus_dm = []
    minus_dm = []
    tr_list = []

    for i in range(1, n):
        high_diff = highs[i] - highs[i - 1]
        low_diff = lows[i - 1] - lows[i]
        plus_dm.append(max(high_diff, 0) if high_diff > low_diff else 0)
        minus_dm.append(max(low_diff, 0) if low_diff > high_diff else 0)
        tr_list.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))

    if len(tr_list) < period:
        return None

    # Smooth with Wilder's method
    atr = np.mean(tr_list[:period])
    plus_di_smooth = np.mean(plus_dm[:period])
    minus_di_smooth = np.mean(minus_dm[:period])

    dx_values = []
    for i in range(period, len(tr_list)):
        atr = atr - (atr / period) + tr_list[i]
        plus_di_smooth = plus_di_smooth - (plus_di_smooth / period) + plus_dm[i]
        minus_di_smooth = minus_di_smooth - (minus_di_smooth / period) + minus_dm[i]

        if atr == 0:
            continue
        plus_di = 100 * plus_di_smooth / atr
        minus_di = 100 * minus_di_smooth / atr
        di_sum = plus_di + minus_di
        if di_sum == 0:
            continue
        dx = 100 * abs(plus_di - minus_di) / di_sum
        dx_values.append(dx)

    if len(dx_values) < period:
        return np.mean(dx_values) if dx_values else None
    return float(np.mean(dx_values[-period:]))


# ───── Momentum Indicators ─────


def compute_rsi(closes: list[float], period: int = 14) -> float | None:
    """Relative Strength Index (Wilder's smoothing)."""
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Wilder's smoothing
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_stochastic(
    highs: list[float], lows: list[float], closes: list[float],
    k_period: int = 14, d_period: int = 3,
) -> tuple[float | None, float | None]:
    """Stochastic Oscillator (%K and %D)."""
    if len(closes) < k_period:
        return None, None

    k_values = []
    for i in range(k_period - 1, len(closes)):
        window_high = max(highs[i - k_period + 1: i + 1])
        window_low = min(lows[i - k_period + 1: i + 1])
        if window_high == window_low:
            k_values.append(50.0)
        else:
            k_values.append(100 * (closes[i] - window_low) / (window_high - window_low))

    pct_k = k_values[-1] if k_values else None
    pct_d = float(np.mean(k_values[-d_period:])) if len(k_values) >= d_period else None
    return pct_k, pct_d


def compute_williams_r(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14,
) -> float | None:
    """Williams %R (-100 to 0)."""
    if len(closes) < period:
        return None
    window_high = max(highs[-period:])
    window_low = min(lows[-period:])
    if window_high == window_low:
        return -50.0
    return -100 * (window_high - closes[-1]) / (window_high - window_low)


def compute_cci(
    highs: list[float], lows: list[float], closes: list[float], period: int = 20,
) -> float | None:
    """Commodity Channel Index."""
    if len(closes) < period:
        return None
    typical_prices = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(closes))]
    tp_slice = typical_prices[-period:]
    tp_mean = np.mean(tp_slice)
    mean_dev = np.mean(np.abs(np.array(tp_slice) - tp_mean))
    if mean_dev == 0:
        return 0.0
    return float((typical_prices[-1] - tp_mean) / (0.015 * mean_dev))


# ───── Volatility Indicators ─────


def compute_bollinger_bands(
    closes: list[float], period: int = 20, std_dev: float = 2.0,
) -> dict:
    """Bollinger Bands — upper, middle (SMA), lower."""
    if len(closes) < period:
        return {"upper": None, "middle": None, "lower": None, "pct_b": None, "bandwidth": None}

    window = closes[-period:]
    middle = float(np.mean(window))
    std = float(np.std(window, ddof=1))

    upper = middle + std_dev * std
    lower = middle - std_dev * std

    # %B: where is price relative to bands (0=lower, 1=upper)
    pct_b = (closes[-1] - lower) / (upper - lower) if upper != lower else 0.5
    # Bandwidth: how wide are the bands (squeeze detection)
    bandwidth = (upper - lower) / middle if middle != 0 else 0

    return {"upper": upper, "middle": middle, "lower": lower, "pct_b": pct_b, "bandwidth": bandwidth}


def compute_atr(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14,
) -> float | None:
    """Average True Range — volatility measure."""
    if len(closes) < period + 1:
        return None
    tr_values = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        tr_values.append(tr)
    return float(np.mean(tr_values[-period:]))


def compute_keltner_channels(
    highs: list[float], lows: list[float], closes: list[float],
    ema_period: int = 20, atr_period: int = 14, multiplier: float = 2.0,
) -> dict:
    """Keltner Channels — EMA ± ATR * multiplier."""
    ema_val = _ema(closes, ema_period)
    atr_val = compute_atr(highs, lows, closes, atr_period)
    if ema_val is None or atr_val is None:
        return {"upper": None, "middle": None, "lower": None}
    return {
        "upper": ema_val + multiplier * atr_val,
        "middle": ema_val,
        "lower": ema_val - multiplier * atr_val,
    }


def compute_supertrend(
    highs: list[float], lows: list[float], closes: list[float],
    period: int = 10, multiplier: float = 3.0,
) -> dict:
    """Supertrend — ATR-based trend-following indicator.

    Returns the current supertrend value and direction (1=bullish, -1=bearish).
    """
    n = len(closes)
    if n < period + 1:
        return {"value": None, "direction": None}

    # Compute ATR series
    tr_list = [highs[0] - lows[0]]
    for i in range(1, n):
        tr_list.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))

    # Wilder's smoothed ATR
    atr = [float(np.mean(tr_list[:period]))]
    for i in range(period, n):
        atr.append((atr[-1] * (period - 1) + tr_list[i]) / period)

    # Supertrend calculation
    upper_band = [0.0] * n
    lower_band = [0.0] * n
    supertrend = [0.0] * n
    direction = [1] * n  # 1=bullish, -1=bearish

    for i in range(period, n):
        atr_idx = i - period
        hl2 = (highs[i] + lows[i]) / 2
        basic_upper = hl2 + multiplier * atr[atr_idx]
        basic_lower = hl2 - multiplier * atr[atr_idx]

        upper_band[i] = min(basic_upper, upper_band[i - 1]) if closes[i - 1] <= upper_band[i - 1] else basic_upper
        lower_band[i] = max(basic_lower, lower_band[i - 1]) if closes[i - 1] >= lower_band[i - 1] else basic_lower

        if closes[i] > upper_band[i]:
            direction[i] = 1
        elif closes[i] < lower_band[i]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

        supertrend[i] = lower_band[i] if direction[i] == 1 else upper_band[i]

    return {"value": supertrend[-1], "direction": direction[-1]}


def compute_donchian_channel(
    highs: list[float], lows: list[float], period: int = 20,
) -> dict:
    """Donchian Channel — highest high / lowest low over N periods."""
    if len(highs) < period:
        return {"upper": None, "middle": None, "lower": None}
    upper = max(highs[-period:])
    lower = min(lows[-period:])
    middle = (upper + lower) / 2
    return {"upper": upper, "middle": middle, "lower": lower}


# ───── Volume Indicators ─────


def compute_obv(closes: list[float], volumes: list[int]) -> list[float]:
    """On-Balance Volume — cumulative volume in direction of price."""
    if len(closes) < 2:
        return []
    obv = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i - 1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    return obv


def compute_vwap(
    highs: list[float], lows: list[float], closes: list[float], volumes: list[int],
) -> float | None:
    """Volume-Weighted Average Price."""
    if len(closes) == 0 or sum(volumes) == 0:
        return None
    typical_prices = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(closes))]
    cum_tp_vol = sum(tp * v for tp, v in zip(typical_prices, volumes))
    cum_vol = sum(volumes)
    return cum_tp_vol / cum_vol if cum_vol != 0 else None


def compute_accumulation_distribution(
    highs: list[float], lows: list[float], closes: list[float], volumes: list[int],
) -> list[float]:
    """Accumulation/Distribution line."""
    ad = [0.0]
    for i in range(len(closes)):
        hl_range = highs[i] - lows[i]
        if hl_range == 0:
            mfm = 0.0
        else:
            mfm = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / hl_range
        mfv = mfm * volumes[i]
        ad.append(ad[-1] + mfv)
    return ad[1:]


# ───── Support / Resistance ─────


def compute_fibonacci_levels(highs: list[float], lows: list[float]) -> dict:
    """Fibonacci retracement levels from recent swing high/low."""
    if len(highs) < 20:
        return {}
    swing_high = max(highs[-60:]) if len(highs) >= 60 else max(highs)
    swing_low = min(lows[-60:]) if len(lows) >= 60 else min(lows)
    diff = swing_high - swing_low
    if diff == 0:
        return {}
    return {
        "level_0": swing_high,
        "level_236": swing_high - 0.236 * diff,
        "level_382": swing_high - 0.382 * diff,
        "level_500": swing_high - 0.500 * diff,
        "level_618": swing_high - 0.618 * diff,
        "level_786": swing_high - 0.786 * diff,
        "level_100": swing_low,
    }


def compute_pivot_points(high: float, low: float, close: float) -> dict:
    """Classic pivot points from prior day's HLC."""
    pivot = (high + low + close) / 3
    return {
        "pivot": pivot,
        "r1": 2 * pivot - low,
        "r2": pivot + (high - low),
        "s1": 2 * pivot - high,
        "s2": pivot - (high - low),
    }


# ──────────────────────────── Main Agent ────────────────────────────


def technicals_agent(state: AgentState) -> dict:
    """Run comprehensive technical analysis on each ticker."""
    tickers = state["metadata"]["tickers"]
    end_date = state["metadata"].get("end_date", datetime.now().strftime("%Y-%m-%d"))
    # Need 200+ days for SMA200
    start_date_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)
    start_date = state["metadata"].get("start_date", start_date_dt.strftime("%Y-%m-%d"))
    # Override start to ensure enough data for SMA200
    if (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days < 250:
        start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")

    signals = {}

    for ticker in tickers:
        prices = get_price_history(ticker, start_date, end_date)
        if len(prices) < 30:
            signals[ticker] = AnalystSignal(
                agent_name="technicals_analyst",
                ticker=ticker,
                signal=Signal.NEUTRAL,
                confidence=10,
                reasoning="Insufficient price history for technical analysis.",
            )
            continue

        closes = [p.close for p in prices]
        highs = [p.high for p in prices]
        lows = [p.low for p in prices]
        volumes = [p.volume for p in prices]
        current_price = closes[-1]

        details = []
        score = 0.0
        checks = 0

        # ━━━ 1. TREND: SMA Crossovers ━━━
        smas = compute_sma_crossovers(closes)
        sma20, sma50, sma200 = smas["sma20"], smas["sma50"], smas["sma200"]

        if sma20 is not None and sma50 is not None:
            checks += 1
            if sma20 > sma50:
                score += 1
                details.append(f"SMA20({sma20:.2f}) > SMA50({sma50:.2f}) 金叉")
            else:
                score -= 1
                details.append(f"SMA20({sma20:.2f}) < SMA50({sma50:.2f}) 死叉")

        if sma200 is not None:
            checks += 1
            if current_price > sma200:
                score += 1
                details.append(f"Price above SMA200({sma200:.2f}) — long-term uptrend")
            else:
                score -= 1
                details.append(f"Price below SMA200({sma200:.2f}) — long-term downtrend")

        # ━━━ 2. TREND: MACD ━━━
        macd = compute_macd(closes)
        if macd["macd_line"] is not None and macd["signal_line"] is not None:
            checks += 1
            if macd["macd_line"] > macd["signal_line"]:
                score += 1
                details.append(f"MACD({macd['macd_line']:.3f}) > Signal({macd['signal_line']:.3f}) — bullish")
            else:
                score -= 1
                details.append(f"MACD({macd['macd_line']:.3f}) < Signal({macd['signal_line']:.3f}) — bearish")
            if macd["histogram"] is not None:
                hist_dir = "expanding" if abs(macd["histogram"]) > 0.01 else "flat"
                details.append(f"MACD histogram: {macd['histogram']:.4f} ({hist_dir})")

        # ━━━ 3. TREND: ADX ━━━
        adx = compute_adx(highs, lows, closes)
        if adx is not None:
            checks += 1
            if adx > 25:
                # Strong trend — amplify existing direction
                direction = "bullish" if score > 0 else "bearish" if score < 0 else "neutral"
                if score > 0:
                    score += 0.5
                elif score < 0:
                    score -= 0.5
                details.append(f"ADX {adx:.1f} — strong trend ({direction})")
            else:
                details.append(f"ADX {adx:.1f} — weak/no trend (range-bound)")

        # ━━━ 4. MOMENTUM: RSI ━━━
        rsi = compute_rsi(closes)
        if rsi is not None:
            checks += 1
            if rsi > 80:
                score -= 1.5
                details.append(f"RSI {rsi:.1f} — strongly overbought")
            elif rsi > 70:
                score -= 1
                details.append(f"RSI {rsi:.1f} — overbought")
            elif rsi < 20:
                score += 1.5
                details.append(f"RSI {rsi:.1f} — strongly oversold")
            elif rsi < 30:
                score += 1
                details.append(f"RSI {rsi:.1f} — oversold")
            else:
                details.append(f"RSI {rsi:.1f} — neutral zone")

        # ━━━ 5. MOMENTUM: Stochastic ━━━
        stoch_k, stoch_d = compute_stochastic(highs, lows, closes)
        if stoch_k is not None and stoch_d is not None:
            checks += 1
            if stoch_k > 80 and stoch_d > 80:
                score -= 1
                details.append(f"Stochastic %K={stoch_k:.1f} %D={stoch_d:.1f} — overbought")
            elif stoch_k < 20 and stoch_d < 20:
                score += 1
                details.append(f"Stochastic %K={stoch_k:.1f} %D={stoch_d:.1f} — oversold")
            elif stoch_k > stoch_d:
                score += 0.5
                details.append(f"Stochastic %K({stoch_k:.1f}) > %D({stoch_d:.1f}) — bullish crossover")
            else:
                score -= 0.5
                details.append(f"Stochastic %K({stoch_k:.1f}) < %D({stoch_d:.1f}) — bearish crossover")

        # ━━━ 6. MOMENTUM: Williams %R ━━━
        williams = compute_williams_r(highs, lows, closes)
        if williams is not None:
            checks += 1
            if williams > -20:
                score -= 0.5
                details.append(f"Williams %R = {williams:.1f} — overbought")
            elif williams < -80:
                score += 0.5
                details.append(f"Williams %R = {williams:.1f} — oversold")
            else:
                details.append(f"Williams %R = {williams:.1f} — neutral")

        # ━━━ 7. MOMENTUM: CCI ━━━
        cci = compute_cci(highs, lows, closes)
        if cci is not None:
            checks += 1
            if cci > 200:
                score -= 1
                details.append(f"CCI {cci:.0f} — extremely overbought")
            elif cci > 100:
                score -= 0.5
                details.append(f"CCI {cci:.0f} — overbought")
            elif cci < -200:
                score += 1
                details.append(f"CCI {cci:.0f} — extremely oversold")
            elif cci < -100:
                score += 0.5
                details.append(f"CCI {cci:.0f} — oversold")

        # ━━━ 8. VOLATILITY: Bollinger Bands ━━━
        bb = compute_bollinger_bands(closes)
        if bb["pct_b"] is not None:
            checks += 1
            if bb["pct_b"] > 1.0:
                score -= 1
                details.append(f"BB %B={bb['pct_b']:.2f} — above upper band (overextended)")
            elif bb["pct_b"] < 0.0:
                score += 1
                details.append(f"BB %B={bb['pct_b']:.2f} — below lower band (oversold)")
            elif bb["pct_b"] < 0.2:
                score += 0.5
                details.append(f"BB %B={bb['pct_b']:.2f} — near lower band")
            elif bb["pct_b"] > 0.8:
                score -= 0.5
                details.append(f"BB %B={bb['pct_b']:.2f} — near upper band")

            # Squeeze detection
            if bb["bandwidth"] is not None and bb["bandwidth"] < 0.05:
                details.append(f"BB squeeze detected (bandwidth={bb['bandwidth']:.4f}) — breakout imminent")

        # ━━━ 9. VOLATILITY: ATR context ━━━
        atr = compute_atr(highs, lows, closes)
        if atr is not None and current_price > 0:
            atr_pct = atr / current_price * 100
            details.append(f"ATR={atr:.3f} ({atr_pct:.1f}% of price) — {'high' if atr_pct > 3 else 'normal'} volatility")

        # ━━━ 10. TREND: Supertrend ━━━
        st = compute_supertrend(highs, lows, closes)
        if st["direction"] is not None:
            checks += 1
            if st["direction"] == 1:
                score += 1
                details.append(f"Supertrend BULLISH (support at {st['value']:.2f})")
            else:
                score -= 1
                details.append(f"Supertrend BEARISH (resistance at {st['value']:.2f})")

        # ━━━ 11. VOLATILITY: Donchian Channel ━━━
        dc = compute_donchian_channel(highs, lows)
        if dc["upper"] is not None:
            checks += 1
            dc_range = dc["upper"] - dc["lower"]
            if dc_range > 0:
                dc_pos = (current_price - dc["lower"]) / dc_range
                if dc_pos > 0.95:
                    score += 0.5
                    details.append(f"Donchian breakout — at 20-day high ({dc['upper']:.2f})")
                elif dc_pos < 0.05:
                    score -= 0.5
                    details.append(f"Donchian breakdown — at 20-day low ({dc['lower']:.2f})")
                else:
                    details.append(f"Donchian position: {dc_pos:.0%} (range {dc['lower']:.2f}-{dc['upper']:.2f})")

        # ━━━ 12. VOLATILITY: Keltner Channel + BB Squeeze ━━━
        kc = compute_keltner_channels(highs, lows, closes)
        if kc["upper"] is not None and bb["upper"] is not None:
            # Keltner-BB squeeze: BB inside KC = low volatility, breakout imminent
            if bb["upper"] < kc["upper"] and bb["lower"] > kc["lower"]:
                details.append("KC-BB squeeze — Bollinger inside Keltner, expect breakout")

        # ━━━ VOLUME: OBV trend ━━━
        obv = compute_obv(closes, volumes)
        if len(obv) >= 20:
            checks += 1
            obv_sma = np.mean(obv[-20:])
            if obv[-1] > obv_sma and obv[-1] > obv[-5]:
                score += 0.5
                details.append("OBV rising — accumulation")
            elif obv[-1] < obv_sma and obv[-1] < obv[-5]:
                score -= 0.5
                details.append("OBV falling — distribution")

        # ━━━ 11. VOLUME: VWAP ━━━
        vwap = compute_vwap(highs[-20:], lows[-20:], closes[-20:], volumes[-20:])
        if vwap is not None:
            checks += 1
            if current_price > vwap * 1.02:
                score += 0.5
                details.append(f"Price({current_price:.2f}) > VWAP({vwap:.2f}) — bullish")
            elif current_price < vwap * 0.98:
                score -= 0.5
                details.append(f"Price({current_price:.2f}) < VWAP({vwap:.2f}) — bearish")

        # ━━━ 12. VOLUME: Volume surge ━━━
        if len(volumes) >= 20:
            recent_vol = np.mean(volumes[-5:])
            avg_vol = np.mean(volumes[-20:])
            if avg_vol > 0:
                vol_ratio = recent_vol / avg_vol
                if vol_ratio > 2.0:
                    details.append(f"Volume surge: {vol_ratio:.1f}x average — strong conviction")
                    if score > 0:
                        score += 0.5
                    elif score < 0:
                        score -= 0.5
                elif vol_ratio > 1.5:
                    details.append(f"Volume elevated: {vol_ratio:.1f}x average")
                elif vol_ratio < 0.5:
                    details.append(f"Volume dry-up: {vol_ratio:.1f}x average — low conviction")

        # ━━━ 13. SUPPORT/RESISTANCE: Fibonacci ━━━
        fib = compute_fibonacci_levels(highs, lows)
        if fib:
            near_support = False
            near_resistance = False
            for level_name, level_val in fib.items():
                dist = abs(current_price - level_val) / current_price
                if dist < 0.02:  # Within 2% of a Fib level
                    if current_price > level_val:
                        near_support = True
                        details.append(f"Near Fib support {level_name}({level_val:.2f})")
                    else:
                        near_resistance = True
                        details.append(f"Near Fib resistance {level_name}({level_val:.2f})")

            if near_support:
                score += 0.3
            if near_resistance:
                score -= 0.3

        # ━━━ 14. Pivot Points ━━━
        if len(prices) >= 2:
            prev = prices[-2]
            pivots = compute_pivot_points(prev.high, prev.low, prev.close)
            if current_price > pivots["r1"]:
                details.append(f"Above R1({pivots['r1']:.2f}) — bullish breakout zone")
            elif current_price < pivots["s1"]:
                details.append(f"Below S1({pivots['s1']:.2f}) — bearish breakdown zone")

        # ━━━ FINAL SCORING ━━━
        if checks == 0:
            signal = Signal.NEUTRAL
            confidence = 15.0
        else:
            ratio = score / checks
            if ratio > 0.25:
                signal = Signal.BULLISH
                confidence = min(30 + ratio * 70, 95)
            elif ratio < -0.25:
                signal = Signal.BEARISH
                confidence = min(30 + abs(ratio) * 70, 95)
            else:
                signal = Signal.NEUTRAL
                confidence = 25 + abs(ratio) * 30

        # Include raw indicator values for portfolio manager
        indicator_snapshot = {
            "sma20": smas.get("sma20"),
            "sma50": smas.get("sma50"),
            "sma200": smas.get("sma200"),
            "rsi": rsi,
            "macd": macd.get("macd_line"),
            "macd_signal": macd.get("signal_line"),
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "williams_r": williams,
            "cci": cci,
            "adx": adx,
            "bb_pct_b": bb.get("pct_b") if bb else None,
            "atr": atr,
            "vwap": vwap,
            "supertrend_dir": st.get("direction"),
            "supertrend_val": st.get("value"),
            "donchian_upper": dc.get("upper") if dc else None,
            "donchian_lower": dc.get("lower") if dc else None,
            "score": round(score, 2),
            "checks": checks,
        }

        signals[ticker] = AnalystSignal(
            agent_name="technicals_analyst",
            ticker=ticker,
            signal=signal,
            confidence=round(confidence, 1),
            reasoning="; ".join(details),
        )

    return {
        "messages": [HumanMessage(content="Technical analysis complete.", name="technicals_analyst")],
        "data": {
            "technicals_signals": {t: s.model_dump() for t, s in signals.items()},
            "technicals_indicators": {t: indicator_snapshot for t in tickers if t in signals},
        },
    }
