"""Backtesting engine for ASX AI Hedge Fund.

Features:
  - Time-step simulation with configurable step size
  - Transaction costs (commission + slippage)
  - Performance metrics: Sharpe ratio, Calmar ratio, max drawdown, profit factor
  - Walk-forward validation for overfitting detection
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import yfinance as yf

from src.graph.graph import run_hedge_fund
from src.tools.asx_data import ensure_asx_ticker


# ──────────────────────── Configuration ────────────────────────

COMMISSION_RATE = 0.005    # 0.5% per trade
SLIPPAGE_RATE = 0.0005     # 0.05% slippage estimate
RISK_FREE_RATE = 0.04      # 4% annual (RBA cash rate proxy)


# ──────────────────────── Core Backtest ────────────────────────


def run_backtest(
    tickers: list[str],
    start_date: str,
    end_date: str,
    initial_cash: float = 100_000,
    analysts: list[str] | None = None,
    step_days: int = 30,
) -> dict:
    """Run a backtest over a date range.

    Steps through time in increments of `step_days`, running the full agent
    analysis at each step and executing recommended trades with realistic costs.
    """
    tickers = [ensure_asx_ticker(t) for t in tickers]
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    portfolio = {
        "cash": initial_cash,
        "positions": {},
        "total_value": initial_cash,
    }

    history = [{"date": start_date, "value": initial_cash}]
    total_trades = 0
    total_commission = 0.0
    winning_trades = 0
    losing_trades = 0
    gross_profit = 0.0
    gross_loss = 0.0
    current = start

    while current < end:
        step_end = min(current + timedelta(days=step_days), end)
        lookback_start = (current - timedelta(days=90)).strftime("%Y-%m-%d")

        result = run_hedge_fund(
            tickers=tickers,
            start_date=lookback_start,
            end_date=current.strftime("%Y-%m-%d"),
            portfolio=portfolio,
            analysts=analysts,
        )

        decisions = result.get("data", {}).get("decisions", [])

        for decision in decisions:
            ticker = decision.get("ticker", "")
            action = decision.get("action", "hold")
            alloc_pct = decision.get("allocation_pct", 0)

            if action == "hold" or alloc_pct <= 0:
                continue

            try:
                stock = yf.Ticker(ticker)
                price_data = stock.history(
                    start=current.strftime("%Y-%m-%d"),
                    end=step_end.strftime("%Y-%m-%d"),
                )
                if price_data.empty:
                    continue
                price = price_data["Close"].iloc[0]
            except Exception:
                continue

            # Apply slippage
            if action == "buy":
                exec_price = price * (1 + SLIPPAGE_RATE)
            else:
                exec_price = price * (1 - SLIPPAGE_RATE)

            # Convert allocation_pct to quantity
            target_value = portfolio["total_value"] * alloc_pct / 100
            quantity = int(target_value / exec_price) if exec_price > 0 else 0
            if quantity <= 0:
                continue

            cost = exec_price * quantity
            commission = cost * COMMISSION_RATE
            total_commission += commission

            if action == "buy" and portfolio["cash"] >= cost + commission:
                portfolio["cash"] -= cost + commission
                pos = portfolio["positions"].get(ticker, {"shares": 0, "avg_cost": 0})
                total_shares = pos["shares"] + quantity
                if total_shares > 0:
                    pos["avg_cost"] = (pos["shares"] * pos["avg_cost"] + cost) / total_shares
                pos["shares"] = total_shares
                portfolio["positions"][ticker] = pos
                total_trades += 1

            elif action == "sell":
                pos = portfolio["positions"].get(ticker)
                if pos and pos["shares"] > 0:
                    sell_qty = min(quantity, pos["shares"])
                    proceeds = sell_qty * exec_price - commission
                    pnl = proceeds - sell_qty * pos["avg_cost"]
                    if pnl > 0:
                        winning_trades += 1
                        gross_profit += pnl
                    else:
                        losing_trades += 1
                        gross_loss += abs(pnl)
                    portfolio["cash"] += proceeds
                    pos["shares"] -= sell_qty
                    if pos["shares"] == 0:
                        del portfolio["positions"][ticker]
                    else:
                        portfolio["positions"][ticker] = pos
                    total_trades += 1

        # Update portfolio value
        total_value = portfolio["cash"]
        for ticker, pos in portfolio["positions"].items():
            try:
                stock = yf.Ticker(ticker)
                price_data = stock.history(
                    start=step_end.strftime("%Y-%m-%d"),
                    end=(step_end + timedelta(days=5)).strftime("%Y-%m-%d"),
                )
                if not price_data.empty:
                    total_value += pos["shares"] * price_data["Close"].iloc[0]
                else:
                    total_value += pos["shares"] * pos["avg_cost"]
            except Exception:
                total_value += pos["shares"] * pos["avg_cost"]

        portfolio["total_value"] = total_value
        history.append({
            "date": step_end.strftime("%Y-%m-%d"),
            "value": total_value,
        })

        current = step_end

    # Compute metrics
    benchmark_return = _get_benchmark_return(start_date, end_date)
    final_value = portfolio["total_value"]
    total_return = ((final_value - initial_cash) / initial_cash) * 100

    values = [h["value"] for h in history]
    returns = _compute_returns(values)
    max_dd = _max_drawdown(values)
    days_total = (end - start).days
    sharpe = _sharpe_ratio(returns, days_total)
    calmar = _calmar_ratio(total_return / 100, max_dd, days_total)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0
    win_rate = winning_trades / (winning_trades + losing_trades) * 100 if (winning_trades + losing_trades) > 0 else 0

    return {
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "alpha": total_return - benchmark_return,
        "final_value": final_value,
        "initial_cash": initial_cash,
        "total_trades": total_trades,
        "total_commission": total_commission,
        "max_drawdown": max_dd * 100,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "history": history,
        "final_portfolio": portfolio,
    }


# ──────────────────────── Walk-Forward Validation ────────────────────────


def run_walk_forward(
    tickers: list[str],
    start_date: str,
    end_date: str,
    n_folds: int = 3,
    train_pct: float = 0.7,
    initial_cash: float = 100_000,
    analysts: list[str] | None = None,
    step_days: int = 30,
) -> dict:
    """Walk-forward validation to detect overfitting.

    Splits the date range into N folds. For each fold, trains on train_pct
    and validates on the remainder. A robust strategy should perform similarly
    in both periods.

    Returns:
        Dict with per-fold results and robustness score.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days
    fold_days = total_days // n_folds

    folds = []
    for i in range(n_folds):
        fold_start = start + timedelta(days=i * fold_days)
        fold_end = fold_start + timedelta(days=fold_days)
        if fold_end > end:
            fold_end = end

        train_days = int(fold_days * train_pct)
        train_end = fold_start + timedelta(days=train_days)
        test_start = train_end

        train_result = run_backtest(
            tickers=tickers,
            start_date=fold_start.strftime("%Y-%m-%d"),
            end_date=train_end.strftime("%Y-%m-%d"),
            initial_cash=initial_cash,
            analysts=analysts,
            step_days=step_days,
        )

        test_result = run_backtest(
            tickers=tickers,
            start_date=test_start.strftime("%Y-%m-%d"),
            end_date=fold_end.strftime("%Y-%m-%d"),
            initial_cash=initial_cash,
            analysts=analysts,
            step_days=step_days,
        )

        folds.append({
            "fold": i + 1,
            "train_period": f"{fold_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}",
            "test_period": f"{test_start.strftime('%Y-%m-%d')} to {fold_end.strftime('%Y-%m-%d')}",
            "train_return": train_result["total_return"],
            "test_return": test_result["total_return"],
            "train_sharpe": train_result["sharpe_ratio"],
            "test_sharpe": test_result["sharpe_ratio"],
        })

    # Robustness score: avg(test_return / train_return) across folds
    ratios = []
    for f in folds:
        if f["train_return"] != 0:
            ratios.append(f["test_return"] / f["train_return"])
        else:
            ratios.append(1.0 if f["test_return"] >= 0 else 0.0)

    avg_ratio = sum(ratios) / len(ratios) if ratios else 0
    if avg_ratio >= 0.8:
        verdict = "ROBUST"
    elif avg_ratio >= 0.5:
        verdict = "MODERATE"
    elif avg_ratio >= 0.2:
        verdict = "WEAK"
    else:
        verdict = "OVERFITTED"

    return {
        "folds": folds,
        "robustness_score": round(avg_ratio, 3),
        "verdict": verdict,
    }


# ──────────────────────── Metrics ────────────────────────


def _compute_returns(values: list[float]) -> list[float]:
    """Compute period-over-period returns from value series."""
    if len(values) < 2:
        return []
    return [(values[i] - values[i - 1]) / values[i - 1] for i in range(1, len(values))]


def _max_drawdown(values: list[float]) -> float:
    """Maximum peak-to-trough drawdown as a fraction (0-1)."""
    if len(values) < 2:
        return 0.0
    peak = values[0]
    max_dd = 0.0
    for v in values[1:]:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _sharpe_ratio(returns: list[float], total_days: int) -> float:
    """Annualized Sharpe ratio.

    Uses step returns (not daily), annualizes based on actual period.
    """
    if len(returns) < 2:
        return 0.0
    n = len(returns)
    mean_r = sum(returns) / n
    var = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
    std_r = math.sqrt(var) if var > 0 else 0

    if std_r == 0:
        return 0.0

    # Annualization: scale by sqrt(periods_per_year)
    periods_per_year = 365 / (total_days / n) if total_days > 0 else 12
    rf_per_period = RISK_FREE_RATE / periods_per_year

    sharpe = (mean_r - rf_per_period) / std_r * math.sqrt(periods_per_year)
    return round(sharpe, 3)


def _calmar_ratio(total_return_frac: float, max_dd: float, total_days: int) -> float:
    """Calmar ratio — annualized return / max drawdown."""
    if max_dd == 0 or total_days == 0:
        return 0.0
    annualized = (1 + total_return_frac) ** (365 / total_days) - 1
    return round(annualized / max_dd, 3)


def _get_benchmark_return(start_date: str, end_date: str) -> float:
    """Get ASX 200 benchmark return over the period."""
    try:
        benchmark = yf.Ticker("IOZ.AX")
        data = benchmark.history(start=start_date, end=end_date)
        if data.empty or len(data) < 2:
            return 0.0
        start_price = data["Close"].iloc[0]
        end_price = data["Close"].iloc[-1]
        return ((end_price - start_price) / start_price) * 100
    except Exception:
        return 0.0
