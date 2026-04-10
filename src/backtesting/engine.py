"""Backtesting engine for ASX AI Hedge Fund."""

from __future__ import annotations


from datetime import datetime, timedelta

import yfinance as yf

from src.graph.graph import run_hedge_fund
from src.tools.asx_data import ensure_asx_ticker


def run_backtest(
    tickers: list[str],
    start_date: str,
    end_date: str,
    initial_cash: float = 100_000,
    analysts: list[str] | None = None,
    step_days: int = 30,
) -> dict:
    """Run a backtest over a date range.

    The engine steps through time in increments of `step_days`, running
    the full agent analysis at each step and executing the recommended trades.

    Args:
        tickers: List of ASX tickers
        start_date: Backtest start (YYYY-MM-DD)
        end_date: Backtest end (YYYY-MM-DD)
        initial_cash: Starting cash in AUD
        analysts: Which analysts to include
        step_days: Days between each analysis step

    Returns:
        Dict of backtest metrics
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
    current = start

    while current < end:
        step_end = min(current + timedelta(days=step_days), end)
        lookback_start = (current - timedelta(days=90)).strftime("%Y-%m-%d")

        # Run analysis
        result = run_hedge_fund(
            tickers=tickers,
            start_date=lookback_start,
            end_date=current.strftime("%Y-%m-%d"),
            portfolio=portfolio,
            analysts=analysts,
        )

        decisions = result.get("data", {}).get("decisions", [])

        # Execute trades at current prices
        for decision in decisions:
            ticker = decision.get("ticker", "")
            action = decision.get("action", "hold")
            quantity = decision.get("quantity", 0)

            if action == "hold" or quantity == 0:
                continue

            # Get current price
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

            cost = price * quantity

            if action == "buy" and portfolio["cash"] >= cost:
                portfolio["cash"] -= cost
                pos = portfolio["positions"].get(ticker, {"shares": 0, "avg_cost": 0})
                total_shares = pos["shares"] + quantity
                if total_shares > 0:
                    pos["avg_cost"] = (
                        (pos["shares"] * pos["avg_cost"] + cost) / total_shares
                    )
                pos["shares"] = total_shares
                portfolio["positions"][ticker] = pos
                total_trades += 1

            elif action == "sell":
                pos = portfolio["positions"].get(ticker)
                if pos and pos["shares"] > 0:
                    sell_qty = min(quantity, pos["shares"])
                    portfolio["cash"] += sell_qty * price
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

    # Compute benchmark return (ASX 200 via IOZ.AX ETF)
    benchmark_return = _get_benchmark_return(start_date, end_date)

    final_value = portfolio["total_value"]
    total_return = ((final_value - initial_cash) / initial_cash) * 100

    return {
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "final_value": final_value,
        "initial_cash": initial_cash,
        "total_trades": total_trades,
        "history": history,
        "final_portfolio": portfolio,
    }


def _get_benchmark_return(start_date: str, end_date: str) -> float:
    """Get ASX 200 benchmark return over the period."""
    try:
        benchmark = yf.Ticker("IOZ.AX")  # iShares ASX 200 ETF
        data = benchmark.history(start=start_date, end=end_date)
        if data.empty or len(data) < 2:
            return 0.0
        start_price = data["Close"].iloc[0]
        end_price = data["Close"].iloc[-1]
        return ((end_price - start_price) / start_price) * 100
    except Exception:
        return 0.0
