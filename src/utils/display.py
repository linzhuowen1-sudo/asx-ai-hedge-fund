"""Display utilities for terminal output."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def display_analysis_results(result: dict) -> None:
    """Display analysis results in a formatted table."""
    data = result.get("data", {})
    decisions = data.get("decisions", [])

    if not decisions:
        console.print("[yellow]No trade decisions generated.[/yellow]")
        return

    # Decisions table
    table = Table(title="ASX AI Hedge Fund — Trade Decisions", show_lines=True)
    table.add_column("Ticker", style="bold cyan")
    table.add_column("Action", style="bold")
    table.add_column("Qty", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Reasoning")

    action_colors = {
        "buy": "green",
        "sell": "red",
        "short": "red",
        "cover": "green",
        "hold": "yellow",
    }

    for d in decisions:
        action = d.get("action", "hold")
        color = action_colors.get(action, "white")
        table.add_row(
            d.get("ticker", ""),
            f"[{color}]{action.upper()}[/{color}]",
            str(d.get("quantity", 0)),
            f"{d.get('confidence', 0):.0f}%",
            d.get("reasoning", "")[:80],
        )

    console.print(table)

    # Signal summary
    signal_keys = [
        ("technicals_signals", "Technicals"),
        ("timeframe_signals", "Multi-Timeframe"),
        ("sentiment_signals", "Sentiment"),
        ("risk_signals", "Risk"),
    ]

    for key, name in signal_keys:
        signals = data.get(key, {})
        if signals:
            sig_table = Table(title=f"{name} Signals", show_lines=True)
            sig_table.add_column("Ticker", style="cyan")
            sig_table.add_column("Signal")
            sig_table.add_column("Confidence", justify="right")
            sig_table.add_column("Reasoning")

            signal_colors = {"bullish": "green", "bearish": "red", "neutral": "yellow"}

            for ticker, s in signals.items():
                sig = s.get("signal", "neutral")
                color = signal_colors.get(sig, "white")
                sig_table.add_row(
                    ticker,
                    f"[{color}]{sig.upper()}[/{color}]",
                    f"{s.get('confidence', 0):.0f}%",
                    s.get("reasoning", "")[:60],
                )

            console.print(sig_table)


def display_backtest_results(metrics: dict) -> None:
    """Display backtesting results."""
    total_return = metrics.get("total_return", 0)
    benchmark = metrics.get("benchmark_return", 0)
    alpha = metrics.get("alpha", total_return - benchmark)
    sharpe = metrics.get("sharpe_ratio", 0)
    calmar = metrics.get("calmar_ratio", 0)
    max_dd = metrics.get("max_drawdown", 0)
    pf = metrics.get("profit_factor", 0)
    win_rate = metrics.get("win_rate", 0)
    commission = metrics.get("total_commission", 0)

    ret_color = "green" if total_return >= 0 else "red"
    alpha_color = "green" if alpha >= 0 else "red"

    console.print(Panel.fit(
        f"[bold]Backtest Results[/bold]\n\n"
        f"  Total Return:      [{ret_color}]{total_return:.2f}%[/]\n"
        f"  Benchmark (ASX200): {benchmark:.2f}%\n"
        f"  Alpha:             [{alpha_color}]{alpha:.2f}%[/]\n"
        f"  Sharpe Ratio:      {sharpe:.3f}\n"
        f"  Calmar Ratio:      {calmar:.3f}\n"
        f"  Max Drawdown:      [red]{max_dd:.2f}%[/]\n"
        f"  Profit Factor:     {pf:.2f}\n"
        f"  Win Rate:          {win_rate:.1f}%\n"
        f"  Total Trades:      {metrics.get('total_trades', 0)}\n"
        f"  Total Commission:  ${commission:,.2f}\n"
        f"  Final Value:       ${metrics.get('final_value', 0):,.2f} AUD",
        title="ASX AI Hedge Fund Backtest",
        border_style="blue",
    ))


def display_walk_forward_results(wf: dict) -> None:
    """Display walk-forward validation results."""
    table = Table(title="Walk-Forward Validation", show_lines=True)
    table.add_column("Fold", style="bold")
    table.add_column("Train Period")
    table.add_column("Train Return", justify="right")
    table.add_column("Test Period")
    table.add_column("Test Return", justify="right")

    for f in wf.get("folds", []):
        tr = f["train_return"]
        te = f["test_return"]
        table.add_row(
            str(f["fold"]),
            f["train_period"],
            f"[{'green' if tr >= 0 else 'red'}]{tr:.2f}%[/]",
            f["test_period"],
            f"[{'green' if te >= 0 else 'red'}]{te:.2f}%[/]",
        )

    console.print(table)

    verdict = wf.get("verdict", "UNKNOWN")
    score = wf.get("robustness_score", 0)
    verdict_colors = {
        "ROBUST": "green", "MODERATE": "yellow",
        "WEAK": "red", "OVERFITTED": "bold red",
    }
    color = verdict_colors.get(verdict, "white")

    console.print(Panel.fit(
        f"  Robustness Score: {score:.3f}\n"
        f"  Verdict:          [{color}]{verdict}[/{color}]",
        title="Walk-Forward Summary",
        border_style="blue",
    ))
