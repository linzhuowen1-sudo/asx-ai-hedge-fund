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
        ("fundamentals_signals", "Fundamentals"),
        ("valuation_signals", "Valuation"),
        ("technicals_signals", "Technicals"),
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
    console.print(Panel.fit(
        f"[bold]Backtest Results[/bold]\n\n"
        f"  Total Return:    [{'green' if metrics.get('total_return', 0) >= 0 else 'red'}]"
        f"{metrics.get('total_return', 0):.2f}%[/]\n"
        f"  Benchmark (ASX200): {metrics.get('benchmark_return', 0):.2f}%\n"
        f"  Alpha:           {metrics.get('total_return', 0) - metrics.get('benchmark_return', 0):.2f}%\n"
        f"  Total Trades:    {metrics.get('total_trades', 0)}\n"
        f"  Final Value:     ${metrics.get('final_value', 0):,.2f} AUD",
        title="ASX AI Hedge Fund Backtest",
        border_style="blue",
    ))
