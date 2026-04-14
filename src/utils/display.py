"""Display utilities — optimized for mobile chat readability."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(width=50)  # Mobile-friendly width

_SIG_ICON = {"bullish": "+", "bearish": "-", "neutral": "~"}
_ACT_ICON = {"buy": "+", "sell": "-", "short": "-", "cover": "+", "hold": "~"}


def display_analysis_results(result: dict) -> None:
    """Display analysis results optimized for mobile chat."""
    data = result.get("data", {})
    decisions = data.get("decisions", [])
    trade_plans = data.get("trade_plans", {})
    company_briefs = data.get("company_briefs", {})

    if not decisions:
        console.print("[yellow]No trade decisions.[/yellow]")
        return

    for d in decisions:
        ticker = d.get("ticker", "")
        action = d.get("action", "hold")
        alloc = d.get("allocation_pct", 0)
        conf = d.get("confidence", 0)
        reasoning = d.get("reasoning", "")

        act_color = {"buy": "green", "sell": "red", "short": "red", "cover": "green", "hold": "yellow"}.get(action, "white")
        icon = _ACT_ICON.get(action, "")

        # ── Header: Company + Decision ──
        lines = []

        # Company intro
        brief = company_briefs.get(ticker, ticker)
        lines.append(f"[dim]{brief}[/dim]")
        lines.append("")

        # Decision
        lines.append(f"[{act_color} bold]{icon} {action.upper()}  {alloc}%[/{act_color} bold]  conf {conf:.0f}%")
        lines.append(f"[dim]{reasoning[:120]}[/dim]")

        # ── Trade Plan ──
        plan = trade_plans.get(ticker)
        if plan and plan.get("current_price"):
            price = plan["current_price"]
            buy_z = plan.get("buy_zone", {})
            sell_z = plan.get("sell_zone", {})
            stop = plan.get("stop_loss")
            short_t = plan.get("short_term_targets", [])
            mid_t = plan.get("mid_term_targets", [])
            long_t = plan.get("long_term_targets", [])
            rr = plan.get("risk_reward")
            validity = plan.get("validity_label", "N/A")

            lines.append("")
            lines.append(f"Price  ${price:.2f}")

            if buy_z.get("low") and buy_z.get("high"):
                lines.append(f"[green]Buy    ${buy_z['low']:.2f} - ${buy_z['high']:.2f}[/green]")

            if sell_z.get("low") and sell_z.get("high"):
                lines.append(f"[red]Sell   ${sell_z['low']:.2f} - ${sell_z['high']:.2f}[/red]")

            if stop:
                risk_pct = abs(price - stop) / price * 100
                lines.append(f"[bold red]Stop   ${stop:.2f}[/bold red] (-{risk_pct:.1f}%)")

            # Targets
            def _fmt_targets(targets, label):
                if not targets:
                    return
                lines.append(f"\n[bold]{label}[/bold]")
                for t in targets:
                    pct = abs(t["price"] - price) / price * 100
                    sign = "+" if t["price"] > price else "-"
                    lines.append(f"  ${t['price']:.2f}  {sign}{pct:.1f}%  {t['label']}")

            _fmt_targets(short_t, "Short  1-5d")
            _fmt_targets(mid_t, "Mid  1-4w")
            _fmt_targets(long_t, "Long  1-3m")

            # Footer
            lines.append("")
            parts = []
            if rr is not None:
                parts.append(f"R/R {rr:.2f}")
            parts.append(validity)
            lines.append("[dim]" + "  |  ".join(parts) + "[/dim]")

        console.print(Panel(
            "\n".join(lines),
            title=f"[bold]{ticker}[/bold]",
            border_style=act_color,
            width=50,
        ))

    # ── Signals Summary (compact) ──
    signal_keys = [
        ("technicals_signals", "Tech"),
        ("timeframe_signals", "TF"),
        ("sentiment_signals", "Sent"),
        ("risk_signals", "Risk"),
    ]

    for ticker in [d.get("ticker") for d in decisions]:
        sig_parts = []
        for key, label in signal_keys:
            signals = data.get(key, {})
            if ticker in signals:
                s = signals[ticker]
                sig = s.get("signal", "neutral")
                conf = s.get("confidence", 0)
                icon = _SIG_ICON.get(sig, "~")
                sig_parts.append(f"{icon}{label} {conf:.0f}%")

        if sig_parts:
            console.print(f"[dim]{ticker}: {' | '.join(sig_parts)}[/dim]")

    console.print()


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

    console.print(Panel(
        f"Return    [{ret_color}]{total_return:.2f}%[/]\n"
        f"ASX200    {benchmark:.2f}%\n"
        f"Alpha     [{alpha_color}]{alpha:.2f}%[/]\n"
        f"Sharpe    {sharpe:.3f}\n"
        f"Calmar    {calmar:.3f}\n"
        f"Max DD    [red]{max_dd:.2f}%[/]\n"
        f"PF        {pf:.2f}\n"
        f"Win Rate  {win_rate:.1f}%\n"
        f"Trades    {metrics.get('total_trades', 0)}\n"
        f"Fees      ${commission:,.2f}\n"
        f"Final     ${metrics.get('final_value', 0):,.2f}",
        title="[bold]Backtest[/bold]",
        border_style="blue",
        width=40,
    ))


def display_walk_forward_results(wf: dict) -> None:
    """Display walk-forward validation results."""
    lines = []
    for f in wf.get("folds", []):
        tr = f["train_return"]
        te = f["test_return"]
        tr_c = "green" if tr >= 0 else "red"
        te_c = "green" if te >= 0 else "red"
        lines.append(
            f"Fold {f['fold']}: "
            f"Train [{tr_c}]{tr:.1f}%[/] "
            f"Test [{te_c}]{te:.1f}%[/]"
        )

    verdict = wf.get("verdict", "UNKNOWN")
    score = wf.get("robustness_score", 0)
    v_color = {"ROBUST": "green", "MODERATE": "yellow", "WEAK": "red", "OVERFITTED": "bold red"}.get(verdict, "white")

    lines.append(f"\nScore  {score:.3f}")
    lines.append(f"[{v_color}]{verdict}[/{v_color}]")

    console.print(Panel(
        "\n".join(lines),
        title="[bold]Walk-Forward[/bold]",
        border_style="blue",
        width=40,
    ))
