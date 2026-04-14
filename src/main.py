"""CLI entry point for ASX AI Hedge Fund."""

import argparse
import sys
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="ASX AI Hedge Fund — AI-powered stock analysis for the Australian Securities Exchange"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        required=True,
        help="Comma-separated ASX tickers (e.g., BHP.AX,CBA.AX,CSL.AX)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
        help="Analysis start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Analysis end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--analysts",
        type=str,
        default=None,
        help="Comma-separated analyst names: technicals,sentiment,timeframe",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=100_000,
        help="Starting cash in AUD (default: 100000)",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run in backtest mode",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run walk-forward validation (overfitting detection)",
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )

    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",")]
    analysts = [a.strip() for a in args.analysts.split(",")] if args.analysts else None

    if args.walk_forward:
        from src.backtesting.engine import run_walk_forward
        from src.utils.display import display_walk_forward_results

        wf = run_walk_forward(
            tickers=tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_cash=args.cash,
            analysts=analysts,
        )
        display_walk_forward_results(wf)
    elif args.backtest:
        from src.backtesting.engine import run_backtest
        from src.utils.display import display_backtest_results

        metrics = run_backtest(
            tickers=tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_cash=args.cash,
            analysts=analysts,
        )
        display_backtest_results(metrics)
    else:
        from src.graph.graph import run_hedge_fund
        from src.utils.display import display_analysis_results

        portfolio = {"cash": args.cash, "positions": {}, "total_value": args.cash}
        result = run_hedge_fund(
            tickers=tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            portfolio=portfolio,
            analysts=analysts,
        )

        if args.output == "json":
            import json
            decisions = result.get("data", {}).get("decisions", [])
            print(json.dumps(decisions, indent=2))
        else:
            display_analysis_results(result)


if __name__ == "__main__":
    main()
