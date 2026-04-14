# ASX AI Hedge Fund — Agent Guidelines

## Project Structure

```
src/
├── agents/          # Investment analyst agents
│   ├── technicals   # 17 technical indicators + scoring
│   ├── timeframe    # Multi-timeframe alignment (W/D/4H)
│   ├── sentiment    # Multi-source news sentiment
│   ├── risk_manager # Portfolio risk constraints
│   └── portfolio_manager  # Final trade decisions (LLM)
├── backtesting/     # Backtest engine + walk-forward validation
├── data/            # Pydantic models and file-based caching
├── graph/           # LangGraph state machine (parallel agents)
├── llm/             # LLM provider abstraction (Ollama default)
├── tools/           # Data sources
│   ├── asx_data     # yfinance wrapper for ASX
│   ├── au_news      # News via opencli (Bloomberg, AFR, Twitter, Reddit)
│   └── tradingview_data  # TradingView real-time indicators
├── utils/           # Rich terminal display
└── main.py          # CLI entry point
skills/
└── asx-hedge-fund/  # OpenClaw skill definition
```

## Architecture

```
           ┌─→ [technicals]    ─┐
[__start__]─┼─→ [timeframe]     ─┼─→ [portfolio_manager] → [END]
           ├─→ [sentiment]     ─┤
           └─→ [risk_manager]  ─┘
```

- **Agents** receive state, perform analysis, return signals — they never fetch data directly
- **Tools** handle all external data (yfinance, TradingView, opencli)
- **Graph** orchestrates parallel agent execution via LangGraph
- **Data layer** provides file-based caching with TTL

## Data Sources

| Source | Tool | API Key? |
|--------|------|----------|
| ASX price/financials | yfinance | No |
| Technical indicators | tradingview-ta | No |
| Bloomberg news | opencli | No |
| AFR news | opencli | No |
| Twitter/X | opencli | No (browser login) |
| Reddit | opencli | No |
| Google News AU | RSS/httpx | No |

## Coding Conventions

- Python 3.11+
- Type hints on all public functions
- Pydantic models for structured data
- All currency values in AUD
- ASX tickers use `.AX` suffix (e.g., `BHP.AX`)
- Default LLM: Ollama (local, no API key needed)

## Running

```bash
# Analysis
python src/main.py --tickers BHP.AX,CBA.AX

# Backtest
python src/main.py --tickers BHP.AX --backtest --start-date 2024-01-01 --end-date 2025-01-01

# Walk-forward validation
python src/main.py --tickers BHP.AX --walk-forward --start-date 2023-01-01 --end-date 2025-01-01
```

## Security

- No API keys required for default setup (Ollama + opencli)
- No real trading execution — analysis only
- All LLM calls use structured JSON output
