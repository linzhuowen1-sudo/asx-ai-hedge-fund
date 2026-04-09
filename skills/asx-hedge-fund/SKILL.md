---
name: asx-hedge-fund
description: AI hedge fund analyst for the Australian Securities Exchange (ASX). Analyzes ASX stocks using fundamental, valuation, technical, and sentiment analysis with multiple AI agents. Generates trade signals (buy/sell/hold) with confidence scores.
user-invocable: true
metadata: {"openclaw": {"emoji": "📈", "os": ["darwin", "linux"], "requires": {"bins": ["python3"], "env": ["OPENAI_API_KEY"]}, "primaryEnv": "OPENAI_API_KEY"}}
---

# ASX AI Hedge Fund Skill

You can analyze Australian Securities Exchange (ASX) stocks using a multi-agent AI system.

## Usage

When the user asks about ASX stocks, Australian shares, or wants stock analysis for Australian companies, use this skill.

### Analyze stocks
Run analysis on one or more ASX tickers:
```bash
cd {baseDir}/../../ && python3 -m src.main --tickers BHP.AX,CBA.AX,CSL.AX --output json
```

### Backtest a strategy
Run historical backtesting:
```bash
cd {baseDir}/../../ && python3 -m src.main --tickers BHP.AX,CBA.AX --start-date 2024-01-01 --end-date 2024-12-31 --backtest
```

### Available analysts
- `fundamentals` — Profitability, growth, financial health scoring
- `valuation` — P/E, P/B, P/S relative to ASX sector benchmarks
- `technicals` — SMA, RSI, MACD, volume analysis
- `sentiment` — News sentiment via LLM analysis

### Select specific analysts
```bash
cd {baseDir}/../../ && python3 -m src.main --tickers BHP.AX --analysts fundamentals,technicals --output json
```

## Common ASX Tickers
- **Banks**: CBA.AX, NAB.AX, WBC.AX, ANZ.AX, MQG.AX
- **Mining**: BHP.AX, RIO.AX, FMG.AX, MIN.AX, NCM.AX
- **Healthcare**: CSL.AX, SHL.AX, COH.AX
- **Tech**: XRO.AX, REA.AX, CPU.AX
- **Consumer**: WOW.AX, WES.AX, COL.AX
- **Energy**: WDS.AX, STO.AX, ORG.AX, AGL.AX

## Notes
- All currency values are in AUD
- Benchmark is S&P/ASX 200 (IOZ.AX ETF)
- The system uses yfinance for data — tickers need `.AX` suffix
- Analysis is for research only, not financial advice
- An LLM API key (OpenAI, Anthropic, etc.) is required for sentiment analysis and portfolio decisions
