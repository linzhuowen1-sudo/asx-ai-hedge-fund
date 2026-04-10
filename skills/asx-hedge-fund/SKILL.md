---
name: asx-hedge-fund
description: AI hedge fund analyst for the Australian Securities Exchange (ASX). Analyzes ASX stocks using fundamental, valuation, technical, and sentiment analysis with multiple AI agents. Sentiment powered by opencli for Bloomberg, AFR, The Australian, Twitter/X, Reddit. Generates trade signals (buy/sell/hold) with confidence scores.
user-invocable: true
metadata: {"openclaw": {"emoji": "📈", "os": ["darwin", "linux"], "requires": {"bins": ["python3", "opencli"], "env": ["OPENAI_API_KEY"]}, "primaryEnv": "OPENAI_API_KEY"}}
---

# ASX AI Hedge Fund Skill

You can analyze Australian Securities Exchange (ASX) stocks using a multi-agent AI system.

## Prerequisites

- **opencli** — required for sentiment analysis (news fetching from Bloomberg, AFR, Twitter, etc.)
  ```bash
  npm install -g @jackwener/opencli
  opencli doctor  # Verify installation
  ```
  Load the Browser Bridge Chrome extension from [GitHub Releases](https://github.com/jackwener/opencli/releases) for AFR, The Australian, Bloomberg access.

- **LLM API key** — OpenAI, Anthropic, or other supported provider

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
- `fundamentals` — Profitability, growth, financial health scoring (14 metrics)
- `valuation` — P/E, P/B, P/S, dividend yield relative to ASX sector benchmarks
- `technicals` — 14 indicators: SMA/EMA, MACD, RSI, Bollinger Bands, Stochastic, ADX, Williams %R, CCI, OBV, VWAP, ATR, Fibonacci, Pivot Points
- `sentiment` — Multi-source news sentiment (30-day window, time-decay weighted):
  - **Tier 1:** Bloomberg, AFR, The Australian (via `opencli browser`)
  - **Tier 2:** Google News AU (via RSS)
  - **Tier 3:** Twitter/X, Reddit (via `opencli twitter/reddit`)

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

## Sentiment Data Flow

```
opencli twitter search → Twitter/X posts (native adapter, no API key)
opencli reddit search  → Reddit posts (native adapter, no API key)
opencli browser open   → Bloomberg / AFR / The Australian (uses Chrome session)
Google News RSS        → Aggregated AU news (always available)
         ↓ all filtered to last 30 days
         ↓ weighted: source_credibility × time_decay
         ↓ sorted by combined weight
         ↓ fed to LLM for sentiment analysis
         → BULLISH / BEARISH / NEUTRAL + confidence
```

## Notes
- All currency values are in AUD
- Benchmark is S&P/ASX 200 (IOZ.AX ETF)
- Price data via yfinance — tickers need `.AX` suffix
- Analysis is for research only, not financial advice
- Bloomberg/AFR access requires being logged in via Chrome (opencli reuses your session)
