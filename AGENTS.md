# ASX AI Hedge Fund — Agent Guidelines

## Project Structure

```
src/
├── agents/          # Investment analyst agents
├── backtesting/     # Historical simulation engine
├── data/            # Data models and caching
├── graph/           # LangGraph state machine
├── llm/             # LLM provider abstraction
├── tools/           # ASX data fetching tools
├── utils/           # Display and helper utilities
├── main.py          # CLI entry point
└── backtester.py    # Backtesting runner
skills/
└── asx-hedge-fund/  # OpenClaw skill definition
scripts/             # Helper scripts for OpenClaw integration
```

## Architecture Boundaries

- **Agents** receive state, perform analysis, return signals — they never fetch data directly
- **Tools** handle all external API calls (yfinance, Alpha Vantage, etc.)
- **Graph** orchestrates agent execution order via LangGraph
- **Data layer** provides caching to avoid redundant API calls

## Coding Conventions

- Python 3.11+
- Type hints on all public functions
- Pydantic models for structured data
- All currency values in AUD unless explicitly stated
- ASX tickers use `.AX` suffix for yfinance (e.g., `BHP.AX`, `CBA.AX`)

## Testing

```bash
poetry run pytest tests/
```

## Running

```bash
# CLI mode
poetry run python src/main.py --tickers BHP.AX,CBA.AX,CSL.AX

# Via OpenClaw
# The skill in skills/asx-hedge-fund/ handles integration
```

## Security

- API keys stored in `.env`, never committed
- No real trading execution
- All LLM calls use structured output to prevent injection
