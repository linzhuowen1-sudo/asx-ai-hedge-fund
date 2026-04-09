"""ASX market data tools using yfinance and other sources."""

import json
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from src.data.cache import get_cache, set_cache
from src.data.models import CompanyInfo, FinancialMetrics, PriceData


def ensure_asx_ticker(ticker: str) -> str:
    """Ensure ticker has .AX suffix for ASX stocks."""
    if not ticker.upper().endswith(".AX"):
        return f"{ticker.upper()}.AX"
    return ticker.upper()


def get_price_history(
    ticker: str,
    start_date: str,
    end_date: str,
) -> list[PriceData]:
    """Fetch historical OHLCV data for an ASX stock.

    Args:
        ticker: ASX ticker (e.g., "BHP" or "BHP.AX")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        List of PriceData objects sorted by date
    """
    ticker = ensure_asx_ticker(ticker)
    cache_key = f"prices:{ticker}:{start_date}:{end_date}"

    cached = get_cache(cache_key)
    if cached:
        return [PriceData(**p) for p in cached]

    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)

    if df.empty:
        return []

    prices = []
    for date, row in df.iterrows():
        prices.append(PriceData(
            date=date.to_pydatetime(),
            open=round(row["Open"], 4),
            high=round(row["High"], 4),
            low=round(row["Low"], 4),
            close=round(row["Close"], 4),
            volume=int(row["Volume"]),
        ))

    set_cache(cache_key, [p.model_dump(mode="json") for p in prices])
    return prices


def get_financial_metrics(ticker: str) -> Optional[FinancialMetrics]:
    """Fetch key financial metrics for an ASX stock."""
    ticker = ensure_asx_ticker(ticker)
    cache_key = f"metrics:{ticker}"

    cached = get_cache(cache_key)
    if cached:
        return FinancialMetrics(**cached)

    stock = yf.Ticker(ticker)
    info = stock.info

    if not info or "symbol" not in info:
        return None

    metrics = FinancialMetrics(
        ticker=ticker,
        market_cap=info.get("marketCap"),
        pe_ratio=info.get("trailingPE"),
        pb_ratio=info.get("priceToBook"),
        ps_ratio=info.get("priceToSalesTrailing12Months"),
        dividend_yield=info.get("dividendYield"),
        roe=info.get("returnOnEquity"),
        roa=info.get("returnOnAssets"),
        debt_to_equity=info.get("debtToEquity"),
        current_ratio=info.get("currentRatio"),
        gross_margin=info.get("grossMargins"),
        operating_margin=info.get("operatingMargins"),
        net_margin=info.get("profitMargins"),
        revenue_growth=info.get("revenueGrowth"),
        earnings_growth=info.get("earningsGrowth"),
        free_cash_flow=info.get("freeCashflow"),
        beta=info.get("beta"),
    )

    set_cache(cache_key, metrics.model_dump())
    return metrics


def get_company_info(ticker: str) -> Optional[CompanyInfo]:
    """Fetch basic company information."""
    ticker = ensure_asx_ticker(ticker)
    cache_key = f"company:{ticker}"

    cached = get_cache(cache_key)
    if cached:
        return CompanyInfo(**cached)

    stock = yf.Ticker(ticker)
    info = stock.info

    if not info or "symbol" not in info:
        return None

    company = CompanyInfo(
        ticker=ticker,
        name=info.get("longName", info.get("shortName", ticker)),
        sector=info.get("sector"),
        industry=info.get("industry"),
        market_cap=info.get("marketCap"),
        currency=info.get("currency", "AUD"),
        exchange="ASX",
    )

    set_cache(cache_key, company.model_dump())
    return company


def get_income_statement(ticker: str) -> Optional[pd.DataFrame]:
    """Fetch income statement data."""
    ticker = ensure_asx_ticker(ticker)
    stock = yf.Ticker(ticker)
    return stock.financials if stock.financials is not None and not stock.financials.empty else None


def get_balance_sheet(ticker: str) -> Optional[pd.DataFrame]:
    """Fetch balance sheet data."""
    ticker = ensure_asx_ticker(ticker)
    stock = yf.Ticker(ticker)
    return stock.balance_sheet if stock.balance_sheet is not None and not stock.balance_sheet.empty else None


def get_cash_flow(ticker: str) -> Optional[pd.DataFrame]:
    """Fetch cash flow statement data."""
    ticker = ensure_asx_ticker(ticker)
    stock = yf.Ticker(ticker)
    return stock.cashflow if stock.cashflow is not None and not stock.cashflow.empty else None


def get_news(ticker: str) -> list[dict]:
    """Fetch recent news for a stock."""
    ticker = ensure_asx_ticker(ticker)
    cache_key = f"news:{ticker}"

    cached = get_cache(cache_key, ttl=1800)  # 30 min cache for news
    if cached:
        return cached

    stock = yf.Ticker(ticker)
    news = stock.news or []

    # Normalize news format
    normalized = []
    for item in news[:20]:
        content = item.get("content", {})
        normalized.append({
            "title": content.get("title", ""),
            "summary": content.get("summary", ""),
            "published": content.get("pubDate", ""),
            "source": content.get("provider", {}).get("displayName", ""),
            "url": content.get("canonicalUrl", {}).get("url", ""),
        })

    set_cache(cache_key, normalized)
    return normalized


def get_asx200_tickers() -> list[str]:
    """Return a curated list of major ASX 200 component tickers."""
    return [
        "BHP.AX",   # BHP Group
        "CBA.AX",   # Commonwealth Bank
        "CSL.AX",   # CSL Limited
        "NAB.AX",   # National Australia Bank
        "WBC.AX",   # Westpac Banking
        "ANZ.AX",   # ANZ Banking Group
        "WES.AX",   # Wesfarmers
        "MQG.AX",   # Macquarie Group
        "GMG.AX",   # Goodman Group
        "FMG.AX",   # Fortescue Metals
        "WOW.AX",   # Woolworths Group
        "RIO.AX",   # Rio Tinto
        "TLS.AX",   # Telstra
        "ALL.AX",   # Aristocrat Leisure
        "REA.AX",   # REA Group
        "WDS.AX",   # Woodside Energy
        "STO.AX",   # Santos
        "JHX.AX",   # James Hardie
        "QBE.AX",   # QBE Insurance
        "SHL.AX",   # Sonic Healthcare
        "COL.AX",   # Coles Group
        "TCL.AX",   # Transurban
        "NCM.AX",   # Newcrest Mining
        "MIN.AX",   # Mineral Resources
        "XRO.AX",   # Xero
        "CPU.AX",   # Computershare
        "ORG.AX",   # Origin Energy
        "AGL.AX",   # AGL Energy
        "IAG.AX",   # Insurance Australia Group
        "SUN.AX",   # Suncorp Group
    ]
