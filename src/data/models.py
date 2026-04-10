"""Data models for ASX AI Hedge Fund."""

from __future__ import annotations


from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Signal(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class Action(str, Enum):
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"
    HOLD = "hold"


class AnalystSignal(BaseModel):
    """Output from an analyst agent."""
    agent_name: str
    ticker: str
    signal: Signal
    confidence: float = Field(ge=0, le=100, description="Confidence 0-100")
    reasoning: str


class TradeDecision(BaseModel):
    """A single trade decision from the portfolio manager."""
    ticker: str
    action: Action
    quantity: int = 0
    confidence: float = Field(ge=0, le=100)
    reasoning: str


class PortfolioDecisions(BaseModel):
    """All trade decisions for a given analysis run."""
    decisions: list[TradeDecision]


class Position(BaseModel):
    """A portfolio position."""
    ticker: str
    shares: int = 0
    avg_cost: float = 0.0
    side: str = "long"  # "long" or "short"

    @property
    def market_value(self) -> float:
        return abs(self.shares) * self.avg_cost


class Portfolio(BaseModel):
    """Portfolio state."""
    cash: float = 100_000.0  # AUD
    positions: dict[str, Position] = Field(default_factory=dict)
    margin_requirement: float = 0.5

    @property
    def total_value(self) -> float:
        position_value = sum(p.market_value for p in self.positions.values())
        return self.cash + position_value


class PriceData(BaseModel):
    """OHLCV price data for a single day."""
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class FinancialMetrics(BaseModel):
    """Key financial metrics for a company."""
    ticker: str
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    franking_pct: Optional[float] = None  # Australian-specific: franking credits
    roe: Optional[float] = None
    roa: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    free_cash_flow: Optional[float] = None
    beta: Optional[float] = None


class CompanyInfo(BaseModel):
    """Basic company information."""
    ticker: str
    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    currency: str = "AUD"
    exchange: str = "ASX"
