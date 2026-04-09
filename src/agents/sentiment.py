"""Sentiment analyst agent — multi-source Australian news sentiment.

Data Sources (weighted by credibility):
  1. AFR (Australian Financial Review) — weight 1.0 (institutional, high quality)
  2. The Australian                    — weight 0.9 (mainstream financial)
  3. Google News AU                    — weight 0.7 (aggregated, mixed quality)
  4. yfinance news                     — weight 0.6 (global, may lack AU context)
  5. Twitter/X                         — weight 0.4 (noisy but real-time sentiment)
  6. Reddit (r/ASX_Bets, r/AusFinance) — weight 0.3 (retail sentiment, contrarian indicator)
"""

import json

from langchain_core.messages import HumanMessage

from src.data.models import AnalystSignal, Signal
from src.graph.state import AgentState
from src.llm.models import get_llm
from src.tools.asx_data import get_company_info, get_news as get_yfinance_news
from src.tools.au_news import fetch_all_au_news


# Source credibility weights — higher = more trusted
SOURCE_WEIGHTS = {
    "AFR": 1.0,
    "The Australian": 0.9,
    "Google News AU": 0.7,
    "yfinance": 0.6,
    "Twitter/X": 0.4,
    "Reddit r/ASX_Bets": 0.3,
    "Reddit r/AusFinance": 0.35,
    "Reddit r/AusStocks": 0.3,
}

# Get weight for a source, defaulting to 0.5
def _get_weight(source: str) -> float:
    for key, weight in SOURCE_WEIGHTS.items():
        if key.lower() in source.lower():
            return weight
    return 0.5


SENTIMENT_PROMPT = """You are a senior financial sentiment analyst specializing in the Australian Securities Exchange (ASX).

You must analyze news and social media about {ticker} ({company_name}) from MULTIPLE Australian sources, considering source credibility.

## Source Credibility Tiers
- **Tier 1 (High trust):** AFR, The Australian — institutional journalism, fact-checked
- **Tier 2 (Medium trust):** Google News AU, yfinance — aggregated, mixed quality
- **Tier 3 (Low trust / contrarian):** Twitter/X, Reddit — retail sentiment, noisy but useful for gauging crowd psychology. Extreme bullishness on Reddit r/ASX_Bets can be a contrarian BEARISH indicator.

## News from each source:

{sources_text}

## Analysis Requirements

1. **Weigh sources by credibility** — a single AFR article is worth more than 10 Reddit posts
2. **Look for consensus vs divergence** — if institutions are bearish but retail is bullish, that's a warning
3. **Identify catalysts** — earnings, regulatory, M&A, commodity prices, RBA decisions
4. **Consider recency** — more recent news carries more weight
5. **Detect sentiment extremes** — overwhelming one-sided sentiment often precedes reversals
6. **Australian context** — RBA rate decisions, AUD movements, China trade relations, commodity cycles

Respond with EXACTLY this JSON format:
{{
    "signal": "bullish" or "bearish" or "neutral",
    "confidence": <number 10-95>,
    "reasoning": "<2-3 sentence explanation covering key findings>",
    "source_breakdown": {{
        "institutional": "bullish/bearish/neutral",
        "mainstream": "bullish/bearish/neutral",
        "social": "bullish/bearish/neutral"
    }},
    "key_catalysts": ["<catalyst 1>", "<catalyst 2>"]
}}
"""


def _format_source_news(source_name: str, articles: list[dict], max_items: int = 8) -> str:
    """Format news articles from a single source."""
    if not articles:
        return f"### {source_name}\nNo articles found.\n"

    lines = [f"### {source_name}"]
    for item in articles[:max_items]:
        title = item.get("title", "").strip()
        if not title:
            continue

        summary = item.get("summary", "").strip()
        source_tag = item.get("source", source_name)
        engagement = item.get("engagement")

        line = f"- [{source_tag}] {title}"
        if summary:
            line += f"\n  Summary: {summary[:150]}"
        if engagement is not None:
            line += f" (engagement: {engagement})"
        lines.append(line)

    return "\n".join(lines) + "\n"


def sentiment_agent(state: AgentState) -> dict:
    """Multi-source sentiment analysis for ASX stocks."""
    tickers = state["metadata"]["tickers"]
    signals = {}
    llm = get_llm()

    for ticker in tickers:
        # Get company name for better search
        company_info = get_company_info(ticker)
        company_name = company_info.name if company_info else ticker.replace(".AX", "")

        # ── Fetch from all Australian sources ──
        au_news = fetch_all_au_news(
            ticker=ticker,
            company_name=company_name,
            max_per_source=8,
        )

        # Also get yfinance news as fallback
        yf_news = get_yfinance_news(ticker)

        # ── Count total articles ──
        total_articles = sum(len(v) for v in au_news.values()) + len(yf_news)

        if total_articles == 0:
            signals[ticker] = AnalystSignal(
                agent_name="sentiment_analyst",
                ticker=ticker,
                signal=Signal.NEUTRAL,
                confidence=10,
                reasoning="No news found from any Australian source.",
            )
            continue

        # ── Format all sources for LLM ──
        sources_text = ""
        sources_text += _format_source_news("AFR (Australian Financial Review)", au_news.get("afr", []))
        sources_text += _format_source_news("The Australian", au_news.get("the_australian", []))
        sources_text += _format_source_news("Google News AU", au_news.get("google_news_au", []))
        sources_text += _format_source_news("yfinance (Global)", yf_news)
        sources_text += _format_source_news("Twitter/X (Social)", au_news.get("twitter", []))
        sources_text += _format_source_news("Reddit (Retail Investors)", au_news.get("reddit", []))

        # ── Build source coverage summary ──
        coverage = {
            "AFR": len(au_news.get("afr", [])),
            "The Australian": len(au_news.get("the_australian", [])),
            "Google News": len(au_news.get("google_news_au", [])),
            "yfinance": len(yf_news),
            "Twitter": len(au_news.get("twitter", [])),
            "Reddit": len(au_news.get("reddit", [])),
        }
        coverage_str = ", ".join(f"{k}: {v}" for k, v in coverage.items() if v > 0)

        prompt = SENTIMENT_PROMPT.format(
            ticker=ticker,
            company_name=company_name,
            sources_text=sources_text,
        )

        try:
            response = llm.invoke(prompt)
            content = response.content.strip()

            # Parse JSON response
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            result = json.loads(content)

            signal_map = {
                "bullish": Signal.BULLISH,
                "bearish": Signal.BEARISH,
                "neutral": Signal.NEUTRAL,
            }

            raw_signal = signal_map.get(result.get("signal", "neutral"), Signal.NEUTRAL)
            raw_confidence = min(max(float(result.get("confidence", 30)), 10), 95)

            # ── Adjust confidence based on source coverage ──
            # More sources = higher confidence
            active_sources = sum(1 for v in coverage.values() if v > 0)
            if active_sources >= 4:
                confidence_boost = 1.1  # 10% boost for 4+ sources
            elif active_sources >= 2:
                confidence_boost = 1.0
            elif active_sources == 1:
                confidence_boost = 0.8  # Penalize single-source analysis
            else:
                confidence_boost = 0.5

            adjusted_confidence = min(raw_confidence * confidence_boost, 95)

            # ── Build reasoning ──
            source_breakdown = result.get("source_breakdown", {})
            catalysts = result.get("key_catalysts", [])

            reasoning = result.get("reasoning", "")
            reasoning += f" [Sources: {coverage_str}]"
            if catalysts:
                reasoning += f" [Catalysts: {', '.join(catalysts[:3])}]"
            if source_breakdown:
                inst = source_breakdown.get("institutional", "n/a")
                social = source_breakdown.get("social", "n/a")
                if inst != social:
                    reasoning += f" [Divergence: institutional={inst}, social={social}]"

            signals[ticker] = AnalystSignal(
                agent_name="sentiment_analyst",
                ticker=ticker,
                signal=raw_signal,
                confidence=round(adjusted_confidence, 1),
                reasoning=reasoning,
            )

        except Exception as e:
            signals[ticker] = AnalystSignal(
                agent_name="sentiment_analyst",
                ticker=ticker,
                signal=Signal.NEUTRAL,
                confidence=10,
                reasoning=f"Sentiment analysis error: {str(e)[:100]}. Sources checked: {coverage_str}",
            )

    return {
        "messages": [HumanMessage(content="Multi-source sentiment analysis complete.", name="sentiment_analyst")],
        "data": {"sentiment_signals": {t: s.model_dump() for t, s in signals.items()}},
    }
