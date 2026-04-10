"""Sentiment analyst agent — multi-source Australian news with time-decay weighting.

Data fetched via opencli (https://github.com/jackwener/opencli) — NO API keys needed.
News is filtered to the last 30 days and weighted by recency:
  - 0~3 days old  → weight × 1.0  (full weight)
  - 4~7 days old  → weight × 0.8
  - 8~14 days old → weight × 0.5
  - 15~21 days old → weight × 0.3
  - 22~30 days old → weight × 0.15

Source credibility weights:
  Bloomberg        → 1.0  (global institutional financial news)
  AFR              → 1.0  (Australian institutional financial journalism)
  The Australian   → 0.9  (mainstream financial news)
  Google News AU   → 0.7  (aggregated, mixed quality)
  Twitter/X        → 0.4  (real-time but noisy social sentiment)
  Reddit           → 0.3  (retail noise, useful as contrarian)
"""

import json

from langchain_core.messages import HumanMessage

from src.data.models import AnalystSignal, Signal
from src.graph.state import AgentState
from src.llm.models import get_llm
from src.tools.asx_data import get_company_info
from src.tools.au_news import fetch_all_au_news


# ──────────────────── Weighting ────────────────────

SOURCE_CREDIBILITY = {
    "Bloomberg": 1.0,
    "AFR": 1.0,
    "The Australian": 0.9,
    "Google News AU": 0.7,
    "Google News": 0.7,
    "Twitter/X": 0.4,
    "Reddit r/ASX_Bets": 0.3,
    "Reddit r/AusFinance": 0.35,
    "Reddit r/AusStocks": 0.3,
}


def _source_weight(source_name: str) -> float:
    """Look up credibility weight by source name (fuzzy match)."""
    for key, w in SOURCE_CREDIBILITY.items():
        if key.lower() in source_name.lower():
            return w
    return 0.5


def _time_decay(days_ago: float | None) -> float:
    """Return a decay multiplier based on how old the article is.

    Uses a piecewise linear schedule so the LLM prompt can explain it:
      0~3 days   → 1.0
      4~7 days   → 0.8
      8~14 days  → 0.5
      15~21 days → 0.3
      22~30 days → 0.15
      >30 days   → 0.0  (should already be filtered out)
    """
    if days_ago is None:
        return 0.5  # Unknown date → treat as mid-weight
    if days_ago <= 3:
        return 1.0
    if days_ago <= 7:
        return 0.8
    if days_ago <= 14:
        return 0.5
    if days_ago <= 21:
        return 0.3
    if days_ago <= 30:
        return 0.15
    return 0.0


def _combined_weight(source: str, days_ago: float | None) -> float:
    """Final weight = source_credibility × time_decay."""
    return _source_weight(source) * _time_decay(days_ago)


# ──────────────────── Prompt ────────────────────

SENTIMENT_PROMPT = """You are a senior financial sentiment analyst specialising in the Australian Securities Exchange (ASX).

Analyse the following news about **{ticker}** ({company_name}) collected from Australian sources over the **last 30 days**.

## How to read the articles

Each article has:
- **Source** and **credibility tier** (Tier 1 = most trustworthy)
- **Age** (days ago) — more recent = more relevant
- **Weight** — pre-computed combined score (source credibility × time decay, 0–1)

Give proportionally MORE attention to articles with higher weight.

## Source Credibility Tiers
- **Tier 1 (High trust):** Bloomberg, AFR, The Australian — institutional financial journalism
- **Tier 2 (Medium trust):** Google News AU — aggregated, mixed quality
- **Tier 3 (Low trust / contrarian):** Twitter/X, Reddit — social/retail sentiment.
  Extreme one-sided sentiment on r/ASX_Bets or Twitter is often a CONTRARIAN indicator.

## Articles (sorted by weight, highest first):

{articles_text}

## Analysis Requirements

1. Articles with weight > 0.5 should drive your conclusion.
2. Articles with weight 0.2–0.5 provide context.
3. Articles with weight < 0.2 are background noise — only notable if they show extreme unanimity.
4. Look for **consensus vs divergence** across tiers.
5. Identify concrete **catalysts** (earnings, RBA, M&A, commodity prices, regulation, China).
6. Factor in **Australian macro**: AUD/USD, RBA cash rate, iron ore / coal / LNG prices, housing.
7. If almost all sources agree → higher confidence. If they diverge → lower confidence.

Respond with EXACTLY this JSON (no other text):
{{
    "signal": "bullish" or "bearish" or "neutral",
    "confidence": <number 10-95>,
    "reasoning": "<2-3 sentences covering the key findings and which sources drove the conclusion>",
    "source_breakdown": {{
        "tier1_institutional": "bullish/bearish/neutral",
        "tier2_aggregated": "bullish/bearish/neutral",
        "tier3_retail": "bullish/bearish/neutral"
    }},
    "key_catalysts": ["<catalyst 1>", "<catalyst 2>"]
}}
"""


# ──────────────────── Formatting ────────────────────

def _format_article(article: dict) -> str:
    """Format a single article with its weight for the LLM."""
    source = article.get("source", "Unknown")
    title = article.get("title", "").strip()
    summary = article.get("summary", "").strip()
    days = article.get("days_ago")
    weight = _combined_weight(source, days)

    age_str = f"{days:.0f}d ago" if days is not None else "date unknown"
    line = f"- **[w={weight:.2f}]** [{source}] ({age_str}) {title}"
    if summary:
        line += f"\n  _{summary[:150]}_"
    return line


def _format_all_articles(all_news: dict[str, list[dict]]) -> str:
    """Merge all sources, compute weights, sort by weight descending."""
    articles = []

    for source_key, items in all_news.items():
        for item in items:
            item["_weight"] = _combined_weight(item.get("source", ""), item.get("days_ago"))
            articles.append(item)

    # Sort by weight descending
    articles.sort(key=lambda x: x.get("_weight", 0), reverse=True)

    if not articles:
        return "No articles found from any source.\n"

    lines = []
    for a in articles:
        formatted = _format_article(a)
        if formatted:
            lines.append(formatted)

    return "\n".join(lines)


# ──────────────────── Agent ────────────────────

def sentiment_agent(state: AgentState) -> dict:
    """Multi-source, time-weighted sentiment analysis for ASX stocks.

    No API keys needed — all data fetched via opencli.
    """
    tickers = state["metadata"]["tickers"]
    signals = {}
    llm = get_llm()

    for ticker in tickers:
        # Get company name for better search
        company_info = get_company_info(ticker)
        company_name = company_info.name if company_info else ticker.replace(".AX", "")

        # ── Fetch from all sources via opencli ──
        au_news = fetch_all_au_news(
            ticker=ticker,
            company_name=company_name,
            max_per_source=10,
        )

        # ── Coverage stats ──
        coverage = {
            "Bloomberg": len(au_news.get("bloomberg", [])),
            "AFR": len(au_news.get("afr", [])),
            "The Australian": len(au_news.get("the_australian", [])),
            "Google News AU": len(au_news.get("google_news_au", [])),
            "Twitter/X": len(au_news.get("twitter", [])),
            "Reddit": len(au_news.get("reddit", [])),
        }
        total = sum(coverage.values())
        coverage_str = ", ".join(f"{k}:{v}" for k, v in coverage.items() if v > 0)

        if total == 0:
            signals[ticker] = AnalystSignal(
                agent_name="sentiment_analyst",
                ticker=ticker,
                signal=Signal.NEUTRAL,
                confidence=10,
                reasoning="No news found in the last 30 days from any source.",
            )
            continue

        # ── Build weighted article list ──
        articles_text = _format_all_articles(au_news)

        prompt = SENTIMENT_PROMPT.format(
            ticker=ticker,
            company_name=company_name,
            articles_text=articles_text,
        )

        try:
            response = llm.invoke(prompt)
            content = response.content.strip()

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

            # ── Confidence adjustment based on coverage breadth ──
            active_sources = sum(1 for v in coverage.values() if v > 0)
            if active_sources >= 5:
                conf_multiplier = 1.15
            elif active_sources >= 3:
                conf_multiplier = 1.0
            elif active_sources == 2:
                conf_multiplier = 0.85
            else:
                conf_multiplier = 0.7

            adjusted_confidence = min(raw_confidence * conf_multiplier, 95)

            # ── Build reasoning ──
            reasoning = result.get("reasoning", "")
            reasoning += f" [Sources({active_sources}): {coverage_str}]"

            catalysts = result.get("key_catalysts", [])
            if catalysts:
                reasoning += f" [Catalysts: {', '.join(catalysts[:3])}]"

            breakdown = result.get("source_breakdown", {})
            t1 = breakdown.get("tier1_institutional", "n/a")
            t3 = breakdown.get("tier3_retail", "n/a")
            if t1 != "n/a" and t3 != "n/a" and t1 != t3:
                reasoning += f" [Divergence: institutional={t1}, retail={t3}]"

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
                reasoning=f"Analysis error: {str(e)[:100]}. Sources: {coverage_str}",
            )

    return {
        "messages": [HumanMessage(
            content="Multi-source time-weighted sentiment analysis complete.",
            name="sentiment_analyst",
        )],
        "data": {"sentiment_signals": {t: s.model_dump() for t, s in signals.items()}},
    }
