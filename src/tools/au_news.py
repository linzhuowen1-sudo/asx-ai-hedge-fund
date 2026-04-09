"""Australian news sources for sentiment analysis.

Sources:
  1. Australian Financial Review (AFR) — via web scraping
  2. The Australian — via web scraping
  3. Twitter/X — via API (v2)
  4. Google News AU — via RSS
  5. ASX Announcements — via yfinance fallback
  6. Reddit (r/ASX_Bets, r/AusFinance) — via API
"""

import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

from src.data.cache import get_cache, set_cache


# ──────────────────────── AFR (Australian Financial Review) ────────────────────────


def fetch_afr_news(
    query: str,
    max_results: int = 10,
) -> list[dict]:
    """Scrape AFR search results for a company/ticker.

    Uses AFR's public search page. No API key required.
    Respects rate limits with caching (1 hour TTL).
    """
    cache_key = f"afr:{query}:{max_results}"
    cached = get_cache(cache_key, ttl=3600)
    if cached:
        return cached

    url = f"https://www.afr.com/search?text={quote_plus(query)}&sortBy=relevance"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-AU,en;q=0.9",
    }

    results = []
    try:
        with httpx.Client(timeout=15, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code != 200:
                return []

            soup = BeautifulSoup(resp.text, "html.parser")

            # AFR search results are typically in article cards
            articles = soup.select("article, [data-testid*='search-result'], .search-result, h3 a")
            for article in articles[:max_results]:
                title_el = article.select_one("h2, h3, .headline, a")
                summary_el = article.select_one("p, .summary, .standfirst")
                time_el = article.select_one("time, .timestamp")

                title = title_el.get_text(strip=True) if title_el else ""
                if not title:
                    title = article.get_text(strip=True)[:120]
                if not title:
                    continue

                link = ""
                link_el = article.select_one("a[href]") or (title_el if title_el and title_el.name == "a" else None)
                if link_el and link_el.get("href"):
                    href = link_el["href"]
                    link = href if href.startswith("http") else f"https://www.afr.com{href}"

                results.append({
                    "title": title,
                    "summary": summary_el.get_text(strip=True) if summary_el else "",
                    "source": "AFR",
                    "url": link,
                    "published": time_el.get("datetime", "") if time_el else "",
                })

    except Exception as e:
        # Silently fail — other sources will compensate
        pass

    set_cache(cache_key, results)
    return results


# ──────────────────────── The Australian ────────────────────────


def fetch_theaustralian_news(
    query: str,
    max_results: int = 10,
) -> list[dict]:
    """Scrape The Australian search results.

    Uses public search page. Paywalled content shows headlines only.
    """
    cache_key = f"theaustralian:{query}:{max_results}"
    cached = get_cache(cache_key, ttl=3600)
    if cached:
        return cached

    url = f"https://www.theaustralian.com.au/search-results?q={quote_plus(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-AU,en;q=0.9",
    }

    results = []
    try:
        with httpx.Client(timeout=15, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code != 200:
                return []

            soup = BeautifulSoup(resp.text, "html.parser")

            articles = soup.select("article, .search-result, [data-testid*='result'], .story-block")
            for article in articles[:max_results]:
                title_el = article.select_one("h3, h2, .headline, a.story-block__heading")
                summary_el = article.select_one("p, .summary, .standfirst")
                time_el = article.select_one("time, .timestamp, .date")

                title = title_el.get_text(strip=True) if title_el else ""
                if not title:
                    title = article.get_text(strip=True)[:120]
                if not title:
                    continue

                link = ""
                link_el = article.select_one("a[href]")
                if link_el and link_el.get("href"):
                    href = link_el["href"]
                    link = href if href.startswith("http") else f"https://www.theaustralian.com.au{href}"

                results.append({
                    "title": title,
                    "summary": summary_el.get_text(strip=True) if summary_el else "",
                    "source": "The Australian",
                    "url": link,
                    "published": time_el.get("datetime", "") if time_el else "",
                })

    except Exception:
        pass

    set_cache(cache_key, results)
    return results


# ──────────────────────── Google News AU (RSS) ────────────────────────


def fetch_google_news_au(
    query: str,
    max_results: int = 10,
) -> list[dict]:
    """Fetch Google News RSS for Australian stock news.

    No API key required. Uses RSS feed.
    """
    cache_key = f"googlenews_au:{query}:{max_results}"
    cached = get_cache(cache_key, ttl=1800)
    if cached:
        return cached

    # Google News RSS — restrict to Australian sources
    url = f"https://news.google.com/rss/search?q={quote_plus(query + ' ASX')}&hl=en-AU&gl=AU&ceid=AU:en"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ASXHedgeFund/1.0)",
    }

    results = []
    try:
        with httpx.Client(timeout=15, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code != 200:
                return []

            soup = BeautifulSoup(resp.text, "xml")
            items = soup.find_all("item")

            for item in items[:max_results]:
                title = item.find("title")
                link = item.find("link")
                pub_date = item.find("pubDate")
                source = item.find("source")

                results.append({
                    "title": title.get_text(strip=True) if title else "",
                    "summary": "",
                    "source": source.get_text(strip=True) if source else "Google News AU",
                    "url": link.get_text(strip=True) if link else "",
                    "published": pub_date.get_text(strip=True) if pub_date else "",
                })

    except Exception:
        pass

    set_cache(cache_key, results)
    return results


# ──────────────────────── Twitter/X API v2 ────────────────────────


def fetch_twitter_sentiment(
    query: str,
    max_results: int = 20,
) -> list[dict]:
    """Fetch recent tweets about an ASX stock via Twitter API v2.

    Requires TWITTER_BEARER_TOKEN in environment.
    Returns tweet text with engagement metrics.
    """
    bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        return []

    cache_key = f"twitter:{query}:{max_results}"
    cached = get_cache(cache_key, ttl=900)  # 15 min cache
    if cached:
        return cached

    # Search for stock-related tweets — filter out spam
    search_query = f'"{query}" (ASX OR stock OR shares OR $) -is:retweet lang:en'
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {
        "query": search_query,
        "max_results": min(max_results, 100),
        "tweet.fields": "created_at,public_metrics,author_id,text",
        "sort_order": "relevancy",
    }
    headers = {
        "Authorization": f"Bearer {bearer_token}",
    }

    results = []
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(url, params=params, headers=headers)
            if resp.status_code != 200:
                return []

            data = resp.json()
            tweets = data.get("data", [])

            for tweet in tweets:
                metrics = tweet.get("public_metrics", {})
                engagement = (
                    metrics.get("like_count", 0)
                    + metrics.get("retweet_count", 0) * 2
                    + metrics.get("reply_count", 0)
                )
                results.append({
                    "title": tweet.get("text", "")[:200],
                    "summary": "",
                    "source": "Twitter/X",
                    "url": f"https://twitter.com/i/web/status/{tweet.get('id', '')}",
                    "published": tweet.get("created_at", ""),
                    "engagement": engagement,
                })

            # Sort by engagement — higher engagement = more influential
            results.sort(key=lambda x: x.get("engagement", 0), reverse=True)

    except Exception:
        pass

    set_cache(cache_key, results)
    return results


# ──────────────────────── Reddit (r/ASX_Bets, r/AusFinance) ────────────────────────


def fetch_reddit_sentiment(
    query: str,
    max_results: int = 10,
) -> list[dict]:
    """Fetch Reddit posts about an ASX stock from Australian finance subs.

    Uses Reddit's public JSON API (no auth required for read).
    """
    cache_key = f"reddit:{query}:{max_results}"
    cached = get_cache(cache_key, ttl=1800)
    if cached:
        return cached

    subreddits = ["ASX_Bets", "AusFinance", "AusStocks"]
    headers = {
        "User-Agent": "ASXHedgeFund/1.0 (research bot)",
    }

    results = []
    for sub in subreddits:
        url = f"https://www.reddit.com/r/{sub}/search.json"
        params = {
            "q": query,
            "restrict_sr": "on",
            "sort": "relevance",
            "t": "week",
            "limit": max_results,
        }

        try:
            with httpx.Client(timeout=10, follow_redirects=True) as client:
                resp = client.get(url, params=params, headers=headers)
                if resp.status_code != 200:
                    continue

                data = resp.json()
                posts = data.get("data", {}).get("children", [])

                for post in posts:
                    pd = post.get("data", {})
                    results.append({
                        "title": pd.get("title", ""),
                        "summary": pd.get("selftext", "")[:300],
                        "source": f"Reddit r/{sub}",
                        "url": f"https://reddit.com{pd.get('permalink', '')}",
                        "published": datetime.fromtimestamp(pd.get("created_utc", 0)).isoformat() if pd.get("created_utc") else "",
                        "engagement": pd.get("score", 0) + pd.get("num_comments", 0),
                    })

        except Exception:
            continue

    # Sort by engagement
    results.sort(key=lambda x: x.get("engagement", 0), reverse=True)

    set_cache(cache_key, results[:max_results])
    return results[:max_results]


# ──────────────────────── Aggregator ────────────────────────


def fetch_all_au_news(
    ticker: str,
    company_name: Optional[str] = None,
    max_per_source: int = 8,
) -> dict[str, list[dict]]:
    """Aggregate news from all Australian sources.

    Returns a dict keyed by source name for weighted analysis.
    """
    # Clean ticker for search (remove .AX suffix)
    search_term = ticker.replace(".AX", "").replace(".ax", "")
    if company_name:
        search_term = f"{company_name} {search_term}"

    return {
        "afr": fetch_afr_news(search_term, max_per_source),
        "the_australian": fetch_theaustralian_news(search_term, max_per_source),
        "google_news_au": fetch_google_news_au(search_term, max_per_source),
        "twitter": fetch_twitter_sentiment(search_term, max_per_source * 2),
        "reddit": fetch_reddit_sentiment(search_term, max_per_source),
    }
