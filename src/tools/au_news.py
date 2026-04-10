"""Australian news sources for sentiment analysis — pure web scraping, no API keys needed.

All sources use httpx + BeautifulSoup to crawl public pages.
Results are filtered to the last 30 days and tagged with publish dates
for time-decay weighting in the sentiment agent.

Sources:
  1. AFR (Australian Financial Review) — institutional financial journalism
  2. The Australian — mainstream financial news
  3. Google News AU — RSS aggregator, Australian edition
  4. ABC News Australia — public broadcaster, free and reliable
  5. MarketIndex.com.au — ASX-focused market news
  6. Reddit (r/ASX_Bets, r/AusFinance) — retail sentiment (public JSON, no auth)
"""

import re
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

from src.data.cache import get_cache, set_cache


# ──────────────────────── Shared ────────────────────────

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-AU,en;q=0.9",
}

# All news must be within this window
NEWS_WINDOW_DAYS = 30


def _parse_date(date_str: str) -> Optional[datetime]:
    """Best-effort parse of various date formats into a datetime."""
    if not date_str:
        return None
    date_str = date_str.strip()

    formats = [
        "%Y-%m-%dT%H:%M:%S%z",       # ISO 8601 with tz
        "%Y-%m-%dT%H:%M:%SZ",        # ISO 8601 UTC
        "%Y-%m-%dT%H:%M:%S.%f%z",    # ISO 8601 with microseconds
        "%Y-%m-%dT%H:%M:%S",         # ISO 8601 naive
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d %b %Y",                  # 10 Apr 2026
        "%d %B %Y",                  # 10 April 2026
        "%B %d, %Y",                 # April 10, 2026
        "%b %d, %Y",                 # Apr 10, 2026
        "%a, %d %b %Y %H:%M:%S %z",  # RFC 2822 (RSS)
        "%a, %d %b %Y %H:%M:%S %Z",  # RFC 2822 variant
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    # Try extracting just a date pattern
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", date_str)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


def _is_within_window(date_str: str, days: int = NEWS_WINDOW_DAYS) -> bool:
    """Check if a date string is within the last N days."""
    dt = _parse_date(date_str)
    if dt is None:
        return True  # Keep articles with unparseable dates (let LLM judge)
    # Make both naive for comparison
    now = datetime.now()
    if dt.tzinfo is not None:
        now = datetime.now(timezone.utc)
    return (now - dt).days <= days


def _days_ago(date_str: str) -> Optional[float]:
    """Return how many days ago a date was, or None if unparseable."""
    dt = _parse_date(date_str)
    if dt is None:
        return None
    now = datetime.now()
    if dt.tzinfo is not None:
        now = datetime.now(timezone.utc)
    delta = (now - dt).total_seconds() / 86400
    return max(delta, 0)


def _get(url: str, headers: dict | None = None, timeout: int = 15) -> Optional[str]:
    """HTTP GET with error handling. Returns response text or None."""
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url, headers=headers or _HEADERS)
            if resp.status_code == 200:
                return resp.text
    except Exception:
        pass
    return None


# ──────────────────────── AFR (Australian Financial Review) ────────────────────────


def fetch_afr_news(query: str, max_results: int = 10) -> list[dict]:
    """Scrape AFR search results. No API key needed."""
    cache_key = f"afr_v2:{query}:{max_results}"
    cached = get_cache(cache_key, ttl=3600)
    if cached:
        return cached

    url = f"https://www.afr.com/search?text={quote_plus(query)}&sortBy=date"
    html = _get(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    results = []

    for article in soup.select("article, [data-testid*='search-result'], .search-result, .story-block"):
        title_el = article.select_one("h2, h3, .headline, a")
        summary_el = article.select_one("p, .summary, .standfirst")
        time_el = article.select_one("time, .timestamp, [datetime]")

        title = title_el.get_text(strip=True) if title_el else article.get_text(strip=True)[:120]
        if not title or len(title) < 10:
            continue

        published = ""
        if time_el:
            published = time_el.get("datetime", "") or time_el.get_text(strip=True)

        if not _is_within_window(published):
            continue

        link = ""
        link_el = article.select_one("a[href]")
        if link_el and link_el.get("href"):
            href = link_el["href"]
            link = href if href.startswith("http") else f"https://www.afr.com{href}"

        results.append({
            "title": title,
            "summary": summary_el.get_text(strip=True)[:200] if summary_el else "",
            "source": "AFR",
            "url": link,
            "published": published,
            "days_ago": _days_ago(published),
        })

        if len(results) >= max_results:
            break

    set_cache(cache_key, results)
    return results


# ──────────────────────── The Australian ────────────────────────


def fetch_theaustralian_news(query: str, max_results: int = 10) -> list[dict]:
    """Scrape The Australian search results. No API key needed."""
    cache_key = f"theaustralian_v2:{query}:{max_results}"
    cached = get_cache(cache_key, ttl=3600)
    if cached:
        return cached

    url = f"https://www.theaustralian.com.au/search-results?q={quote_plus(query)}"
    html = _get(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    results = []

    for article in soup.select("article, .search-result, .story-block, [data-testid*='result']"):
        title_el = article.select_one("h3, h2, .headline, a.story-block__heading")
        summary_el = article.select_one("p, .summary, .standfirst")
        time_el = article.select_one("time, .timestamp, .date, [datetime]")

        title = title_el.get_text(strip=True) if title_el else article.get_text(strip=True)[:120]
        if not title or len(title) < 10:
            continue

        published = ""
        if time_el:
            published = time_el.get("datetime", "") or time_el.get_text(strip=True)

        if not _is_within_window(published):
            continue

        link = ""
        link_el = article.select_one("a[href]")
        if link_el and link_el.get("href"):
            href = link_el["href"]
            link = href if href.startswith("http") else f"https://www.theaustralian.com.au{href}"

        results.append({
            "title": title,
            "summary": summary_el.get_text(strip=True)[:200] if summary_el else "",
            "source": "The Australian",
            "url": link,
            "published": published,
            "days_ago": _days_ago(published),
        })

        if len(results) >= max_results:
            break

    set_cache(cache_key, results)
    return results


# ──────────────────────── Google News AU (RSS) ────────────────────────


def fetch_google_news_au(query: str, max_results: int = 15) -> list[dict]:
    """Google News AU RSS feed. No API key needed, excellent coverage."""
    cache_key = f"googlenews_v2:{query}:{max_results}"
    cached = get_cache(cache_key, ttl=1800)
    if cached:
        return cached

    # when= parameter limits to recent results
    url = (
        f"https://news.google.com/rss/search"
        f"?q={quote_plus(query + ' ASX')}&hl=en-AU&gl=AU&ceid=AU:en"
    )
    html = _get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ASXHedgeFund/1.0)"})
    if not html:
        return []

    soup = BeautifulSoup(html, "xml")
    results = []

    for item in soup.find_all("item"):
        title = item.find("title")
        link = item.find("link")
        pub_date = item.find("pubDate")
        source_el = item.find("source")

        published = pub_date.get_text(strip=True) if pub_date else ""
        if not _is_within_window(published):
            continue

        results.append({
            "title": title.get_text(strip=True) if title else "",
            "summary": "",
            "source": source_el.get_text(strip=True) if source_el else "Google News AU",
            "url": link.get_text(strip=True) if link else "",
            "published": published,
            "days_ago": _days_ago(published),
        })

        if len(results) >= max_results:
            break

    set_cache(cache_key, results)
    return results


# ──────────────────────── ABC News Australia ────────────────────────


def fetch_abc_news(query: str, max_results: int = 10) -> list[dict]:
    """Scrape ABC News (Australian Broadcasting Corporation) search results.

    Free, publicly funded, no paywall. Reliable source.
    """
    cache_key = f"abc_v2:{query}:{max_results}"
    cached = get_cache(cache_key, ttl=3600)
    if cached:
        return cached

    url = f"https://search-beta.abc.net.au/index.html?siteTitle=news&query={quote_plus(query)}"
    html = _get(url)
    if not html:
        # Fallback: try RSS
        return _fetch_abc_rss(query, max_results)

    soup = BeautifulSoup(html, "html.parser")
    results = []

    for article in soup.select("article, .search-result, [data-component='CardResult'], .doctype-article"):
        title_el = article.select_one("h3, h2, .title, a")
        summary_el = article.select_one("p, .synopsis, .description")
        time_el = article.select_one("time, .timestamp, [datetime]")

        title = title_el.get_text(strip=True) if title_el else ""
        if not title or len(title) < 10:
            continue

        published = ""
        if time_el:
            published = time_el.get("datetime", "") or time_el.get_text(strip=True)

        if not _is_within_window(published):
            continue

        link = ""
        link_el = article.select_one("a[href]")
        if link_el and link_el.get("href"):
            href = link_el["href"]
            link = href if href.startswith("http") else f"https://www.abc.net.au{href}"

        results.append({
            "title": title,
            "summary": summary_el.get_text(strip=True)[:200] if summary_el else "",
            "source": "ABC News",
            "url": link,
            "published": published,
            "days_ago": _days_ago(published),
        })

        if len(results) >= max_results:
            break

    if not results:
        results = _fetch_abc_rss(query, max_results)

    set_cache(cache_key, results)
    return results


def _fetch_abc_rss(query: str, max_results: int = 10) -> list[dict]:
    """Fallback: ABC News RSS for business/finance."""
    url = "https://www.abc.net.au/news/feed/2942460/rss.xml"  # ABC Business RSS
    html = _get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ASXHedgeFund/1.0)"})
    if not html:
        return []

    soup = BeautifulSoup(html, "xml")
    results = []
    query_lower = query.lower()

    for item in soup.find_all("item"):
        title = item.find("title")
        title_text = title.get_text(strip=True) if title else ""
        description = item.find("description")
        desc_text = description.get_text(strip=True) if description else ""

        # Filter by relevance
        if query_lower not in title_text.lower() and query_lower not in desc_text.lower():
            continue

        pub_date = item.find("pubDate")
        published = pub_date.get_text(strip=True) if pub_date else ""
        if not _is_within_window(published):
            continue

        link = item.find("link")
        results.append({
            "title": title_text,
            "summary": desc_text[:200],
            "source": "ABC News",
            "url": link.get_text(strip=True) if link else "",
            "published": published,
            "days_ago": _days_ago(published),
        })

        if len(results) >= max_results:
            break

    return results


# ──────────────────────── MarketIndex.com.au ────────────────────────


def fetch_marketindex_news(ticker: str, max_results: int = 10) -> list[dict]:
    """Scrape MarketIndex.com.au — ASX-specific market news and analysis.

    Ticker-specific pages have direct news feeds.
    """
    cache_key = f"marketindex_v2:{ticker}:{max_results}"
    cached = get_cache(cache_key, ttl=3600)
    if cached:
        return cached

    # MarketIndex uses plain ticker without .AX
    clean_ticker = ticker.replace(".AX", "").replace(".ax", "").lower()
    url = f"https://www.marketindex.com.au/asx/{clean_ticker}/news"
    html = _get(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    results = []

    for article in soup.select("article, .news-item, .media-body, tr, .card"):
        title_el = article.select_one("h3, h4, h2, a, .title, .headline")
        time_el = article.select_one("time, .date, .timestamp, small, [datetime]")

        title = title_el.get_text(strip=True) if title_el else ""
        if not title or len(title) < 10:
            continue

        published = ""
        if time_el:
            published = time_el.get("datetime", "") or time_el.get_text(strip=True)

        if not _is_within_window(published):
            continue

        link = ""
        link_el = article.select_one("a[href]")
        if link_el and link_el.get("href"):
            href = link_el["href"]
            link = href if href.startswith("http") else f"https://www.marketindex.com.au{href}"

        results.append({
            "title": title,
            "summary": "",
            "source": "MarketIndex",
            "url": link,
            "published": published,
            "days_ago": _days_ago(published),
        })

        if len(results) >= max_results:
            break

    set_cache(cache_key, results)
    return results


# ──────────────────────── Reddit (public JSON, no auth) ────────────────────────


def fetch_reddit_posts(query: str, max_results: int = 10) -> list[dict]:
    """Fetch Reddit posts from Australian finance subs via public JSON API.

    No API key or OAuth needed — Reddit's .json endpoints are public.
    """
    cache_key = f"reddit_v2:{query}:{max_results}"
    cached = get_cache(cache_key, ttl=1800)
    if cached:
        return cached

    subreddits = ["ASX_Bets", "AusFinance", "AusStocks"]
    headers = {"User-Agent": "ASXHedgeFund/1.0 (research bot)"}
    results = []

    for sub in subreddits:
        url = f"https://www.reddit.com/r/{sub}/search.json"
        params = {
            "q": query,
            "restrict_sr": "on",
            "sort": "relevance",
            "t": "month",  # Last month
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
                    created = pd.get("created_utc", 0)
                    published = datetime.fromtimestamp(created).isoformat() if created else ""

                    if not _is_within_window(published):
                        continue

                    results.append({
                        "title": pd.get("title", ""),
                        "summary": pd.get("selftext", "")[:300],
                        "source": f"Reddit r/{sub}",
                        "url": f"https://reddit.com{pd.get('permalink', '')}",
                        "published": published,
                        "days_ago": _days_ago(published),
                        "engagement": pd.get("score", 0) + pd.get("num_comments", 0),
                    })

        except Exception:
            continue

    results.sort(key=lambda x: x.get("engagement", 0), reverse=True)
    set_cache(cache_key, results[:max_results])
    return results[:max_results]


# ──────────────────────── Aggregator ────────────────────────


def fetch_all_au_news(
    ticker: str,
    company_name: Optional[str] = None,
    max_per_source: int = 10,
) -> dict[str, list[dict]]:
    """Aggregate news from all Australian sources. No API keys required.

    All results are filtered to the last 30 days and include `days_ago`
    for time-decay weighting.
    """
    search_term = ticker.replace(".AX", "").replace(".ax", "")
    if company_name:
        search_term = f"{company_name} {search_term}"

    return {
        "afr": fetch_afr_news(search_term, max_per_source),
        "the_australian": fetch_theaustralian_news(search_term, max_per_source),
        "google_news_au": fetch_google_news_au(search_term, max_per_source),
        "abc_news": fetch_abc_news(search_term, max_per_source),
        "marketindex": fetch_marketindex_news(ticker, max_per_source),
        "reddit": fetch_reddit_posts(search_term, max_per_source),
    }
