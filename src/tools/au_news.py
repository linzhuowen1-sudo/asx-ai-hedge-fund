"""Australian news sources for sentiment analysis — powered by OpenCLI.

Uses `opencli` (https://github.com/jackwener/opencli) to fetch data from:
  1. AFR (Australian Financial Review) — via opencli browser
  2. The Australian                    — via opencli browser
  3. Bloomberg                         — via opencli browser
  4. Twitter/X                         — via opencli twitter (native adapter)
  5. Reddit (r/ASX_Bets, r/AusFinance) — via opencli reddit (native adapter)
  6. Google News AU                    — via RSS (no opencli needed)

No API keys needed — opencli reuses your Chrome login session.

Install opencli:
    npm install -g @jackwener/opencli

Results are filtered to the last 30 days and tagged with `days_ago`
for time-decay weighting in the sentiment agent.
"""

import json
import re
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

from src.data.cache import get_cache, set_cache


# ──────────────────────── Shared Helpers ────────────────────────

NEWS_WINDOW_DAYS = 30

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-AU,en;q=0.9",
}


def _parse_date(date_str: str) -> Optional[datetime]:
    """Best-effort parse of various date formats."""
    if not date_str:
        return None
    date_str = date_str.strip()

    formats = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d %b %Y",
        "%d %B %Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", date_str)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


def _is_within_window(date_str: str, days: int = NEWS_WINDOW_DAYS) -> bool:
    dt = _parse_date(date_str)
    if dt is None:
        return True  # Keep if unparseable (let LLM judge)
    now = datetime.now()
    if dt.tzinfo is not None:
        now = datetime.now(timezone.utc)
    return (now - dt).days <= days


def _days_ago(date_str: str) -> Optional[float]:
    dt = _parse_date(date_str)
    if dt is None:
        return None
    now = datetime.now()
    if dt.tzinfo is not None:
        now = datetime.now(timezone.utc)
    return max((now - dt).total_seconds() / 86400, 0)


# ──────────────────────── OpenCLI Runner ────────────────────────


def _run_opencli(args: list[str], timeout: int = 30) -> Optional[list | dict]:
    """Run an opencli command and return parsed JSON output.

    Returns None on any failure (opencli not installed, timeout, parse error).
    """
    cmd = ["opencli"] + args + ["-f", "json"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return None
        output = result.stdout.strip()
        if not output:
            return None
        return json.loads(output)
    except FileNotFoundError:
        # opencli not installed
        return None
    except subprocess.TimeoutExpired:
        return None
    except json.JSONDecodeError:
        return None
    except Exception:
        return None


def _opencli_browser_fetch(url: str, selectors: dict, timeout: int = 45) -> list[dict]:
    """Use opencli browser to open a URL and extract content via CSS selectors.

    Args:
        url: The page URL to open
        selectors: Dict mapping field names to CSS selectors, e.g.:
            {"title": "h3.headline", "summary": "p.standfirst", "time": "time"}

    Returns:
        List of extracted article dicts
    """
    # Step 1: Open the URL
    open_result = _run_opencli(["browser", "open", url], timeout=timeout)
    if open_result is None:
        # Try without -f json for browser open (it may not return JSON)
        try:
            subprocess.run(
                ["opencli", "browser", "open", url],
                capture_output=True, text=True, timeout=timeout,
            )
        except Exception:
            return []

    # Step 2: Wait for page load
    try:
        subprocess.run(
            ["opencli", "browser", "wait", "networkidle"],
            capture_output=True, text=True, timeout=15,
        )
    except Exception:
        pass

    # Step 3: Extract content via selectors
    articles = []
    title_selector = selectors.get("container", "article")

    try:
        result = subprocess.run(
            ["opencli", "browser", "eval", f"""
                (() => {{
                    const articles = [];
                    document.querySelectorAll('{title_selector}').forEach((el, i) => {{
                        if (i >= 15) return;
                        const title = el.querySelector('{selectors.get("title", "h3, h2, a")}');
                        const summary = el.querySelector('{selectors.get("summary", "p")}');
                        const time = el.querySelector('{selectors.get("time", "time")}');
                        const link = el.querySelector('a[href]');
                        articles.push({{
                            title: title ? title.innerText.trim() : '',
                            summary: summary ? summary.innerText.trim().slice(0, 200) : '',
                            published: time ? (time.getAttribute('datetime') || time.innerText.trim()) : '',
                            url: link ? link.href : '',
                        }});
                    }});
                    return JSON.stringify(articles);
                }})()
            """],
            capture_output=True, text=True, timeout=20,
        )
        if result.returncode == 0 and result.stdout.strip():
            raw = result.stdout.strip()
            # opencli browser eval may wrap output
            if raw.startswith('"') and raw.endswith('"'):
                raw = json.loads(raw)  # Unwrap string
            articles = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        pass

    return [a for a in articles if a.get("title") and len(a["title"]) >= 10]


# ──────────────────────── Twitter/X (native opencli adapter) ────────────────────────


def fetch_twitter_news(query: str, max_results: int = 15) -> list[dict]:
    """Fetch tweets via `opencli twitter search`. No API key needed.

    Uses your Chrome login session via the Browser Bridge.
    """
    cache_key = f"twitter_opencli:{query}:{max_results}"
    cached = get_cache(cache_key, ttl=900)  # 15 min cache
    if cached:
        return cached

    data = _run_opencli([
        "twitter", "search",
        "--query", f"{query} ASX stock",
        "--limit", str(max_results),
    ])

    if not data or not isinstance(data, list):
        return []

    results = []
    for tweet in data:
        # opencli twitter returns: text, author, created_at, likes, retweets, etc.
        text = tweet.get("text") or tweet.get("content") or tweet.get("title", "")
        created = tweet.get("created_at") or tweet.get("date") or tweet.get("time", "")
        likes = tweet.get("likes", 0) or tweet.get("like_count", 0) or 0
        retweets = tweet.get("retweets", 0) or tweet.get("retweet_count", 0) or 0
        author = tweet.get("author") or tweet.get("user") or tweet.get("username", "")

        if not text or len(text) < 10:
            continue
        if not _is_within_window(created):
            continue

        results.append({
            "title": text[:200],
            "summary": f"@{author}" if author else "",
            "source": "Twitter/X",
            "url": tweet.get("url", ""),
            "published": created,
            "days_ago": _days_ago(created),
            "engagement": int(likes) + int(retweets) * 2,
        })

    results.sort(key=lambda x: x.get("engagement", 0), reverse=True)
    set_cache(cache_key, results)
    return results


# ──────────────────────── Reddit (native opencli adapter) ────────────────────────


def fetch_reddit_posts(query: str, max_results: int = 10) -> list[dict]:
    """Fetch Reddit posts via `opencli reddit search/hot`. No API key needed."""
    cache_key = f"reddit_opencli:{query}:{max_results}"
    cached = get_cache(cache_key, ttl=1800)
    if cached:
        return cached

    subreddits = ["ASX_Bets", "AusFinance", "AusStocks"]
    results = []

    for sub in subreddits:
        # Try search first
        data = _run_opencli([
            "reddit", "search",
            "--query", query,
            "--subreddit", sub,
            "--limit", str(max_results),
        ])

        if not data or not isinstance(data, list):
            # Fallback: try hot posts
            data = _run_opencli([
                "reddit", "hot",
                "--subreddit", sub,
                "--limit", str(max_results),
            ])

        if not data or not isinstance(data, list):
            continue

        for post in data:
            title = post.get("title", "")
            if not title:
                continue

            created = post.get("created") or post.get("date") or post.get("time", "")
            if not _is_within_window(created):
                continue

            score = post.get("score") or post.get("upvotes") or post.get("points", 0)
            comments = post.get("comments") or post.get("num_comments") or post.get("comment_count", 0)

            results.append({
                "title": title,
                "summary": (post.get("selftext") or post.get("body") or "")[:300],
                "source": f"Reddit r/{sub}",
                "url": post.get("url") or post.get("permalink", ""),
                "published": created,
                "days_ago": _days_ago(created),
                "engagement": int(score) + int(comments),
            })

    results.sort(key=lambda x: x.get("engagement", 0), reverse=True)
    set_cache(cache_key, results[:max_results])
    return results[:max_results]


# ──────────────────────── AFR (via opencli browser) ────────────────────────


def fetch_afr_news(query: str, max_results: int = 10) -> list[dict]:
    """Fetch AFR news via `opencli browser`."""
    cache_key = f"afr_opencli:{query}:{max_results}"
    cached = get_cache(cache_key, ttl=3600)
    if cached:
        return cached

    url = f"https://www.afr.com/search?text={quote_plus(query)}&sortBy=date"
    articles = _opencli_browser_fetch(url, {
        "container": "article, [data-testid*='search-result'], .search-result",
        "title": "h2, h3, .headline, a",
        "summary": "p, .summary, .standfirst",
        "time": "time, .timestamp, [datetime]",
    })

    results = []
    for a in articles[:max_results]:
        published = a.get("published", "")
        if not _is_within_window(published):
            continue

        link = a.get("url", "")
        if link and not link.startswith("http"):
            link = f"https://www.afr.com{link}"

        results.append({
            "title": a["title"],
            "summary": a.get("summary", ""),
            "source": "AFR",
            "url": link,
            "published": published,
            "days_ago": _days_ago(published),
        })

    set_cache(cache_key, results)
    return results


# ──────────────────────── The Australian (via opencli browser) ────────────────────────


def fetch_theaustralian_news(query: str, max_results: int = 10) -> list[dict]:
    """Fetch The Australian news via `opencli browser`."""
    cache_key = f"theaustralian_opencli:{query}:{max_results}"
    cached = get_cache(cache_key, ttl=3600)
    if cached:
        return cached

    url = f"https://www.theaustralian.com.au/search-results?q={quote_plus(query)}"
    articles = _opencli_browser_fetch(url, {
        "container": "article, .search-result, .story-block, [data-testid*='result']",
        "title": "h3, h2, .headline, a.story-block__heading",
        "summary": "p, .summary, .standfirst",
        "time": "time, .timestamp, .date, [datetime]",
    })

    results = []
    for a in articles[:max_results]:
        published = a.get("published", "")
        if not _is_within_window(published):
            continue

        link = a.get("url", "")
        if link and not link.startswith("http"):
            link = f"https://www.theaustralian.com.au{link}"

        results.append({
            "title": a["title"],
            "summary": a.get("summary", ""),
            "source": "The Australian",
            "url": link,
            "published": published,
            "days_ago": _days_ago(published),
        })

    set_cache(cache_key, results)
    return results


# ──────────────────────── Bloomberg (via opencli browser) ────────────────────────


def fetch_bloomberg_news(query: str, max_results: int = 10) -> list[dict]:
    """Fetch Bloomberg news via `opencli browser`.

    Uses your Bloomberg login session via Chrome for paywalled content.
    """
    cache_key = f"bloomberg_opencli:{query}:{max_results}"
    cached = get_cache(cache_key, ttl=3600)
    if cached:
        return cached

    url = f"https://www.bloomberg.com/search?query={quote_plus(query)}"
    articles = _opencli_browser_fetch(url, {
        "container": "article, [data-component='headline'], .storyItem, .search-result",
        "title": "h3, h2, a, .headline",
        "summary": "p, .summary, .abstract",
        "time": "time, [datetime], .date, .publishedAt",
    })

    results = []
    for a in articles[:max_results]:
        published = a.get("published", "")
        if not _is_within_window(published):
            continue

        link = a.get("url", "")
        if link and not link.startswith("http"):
            link = f"https://www.bloomberg.com{link}"

        results.append({
            "title": a["title"],
            "summary": a.get("summary", ""),
            "source": "Bloomberg",
            "url": link,
            "published": published,
            "days_ago": _days_ago(published),
        })

    set_cache(cache_key, results)
    return results


# ──────────────────────── Google News AU (RSS fallback) ────────────────────────


def fetch_google_news_au(query: str, max_results: int = 15) -> list[dict]:
    """Google News AU via RSS. No opencli needed, always works."""
    cache_key = f"googlenews_v3:{query}:{max_results}"
    cached = get_cache(cache_key, ttl=1800)
    if cached:
        return cached

    url = (
        f"https://news.google.com/rss/search"
        f"?q={quote_plus(query + ' ASX')}&hl=en-AU&gl=AU&ceid=AU:en"
    )

    results = []
    try:
        with httpx.Client(timeout=15, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible)"})
            if resp.status_code != 200:
                return []

        soup = BeautifulSoup(resp.text, "xml")
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
    except Exception:
        pass

    set_cache(cache_key, results)
    return results


# ──────────────────────── Aggregator ────────────────────────


def fetch_all_au_news(
    ticker: str,
    company_name: Optional[str] = None,
    max_per_source: int = 10,
) -> dict[str, list[dict]]:
    """Aggregate news from all Australian sources via opencli.

    All results are filtered to the last 30 days and include `days_ago`
    for time-decay weighting.

    Sources:
        - AFR, The Australian, Bloomberg → opencli browser
        - Twitter/X, Reddit → opencli native adapters
        - Google News AU → RSS (always available as fallback)
    """
    search_term = ticker.replace(".AX", "").replace(".ax", "")
    if company_name:
        search_term = f"{company_name} {search_term}"

    return {
        "afr": fetch_afr_news(search_term, max_per_source),
        "the_australian": fetch_theaustralian_news(search_term, max_per_source),
        "bloomberg": fetch_bloomberg_news(search_term, max_per_source),
        "twitter": fetch_twitter_news(search_term, max_per_source),
        "reddit": fetch_reddit_posts(search_term, max_per_source),
        "google_news_au": fetch_google_news_au(search_term, max_per_source),
    }
