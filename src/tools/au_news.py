"""Australian news sources for sentiment analysis — powered by OpenCLI.

Uses `opencli` (https://github.com/jackwener/opencli) to fetch data:
  1. Bloomberg       — `opencli bloomberg markets/industries` (public RSS, no browser)
  2. AFR             — `opencli web read` (browser, uses Chrome session)
  3. The Australian  — `opencli web read` (browser, uses Chrome session)
  4. Twitter/X       — `opencli twitter search` (browser, needs Twitter login)
  5. Reddit          — `opencli reddit search` (browser, uses Chrome session)
  6. Google News AU  — RSS via httpx (always available fallback)

No API keys needed — opencli reuses your Chrome login session.

Install:  npm install -g @jackwener/opencli

Correct command syntax (verified):
  opencli bloomberg markets --limit 10 -f json
  opencli twitter search "BHP ASX" --limit 10 -f json
  opencli reddit search "BHP" --subreddit ASX_Bets --time month --limit 10 -f json
  opencli web read --url "https://..." -f json --download-images false
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

from src.data.cache import get_cache, set_cache


# ──────────────────────── Shared Helpers ────────────────────────

NEWS_WINDOW_DAYS = 30


def _parse_date(date_str: str) -> Optional[datetime]:
    """Best-effort parse of various date formats."""
    if not date_str:
        return None
    date_str = date_str.strip()

    # Strip common prefixes
    date_str = re.sub(r"^(Updated |Published )", "", date_str).strip()

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
        "%a %b %d %H:%M:%S %z %Y",     # Twitter: "Tue Apr 07 09:48:24 +0000 2026"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    # "Apr 8, 2026" or "Updated Apr 9, 2026"
    m = re.search(r"(\w+ \d{1,2}, \d{4})", date_str)
    if m:
        try:
            return datetime.strptime(m.group(1), "%b %d, %Y")
        except ValueError:
            try:
                return datetime.strptime(m.group(1), "%B %d, %Y")
            except ValueError:
                pass

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
        return True  # Keep if unparseable
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

    Returns None on any failure.
    """
    cmd = ["opencli"] + args
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
        # opencli may append update notices after JSON — find the JSON part
        # JSON starts with [ or {
        for i, ch in enumerate(output):
            if ch in "[{":
                output = output[i:]
                break
        # Find the end of JSON
        depth = 0
        end = 0
        in_string = False
        escape = False
        for i, ch in enumerate(output):
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in '[{':
                depth += 1
            elif ch in ']}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > 0:
            output = output[:end]
        return json.loads(output)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    except Exception:
        return None


def _read_opencli_markdown(url: str, timeout: int = 30) -> Optional[str]:
    """Use `opencli web read` to fetch a URL as Markdown. Returns the markdown text."""
    # Use a temp output dir
    out_dir = Path("/tmp/asx-hedge-fund-web")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            ["opencli", "web", "read",
             "--url", url,
             "--output", str(out_dir),
             "--download-images", "false"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return None

    # Find the generated .md file (opencli creates a subfolder)
    md_files = list(out_dir.rglob("*.md"))
    if not md_files:
        return None

    # Read the most recently modified one
    md_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    content = md_files[0].read_text(errors="ignore")

    # Clean up
    try:
        import shutil
        for f in md_files:
            f.unlink(missing_ok=True)
    except Exception:
        pass

    return content


# ──────────────────────── Bloomberg (native RSS — public, no browser) ────────────────────────


def fetch_bloomberg_news(query: str, max_results: int = 10) -> list[dict]:
    """Fetch Bloomberg news via `opencli bloomberg` native RSS commands.

    Public RSS feeds — no browser or login needed.
    Fetches from markets + industries feeds for broad coverage.
    """
    cache_key = f"bloomberg_cli:{query}:{max_results}"
    cached = get_cache(cache_key, ttl=3600)
    if cached:
        return cached

    results = []
    query_lower = query.lower()

    # Fetch from multiple Bloomberg RSS feeds
    for feed in ["markets", "industries", "main"]:
        data = _run_opencli(["bloomberg", feed, "--limit", "15", "-f", "json"])
        if not data or not isinstance(data, list):
            continue

        for item in data:
            title = item.get("title", "")
            summary = item.get("summary", "")

            # Filter by relevance to query
            combined = (title + " " + summary).lower()
            if query_lower and not any(
                term in combined
                for term in query_lower.split()
            ):
                continue

            results.append({
                "title": title,
                "summary": summary[:200],
                "source": "Bloomberg",
                "url": item.get("link", ""),
                "published": "",  # RSS feeds don't always include dates
                "days_ago": 1.0,  # Assume recent (RSS is current news)
            })

    # If no query-filtered results, return top stories as general market context
    if not results:
        data = _run_opencli(["bloomberg", "markets", "--limit", str(max_results), "-f", "json"])
        if data and isinstance(data, list):
            for item in data:
                results.append({
                    "title": item.get("title", ""),
                    "summary": item.get("summary", "")[:200],
                    "source": "Bloomberg",
                    "url": item.get("link", ""),
                    "published": "",
                    "days_ago": 1.0,
                })

    set_cache(cache_key, results[:max_results])
    return results[:max_results]


# ──────────────────────── AFR (via opencli web read) ────────────────────────


def fetch_afr_news(query: str, max_results: int = 10) -> list[dict]:
    """Fetch AFR news via `opencli web read` which renders the page as Markdown.

    Parses article titles, dates, and summaries from the Markdown output.
    """
    cache_key = f"afr_cli:{query}:{max_results}"
    cached = get_cache(cache_key, ttl=3600)
    if cached:
        return cached

    url = f"https://www.afr.com/search?text={quote_plus(query)}&sortBy=date"
    md = _read_opencli_markdown(url, timeout=30)
    if not md:
        return []

    results = _parse_afr_markdown(md, max_results)
    set_cache(cache_key, results)
    return results


def _parse_afr_markdown(md: str, max_results: int) -> list[dict]:
    """Parse AFR search results from Markdown output.

    The markdown from `opencli web read` contains patterns like:
        ### [Article Title](/path/to/article)
        Summary text
        - Apr 8, 2026
        - Author Name
    """
    results = []
    lines = md.split("\n")

    i = 0
    while i < len(lines) and len(results) < max_results:
        line = lines[i].strip()

        # Look for article headings: ### [Title](url)
        heading_match = re.match(r"^#{1,4}\s*\[(.+?)\]\((.+?)\)", line)
        if heading_match:
            title = heading_match.group(1).strip()
            link = heading_match.group(2).strip()
            if not link.startswith("http"):
                link = f"https://www.afr.com{link}"

            # Look ahead for summary and date (skip empty lines)
            summary = ""
            published = ""
            for j in range(i + 1, min(i + 12, len(lines))):
                next_line = lines[j].strip()
                if next_line.startswith("#"):
                    break
                if not next_line:
                    continue
                # Date patterns (e.g. "- Apr 8, 2026" or "Apr 8, 2026")
                clean_line = re.sub(r"^-\s*", "", next_line)
                date_match = re.search(
                    r"((?:Updated )?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4})",
                    clean_line,
                )
                if date_match:
                    published = date_match.group(1).replace("Updated ", "")
                elif not summary and len(next_line) > 20 and not next_line.startswith("-"):
                    summary = next_line[:200]

            if title and len(title) >= 10:
                if _is_within_window(published):
                    results.append({
                        "title": title,
                        "summary": summary,
                        "source": "AFR",
                        "url": link,
                        "published": published,
                        "days_ago": _days_ago(published),
                    })
        i += 1

    return results


# ──────────────────────── The Australian (via opencli web read) ────────────────────────


def fetch_theaustralian_news(query: str, max_results: int = 10) -> list[dict]:
    """Fetch The Australian news via `opencli web read`."""
    cache_key = f"theaustralian_cli:{query}:{max_results}"
    cached = get_cache(cache_key, ttl=3600)
    if cached:
        return cached

    url = f"https://www.theaustralian.com.au/search-results?q={quote_plus(query)}"
    md = _read_opencli_markdown(url, timeout=30)
    if not md:
        return []

    results = _parse_generic_news_markdown(md, "The Australian", "https://www.theaustralian.com.au", max_results)
    set_cache(cache_key, results)
    return results


def _parse_generic_news_markdown(
    md: str, source: str, base_url: str, max_results: int
) -> list[dict]:
    """Generic parser for news search results rendered as Markdown."""
    results = []
    lines = md.split("\n")

    i = 0
    while i < len(lines) and len(results) < max_results:
        line = lines[i].strip()

        heading_match = re.match(r"^#{1,4}\s*\[(.+?)\]\((.+?)\)", line)
        if heading_match:
            title = heading_match.group(1).strip()
            link = heading_match.group(2).strip()
            if not link.startswith("http"):
                link = f"{base_url}{link}"

            summary = ""
            published = ""
            for j in range(i + 1, min(i + 12, len(lines))):
                next_line = lines[j].strip()
                if next_line.startswith("#"):
                    break
                if not next_line:
                    continue
                clean_line = re.sub(r"^-\s*", "", next_line)
                date_match = re.search(
                    r"((?:Updated )?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4})",
                    clean_line,
                )
                if date_match:
                    published = date_match.group(1).replace("Updated ", "")
                elif not summary and len(next_line) > 20 and not next_line.startswith("-"):
                    summary = next_line[:200]

            if title and len(title) >= 10:
                if _is_within_window(published):
                    results.append({
                        "title": title,
                        "summary": summary,
                        "source": source,
                        "url": link,
                        "published": published,
                        "days_ago": _days_ago(published),
                    })
        i += 1

    return results


# ──────────────────────── Twitter/X (native opencli adapter) ────────────────────────


def fetch_twitter_news(query: str, max_results: int = 15) -> list[dict]:
    """Fetch tweets via `opencli twitter search "query" --limit N -f json`.

    Requires Twitter login in Chrome. Gracefully returns [] if not logged in.
    """
    cache_key = f"twitter_cli:{query}:{max_results}"
    cached = get_cache(cache_key, ttl=900)
    if cached:
        return cached

    data = _run_opencli(
        ["twitter", "search", f"{query} ASX", "--limit", str(max_results), "-f", "json"],
        timeout=30,
    )

    if not data or not isinstance(data, list):
        return []

    results = []
    for tweet in data:
        text = tweet.get("text", "")
        created = tweet.get("created_at", "")
        likes = int(tweet.get("likes", 0) or 0)
        views = int(tweet.get("views", 0) or 0)
        author = tweet.get("author", "")
        url = tweet.get("url", "")

        if not text or len(text) < 10:
            continue
        if not _is_within_window(created):
            continue

        results.append({
            "title": text[:200],
            "summary": f"@{author}" if author else "",
            "source": "Twitter/X",
            "url": url,
            "published": created,
            "days_ago": _days_ago(created),
            "engagement": likes + (views // 100),
        })

    results.sort(key=lambda x: x.get("engagement", 0), reverse=True)
    set_cache(cache_key, results)
    return results


# ──────────────────────── Reddit (native opencli adapter) ────────────────────────


def fetch_reddit_posts(query: str, max_results: int = 10) -> list[dict]:
    """Fetch Reddit posts via `opencli reddit search "query" --subreddit X -f json`."""
    cache_key = f"reddit_cli:{query}:{max_results}"
    cached = get_cache(cache_key, ttl=1800)
    if cached:
        return cached

    subreddits = ["ASX_Bets", "AusFinance", "AusStocks"]
    results = []

    for sub in subreddits:
        data = _run_opencli(
            ["reddit", "search", query,
             "--subreddit", sub,
             "--time", "month",
             "--sort", "relevance",
             "--limit", str(max_results),
             "-f", "json"],
            timeout=30,
        )

        if not data or not isinstance(data, list):
            continue

        for post in data:
            title = post.get("title", "")
            if not title:
                continue

            score = int(post.get("score", 0) or 0)
            comments = int(post.get("comments", 0) or 0)

            results.append({
                "title": title,
                "summary": "",
                "source": f"Reddit r/{sub}",
                "url": post.get("url", ""),
                "published": "",  # Reddit search doesn't reliably return dates
                "days_ago": 7.0,  # Assume within the month (filtered by --time month)
                "engagement": score + comments,
            })

    results.sort(key=lambda x: x.get("engagement", 0), reverse=True)
    set_cache(cache_key, results[:max_results])
    return results[:max_results]


# ──────────────────────── Google News AU (RSS — always-available fallback) ────────────────────────


def fetch_google_news_au(query: str, max_results: int = 15) -> list[dict]:
    """Google News AU via RSS. No opencli needed, always works."""
    cache_key = f"googlenews_v4:{query}:{max_results}"
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

    All results filtered to last 30 days with `days_ago` for time-decay weighting.

    Command mapping:
        Bloomberg      → opencli bloomberg markets/industries (public RSS, no login)
        AFR            → opencli web read (browser, Chrome session)
        The Australian → opencli web read (browser, Chrome session)
        Twitter/X      → opencli twitter search (browser, needs Twitter login)
        Reddit         → opencli reddit search (browser, Chrome session)
        Google News AU → RSS via httpx (always available, no opencli needed)
    """
    search_term = ticker.replace(".AX", "").replace(".ax", "")
    if company_name:
        search_term = f"{company_name} {search_term}"

    from concurrent.futures import ThreadPoolExecutor, as_completed

    sources = {
        "bloomberg": lambda: fetch_bloomberg_news(search_term, max_per_source),
        "afr": lambda: fetch_afr_news(search_term, max_per_source),
        "the_australian": lambda: fetch_theaustralian_news(search_term, max_per_source),
        "twitter": lambda: fetch_twitter_news(search_term, max_per_source),
        "reddit": lambda: fetch_reddit_posts(search_term, max_per_source),
        "google_news_au": lambda: fetch_google_news_au(search_term, max_per_source),
    }

    results = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(fn): name for name, fn in sources.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception:
                results[name] = []

    return results
