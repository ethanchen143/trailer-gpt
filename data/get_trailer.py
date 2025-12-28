#!/usr/bin/env python3
import os, re, time
from urllib.parse import urljoin, urlparse, parse_qs
import requests
from bs4 import BeautifulSoup

BASE = "https://transcripts.foreverdreaming.org"
FORUM = f"{BASE}/viewforum.php?f=247"
OUT_DIR = "downloads"
DELAY_SEC = 1.2

session = requests.Session()
session.headers.update({
    # Mimic a real browser to avoid basic anti-bot 403 blocks.
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": BASE + "/",
    "Connection": "keep-alive",
})

def _apply_cookies_from_env():
    """Optional: set cookies via FD_COOKIES env var (e.g. copied from browser)."""
    raw = os.environ.get("FD_COOKIES")
    if not raw:
        return
    for part in raw.split(";"):
        if "=" not in part:
            continue
        k, v = part.strip().split("=", 1)
        session.cookies.set(k, v)

_apply_cookies_from_env()

def get_soup(url: str) -> BeautifulSoup:
    resp = session.get(url, timeout=20)
    # If blocked, surface a clearer message so the user can add cookies.
    if resp.status_code == 403:
        raise RuntimeError(
            "Got 403 Forbidden. Try setting FD_COOKIES to your browser cookies "
            "for the site (e.g. export FD_COOKIES='key1=val1; key2=val2')."
        )
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")

def iter_forum_pages():
    """Yield soups for each forum page (handles pagination)."""
    seen = set()
    next_url = FORUM
    while next_url and next_url not in seen:
        seen.add(next_url)
        soup = get_soup(next_url)
        yield soup
        nxt = soup.select_one("a[rel='next']")
        next_url = urljoin(BASE, nxt["href"]) if nxt and nxt.get("href") else None
        time.sleep(DELAY_SEC)

def extract_topic_links(soup: BeautifulSoup):
    for a in soup.select("a.topictitle"):
        href = a.get("href")
        if not href:
            continue
        yield urljoin(BASE, href)

def iter_topic_pages(topic_url: str):
    """Iterate through all pages of a topic."""
    seen = set()
    next_url = topic_url
    while next_url and next_url not in seen:
        seen.add(next_url)
        soup = get_soup(next_url)
        yield soup
        nxt = soup.select_one("a[rel='next']")
        next_url = urljoin(BASE, nxt["href"]) if nxt and nxt.get("href") else None
        time.sleep(DELAY_SEC)

def sanitize(name: str) -> str:
    name = re.sub(r"[^\w\s.-]", "", name).strip()
    return re.sub(r"\s+", "_", name)[:80] or "untitled"

def topic_id_from_url(url: str) -> str:
    qs = parse_qs(urlparse(url).query)
    return qs.get("t", ["unknown"])[0]

def scrape_topic(topic_url: str):
    texts = []
    title = "untitled"
    for soup in iter_topic_pages(topic_url):
        if title == "untitled":
            h = soup.select_one("h2.topic-title")
            if h:
                title = h.get_text(strip=True)
        for post in soup.select("div.postbody div.content"):
            texts.append(post.get_text("\n", strip=True))
    return title, texts

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for forum_soup in iter_forum_pages():
        for topic_url in extract_topic_links(forum_soup):
            tid = topic_id_from_url(topic_url)
            title, posts = scrape_topic(topic_url)
            if not posts:
                continue
            fname = f"{tid}_{sanitize(title)}.txt"
            path = os.path.join(OUT_DIR, fname)
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n\n---\n\n".join(posts))
            print(f"Saved {path}")

if __name__ == "__main__":
    main()