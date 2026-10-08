"""
arxiv_fetcher.py — harvest paper metadata and download PDFs from arXiv.

Rate-limited per arXiv's API terms (>= 3 s between requests, single connection),
via the `arxiv` client's built-in delay.
"""
from __future__ import annotations

import os
import re
import time

import arxiv
import requests

from backend.app import config

DELAY_SECONDS = 3.0
USER_AGENT = "autonomous-research-agent/2.0 (academic RAG demo)"
_ID_VERSION = re.compile(r"^(?P<id>.+?)(?:v(?P<ver>\d+))?$")


def split_id_version(short_id: str) -> tuple[str, str]:
    """'2101.00001v2' -> ('2101.00001', 'v2'); no suffix -> 'v1'."""
    m = _ID_VERSION.match(short_id)
    return m.group("id"), f"v{m.group('ver')}" if m.group("ver") else "v1"


def result_to_meta(r: arxiv.Result) -> dict:
    base, version = split_id_version(r.get_short_id())
    return {
        "arxiv_id": base,
        "version": version,
        "title": " ".join(r.title.split()),
        "author": [a.name for a in r.authors],
        "summary": " ".join(r.summary.split()),
        "category": r.primary_category,
        "categories": list(r.categories),
        "published_date": r.published.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pdf_url": r.pdf_url,
    }


def _client() -> arxiv.Client:
    return arxiv.Client(page_size=50, delay_seconds=DELAY_SECONDS, num_retries=3)


def harvest(queries: list[str], max_papers: int = 100,
            categories: list[str] | None = None, per_query: int | None = None) -> list[dict]:
    """
    Collect up to `max_papers` unique papers (by base arXiv id) across `queries`,
    optionally restricted to arXiv `categories` (e.g. ["cs.LG", "cs.CL"]).
    """
    per_query = per_query or max(10, -(-max_papers // max(len(queries), 1)))
    seen: dict[str, dict] = {}
    client = _client()
    for q in queries:
        if len(seen) >= max_papers:
            break
        full_q = q
        if categories:
            cats = " OR ".join(f"cat:{c}" for c in categories)
            full_q = f"({q}) AND ({cats})"
        search = arxiv.Search(query=full_q, max_results=per_query,
                              sort_by=arxiv.SortCriterion.Relevance)
        for r in client.results(search):
            meta = result_to_meta(r)
            prev = seen.get(meta["arxiv_id"])
            if prev is None or int(meta["version"][1:]) > int(prev["version"][1:]):
                seen[meta["arxiv_id"]] = meta
            if len(seen) >= max_papers:
                break
    return list(seen.values())[:max_papers]


_last_download = 0.0


def download_pdf(meta: dict, dest_dir: str | None = None) -> str:
    """Download the PDF for `meta` (exact version), >= DELAY_SECONDS apart, return its path."""
    global _last_download
    dest_dir = dest_dir or config.RAW_DIR
    os.makedirs(dest_dir, exist_ok=True)
    safe = meta["arxiv_id"].replace("/", "_")
    path = os.path.join(dest_dir, f"{safe}_{meta['version']}.pdf")
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        return path

    wait = DELAY_SECONDS - (time.time() - _last_download)
    if wait > 0:
        time.sleep(wait)
    url = f"https://arxiv.org/pdf/{meta['arxiv_id']}{meta['version']}"
    resp = requests.get(url, timeout=60, headers={"User-Agent": USER_AGENT}, allow_redirects=True)
    _last_download = time.time()
    resp.raise_for_status()
    if not resp.content.startswith(b"%PDF"):
        raise ValueError(f"response from {url} is not a PDF")
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(resp.content)
    os.replace(tmp, path)
    return path
