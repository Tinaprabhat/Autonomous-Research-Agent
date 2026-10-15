"""
observability.py — structured JSONL query log + p50/p95 latency summary.

Deliberately a flat file at this scale (10 users): no metrics stack.
Cost: Groq's free tier is $0, so we record estimated tokens and a cost figure
computed from configurable per-million-token prices (default 0).
"""
from __future__ import annotations

import json
import os
import threading
import time

from backend.app import config

_lock = threading.Lock()
PRICE_IN_PER_M  = float(os.getenv("LLM_PRICE_IN_PER_M", "0"))
PRICE_OUT_PER_M = float(os.getenv("LLM_PRICE_OUT_PER_M", "0"))


def est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def log_query(record: dict, path: str | None = None) -> None:
    path = path or config.QUERY_LOG
    record = {"ts": round(time.time(), 3), **record}
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with _lock, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass  # logging must never break a request


def cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return round(prompt_tokens * PRICE_IN_PER_M / 1e6 + completion_tokens * PRICE_OUT_PER_M / 1e6, 6)


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))
    return s[idx]


def summarize(path: str | None = None) -> dict:
    path = path or config.QUERY_LOG
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    lat = [r["latency_ms"] for r in rows if "latency_ms" in r]
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[r.get("status", "?")] = statuses.get(r.get("status", "?"), 0) + 1
    return {
        "queries": len(rows),
        "latency_ms_p50": _pct(lat, 50),
        "latency_ms_p95": _pct(lat, 95),
        "total_cost_usd": round(sum(r.get("cost_usd", 0) for r in rows), 6),
        "avg_cost_usd": round(sum(r.get("cost_usd", 0) for r in rows) / len(rows), 6) if rows else 0.0,
        "status_counts": statuses,
        # a hedge-only answer is status "ok" but gives the user nothing: watch this rate for false refusals
        "hedged_answers": sum(1 for r in rows if r.get("hedged")),
        "hedged_rate": round(sum(1 for r in rows if r.get("hedged")) / len(rows), 3) if rows else 0.0,
    }
