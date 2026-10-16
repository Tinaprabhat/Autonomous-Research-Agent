"""
auth.py — API-key authentication and per-user limits.

Keys come from the API_KEYS env var: comma-separated `name:key[:per_minute[:per_day]]`.
    API_KEYS="alice:k-3f9a...:30:300,bob:k-77c1..."
When API_KEYS is empty, auth is OFF (local dev / tests) and the per-IP limiter applies instead.
A client sends its key as `X-API-Key: <key>` or `Authorization: Bearer <key>`.

Limits are per user, not per IP: a per-minute sliding window and a per-UTC-day query quota.
Both are in memory (reset on restart; one worker) — see docs/case_study.md section 5.
"""
from __future__ import annotations

import hmac
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from backend.app import config


@dataclass(frozen=True)
class User:
    name: str
    key: str
    per_minute: int
    per_day: int


def parse_api_keys(spec: str | None, default_per_minute: int | None = None,
                   default_per_day: int | None = None) -> dict[str, User]:
    """'alice:key:30:300,bob:key2' -> {name: User}. Raises ValueError on malformed or duplicate entries."""
    rpm_default = default_per_minute or config.USER_RATE_LIMIT_PER_MIN
    day_default = default_per_day or config.USER_DAILY_QUOTA
    users: dict[str, User] = {}
    seen_keys: set[str] = set()
    for raw in (spec or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(":")
        if len(parts) < 2 or len(parts) > 4 or not parts[0].strip() or not parts[1].strip():
            raise ValueError(f"bad API_KEYS entry {parts[0]!r}: expected name:key[:per_minute[:per_day]]")
        name, key = parts[0].strip(), parts[1].strip()
        if len(key) < 16:
            raise ValueError(f"API key for {name!r} is shorter than 16 characters")
        if name in users or key in seen_keys:
            raise ValueError(f"duplicate user name or key for {name!r}")
        try:
            rpm = int(parts[2]) if len(parts) > 2 and parts[2] else rpm_default
            day = int(parts[3]) if len(parts) > 3 and parts[3] else day_default
        except ValueError:
            raise ValueError(f"non-numeric limit for {name!r}") from None
        if rpm < 1 or day < 1:
            raise ValueError(f"limits for {name!r} must be >= 1")
        seen_keys.add(key)
        users[name] = User(name, key, rpm, day)
    return users


def extract_key(headers) -> str | None:
    key = headers.get("x-api-key")
    if key:
        return key.strip()
    auth = headers.get("authorization", "")
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip() or None
    return None


class Authenticator:

    def __init__(self, users: dict[str, User]):
        self.users = users

    @property
    def enabled(self) -> bool:
        return bool(self.users)

    def authenticate(self, key: str | None) -> User | None:
        """Constant-time comparison against every configured key (no early exit on a match)."""
        if not key:
            return None
        found = None
        for u in self.users.values():
            if hmac.compare_digest(u.key.encode(), key.encode()):
                found = u
        return found


class UserLimiter:
    """Per-user sliding-minute window + per-UTC-day query counter."""

    def __init__(self):
        self._minute: dict[str, deque] = defaultdict(deque)
        self._day: dict[str, tuple[int, int]] = {}     # name -> (day_number, count)
        self._lock = threading.Lock()

    @staticmethod
    def _day_number(now: float) -> int:
        return int(now // 86400)

    def check_minute(self, user: User, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            q = self._minute[user.name]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= user.per_minute:
                return False
            q.append(now)
            return True

    def consume_day(self, user: User, now: float | None = None) -> bool:
        """Count one query against the daily quota. False once the quota is spent."""
        now = time.time() if now is None else now
        today = self._day_number(now)
        with self._lock:
            day, used = self._day.get(user.name, (today, 0))
            if day != today:
                used = 0
            if used >= user.per_day:
                self._day[user.name] = (today, used)
                return False
            self._day[user.name] = (today, used + 1)
            return True

    def remaining_day(self, user: User, now: float | None = None) -> int:
        now = time.time() if now is None else now
        with self._lock:
            day, used = self._day.get(user.name, (self._day_number(now), 0))
            if day != self._day_number(now):
                used = 0
            return max(user.per_day - used, 0)

    @staticmethod
    def seconds_until_reset(now: float | None = None) -> int:
        now = time.time() if now is None else now
        return int(86400 - (now % 86400))
