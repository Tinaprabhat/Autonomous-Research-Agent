"""API-key auth and per-user limits (no models, no network)."""
import pytest
from fastapi.testclient import TestClient

from backend.app.api import create_app
from backend.app.auth import Authenticator, UserLimiter, extract_key, parse_api_keys
from backend.app.pipeline import PipelineResult

ALICE = "alice-key-0123456789abcdef"
BOB = "bob-key-0123456789abcdefgh"


class FakePipeline:
    class llm:
        model = "fake"

    def answer(self, q, top_k=5, filters=None):
        return PipelineResult(status="ok", answer="An answer [1].", citations=[{"n": 1}], sources_used=1)

    def answer_stream(self, q, top_k=5, filters=None):
        yield {"event": "final", "result": self.answer(q).to_dict()}


class FakeStore:
    def count(self): return 3
    def list_papers(self): return []


def make(keys, **kw):
    return TestClient(create_app(pipeline=FakePipeline(), store=FakeStore(), api_keys=keys,
                                 rate_limit_per_min=1000, **kw))


def q(client, key=None, path="/query"):
    h = {"X-API-Key": key} if key else {}
    return client.post(path, json={"query": "what is lora"}, headers=h)


# ── parsing ───────────────────────────────────────────────────────────────────

def test_parse_keys_defaults_and_overrides():
    users = parse_api_keys(f"alice:{ALICE}:30:300,bob:{BOB}", default_per_minute=20, default_per_day=200)
    assert (users["alice"].per_minute, users["alice"].per_day) == (30, 300)
    assert (users["bob"].per_minute, users["bob"].per_day) == (20, 200)


@pytest.mark.parametrize("spec", [
    "alice", "alice:short", f"alice:{ALICE}:x", f"alice:{ALICE}:0",
    f"alice:{ALICE},alice:{BOB}", f"alice:{ALICE},bob:{ALICE}"])
def test_parse_keys_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        parse_api_keys(spec)


def test_empty_spec_means_auth_off():
    assert not Authenticator(parse_api_keys("")).enabled


def test_extract_key_from_header_or_bearer():
    assert extract_key({"x-api-key": " abc "}) == "abc"
    assert extract_key({"authorization": "Bearer tok"}) == "tok"
    assert extract_key({"authorization": "Basic zzz"}) is None
    assert extract_key({}) is None


# ── enforcement ───────────────────────────────────────────────────────────────

def test_auth_off_allows_everything():
    assert q(make("")).status_code == 200


def test_missing_and_wrong_key_rejected():
    c = make(f"alice:{ALICE}")
    assert q(c).status_code == 401
    assert q(c, "wrong-key-0123456789").status_code == 401
    assert c.get("/papers").status_code == 401
    assert c.get("/metrics").status_code == 401


def test_valid_key_accepted_header_and_bearer():
    c = make(f"alice:{ALICE}")
    assert q(c, ALICE).status_code == 200
    assert c.post("/query", json={"query": "x"}, headers={"Authorization": f"Bearer {ALICE}"}).status_code == 200
    assert q(c, ALICE, "/query/stream").status_code == 200


def test_public_paths_need_no_key():
    c = make(f"alice:{ALICE}")
    assert c.get("/health").status_code == 200
    assert c.get("/").status_code == 200


def test_cors_preflight_not_blocked():
    c = make(f"alice:{ALICE}")
    r = c.options("/query", headers={"Origin": "http://x", "Access-Control-Request-Method": "POST",
                                     "Access-Control-Request-Headers": "x-api-key"})
    assert r.status_code == 200


def test_401_carries_cors_headers():
    r = make(f"alice:{ALICE}").post("/query", json={"query": "x"}, headers={"Origin": "http://x"})
    assert r.status_code == 401 and r.headers.get("access-control-allow-origin") == "*"


def test_failed_auth_attempts_are_throttled(monkeypatch):
    monkeypatch.setattr("backend.app.config.AUTH_FAIL_PER_MIN", 3)
    c = make(f"alice:{ALICE}")
    codes = [q(c, "wrong-key-0123456789").status_code for _ in range(5)]
    assert codes == [401, 401, 401, 429, 429]
    assert q(c, ALICE).status_code == 200            # a valid key is never throttled by the failure counter


# ── per-user limits ───────────────────────────────────────────────────────────

def test_per_minute_limit_is_per_user_not_shared():
    c = make(f"alice:{ALICE}:2:100,bob:{BOB}:2:100")
    assert [q(c, ALICE).status_code for _ in range(3)] == [200, 200, 429]
    assert q(c, BOB).status_code == 200              # bob unaffected by alice's burst


def test_daily_quota_blocks_then_reports_reset():
    c = make(f"alice:{ALICE}:100:2")
    assert [q(c, ALICE).status_code for _ in range(3)] == [200, 200, 429]
    r = q(c, ALICE)
    assert "Daily quota of 2" in r.json()["detail"] and int(r.headers["retry-after"]) > 0


def test_feedback_does_not_spend_daily_quota():
    c = make(f"alice:{ALICE}:100:1")
    for _ in range(3):
        assert c.post("/feedback", json={"kind": "thumbs_up"}, headers={"X-API-Key": ALICE}).status_code == 200
    assert q(c, ALICE).status_code == 200


def test_me_reports_remaining_quota():
    c = make(f"alice:{ALICE}:100:5")
    q(c, ALICE)
    j = c.get("/me", headers={"X-API-Key": ALICE}).json()
    assert j == {"auth": "api_key", "user": "alice", "per_minute": 100, "per_day": 5, "remaining_today": 4}
    assert make("").get("/me").json() == {"auth": "disabled"}


def test_daily_counter_rolls_over_at_utc_midnight():
    u = parse_api_keys(f"alice:{ALICE}:100:1")["alice"]
    lim = UserLimiter()
    day1, day2 = 86400 * 10 + 100, 86400 * 11 + 100
    assert lim.consume_day(u, day1) and not lim.consume_day(u, day1)
    assert lim.consume_day(u, day2)
    assert lim.remaining_day(u, day2) == 0
