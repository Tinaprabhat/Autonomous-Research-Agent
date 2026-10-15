"""
api.py — FastAPI backend for the Research Agent.

Run from project root:
    uvicorn backend.app.api:app --port 8000

Serves the JSON/SSE API and the static frontend from the same process
(single-container deployment).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.app import config
from backend.app.auth import Authenticator, UserLimiter, extract_key, parse_api_keys
from backend.app.pipeline import RetrievalUnavailable
from backend.app.telemetry import observability

FRONTEND_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../frontend/Research_ui.html"))

RATE_LIMITED_PATHS = ("/query", "/query/stream", "/feedback")
QUERY_PATHS = ("/query", "/query/stream")          # these also spend the user's daily quota
PUBLIC_PATHS = ("/", "/health", "/docs", "/redoc", "/openapi.json")   # reachable without a key


# ── Schemas ───────────────────────────────────────────────────────────────────

class Filters(BaseModel):
    category:  str | None = Field(None, max_length=40)
    author:    str | None = Field(None, max_length=100)
    section:   str | None = Field(None, max_length=60)
    arxiv_id:  str | None = Field(None, max_length=30)
    date_from: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    date_to:   str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")

    def as_dict(self) -> dict | None:
        d = {k: v for k, v in self.model_dump().items() if v}
        return d or None


class QueryRequest(BaseModel):
    query:   str = Field(..., min_length=1, max_length=config.MAX_QUERY_CHARS)
    top_k:   int = Field(5, ge=1, le=10)
    filters: Filters | None = None


# [DEAD CODE] legacy `/ask` request model (checked 2026-10-15): the UI uses /query/stream; only tests called /ask.
# class AskRequest(BaseModel):
#     question: str = Field(..., min_length=1, max_length=config.MAX_QUERY_CHARS)


class FeedbackRequest(BaseModel):
    query:    str = Field("", max_length=config.MAX_QUERY_CHARS)
    answer:   str = Field("", max_length=4000)
    kind:     str = Field(..., pattern=r"^(thumbs_up|thumbs_down|wrong_citation)$")
    citation: int | None = Field(None, ge=1, le=50)
    comment:  str = Field("", max_length=500)


# ── Per-IP rate limiter (sliding window, in-memory) ───────────────────────────

class RateLimiter:

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            return True


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(pipeline=None, store=None, rate_limit_per_min: int | None = None,
               api_keys: str | None = None) -> FastAPI:
    """Pass `pipeline` (and optionally `store`) to inject fakes in tests.
    `api_keys` overrides the API_KEYS env var ("name:key[:per_minute[:per_day]],..."); empty = auth off."""
    state = {"pipeline": pipeline, "store": store}
    auth = Authenticator(parse_api_keys(config.API_KEYS if api_keys is None else api_keys))
    user_limiter = UserLimiter()
    auth_fail_limiter = RateLimiter(config.AUTH_FAIL_PER_MIN)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["pipeline"] is None:
            from backend.app.services import build_pipeline
            print("Loading pipeline...")
            state["pipeline"], state["store"] = build_pipeline()
            print(f"Ready. {state['store'].count()} chunks loaded.")
        yield

    app = FastAPI(title="Research Agent API", lifespan=lifespan)
    # [DEAD CODE] `app.state.pipeline_state` was never read (checked 2026-10-08); commented out.
    # app.state.pipeline_state = state
    limiter = RateLimiter(rate_limit_per_min or config.RATE_LIMIT_PER_MIN)
    app.state.limiter = limiter

    @app.middleware("http")
    async def access_control(request: Request, call_next):
        path = request.url.path
        if request.method == "OPTIONS" or path in PUBLIC_PATHS:
            return await call_next(request)

        if auth.enabled:
            ip = _client_ip(request)
            user = auth.authenticate(extract_key(request.headers))
            if user is None:
                # throttle bad-key guessing per IP, then reject
                if not auth_fail_limiter.allow(ip):
                    return JSONResponse({"detail": "Too many failed authentication attempts."},
                                        status_code=429, headers={"Retry-After": "60"})
                return JSONResponse({"detail": "Missing or invalid API key."}, status_code=401,
                                    headers={"WWW-Authenticate": "Bearer"})
            request.state.user = user
            if request.method == "POST" and path in RATE_LIMITED_PATHS:
                if not user_limiter.check_minute(user):
                    return JSONResponse(
                        {"detail": f"Rate limit exceeded ({user.per_minute}/min). Try again in a minute."},
                        status_code=429, headers={"Retry-After": "60"})
                if path in QUERY_PATHS and not user_limiter.consume_day(user):
                    wait = user_limiter.seconds_until_reset()
                    return JSONResponse(
                        {"detail": f"Daily quota of {user.per_day} queries used. Resets in {wait // 3600}h "
                                   f"{wait % 3600 // 60}m (00:00 UTC)."},
                        status_code=429, headers={"Retry-After": str(wait)})
            return await call_next(request)

        if request.method == "POST" and path in RATE_LIMITED_PATHS:
            if not limiter.allow(_client_ip(request)):
                return JSONResponse({"detail": "Rate limit exceeded. Try again in a minute."},
                                    status_code=429, headers={"Retry-After": "60"})
        return await call_next(request)

    # Added last so it is the outermost layer: 401/429 responses still carry CORS headers.
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=False,
        allow_methods=["*"], allow_headers=["*"])

    def pipe():
        if state["pipeline"] is None:
            raise HTTPException(status_code=503, detail="Pipeline is still loading")
        return state["pipeline"]

    # ── Endpoints ────────────────────────────────────────────────────────────

    @app.get("/", include_in_schema=False)
    def index():
        if os.path.exists(FRONTEND_FILE):
            return FileResponse(FRONTEND_FILE, media_type="text/html")
        return {"message": "Research Agent API running"}

    @app.get("/health")
    def health():
        st = state["store"]
        p = state["pipeline"]
        status, chunks = ("ok" if p is not None else "loading"), 0
        if st is not None:
            try:
                chunks = st.count()
            except Exception:                    # vector DB unreachable: report it, don't 500 the probe
                status = "degraded"
        return {
            "status": status,
            "chunks": chunks,
            "model": getattr(getattr(p, "llm", None), "model", None),
        }

    @app.get("/me")
    def me(request: Request):
        user = getattr(request.state, "user", None)
        if user is None:
            return {"auth": "disabled"}
        return {"auth": "api_key", "user": user.name, "per_minute": user.per_minute,
                "per_day": user.per_day, "remaining_today": user_limiter.remaining_day(user)}

    @app.get("/papers")
    def papers():
        st = state["store"]
        if st is None:
            raise HTTPException(status_code=503, detail="Store not ready")
        try:
            rows = st.list_papers()
        except Exception:
            raise HTTPException(status_code=503, detail="Vector database unavailable") from None
        return {
            "papers": rows,
            "categories": sorted({r["category"] for r in rows if r.get("category")}),
            "sections": ["Abstract", "Introduction", "Related Work", "Background", "Method",
                         "Experiments", "Results", "Discussion", "Conclusion", "Appendix"],
        }

    @app.post("/query")
    def query(req: QueryRequest):
        if not req.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        filters = req.filters.as_dict() if req.filters else None
        try:
            return pipe().answer(req.query, req.top_k, filters).to_dict()
        except RetrievalUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e)) from None

    @app.post("/query/stream")
    def query_stream(req: QueryRequest):
        if not req.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        filters = req.filters.as_dict() if req.filters else None
        p = pipe()

        def gen():
            try:
                for ev in p.answer_stream(req.query, req.top_k, filters):
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            except RetrievalUnavailable as e:
                err = {"event": "error", "detail": str(e)}
                yield f"data: {json.dumps(err)}\n\n"
            except Exception as e:  # never leave the client hanging mid-stream
                err = {"event": "error", "detail": f"{type(e).__name__}"}
                yield f"data: {json.dumps(err)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # [DEAD CODE] legacy `/ask` endpoint (checked 2026-10-15): nothing but two tests called it; /query replaces it.
    # @app.post("/ask")
    # def ask(req: AskRequest):
    #     if not req.question.strip():
    #         raise HTTPException(status_code=400, detail="Question cannot be empty")
    #     try:
    #         res = pipe().answer(req.question)
    #     except RetrievalUnavailable as e:
    #         raise HTTPException(status_code=503, detail=str(e)) from None
    #     return {"question": req.question, "answer": res.answer}

    @app.post("/feedback")
    def feedback(req: FeedbackRequest):
        rec = {"ts": round(time.time(), 3), **req.model_dump()}
        try:
            os.makedirs(os.path.dirname(config.FEEDBACK_LOG) or ".", exist_ok=True)
            with open(config.FEEDBACK_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            raise HTTPException(status_code=500, detail="Could not record feedback")
        return {"ok": True}

    @app.get("/metrics")
    def metrics():
        return observability.summarize()

    return app


app = create_app()
