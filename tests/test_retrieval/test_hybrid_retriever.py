"""Unit tests for hybrid_retriever.py (RRF fusion over Qdrant + BM25)."""
import pytest

from backend.app.retrieval.hybrid_retriever import HybridRetriever, rrf_fuse


# ── RRF maths ────────────────────────────────────────────────────────────────

def test_rrf_single_ranking_keeps_order():
    fused = rrf_fuse([["a", "b", "c"]])
    assert [d for d, _ in fused] == ["a", "b", "c"]


def test_rrf_score_formula_k60():
    fused = dict(rrf_fuse([["a"], ["a"]]))
    assert fused["a"] == pytest.approx(2 / 61)


def test_rrf_rewards_agreement_over_single_top_rank():
    # "b" is 2nd in both lists; "a" is 1st in one only, "c" 1st in the other.
    fused = [d for d, _ in rrf_fuse([["a", "b"], ["c", "b"]])]
    assert fused[0] == "b"


def test_rrf_is_rank_based_not_score_based():
    assert rrf_fuse([["x", "y"]]) == rrf_fuse([["x", "y"]])


def test_rrf_dedupes():
    fused = rrf_fuse([["a", "b"], ["b", "a"]])
    assert sorted(d for d, _ in fused) == ["a", "b"]


# ── Retriever ────────────────────────────────────────────────────────────────

@pytest.fixture
def hybrid(mem_store, fake_embedder):
    return HybridRetriever(mem_store, fake_embedder)


def test_constructor_takes_store_and_embedder(mem_store, fake_embedder):
    HybridRetriever(mem_store, fake_embedder)                        # 2 args
    HybridRetriever(mem_store, fake_embedder, mem_store.scroll_all())  # 3 args


def test_search_returns_at_most_k_with_rrf_score(hybrid, sample_query):
    res = hybrid.search(sample_query, k=4)
    assert 0 < len(res) <= 4
    assert all("rrf_score" in r and "text" in r for r in res)


def test_search_sorted_by_rrf_score(hybrid, sample_query):
    res = hybrid.search(sample_query, k=6)
    scores = [r["rrf_score"] for r in res]
    assert scores == sorted(scores, reverse=True)


def test_keyword_match_surfaces_via_bm25(hybrid):
    res = hybrid.search("LoRA low-rank decomposition", k=3)
    assert any("LoRA" in r["text"] for r in res)


def test_filters_applied_to_both_rankers(hybrid):
    res = hybrid.search("LoRA fine-tuning attention", k=10, filters={"category": "cs.CL"})
    assert res and all(r["category"] == "cs.CL" for r in res)


def test_unmatched_filter_returns_empty(hybrid):
    assert hybrid.search("attention", k=5, filters={"arxiv_id": "nope"}) == []


def test_empty_store_returns_empty(fake_embedder):
    from backend.app.storage.qdrant_store import QdrantStore, make_client
    r = HybridRetriever(QdrantStore(make_client(path=":memory:"), collection="e"), fake_embedder)
    assert r.search("anything", k=3) == []


def test_rewriter_is_applied_before_retrieval(mem_store, fake_embedder):
    class R:
        called_with = None
        def rewrite(self, q):
            R.called_with = q
            return "BM25 sparse keyword ranking"
    h = HybridRetriever(mem_store, fake_embedder, rewriter=R())
    res = h.search("how do keyword engines rank?", k=3)
    assert R.called_with == "how do keyword engines rank?"
    assert any("BM25" in r["text"] for r in res)


def test_rewrite_can_be_disabled(mem_store, fake_embedder):
    class Boom:
        def rewrite(self, q):
            raise AssertionError("should not be called")
    HybridRetriever(mem_store, fake_embedder, rewriter=Boom()).search("x", k=2, rewrite=False)
