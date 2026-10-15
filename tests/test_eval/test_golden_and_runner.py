"""Tests for eval/golden.py, eval/runner.py (retrieval ablation with fakes) and eval/dashboard.py."""
import json

import pytest

from backend.app.retrieval.hybrid_retriever import HybridRetriever
from eval import dashboard, golden, runner


def case(**kw):
    base = dict(id="a1", question="q?", answerable=True, gold_chunk_ids=["x"])
    base.update(kw)
    return golden.GoldenCase(**base)


# ── golden set ───────────────────────────────────────────────────────────────

def test_valid_set_passes():
    golden.validate([case(id="a1"), case(id="u1", answerable=False, gold_chunk_ids=[])])


def test_duplicate_ids_rejected():
    with pytest.raises(golden.GoldenSetError, match="duplicate"):
        golden.validate([case(id="a1"), case(id="a1")])


def test_answerable_needs_gold():
    with pytest.raises(golden.GoldenSetError, match="needs gold"):
        golden.validate([case(gold_chunk_ids=[])])


def test_unanswerable_must_not_have_gold():
    with pytest.raises(golden.GoldenSetError, match="no gold"):
        golden.validate([case(answerable=False)])


def test_empty_question_rejected():
    with pytest.raises(golden.GoldenSetError):
        golden.validate([case(question="  ")])


def test_roundtrip_and_filters(tmp_path):
    p = str(tmp_path / "g.json")
    golden.save([case(id="a1", reviewed=True, regression=True), case(id="a2")], p, backup=False)
    assert len(golden.load(p)) == 2
    assert [c.id for c in golden.load(p, only_reviewed=True)] == ["a1"]
    assert [c.id for c in golden.load(p, regression_only=True)] == ["a1"]


# ── retrieval ablation ───────────────────────────────────────────────────────

class RR:
    def rerank(self, q, docs, top_k=5):
        return sorted(docs, key=lambda d: "LoRA" not in d["text"])[:top_k]


class P:
    pass


@pytest.fixture
def pipe(mem_store, fake_embedder):
    p = P()
    p.retriever = HybridRetriever(mem_store, fake_embedder)
    p.reranker = RR()
    return p


def test_eval_retrieval_reports_all_configs(pipe, mem_store):
    gold = next(d for d in mem_store.scroll_all() if "LoRA" in d["text"])
    cases = [golden.GoldenCase(id="a1", question="LoRA low-rank decomposition matrices",
                               gold_chunk_ids=[gold["chunk_id"]], gold_arxiv_ids=[gold["arxiv_id"]]),
             golden.GoldenCase(id="u1", question="irrelevant", answerable=False)]
    rep = runner.eval_retrieval(pipe, cases, k=3)
    assert set(rep) == {"dense", "bm25", "hybrid", "hybrid+rerank"}
    assert rep["hybrid"]["chunk_recall@3"] == 1.0 and rep["hybrid"]["chunk_mrr"] > 0
    assert rep["bm25"]["paper_recall@3"] == 1.0
    assert all(v["n"] == 1 for v in rep.values())      # unanswerable cases excluded


def test_unknown_config_raises(pipe):
    with pytest.raises(ValueError):
        runner.retrieve_for_config(pipe, "nope", "q", 3)


def test_composite_is_mean_of_headline_metrics():
    gen = {"metrics": {"+guard (full)": {"faithfulness": 0.9, "correctness": 0.8,
                                         "completeness": 0.7, "hallucination": 0.1}}}
    assert runner.composite(gen) == pytest.approx(0.8)
    assert runner.composite({"metrics": {}}) is None


# ── dashboard ────────────────────────────────────────────────────────────────

def test_dashboard_embeds_report_and_escapes_script_tags(tmp_path):
    rep = tmp_path / "latest.json"
    rep.write_text(json.dumps({"timestamp": "t</script><b>", "k": 5, "n_cases": 1,
                               "n_answerable": 1, "n_unanswerable": 0}))
    out = dashboard.build(str(rep), str(tmp_path / "d.html"))
    html = open(out, encoding="utf-8").read()
    assert "</script><b>" not in html.split("<script>")[1]      # injected closing tag neutralised
    assert "const R = {" in html


def test_dashboard_without_report(tmp_path):
    out = dashboard.build(str(tmp_path / "missing.json"), str(tmp_path / "d.html"))
    assert "const R = null" in open(out, encoding="utf-8").read()
