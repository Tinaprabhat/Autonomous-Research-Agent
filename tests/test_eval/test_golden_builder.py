"""Tests for the automatic golden-set builder's filters/checks (no network, no models)."""
import json

import pytest

from eval import golden
from eval.build_golden_set import (
    CONTEXT_LEAKS, build_answerable, build_unanswerable, is_prose, jaccard, jparse)

PROSE = ("Alpha MAML adapts the inner-loop learning rate and the meta-learning rate during training, "
         "which removes much of the hyper-parameter tuning that vanilla MAML requires and speeds up "
         "convergence on few-shot image recognition benchmarks such as Omniglot and Mini-ImageNet, and the "
         "authors also report lower variance across random seeds when comparing against tuned baselines.")


def chunk(i=0, section="Method", text=PROSE, aid="2401.00001"):
    return {"text": text, "chunk_id": f"{aid}_v1_{i}", "arxiv_id": aid, "section": section, "page": 3, "title": "T"}


@pytest.mark.parametrize("q", [
    "Which referenced work studies a transductive setting?",
    "What metrics assess the retrieval model in the described paradigm?",
    "What does this paper propose?", "What does the passage say about X?",
    "What does reference [2] study?", "According to the authors, why does it work?",
])
def test_context_dependent_questions_rejected(q):
    assert CONTEXT_LEAKS.search(q)


@pytest.mark.parametrize("q", [
    "How does Alpha MAML adapt the inner-loop learning rate?",
    "What split did the BBBP dataset use in Relative Molecule Self-Attention?",
])
def test_standalone_questions_pass_filter(q):
    assert not CONTEXT_LEAKS.search(q)


def test_is_prose_filters_tables_garble_and_back_matter():
    assert is_prose(chunk())
    assert not is_prose(chunk(text="62.21% 65.93% 67.25% " * 30))
    assert not is_prose(chunk(section="Appendix"))
    assert not is_prose(chunk(text=PROSE.replace("Alpha", "?") + " ? ? ?"))
    assert not is_prose(chunk(text="Figure 3: " + PROSE))


def test_jaccard_and_jparse():
    assert jaccard("a b c", "a b c") == 1.0 and jaccard("a b", "c d") == 0.0
    assert jparse('noise {"question": "q", "answer": "a"} tail') == {"question": "q", "answer": "a"}
    with pytest.raises(ValueError):
        jparse("no json here")


class ScriptLLM:
    def __init__(self, replies):
        self.replies = list(replies)
    def generate(self, prompt):
        return self.replies.pop(0)


def both(replies):
    """One scripted LLM playing both generator and checker (replies interleave in call order)."""
    llm = ScriptLLM(replies)
    return llm, llm


class Verifier:
    def __init__(self, p): self.p = p
    def entailment_batch(self, pairs): return [self.p] * len(pairs)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)


Q = json.dumps({"question": "How does Alpha MAML adapt the meta-learning rate during training?",
                "answer": "It adapts both learning rates during training."})
OK = json.dumps({"answerable_from_passage": True, "answer_correct_and_supported": True, "standalone": True})


def test_answerable_case_kept_when_all_checks_pass():
    cases = build_answerable(ScriptLLM([Q]), ScriptLLM([OK]), Verifier(0.9), [chunk()], 1, lambda m: None)
    assert len(cases) == 1 and cases[0].gold_chunk_ids == ["2401.00001_v1_0"]
    assert cases[0].verification == "auto" and cases[0].regression and cases[0].source_page == 3


def test_answerable_rejected_by_nli():
    assert build_answerable(ScriptLLM([Q]), ScriptLLM([]), Verifier(0.1), [chunk()], 1, lambda m: None) == []


@pytest.mark.parametrize("bad", [
    {"answerable_from_passage": False, "answer_correct_and_supported": True, "standalone": True},
    {"answerable_from_passage": True, "answer_correct_and_supported": False, "standalone": True},
    {"answerable_from_passage": True, "answer_correct_and_supported": True, "standalone": False},
])
def test_answerable_rejected_by_llm_crosscheck(bad):
    assert build_answerable(ScriptLLM([Q]), ScriptLLM([json.dumps(bad)]), Verifier(0.9), [chunk()], 1, lambda m: None) == []


def test_llm_garbage_does_not_crash_the_run():
    cases = build_answerable(*both(["not json", Q, OK]), Verifier(0.9),
                             [chunk(0), chunk(1, aid="2401.00002")], 1, lambda m: None)
    assert len(cases) == 1


def test_at_most_two_cases_per_paper():
    qs = ["How does Alpha MAML adapt the meta-learning rate during training?",
          "Which benchmarks demonstrate faster convergence for Reptile with warm restarts?",
          "What seed variance reduction does Prototypical Networks report against baselines?"]
    replies = []
    for q in qs:
        replies += [json.dumps({"question": q, "answer": "x"}), OK]
    chunks = [chunk(i) for i in range(3)]          # same paper
    assert len(build_answerable(*both(replies), Verifier(0.9), chunks, 5, lambda m: None)) == 2


def test_duplicate_questions_dropped():
    replies = [Q, OK, Q, OK]
    cases = build_answerable(*both(replies), Verifier(0.9),
                             [chunk(0, aid="a.1"), chunk(1, aid="b.2")], 2, lambda m: None)
    assert len(cases) == 1


class Retriever:
    def search(self, q, k=6, rewrite=False):
        return [{"text": "some passage"}]


def test_unanswerable_kept_only_if_retrieval_cannot_answer():
    out = "What is the exact zero-shot accuracy of Alpha MAML on the MMLU benchmark at 0.5% data?\n" \
          "What is the exact latency of Reptile on the ImageNet-21k benchmark with 8 GPUs today?"
    replies = [out, json.dumps({"answerable": False}), json.dumps({"answerable": True})]
    cases = build_unanswerable(*both(replies), Retriever(), ["T"], 2, lambda m: None)
    assert len(cases) == 1 and not cases[0].answerable and cases[0].gold_chunk_ids == []


def test_golden_verified_filter_includes_auto_cases(tmp_path):
    p = str(tmp_path / "g.json")
    golden.save([golden.GoldenCase(id="a1", question="q?", gold_chunk_ids=["x"], verification="auto"),
                 golden.GoldenCase(id="a2", question="q2?", gold_chunk_ids=["y"]),
                 golden.GoldenCase(id="a3", question="q3?", gold_chunk_ids=["z"], reviewed=True)], p, backup=False)
    assert [c.id for c in golden.load(p, only_reviewed=True)] == ["a1", "a3"]


# ── resume + daily-quota behaviour ───────────────────────────────────────────

def test_checkpoint_persists_cases_and_tried_chunks(tmp_path):
    from eval.build_golden_set import Checkpoint
    path = str(tmp_path / "partial.jsonl")
    ck = Checkpoint(path)
    build_answerable(*both([Q, OK]), Verifier(0.9), [chunk()], 1, lambda m: None, ck)
    ck2 = Checkpoint(path)
    assert [c.id for c in ck2.cases] == ["a001"] and "2401.00001_v1_0" in ck2.tried


def test_resume_skips_tried_chunks_and_keeps_existing_cases(tmp_path):
    from eval.build_golden_set import Checkpoint
    path = str(tmp_path / "partial.jsonl")
    build_answerable(*both([Q, OK]), Verifier(0.9), [chunk(0)], 1, lambda m: None, Checkpoint(path))
    # second run wants 2 cases; chunk 0 is already done, only chunk 1 (another paper) may be spent
    q2 = json.dumps({"question": "Which benchmarks demonstrate faster convergence for Reptile with warm restarts?",
                     "answer": "x"})
    cases = build_answerable(*both([q2, OK]), Verifier(0.9), [chunk(0), chunk(1, aid="b.2")], 2,
                             lambda m: None, Checkpoint(path))
    assert [c.id for c in cases] == ["a001", "a002"]


def test_torn_checkpoint_line_is_ignored(tmp_path):
    from eval.build_golden_set import Checkpoint
    p = tmp_path / "partial.jsonl"
    p.write_text(json.dumps({"kind": "tried", "chunk_id": "x"}) + "\n" + '{"kind": "case", "ca')
    assert Checkpoint(str(p)).tried == {"x"}


def test_daily_quota_raises_budget_exhausted_without_losing_progress(tmp_path):
    from backend.app.llm.groq_llm import LLMRateLimitError
    from eval.build_golden_set import BudgetExhausted, Checkpoint

    class Quota(ScriptLLM):
        def generate(self, prompt):
            if not self.replies:
                raise LLMRateLimitError("TPD", retry_after=587, daily=True)
            return super().generate(prompt)

    path = str(tmp_path / "partial.jsonl")
    llm = Quota([Q, OK])
    chunks = [chunk(0), chunk(1, aid="b.2")]
    with pytest.raises(BudgetExhausted):
        build_answerable(llm, llm, Verifier(0.9), chunks, 5, lambda m: None, Checkpoint(path))
    assert len(Checkpoint(path).cases) == 1          # the first verified case survived


def test_per_minute_limit_waits_then_succeeds():
    from backend.app.llm.groq_llm import LLMRateLimitError
    from eval.build_golden_set import call

    class Flaky:
        n = 0
        def generate(self, p):
            Flaky.n += 1
            if Flaky.n == 1:
                raise LLMRateLimitError("TPM", retry_after=2, daily=False)
            return "ok"
    assert call(Flaky(), "x") == "ok" and Flaky.n == 2


# ── deterministic question gate ──────────────────────────────────────────────

@pytest.mark.parametrize("q", [
    "In few-shot learning, what aspect is revisited in the concurrent works [31, 27] with improved design?",
    "Why was the Estrogen?? dataset excluded from the experiments in the Relative Molecule Self-Attention study?",
    "Which method is mentioned in the paper as the strongest baseline for retrieval evaluation?",
    "Which approach is discussed in the related works as the baseline for cohort retrieval?",
    "What is the result?",
    "Tell me about meta-learning in detail please",
])
def test_question_gate_rejects(q):
    from eval.build_golden_set import question_ok
    assert not question_ok(q)


@pytest.mark.parametrize("q", [
    "Which three conferences concentrate the majority of meta-reinforcement learning contributions after 2016?",
    "What specific top-k metrics evaluate retrieval models for Large Language Model answer generation?",
])
def test_question_gate_accepts(q):
    from eval.build_golden_set import question_ok
    assert question_ok(q)


def test_resumed_cases_are_refiltered(tmp_path):
    from eval.build_golden_set import Checkpoint
    ck = Checkpoint(str(tmp_path / "p.jsonl"))
    ck.add_case(golden.GoldenCase(id="a001", question="What does reference [31, 27] study in this setting?",
                                  gold_chunk_ids=["x"], verification="auto"))
    ck.add_case(golden.GoldenCase(id="a002", question="Which benchmarks demonstrate Alpha MAML convergence speed gains?",
                                  gold_chunk_ids=["y"], verification="auto"))
    assert [c.id for c in Checkpoint(str(tmp_path / "p.jsonl")).cases] == ["a002"]
