"""
evaluate_llm.py
---------------
Evaluates LocalLLM across 4 dimensions:
  1. Relevance      - does the answer address the question?
  2. Faithfulness   - does it stick to the context, no hallucination?
  3. Conciseness    - is the answer focused, not padded?
  4. Latency        - how fast is the response?

Scoring: each dimension 0-3, total = 12 per test case.
"""

import sys
import os
import time
import json
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.llm.local_llm import LocalLLM


# ── Test Cases ────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "id": 1,
        "topic": "Meta-learning basics",
        "context": (
            "Meta-learning, or 'learning to learn', refers to algorithms that improve "
            "their learning efficiency across tasks. MAML (Model-Agnostic Meta-Learning) "
            "optimizes for fast adaptation by computing gradients of gradients. "
            "Prototypical Networks use embedding spaces to classify new examples from "
            "few shots by comparing distances to class prototypes."
        ),
        "question": "What is meta-learning and name two approaches?",
        "reference": "Meta-learning improves learning efficiency across tasks. Two approaches are MAML and Prototypical Networks.",
    },
    {
        "id": 2,
        "topic": "Faithfulness / hallucination test",
        "context": (
            "Transformer models use self-attention mechanisms to relate tokens across "
            "a sequence. BERT is pre-trained using masked language modeling. "
            "GPT models are trained autoregressively to predict the next token."
        ),
        "question": "Does the context mention reinforcement learning?",
        "reference": "No, the context does not mention reinforcement learning.",
    },
    {
        "id": 3,
        "topic": "Conciseness test",
        "context": (
            "Gradient descent updates model parameters by computing the gradient of "
            "the loss with respect to each parameter and moving in the negative "
            "gradient direction. The learning rate controls the step size."
        ),
        "question": "In one sentence, what does the learning rate control?",
        "reference": "The learning rate controls the step size during gradient descent parameter updates.",
    },
    {
        "id": 4,
        "topic": "Out-of-context robustness",
        "context": (
            "This paper studies sample efficiency in model-based reinforcement learning. "
            "The proposed method reduces environment interactions by 40% on MuJoCo benchmarks."
        ),
        "question": "What is the boiling point of water?",
        "reference": "The context does not contain information about the boiling point of water.",
    },
    {
        "id": 5,
        "topic": "Multi-fact synthesis",
        "context": (
            "Few-shot learning aims to generalize from very few labeled examples. "
            "Data augmentation, transfer learning, and metric learning are three "
            "common strategies. Metric learning trains an embedding where similar "
            "classes are close and dissimilar classes are far apart."
        ),
        "question": "List the strategies for few-shot learning and explain metric learning.",
        "reference": (
            "The three strategies are data augmentation, transfer learning, and metric learning. "
            "Metric learning trains embeddings so similar classes are close and dissimilar ones are far apart."
        ),
    },
]

# ── Prompts ───────────────────────────────────────────────────────────────────

RAG_PROMPT = """Answer the question using ONLY the context below.
If the answer is not in the context, say so clearly.

Context: {context}

Question: {question}
Answer:"""

# Short prompt so judge response fits within max_tokens=80 and low timeout
JUDGE_PROMPT = """Score this Q&A. Output ONLY a JSON object, no other text.

Q: {question}
Expected: {reference}
Got: {answer}

Score 0-3 each: relevance, faithfulness, conciseness. Add a one-line comment.
Example: {{"relevance":3,"faithfulness":2,"conciseness":3,"comment":"good but slightly verbose"}}"""


# ── Core Functions ────────────────────────────────────────────────────────────

def run_rag_query(llm, context, question):
    prompt = RAG_PROMPT.format(context=context, question=question)
    start = time.time()
    answer = llm.generate(prompt)
    latency = round(time.time() - start, 2)
    return answer, latency


def judge_answer(judge, question, reference, answer):
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)
    raw = judge.generate(prompt)
    try:
        # Extract JSON even if model wraps it in markdown fences
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON found")
        return json.loads(match.group())
    except Exception:
        print(f"  [warn] Judge returned non-JSON: {raw[:100]}")
        return {"relevance": 0, "faithfulness": 0, "conciseness": 0, "comment": "parse error"}


def latency_score(latency):
    if latency < 5:  return 3
    if latency < 15: return 2
    if latency < 30: return 1
    return 0


def print_result(case, answer, scores, latency):
    ls = latency_score(latency)
    total = scores["relevance"] + scores["faithfulness"] + scores["conciseness"] + ls
    print(f"\n{'='*60}")
    print(f"Test {case['id']}: {case['topic']}")
    print(f"{'='*60}")
    print(f"Q:        {case['question']}")
    print(f"Answer:   {answer[:300]}{'...' if len(answer) > 300 else ''}")
    print(f"Latency:  {latency}s  (score: {ls}/3)")
    print(f"Scores -> Relevance: {scores['relevance']}/3 | "
          f"Faithfulness: {scores['faithfulness']}/3 | "
          f"Conciseness: {scores['conciseness']}/3")
    print(f"Comment:  {scores.get('comment', '-')}")
    print(f"TOTAL:    {total}/12")
    return total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n[EVAL] LLM Evaluation - Autonomous Research Agent")
    print("Model: mistral via Ollama\n")

    # Answer LLM: generous timeout for full responses
    llm = LocalLLM(model="mistral", temperature=0.3, max_tokens=300, timeout=180)

    # Judge LLM: only outputs a tiny JSON blob so max_tokens is tight
    judge = LocalLLM(model="mistral", temperature=0.0, max_tokens=80, timeout=120)

    totals = []
    all_latencies = []

    for case in TEST_CASES:
        print(f"\n[{case['id']}/{len(TEST_CASES)}] Running: {case['topic']}...")

        # Step 1: Generate answer
        answer, latency = run_rag_query(llm, case["context"], case["question"])
        all_latencies.append(latency)
        print(f"  Answer received in {latency}s. Judging...")

        # Step 2: Judge the answer
        scores = judge_answer(judge, case["question"], case["reference"], answer)

        # Step 3: Display
        total = print_result(case, answer, scores, latency)
        totals.append(total)

    # ── Summary ───────────────────────────────────────────────────────────────
    avg = round(sum(totals) / len(totals), 2)
    avg_latency = round(sum(all_latencies) / len(all_latencies), 2)
    rating = "Good" if avg >= 9 else "Acceptable" if avg >= 6 else "Needs Work"

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    for case, score in zip(TEST_CASES, totals):
        bar = "#" * score + "." * (12 - score)
        print(f"  Test {case['id']} [{bar}] {score}/12 - {case['topic']}")
    print(f"\n  Average Score:   {avg}/12")
    print(f"  Average Latency: {avg_latency}s")
    print(f"  Rating:          {rating}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()