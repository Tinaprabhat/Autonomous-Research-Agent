# Case study: a citation-grounded research assistant on free tiers

*Problem, solution, results, and everything that went wrong on the way.* Measured figures come from this repository's own
runs (2026-10-08 to 2026-10-16). Estimates are marked as estimates. The reference numbers live in `docs/info.md`; the weaknesses are listed in section 5 below.

**In one paragraph.** We built a question-answering app over 99 arXiv papers in which every claim carries a `[N]` citation and
a citation guard checks each cited sentence before the user sees it. Retrieval works well (recall@5 0.912, paper recall@5 1.000).
The guard turned out to be much weaker than we hoped: in a human-labelled sample, 5 of 30 raw answers (16.7%) contained an
unsupported claim, and after the guard 3 to 4 of the 25 answers returned without a "not sure" tag (12 to 16%) still did. Most of the work went
into finding that out honestly: replacing a guard that over-rejected, discovering that our evaluation judge scored everything 1.0, and
removing the NLI model completely. The system is useful and safe against mechanical errors. It is **not** proven to be more
faithful than the raw model.

---

## 1. The problem

**Task.** Let a researcher ask questions about about 100 arXiv papers and get an answer in which each claim can be traced to a passage.

**The failure we designed against.** Not "no answer", but a *confident answer with a plausible citation that does not support it*.
A wrong answer with a real-looking `[3]` is worse than no answer, because the citation invites trust.

**Hard constraints** (they shaped every decision):

| Constraint | Consequence |
|---|---|
| Near-zero budget: Groq free tier, Qdrant Cloud free cluster, CPU-only containers, and a US$100 AWS free credit for the demo host | 8,000 tokens/min and 200,000 tokens/day **per model**; about 100 answered queries/day/model (estimate) |
| Small, fixed corpus (99 papers, 19,116 chunks) | Cannot answer outside it; no admin flow for new papers |
| No GPU | Small local models only (MiniLM embedder, MiniLM reranker, DeBERTa-base classifier) |
| One owner doing the human labelling | 30 labelled answers; confidence intervals of roughly 4 to 35% |

**Version 1 and why it was not enough.** v1 was FAISS + a union of dense and BM25 results + a local LLM. Its "citation guard"
reused a relevance cross-encoder (so a relevant-but-wrong citation passed) and silently fell back to the raw answer when it failed.
Chunk ids were per-paper counters that collided across papers. It had no safety layers and no real evaluation.

---

## 2. The solution that was built (v2)

```
browser -> rate limit / auth -> input guard (regex + injection classifier)
        -> hybrid retrieve: dense (Qdrant) + BM25 -> RRF fusion (k=60) -> cross-encoder rerank (20 -> 5)
        -> screen retrieved text for injection -> generate (Groq chain -> Ollama), retry once
        -> citation guard (citation check -> overlap + specifics -> LLM entailment)
        -> output guard (secrets blocked, PII redacted) -> answer + sources
```

| Area | v1 | v2 |
|---|---|---|
| Chunk identity | per-paper counter (collides) | `{arxiv_id}_{version}_{idx}` + page, section, character offsets |
| Chunking | fixed windows | semantic and section-aware, 500 characters, 70 overlap, References dropped |
| Store | FAISS flat | Qdrant: HNSW m=16 / ef_construct=100, 5 payload indexes, server-side filters |
| Fusion | union of dense and BM25 | Reciprocal Rank Fusion + cross-encoder rerank |
| Verification | relevance score, raw-answer fallback | layered guard (below), visible warning when it cannot run |
| Failure policy | silent | retry once, then a visible `unverified` warning; Qdrant down gives 503, never an invented answer |
| Safety | none | injection screen on queries *and* retrieved paper text; secret blocking and PII redaction on output |
| Ingestion | one script | SQLite job queue (retries, resumable, version-aware), OCR fallback |
| Access | none | API-key auth, per-user per-minute and per-day limits |
| Operations | none | Qdrant snapshot + a *tested* restore, failure-injection tests, a live end-to-end test |

**The guard as shipped:** per sentence, (0) pass honest hedges ("the sources do not say..."); (1) require a valid `[N]`; (A) require
that at least 30% of the sentence's content words, and **all** of its numbers, acronyms and names, appear in the cited chunk(s);
(B) one batched LLM call asks whether the cited passages state or directly imply each surviving claim. If more than half the
sentences are dropped, the answer is still shown, tagged "Not sure". If the LLM cannot run, the answer is shown with a visible
`lexical_only` warning. There is no fallback verifier.

---

## 3. The journey: what went wrong, and what each failure taught

Problems are in roughly the order we hit them. Each has the symptom, the cause, the fix and what we would tell someone else.

### 3.1 Early build problems

| # | Problem | Cause and fix | Lesson |
|---|---|---|---|
| 1 | **The first NLI model rejected nearly everything.** `cross-encoder/nli-deberta-v3-base` scored correct multi-fact sentences near 0.00; on the 12-paper prototype it abstained on 3 of 4 answerable questions | Strict entailment punishes paraphrase and fusion of facts. Switched to a FEVER/ANLI-trained model, 3-sentence premise windows and a "short single-claim sentences" prompt | Measure a verifier on real answers before trusting a threshold |
| 2 | **The injection classifier flagged about 4% of benign paper chunks** at the obvious 0.85 threshold (numeric tables, reference lists) | Found because chunks were quarantined in a live run. Queries stay at 0.85; retrieved paper text uses 0.999 on prose chunks only, with regex first | Query screening and document screening need different operating points |
| 3 | **Free-tier quotas were the real capacity limit.** The golden-set build died mid-run on the daily cap and **lost 60 verified cases that were held in memory** | Everything long (ingestion, golden build, evaluation) now checkpoints per item and resumes. The router chains `gpt-oss-120b` to `gpt-oss-20b` to `qwen3.8-27b` to Ollama and cools down rate-limited providers | Budget tokens like money; never keep hours of work in memory |
| 4 | **Evaluation bug:** scoring an abstention ("Not enough evidence...") as an unfaithful answer made the guard look *worse* than no guard | Abstentions are excluded from quality metrics and reported as a recall cost | A metric can punish the safe behaviour it should reward |
| 5 | **A test wrote to the production vector database.** An early smoke test ran with `.env` pointing at the cloud cluster | Fixed structurally: an explicit local path can never resolve to a remote, and tests force local mode. The later live test is read-only by design | Make the dangerous mistake impossible, not just unlikely |
| 6 | **Planned model and tooling did not exist as planned.** `llama-3.3-70b-versatile` returned 404 for our key; `guardrails-ai` Hub validators are not installable from PyPI | Defaults were verified against the account's real model list; the three guardrail checks were implemented locally with a Hugging Face classifier | Verify every planned dependency before building on it |
| 7 | **Section labels were wrong.** Appendices were being labelled `Conclusion` (16% of chunks) | Back-matter heuristic: text after a long Conclusion becomes `Appendix` (Conclusion fell to 3.5%). Side effect: `Appendix` is now a catch-all, about 27% of chunks | A fix can move the error somewhere else; document where |
| 8 | **Deployment detours.** Hugging Face Space files and a Render blueprint were added, then removed | Free hosts did not fit a 1.5 to 2 GB RAM, 3-minute-start service. The container was deployed to AWS ECS (ap-southeast-2) instead, paid from a US$100 free credit | Pick the host before tuning the memory profile; "free" hosting rarely fits a model-heavy service |

### 3.2 The evaluation story: how we learned the numbers were wrong

This is the longest part of the project, and the most valuable.

**Act 1: a flattering result (2026-10-08).** DeepEval with a Groq judge reported faithfulness **1.0**, answer relevancy 0.95, citation
correctness 0.90 on the cases that finished. The judge's daily quota ran out after only 9 of 20 planned cases, so n was tiny. At n=7
the guard showed no faithfulness gain over the unguarded answer and cost recall (7 of 9 answered). We wrote that down and moved on,
but "faithfulness 1.0" should have been a warning in itself.

**Act 2: asking whether the judge works (2026-10-13).** The owner hand-labelled all 30 cached model answers, one at a time, without seeing any
scores: 25 faithful, **5 unfaithful** (a023, a037, a047, a048, a053). Against those labels the DeepEval judge scored **all 30
answers 1.0**. It passed 5 of 5 unfaithful answers; Cohen's kappa was 0.0, exactly "always pass". Every earlier faithfulness figure
therefore carried no evidence of faithfulness. A second, independent labelling by an LLM (Claude, blind to the scores) agreed with
the human on only 80% of answers (kappa 0.14) and was clearly more lenient, passing four of the five answers the human rejected. LLM
labels cannot replace human ones for this task.

**Act 3: diagnosing DeepEval.** Two causes:
1. *A configuration bug of ours:* by default DeepEval counts "borderline" claims as faithful (`penalize_ambiguous_claims=False`), and we
   never set it. Fixing it moved flagged unfaithful answers from 1 of 5 to 2 of 5.
2. *Judge leniency and noise that no setting fixes:* on three of the five unfaithful answers the judge marked every claim "yes", and
   the same answer scored 1.0 in one run and 0.667 in another at temperature 0.

We had set the rule in advance: fix it or drop it. The fix was not enough, so DeepEval was removed from the code, CI gate and requirements.

**Act 4: RAGAS (2026-10-14).** We adopted RAGAS 0.4.3 (judge `gpt-oss-120b`) for faithfulness, hallucination (1 minus faithfulness),
completeness and correctness. Getting it to run was its own saga:
* RAGAS 0.4.3 would not import (`langchain_community.chat_models.vertexai` no longer exists). A stub is installed for that one module
  in the evaluation code only; a regression test imports RAGAS for real and *fails* rather than skips if it breaks.
* Reasoning effort mattered: at `low` the judge missed an invented claim, at the default it caught it.
* Our first adapter had three bugs, found by running it: a new event loop per call broke the HTTP client, a score cache keyed on `id()`
  could return another case's scores, and a stray backspace character from a scripted edit silently broke a hedge regex.
* Cost: about 10 to 13k judge tokens per case, so 30 cases is about 300k tokens against a 200k/day, 8k/min limit. Runs are cached and resume,
  and stop cleanly when the daily quota is gone.

**Act 5: does RAGAS fix the problem? Partly.** On the 30 raw answers it scored only **2 of 5** unfaithful answers below 1.0 (a023 = 0.75,
a053 = 0.90), falsely flagged 3 of 25 faithful ones (one correct one-sentence answer scored 0.0), and had AUROC 0.63 (CI 0.42 to 0.87,
includes chance). It removes the "always 1.0" symptom but is not a validated faithfulness detector, because an LLM judge from the generator's own family
approves subtle over-reach. RAGAS itself is also non-deterministic: the same answer scored 0.90 and 0.71 at temperature 0.

**Act 6: a measurement artefact (2026-10-15).** Our first full pass gave **12 of 30 cases an exact 0.0 for both correctness and completeness**,
including plainly correct answers. RAGAS recall counts only *response* claims that the reference supports; if none is, recall is 0 even when
the response covers every reference claim. Our references are terse extracts, and good answers restate the question's context, so the
strict check marked them unsupported. We redefined completeness as *reference-claim coverage* (0.589 became 0.783) and correctness as the F1 of
precision and coverage. The original RAGAS recall is stored for transparency. Correctness (0.532) is still an **underestimate**: precision against
a terse reference still has 12 exact zeros.

**Final RAGAS numbers (30 guarded responses, one judge):**

| Metric | Score | 95% CI |
|---|---|---|
| Faithfulness | 0.937 | 0.862 to 0.995 |
| Hallucination | 0.063 | 0.005 to 0.138 |
| Completeness | 0.783 | 0.650 to 0.900 |
| Correctness (lower bound) | 0.532 | 0.372 to 0.708 |
| Responses with an unsupported statement | 4 of 30 (13.3%) | 5.3 to 29.7% |

**What we conclude about evaluation.** The reliable results are the *retrieval metrics* and the *human-labelled error rates*. The LLM-judge
scores are indicative. The best single decision of the project was labelling by hand before trusting any judge.

### 3.3 The guard story: from NLI to "no NLI at all"

**Guard v1 (NLI).** Each cited sentence was checked by a local NLI entailment model (DeBERTa trained on MNLI/FEVER/ANLI, threshold 0.5, 3-sentence
premise windows). Measured on the human labels:
* it **dropped 8 of 36 supported sentences (22%)**, some plainly worded ones scoring 0.001;
* it **kept 2 of the 4 truly unsupported sentences** (NLI scores 0.58 and 0.94) and caught 2 (0.07 and 0.04);
* the kept ones were an unstated inference and a conflation of two different uses of Spearman's coefficient. NLI does not catch claims that
  reuse the passage's words but extend or combine them;
* lowering the threshold fixed the false drops but the high-scoring unsupported sentences stayed in at every threshold below 0.6;
* it was also the latency bottleneck: about 6.1 s of an 8.6 s median query.
With only 4 unsupported sentences, "best threshold" was not statistically meaningful, so the threshold was not changed.

**A wrong turn: guard v2 (embedding cosine plus word overlap).** A faster replacement looked better (77.5% vs 75.0% accuracy). We then admitted
that its thresholds had been tuned on the very 40 sentences used to judge it, so the "win" meant nothing, and that cosine and word overlap
measure word reuse, not entailment, which is exactly how the unfaithful answers had slipped through. It was withdrawn.

**The stress test.** To get an evaluation that is not the tuning data, we mutated the 36 human-labelled *supported* sentences so that the
"unsupported" label holds by construction: number swap, antonym flip, negation, added qualifier, conflation with another paper's clause, wrong-chunk
citation (201 items). Findings:
* **Embedding cosine had no signal** (AUC 0.45, chance; real catch 0 of 4). Its only detections were wrong-chunk citations, which overlap catches
  100% of. Deleted.
* **Word overlap is weak exactly where it matters:** number swaps 9%, negations 28%, qualifiers 42%.
* **A "specifics" check is cheap and precise** (6% false drops; 100% of number swaps and foreign-claim conflations).
* **Neither NLI nor an LLM caught the subtle real errors:** both caught the *same* two of the four real unsupported sentences (a048#2, a053#2)
  and missed the inference (a023#1) and the conflation (a047#1). No combination tried caught more than 2 of 4.
* A bug the stress test found: the verifier's replies were truncated (the 400-token limit also covers hidden reasoning), so about 10% of batched calls
  returned unusable JSON. The budget was raised to 1,500 tokens; afterwards 0 of 21 calls failed.

**What shipped:** layer A (overlap floor 0.3 plus the all-specifics gate) followed by one batched LLM entailment call. On the stress set this gave 17%
false drops, 2 of 4 real errors and 100% of the synthetic sample; the NLI-based version had 19 to 25% false drops, 2 of 4 real, 89 to 95% synthetic.

**The NLI removal.** At first the NLI model was kept as a fallback for when the LLM verifier could not run. The owner then said: *"remove nli
completely, i dont want it as fallback."* The reasoning is sound: the NLI model caught exactly the same two real errors as the LLM, dropped 22% of good
sentences, made every query about 6 s slower on CPU, and added about 370 MB of model weights to the image. A fallback that is nearly as wrong as the
primary, and slow, is mostly a source of false confidence. We removed it from the runtime, configuration, Docker image, evaluation code and tests.
The frozen NLI scores stay in the data files as history only.

What this changed, honestly:
* **Gained:** a smaller image, no 6 s CPU step, fewer moving parts, and an honest failure mode.
* **Lost:** the only offline, zero-token verifier. When the LLM is unavailable (including free-tier quota exhaustion, the binding limit), answers
  fall to layer A alone, which caught **0 of 4** real unsupported sentences. They are shown with a visible `lexical_only` warning rather than as verified.
* **Kept an uncomfortable fact:** the LLM verifier is the generator's own model family, so there is now no independent check at all.

### 3.4 What the guard really does to answers

Replaying the 30 labelled raw answers through the shipped guard (twice, identical inputs, temperature 0):

| Measure (n = 30, Wilson 95% CI) | Result |
|---|---|
| Raw answers with an unsupported claim (before guard) | 5 / 30 = 16.7% (7.3 to 33.6%) |
| Error among answered responses (after guard) | 4 / 25 = 16.0% or 3 / 25 = 12.0% (CIs about 4 to 35%) |
| Answer rate | 25 / 30 = 83.3% (66.4 to 92.7%) |
| False-refusal rate | 1 / 30 = 3.3% (0.6 to 16.7%); a retrieval miss, not a guard decision |
| Errors the guard fixed or flagged | 1 to 2 of 5 |
| Correct answers wrongly tagged "not sure" | 4 / 25 = 16% |
| Supported sentences wrongly removed | 6 / 37 = 16% |
| Does the "not sure" tag track correctness? | **No.** All four tagged answers were clean; no untagged error was flagged; guard-score AUROC 0.49 |

(Note: 36 supported sentences were used for the NLI and stress-test analysis in section 3.3; the shipped-guard replay above covers 37.)

Findings: **the guard did not measurably reduce the error rate**; the two runs disagreed on one case (a048), so the verifier is not deterministic;
and a calibrated confidence label cannot be fitted on 3 to 5 erroneous answers.

A **production finding** from the replay: at 8,000 tokens/minute, the first replay hit rate limits so often that 7 of 30 cases silently degraded
to `lexical_only`. Those results were discarded and redone with pacing; the verifier now waits up to 15 s once for a per-minute limit (not a daily one).
Under real traffic some answers will carry the `lexical_only` warning.

### 3.5 Smaller bugs found by labelling and live testing

* *Sentence splitter* cut at "J.S. Bach" and "Data augm. + CL", producing fragments the guard would drop. Fixed (initials and abbreviations), 3 regression tests.
* *Honest non-answers were penalised.* "None of the provided sources contain..." was not recognised as a hedge, so it was tagged "not sure". Hedge patterns extended.
* *Router fall-through:* under token pressure the router served generation and verification from `gpt-oss-20b`, so the verifier is not always the model we evaluated.
* *Qdrant outage* used to produce an unhandled 500 and a crashing `/health`. Now 503 and `degraded`, with 16 failure-injection tests.
* *Memory:* the local server was killed once by the operating system when memory ran critically low (it needs 1.5 to 2 GB).

### 3.6 Housekeeping and security

* Repository clean-up (2026-10-15): kept only live system, testing, evaluation, guard and guardrails; **commented out** dead code (legacy `/ask`, unused cache helpers)
  rather than deleting; removed one-off scripts and duplicate snapshots; archived DeepEval-era scores so the dashboard could not mislabel them as RAGAS.
* A new opt-in **live end-to-end test** (real models, real Qdrant read-only, real Groq): **10 of 10 passed in 112 s**. Offline suite: **440 passed, 0 failed**.
* `.env` held live keys that were visible in a working session. We advised rotating them, hardened `.gitignore`, and scanned the working tree and entire git
  history for the real values (0 hits) before pushing.
* A Qdrant snapshot and a real **restore drill** passed (19,116 of 19,116 points identical; 20 of 20 queries returned the same top 5).
* The demo is deployed on AWS ECS from the repository's Dockerfile. Image size, task size and ECS latency have not been measured yet.

---

## 4. Effective results

### 4.1 What works well

| Area | Result |
|---|---|
| Corpus | 99 of 100 papers ingested (1 genuine arXiv 404); 19,116 chunks; 2014 to 2026 |
| **Retrieval (80 answerable questions, k=5)** | Hybrid + rerank: chunk recall@5 **0.912**, precision@5 0.183 (ceiling 0.20: one gold chunk per question), MRR **0.856**, paper recall@5 **1.000** |
| Ablation | Dense 0.750, BM25 0.900, hybrid 0.887 recall@5. RRF alone *lowers* MRR vs BM25 (0.722 vs 0.830); the reranker repairs it (0.856) |
| Unanswerable questions | 19 of 20 (95%) correctly refused (measured 2026-10-08 with the old guard; not re-run) |
| Mechanical errors | Layer A drops 100% of number swaps and foreign-claim conflations and all wrong-chunk citations in the stress set |
| Safety | Injection rejected before any retrieval or LLM call; credential-shaped output blocked; PII redacted (live-tested) |
| Reliability | LLM down gives `unverified` with a warning; Qdrant down gives 503; backups restore verified |
| Tests | 440 offline passed; 10 of 10 live; 16 failure-injection tests |
| Cost | LLM and vector database: $0 (free tiers); AWS ECS hosting covered by a US$100 free credit |

### 4.2 Where it falls short

* **Faithfulness is not proven.** See section 3.4: 16.7% raw error vs 12 to 16% after the guard, intervals overlapping heavily.
* **RAGAS says faithfulness 0.937 and hallucination 0.063**, but that judge flags only 2 of 5 known unfaithful answers (AUROC 0.63), so treat it as an optimistic indicator.
* **Completeness 0.783, correctness 0.532 (a lower bound).**
* **Latency.** Typical query about 9 to 13 s on a CPU laptop (LLM calls only 1.5 to 2 s of that; the rest is local CPU work and Qdrant Cloud round trips), with tails over 30 s. The earlier "8.6 s median" was measured with the retired NLI guard. Latency on the deployed ECS task has not been measured.

---

## 5. Limits of the offered solution

The ones that matter most:

| Limit | Why it matters |
|---|---|
| **The guard filters; it does not guarantee** | It catches wrong numbers, names and citations and contradictions, but not unstated inferences or conflations (it missed 2 of 4 real errors) |
| **No independent verifier** | The checker is the generator's model family; with NLI gone, nothing offline backs it up |
| **Degrades under load** | At 8,000 tokens/min the check often cannot run; answers then carry `lexical_only` |
| **Not deterministic** | Identical replays differed on one of 30 cases |
| **The "not sure" tag is noise** | 16% of correct answers tagged; none of the untagged errors flagged |
| **Free-tier capacity** | Roughly 300 answered queries/day across the model chain and 3 to 5 per minute (estimates); "100 users" works only at a few queries each |
| **Small evaluation** | 30 labelled answers, one labeller, 4 to 5 real errors; thresholds are in-sample; the golden set is machine-verified, not human-reviewed, and favours BM25 |
| **Judge bias** | Same provider and family generates and judges; scores likely optimistic |
| **Grounded is not on-topic** | "Capital of France" was answered "Paris" from an illustrative sentence in a paper |
| **Injection defence is layered, not complete** | Paraphrased attacks can pass the regex; paper-text classifier is deliberately conservative (0.999) |
| **Parsing** | Heuristic sections; maths and tables extract poorly; `Appendix` is a 27% catch-all; References dropped |
| **Operations** | In-memory limits (one worker); static API keys; manual snapshots; no quantization; no admin flow for new corpora; single ECS task funded by a US$100 credit |
| **Not measured** | Total ingestion time, clean-machine latency, latency and image size on ECS, the guarded pipeline on the other 50 answerable questions, human-judged relevance and completeness, any verifier model other than the primary |

---

## 6. What we would do next

1. Label a few hundred answers (not 30) so error rates, a calibrated confidence label and the guard's effect can be measured, not guessed.
2. Use an **independent** verifier (a different model family, or a human-reviewed subset) so the checker's mistakes are not the generator's.
3. Review the golden set by hand and add vaguer, real-world questions so retrieval is not flattered by BM25-friendly wording.
4. Run the guarded pipeline over all 80 answerable questions (needs a paid tier or several days of free quota).
5. Move to a paid Groq tier before more than a handful of daily users.

## 7. Lessons

1. **Label by hand before trusting any judge.** A judge that scores everything 1.0 looks like success.
2. **Beware tuning and testing on the same data.** With four real errors, any "win" is noise; build a stress set whose labels hold by construction.
3. **Remove what is not earning its place.** A fallback as wrong as the primary adds cost and false comfort.
4. **Fail visibly.** A warning is better than a silent "verified".
5. **Make dangerous mistakes impossible** (tests that cannot touch production data), and **checkpoint everything long-running.**
6. **Report the caveats next to the number.** The honest headline of this project is an error rate, not a score.
