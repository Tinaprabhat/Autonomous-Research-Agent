# Known limits

Honest list of what this system does not do well. Numbers come from runs on this repo's corpus
(see `eval/reports/latest.json` once generated).

## The citation guard filters, it does not guarantee
Measured on a human-labelled sample (info.md section 11.4, 11.5): 5 of 30 unguarded answers (17%) contained a claim the
cited passages do not support, and the best guard variant caught 2 of the 4 real unsupported sentences. Both the LLM
verifier caught two of them (an earlier local NLI model, since removed, caught the same two) and missed an unstated
inference and a conflation of two facts.
The guard is layered: a free screen (overlap floor, numbers/acronyms/names must occur in the cited chunks), then an
entailment check (one batched LLM call; there is no fallback verifier). If the LLM check cannot run, sentences are kept
but flagged `lexical_only` with a visible warning. Caveats: the LLM verifier is the generator's own model family; thresholds
were set on the same 40 labelled sentences (in-sample); the mutation stress set is mechanical; the verifier adds an API call
per answer (estimated, not measured). A hedge-only answer is returned as `ok` (`hedged: true`) and tracked as a possible
false refusal; the false-refusal rate on answerable questions has not been measured yet.

## (history, NLI guard v1, since removed) Verification is strict, so it over-abstains on abstractive answers
The original NLI guard asked "does the cited passage *entail* this sentence?". Models paraphrase and combine facts
across passages, and NLI scores long multi-fact sentences near zero even when a human would call them
supported. Observed during development (12-paper corpus, 4 answerable + 1 off-corpus question): the first NLI model
(`cross-encoder/nli-deberta-v3-base`) scored almost every correct sentence ≈0.00 — it abstained on 3 of the 4
answerable questions and kept only half the sentences on the fourth. Switching to a FEVER/ANLI-trained model,
windowing premises into 3-sentence spans and prompting for short single-claim sentences gave 3 of 4 answerable
questions answered with faithfulness 1.0 (the fourth abstained because the model added claims the passages don't
make) and the off-corpus question correctly abstained. This is a tiny sample, not a benchmark — the golden-set
run is the real measure. Expect false "not sure" tags on correct answers; the trade-off is deliberate (a tagged answer is cheaper than a
confident wrong citation, and the user still gets the answer and its sources). The threshold (0.5) is tunable in `citation_guard.check_citations`.
Cross-sentence claims (a fact supported only by two different passages) are scored per cited chunk, not jointly.

## Tagged answers can still be wrong or misleading
A `low_confidence` answer is the model's unverified output. The tag and the source list are the safeguard; users who ignore
the tag can be misled. Off-corpus questions now get a tagged answer (usually "the sources don't contain this") rather than a
refusal. Evaluation counts a tagged answer to an unanswerable question as handled, and a tagged answer to an answerable
question as *not* answered.

## Grounded does not mean on-topic
The guard checks that an answer is *supported by a retrieved passage*, not that the question is within the corpus's
subject. Asking "What is the capital of France?" returned a correct, cited "Paris" because one paper uses that sentence
as an illustrative example. The system answered faithfully from the corpus; it has no topical relevance gate.
Generation runs at temperature 0 so abstain/answer decisions don't flip between identical requests.

## Prompt-injection defence is layered, not complete
* Query side: regex + a classifier. Paraphrased attacks that avoid trigger phrases can still pass the regex; the
  classifier catches most but not all (a subtle "Note to the AI assistant…" payload scored 0.005 in testing).
* Paper side: only chunks that *retrieve* are screened, regex-first. The classifier is deliberately conservative on
  paper text (≥0.999, prose only) because at 0.85 it flagged ~4% of benign chunks (numeric tables, reference
  lists). That means a well-disguised injection in a paper is likelier to get through than one in a query.
  Defence in depth: the prompt marks sources as untrusted, outputs pass the citation guard, and the output guard
  blocks credential-shaped strings. Treat the corpus as trusted-ish (curated arXiv harvest), not hostile.
* A paper that legitimately *discusses* prompt injection ("ignore previous instructions") will have those chunks
  quarantined.

## Judge bias
The generation metrics need an LLM judge (RAGAS now; DeepEval was dropped, info.md 11.8). We use Groq (`openai/gpt-oss-120b`) — the same provider and family that generates the
answers — to avoid a second API key / cost. Mild self-preference bias is possible; absolute scores are
optimistic, relative comparisons (ablation, regressions) are more trustworthy. The hand-labelling CLI
(`eval/hand_label.py`) exists to spot-check the judge against human faithfulness verdicts.

## Evaluation scope
* The (old, DeepEval) generation numbers in `case_study.md` cover **9 answerable cases** (7 answered) because the judge's free-tier
  daily quota ran out; the other 71 answerable cases are not yet scored. Resume with `python -m eval.run_all --local`.
* Ground truth is machine-verified (NLI + a cross-checking LLM), not human-labelled; a few questions are mildly
  context-dependent, and `u020`-style "unanswerable" labels can be partially answerable. Relative comparisons are more
  trustworthy than absolute scores.
* On the cases measured, the citation guard showed no faithfulness gain over the unguarded answer (both 1.0 at n=7) —
  it costs recall (78% answered) and ~6 s per query; its value is protection against the ungrounded/adversarial
  cases the sample does not stress.

## Latency
Measured on 2026-10-08 with the old local-NLI guard, uncontended on a CPU laptop: median ≈ 8.6 s, worst ≈ 15 s (NLI
verification ≈ 6 s median, the bottleneck; injection screen ≈ 1.2 s). The NLI model has since been removed; the guard now
adds one LLM call (about the cost of the generation call) plus negligible CPU work, but end-to-end latency has not been
re-measured.

## Golden set
`eval/build_golden_set.py` drafts questions from corpus chunks with the gold chunk set to the source chunk.
Other chunks may also answer the question, so chunk-level recall@k is a **lower bound**; paper-level recall is also
reported. Drafts are `reviewed: false` until a human checks them; the runner only scores reviewed cases by default.

## Parsing
* Section detection is heuristic (font size / bold / known names). Two-column layouts, unusual templates and
  scanned PDFs degrade it; non-canonical headings are stored as-is, so the "section" filter works best for
  standard names (Introduction, Method, Experiments…).
* Back-matter heuristic: text that follows a *Conclusion* heading is relabelled `Appendix` once the Conclusion exceeds
  ~3,500 characters, and any unrecognised heading after the Conclusion is also `Appendix`. This fixed appendices being
  mislabelled `Conclusion` (16% → 3.5% of chunks), but a long genuine conclusion could be cut short, and `Appendix` is now
  a large (~27%) catch-all. (The one-off script that re-labelled stored chunks without re-embedding was removed in the 2026-10-15 cleanup.)
* Math and tables extract poorly (symbols become `?`/garbled glyphs, tables become number soup). Numeric table
  chunks are stored but rarely useful; they are excluded from the prose-only injection classifier.
* OCR needs the Tesseract binary (installed in the Docker image; not on a typical Windows dev box). Without it,
  image-only PDFs fail with a clear `OcrUnavailable` error and are recorded as `failed` in the job queue.
* The References section is dropped at chunk time.

## Retrieval
* BM25 is held in memory, built from a Qdrant scroll at startup (fine at ~40k chunks; refresh via restart).
* The BM25 side applies filters by first fetching the allowed chunk ids from Qdrant; for very broad filters on a
  large collection that scroll would become the bottleneck.
* Embedded-local Qdrant (dev default) does not use payload indexes; index creation/verification is only
  meaningful against Docker / Qdrant Cloud.
* Single embedding model (`all-MiniLM-L6-v2`, 384-d); no multilingual support.

## Groq free-tier quotas are the binding capacity limit
Measured on this account (2026-10-08): every Groq model has its **own 200,000-token DAILY cap** (plus ~8,000 tokens/min
and 1,000 requests/day). One answered query costs roughly 1.5-2.5k tokens (5 passages + answer + reasoning), so a single
model serves only ~100 queries/day. The router therefore chains `gpt-oss-120b → gpt-oss-20b → qwen3.8-27b → Ollama`
(`GROQ_FALLBACK_MODELS`), each with its own quota, for roughly 3x headroom; every answer still passes the same citation
guard, but quality may drop on the smaller models. On a Hugging Face Space there is no Ollama, so once all three Groq
models are exhausted queries fail with a visible error until the quotas roll over. For more than a handful of daily
users, use a paid Groq tier. The same quotas make the RAGAS run slow/staged: it costs ~15-20k judge tokens per case,
so `eval/` checkpoints every finished case and resumes the next day (`python -m eval.run_all --local`).

## Operations
* Rate limiting is per-process in memory (resets on restart; one worker). Fine for one HF Space container.
* Streaming shows an unverified draft before the verified answer replaces it; users can briefly see text that is
  later removed. This is labelled in the UI.
* Qdrant Cloud free clusters pause after ~7 days idle and have no backup SLA; recovery path is re-ingestion from
  `data/manifest.json`. HF CPU Basic cold-starts are slow (model load ≈ 1–2 min).
* Local CPU latency is now dominated by the jailbreak classifier and reranker passes, plus the LLM calls (see `/metrics` p50/p95).
* Auth is optional. With `API_KEYS` set, every endpoint except `/`, `/health` and the docs needs `X-API-Key` (or
  `Authorization: Bearer`); each user has a per-minute window and a per-UTC-day query quota, and bad-key attempts are
  throttled per IP. Keys are static strings from the environment (no signup, rotation or revocation without a restart),
  and the limiter state is in memory, so quotas reset on restart and are not shared between workers. With `API_KEYS`
  empty the API is public and only the per-IP limit and query-length cap apply.
* The static UI (`/`) is served without a key; the key is entered in the browser and stored in `localStorage`.
* Groq model availability changes: the defaults in `config.py` were verified against the account's model list on
  2026-10-08 (the originally planned `llama-3.3-70b-versatile` returned 404 for this key).

## Not implemented (explicit non-goals at this scale)
Object storage, vector quantization, admin flow for new corpora, second dataset, backups — see the plan.
