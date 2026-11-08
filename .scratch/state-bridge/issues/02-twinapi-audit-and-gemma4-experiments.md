# 02 — TwinAPI Audit, Real Dual-Model Gemma 4 Matrix, and SSOT Compiler

**What to build:** Build the `TwinAPI` Stateful False-Friend Audit and 256-d embedding catalog (`scripts/build_twinapi_audit.py` → `artifacts/twinapi_embeddings_256d.json` and `artifacts/twinapi_audit.json`), run the real dual-model inference matrix (`scripts/run_gemma4_matrix.py` evaluating `Arm 2`, `Arm 3`, `Arm 4`, and `Arm 5` across all 30 workflows on real `google/models/gemma-4-31b-it` and `google/models/gemma-4-26b-a4b-it` with `max_tokens=8192`, `temperature=0.0`, `<thought>` extraction and token accounting, `min_interval=2.0s` pacing, and atomic `os.fsync` checkpointing into `artifacts/raw_llm_generations.jsonl`), and build the SSOT replay and statistical compiler (`scripts/build_ssot.py` → `artifacts/runs.jsonl`, `artifacts/metrics_ssot.json`, and `artifacts/explorer_bundle.json`).

**Blocked by:** 01 — StateBridge Core Kernel, 30-Workflow Benchmark, and PBT Suite

**Status:** `done`

**Needs:** Local Magpie gateway (`http://127.0.0.1:8317/v1`) serving `google/models/gemma-4-31b-it`, `google/models/gemma-4-26b-a4b-it`, and `google/models/gemini-embedding-001` (`dimensions=256`).

**Validation method:** Run `uv run --with requests==2.34.2 --with httpx==0.28.1 --with pytest python3 -m pytest tests/test_ssot_pipeline.py -v` and `uv run --with requests==2.34.2 --with httpx==0.28.1 python3 scripts/build_ssot.py --verify-replay`.

**Evidence required:** `artifacts/twinapi_audit.json` and `artifacts/twinapi_embeddings_256d.json` populated with real `requests==2.34.2` and `httpx==0.28.1` API pairs and 256-d vectors; `artifacts/raw_llm_generations.jsonl` containing all 240 real LLM generations (`30 workflows × 4 LLM arms × 2 models`) with non-empty prompts, raw responses, extracted code, and token counts; `artifacts/runs.jsonl` containing all 270 live Loopback HTTP execution receipts (`240 LLM runs + 30 Arm 1 deterministic AST codemod runs`) completed in `< 15 seconds`; `artifacts/metrics_ssot.json` containing per-arm and per-family solve rates, First-Divergence distributions, paired McNemar exact $p$-values, and 10,000-draw paired bootstrap 95% confidence intervals.

**Tests:** `required`

**Test seam:** `scripts/build_twinapi_audit.py`, `scripts/run_gemma4_matrix.py` (code extractor and checkpoint loader), and `scripts/build_ssot.py` public CLI/functions.

**Regression scope:** Ensure `<thought>[\s\S]*?</thought>` blocks and markdown code fences are cleanly stripped before AST parsing without losing valid Python code, truncated responses (`finish_reason == "length"` or unclosed `<thought>`) are detected and retried with backoff, and re-running `build_ssot.py` produces deterministic, identical `metrics_ssot.json` outputs from `raw_llm_generations.jsonl` — Source: Story 1, Story 2, Story 6.

- [x] [Contract] `scripts/build_twinapi_audit.py` inspects `requests==2.34.2` and `httpx==0.28.1`, computes normalized edit distance, parameter-masked signature isomorphism, and 256-d embeddings (`dimensions=256`), and writes `artifacts/twinapi_embeddings_256d.json` and `artifacts/twinapi_audit.json` — Source: Story 2
- [x] [Confirmation] `scripts/run_gemma4_matrix.py` strips `<thought>[\s\S]*?</thought>` before extracting Python code fences, records `thought_chars`/`reasoning_tokens` and `visible_completion_tokens` separately, paces calls (`max_concurrency=2`, `min_interval=2.0s`), and atomically flushes (`os.fsync`) each completed generation to `artifacts/raw_llm_generations.jsonl` so interrupted runs resume without duplicate API calls — Source: Story 2, Story 5 — Failure: F1, F4
- [x] [Confirmation] Malformed LLM outputs (missing markdown code fences, unclosed `<thought>` tags, or syntax errors) never crash the batch runner and are deterministically recorded as `PARSE_OR_SYNTAX_ERROR` in `artifacts/runs.jsonl` — Source: Story 5 — Failure: F2, F4
- [x] [Contract] `scripts/build_ssot.py` replays all 270 candidate programs (`30 workflows × (1 Codemod + 4 LLM Arms × 2 Models)`) against the live `127.0.0.1:0` Loopback HTTP State Machine in `< 15 seconds` without network access or API keys, writing `artifacts/runs.jsonl`, `artifacts/metrics_ssot.json`, and `artifacts/explorer_bundle.json` — Source: Story 1, Story 6

## Verification Log

- **Ticket 02 YAGNI Runner (`2026-11-09`)**: `ticket-yagni-runner.py` returned `unreviewed: diff-truncated` (`error.json` under `call_6728779684531732151__tsk16__cbPJmQNuo3n3tU6R__02/manual/`) because generated `artifacts/*.json(l)` (`18,371` lines) sort alphabetically before `scripts/`. Recorded per `/implement` Inline Step 3 without retrying with different arguments; full scope/YAGNI audit will run at Step 3 closeout (`review-implement`).
- **Output Token Budget Adjustment (`2026-11-09`)**: 5/240 runs on `gemma-4-26b-a4b-it` hit `finish_reason=length` at `max_tokens=8192` due to verbose `<thought>` traces (`~14.8k` chars mean on Arm 4); raised `max_tokens` to `16384` per `model-onboard` output-budget rule, achieving `240/240` `status: ok` completions.

