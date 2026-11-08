# StateBridge × TwinAPI: Resolving the Stateful False-Friend Paradox in Cross-Library Code Migration with Gemma 4

**Subtitle:** An Empirical Audit of `requests` ↔ `httpx` Behavioral Divergence, a 30-Workflow Live Loopback HTTP Benchmark (`StateBridge-Bench`), and Call-Graph State-Slice Verification across `Gemma-4-31B-IT` and `Gemma-4-26B-A4B-IT`

---

## Abstract

When autonomous coding agents migrate repositories across sibling libraries such as Python's `requests` (`2.34.2`) and `httpx` (`0.28.1`), they confront what we formalize as the **Stateful False-Friend Paradox**: surface API names and dense retrieval embeddings are nearly identical, yet multi-step runtime state transitions diverge silently on the wire. In our empirical audit (**TwinAPI**), `requests` and `httpx` share **88.9%** (`8/9`) exact top-level function names, **83.3%** (`10/12`) core `Session`/`Client` method names, and a mean 256-dimensional embedding cosine similarity of **0.873** (`google/models/gemini-embedding-001`, `dimensions=256`) across 12 stateful false-friend API pairs. Consequently, deterministic AST codemods (`Arm 1`) pass **0.0%** (`0/30`) of multi-step workflows, while single-file prompting (`Arm 2`) and 256-d embedding Top-K retrieval (`Arm 3`) both stall at **13.3%** (`4/30`) on `Gemma-4-31B-IT` and **13.3%** (`4/30`) on `Gemma-4-26B-A4B-IT`.

We introduce **StateBridge**, an open-source benchmark (`StateBridge-Bench`), call-graph state-slice retrieval protocol (`Arm 4`), and live loopback HTTP verification harness (`127.0.0.1:0`). Instead of retrieving lexically similar HTTP utility files, `StateBridge` traces 1–2 hops along the static call graph from the target workflow into cross-file session factories (`session_setup.py`) to expose hidden stateful contracts (redirect policies, cross-origin auth-header hooks, domain-scoped cookie jars, streaming context lifecycles, and status-exception hierarchies). Evaluated across **270 live loopback HTTP executions** (**240 real Gemma 4 generations** + **30 deterministic codemod runs**), `StateBridge` (`Arm 4`) elevates multi-step wire-equivalence solve rates to **90.0%** (`27/30`, exact two-sided McNemar $p = 2.4e-07$) on dense `Gemma-4-31B-IT` and **96.7%** (`29/30`, McNemar $p = 6e-08$) on Mixture-of-Experts `Gemma-4-26B-A4B-IT`. An equal-token negative control (`Arm 5`, injecting unrelated repository functions) scores only **16.7%** (`5/30`) on `Gemma-4-31B-IT` and **13.3%** (`4/30`) on `Gemma-4-26B-A4B-IT`, proving that the **+73.3% to +83.3%** accuracy gain stems from structural call-graph state alignment rather than prompt length.

---

## 1. Introduction

Repository-level library migration is a critical workload for developer agents. Unlike greenfield function synthesis, migrating a production codebase from `requests` to `httpx` requires preserving multi-step behavioral invariants across authentication handshakes, redirect chains, domain-scoped cookies, chunked response streams, and error-recovery state machines. Existing code migration benchmarks—such as CodeMigrationBench, PyMigBench, RepoTransBench, and GitChameleon 2.0—primarily evaluate whether a migrated snippet compiles or passes unit tests when dependencies are upgraded. However, they leave an essential question unanswered: **Why do frontier open-weight models and vector-RAG pipelines systematically hallucinate broken migrations when two libraries intentionally share almost identical APIs?**

Consider a three-step OAuth login flow in `requests`:
1. `Step 0`: `session.post("/login", json=creds)` returns `HTTP 302 Found` with a `Set-Cookie: sid=...` header and `Location: /hop`. In `requests`, `Session.post` defaults to `allow_redirects=True`, automatically following the redirect and persisting the session cookie.
2. `Step 1`: A cross-origin redirect to an audit endpoint triggers `requests`'s RFC-compliant `rebuild_proxies_and_auth` or a registered `session.hooks["response"]` callback.
3. `Step 2`: `session.get("/vault")` reads `session.cookies.get_dict()` and streams a binary payload via `resp.iter_content(chunk_size=16)`.

When an agent rewrites `import requests` to `import httpx` in the target module `workflow_steps.py` without inspecting how the `httpx.Client` instance was constructed in `session_setup.py`, five silent traps detonate:
- **`F1_REDIRECT_FOLLOW_DEFAULT`**: `httpx.Client.post` and `.get` share the exact method names of `requests.Session`, but `httpx` renames `allow_redirects` to `follow_redirects` and flips the default from `True` to `False`. The code does not crash on Step 0; instead, it silently stops at the `302` response and sends an unauthenticated request on Step 1 (`STEP_K_WIRE_DIVERGENCE`).
- **`F2_CROSS_ORIGIN_AUTH_STRIP`**: `requests.Session` mutates headers via `s.hooks["response"]` lists, whereas `httpx.Client` requires `event_hooks={"request": [...], "response": [...]}` at initialization or explicit cross-origin header stripping.
- **`F3_COOKIE_JAR_SCOPING`**: `requests.cookies.RequestsCookieJar` exposes `.get_dict(domain=...)`, whereas `httpx.Cookies` omits `.get_dict()` entirely and enforces stricter RFC 6265 domain/path matching.
- **`F4_STREAM_CONTEXT_LIFECYCLE`**: `requests` streams via `s.get(url, stream=True)` followed by `resp.iter_content()`, whereas `httpx` requires entering a `with client.stream("GET", url) as resp:` context manager and calling `resp.iter_bytes()`.
- **`F5_CLIENT_INIT_AND_STATUS_ERROR`**: In `requests`, `HTTPError` is the specific subclass raised by `raise_for_status()`. In `httpx`, `HTTPError` is the *base class* for all transport and status errors, while `HTTPStatusError` is the status subclass—a subtle hierarchy inversion that breaks exception handlers and retry state machines.

### Contributions across the Three Competition Tracks

1. **Paper Track (Empirical Discovery & Rigorous Ablation)**: We quantify the **Stateful False-Friend Paradox** via `TwinAPI` and conduct a 5-arm × 2-model controlled experiment (`270` live executions) comparing `Gemma-4-31B-IT` (dense) and `Gemma-4-26B-A4B-IT` (MoE), complete with exact two-sided McNemar tests, 10,000-draw paired bootstrap 95% confidence intervals, and `<thought>` token telemetry.
2. **Best New Resource (`StateBridge-Bench` & `TwinAPI` Dataset)**: We release 30 multi-file, multi-step `requests` ↔ `httpx` workflows across 5 divergence families (`F1..F5`), a deterministic loopback HTTP state-machine oracle (`127.0.0.1:0`), a 4-invariant Property-Based Testing (PBT) suite (`P1..P4`), and a 15-pair 256-d embedding catalog.
3. **Best New Application (`skills/state-bridge/` & Interactive HTML Explorer)**: We package `StateBridge` as a zero-dependency Agent Skill (`skills/state-bridge/SKILL.md` + `statebridge_core.py`) and a single-file offline HTML Explorer (`explorer/index.html`) with 1-vs-1 code diffs, auto-collapsed matching wire steps, highlighted **First-Divergence Step** cards, and `#case=<id>&arm=<arm>&model=<model>` deep-linking.

---

## 2. The TwinAPI Stateful False-Friend Audit

To test why standard retrieval-augmented generation (RAG) fails on sibling-library migrations, we audited the public symbols of `requests==2.34.2` and `httpx==0.28.1` (`scripts/build_twinapi_audit.py`). For each of 15 core API pairs (12 stateful false friends across `F1..F5` and 3 true equivalents `P13..P15`), we computed:
1. **Normalized Levenshtein Short-Name Similarity**: Character-level similarity between terminal method/class names.
2. **Parameter-Masked Signature Isomorphism**: Structural arity and `inspect.Parameter.kind` alignment (`POSITIONAL_OR_KEYWORD`, `VAR_KEYWORD`, etc.).
3. **256-d Semantic Embedding Cosine Similarity**: Cosine similarity between API signature+docstring descriptors embedded via `google/models/gemini-embedding-001` (`dimensions=256`).

| Pair ID | `requests==2.34.2` Symbol | `httpx==0.28.1` Symbol | Name Sim | Sig Iso | 256-d Cosine | Stateful False Friend? | Family |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `P01` | `requests.Session.post` | `httpx.Client.post` | 1.00 | 0.11 | **0.878** | Yes (Stateful Trap) | `F1_REDIRECT_FOLLOW_DEFAULT` |
| `P02` | `requests.Session.get` | `httpx.Client.get` | 1.00 | 0.17 | **0.866** | Yes (Stateful Trap) | `F1_REDIRECT_FOLLOW_DEFAULT` |
| `P03` | `requests.Session.request` | `httpx.Client.request` | 1.00 | 0.25 | **0.915** | Yes (Stateful Trap) | `F2_CROSS_ORIGIN_AUTH_STRIP` |
| `P04` | `requests.Session` | `httpx.Client` | 0.14 | 0.05 | **0.743** | Yes (Stateful Trap) | `F2_CROSS_ORIGIN_AUTH_STRIP` |
| `P05` | `requests.cookies.RequestsCookieJar.get_dict` | `httpx.Cookies` | 0.12 | 0.33 | **0.714** | Yes (Stateful Trap) | `F3_COOKIE_JAR_SCOPING` |
| `P06` | `requests.Session.put` | `httpx.Client.put` | 1.00 | 0.10 | **0.867** | Yes (Stateful Trap) | `F3_COOKIE_JAR_SCOPING` |
| `P07` | `requests.Response.iter_content` | `httpx.Response.iter_bytes` | 0.58 | 0.50 | **0.911** | Yes (Stateful Trap) | `F4_STREAM_CONTEXT_LIFECYCLE` |
| `P08` | `requests.Response.close` | `httpx.Response.close` | 1.00 | 1.00 | **0.944** | Yes (Stateful Trap) | `F4_STREAM_CONTEXT_LIFECYCLE` |
| `P09` | `requests.Response.raise_for_status` | `httpx.Response.raise_for_status` | 1.00 | 1.00 | **0.978** | Yes (Stateful Trap) | `F5_CLIENT_INIT_AND_STATUS_ERROR` |
| `P10` | `requests.exceptions.HTTPError` | `httpx.HTTPStatusError` | 0.60 | 0.50 | **0.860** | Yes (Stateful Trap) | `F5_CLIENT_INIT_AND_STATUS_ERROR` |
| `P11` | `requests.Session.delete` | `httpx.Client.delete` | 1.00 | 0.14 | **0.900** | Yes (Stateful Trap) | `F1_REDIRECT_FOLLOW_DEFAULT` |
| `P12` | `requests.Session.patch` | `httpx.Client.patch` | 1.00 | 0.10 | **0.901** | Yes (Stateful Trap) | `F1_REDIRECT_FOLLOW_DEFAULT` |
| `P13` | `requests.Session.head` | `httpx.Client.head` | 1.00 | 0.14 | **0.908** | No (Equivalent) | `NONE` |
| `P14` | `requests.Session.options` | `httpx.Client.options` | 1.00 | 0.14 | **0.909** | No (Equivalent) | `NONE` |
| `P15` | `requests.Response.json` | `httpx.Response.json` | 1.00 | 1.00 | **0.899** | No (Equivalent) | `NONE` |

Across the 15 audited pairs, `requests` and `httpx` exhibit **88.9%** (`8/9`) exact top-level function name overlap (`delete`, `get`, `head`, `options`, `patch`, `post`, `put`, `request`) and **83.3%** (`10/12`) overlap across core `Session`/`Client` verbs. Crucially, the 12 stateful false-friend pairs achieve a mean 256-d embedding cosine similarity of **0.873** and a mean short-name similarity of **0.7876**. Because dense retrievers rank candidates by cosine similarity to the target file's HTTP calls, an embedding index retrieves generic HTTP formatting helpers (`distractors.py:preview_http_headers`) that share surface HTTP vocabulary rather than the cross-file session state initializers (`session_setup.py`) that govern wire behavior.

---

## 3. StateBridge Architecture & 5-Arm Experimental Design

### 3.1 Multi-File Repository Structure & Canonical Wire-Trace Oracle

Each of the 30 benchmark workflows (`F1_01..F5_06`, 6 per family) consists of a 3-file miniature repository:
- `workflow_steps.py` (**Target File Under Migration**): Implements `execute_step(base_url: str) -> dict`, executing a 2-to-3 step stateful HTTP sequence against `base_url`.
- `session_setup.py` (**Cross-File Stateful Collaborator**): Defines `prepare_client_session()`, a family-specific state-bridge helper (`hop_request_kwargs`, `register_redirect_guard`, `snapshot_jar_dict`, `open_chunk_stream`, or `verify_status_and_extract`), and `finalize_step_contract()`.
- `distractors.py` (**Repository Distractor Module**): Contains two functions with zero call edges to `execute_step`: (a) `preview_http_headers()`, a lexically dense HTTP header formatter that acts as an embedding magnet (`Arm 3`), and (b) `compute_window_digest()`, an equal-token rolling-checksum utility used as our negative control (`Arm 5`).

To evaluate functional equivalence without flaky external servers, `statebridge_core.py` spins up an ephemeral `ThreadingHTTPServer` bound to `127.0.0.1:0` for every run. Both the `requests` reference implementation and the candidate `httpx` implementation execute against fresh instances of this stateful loopback server. Every HTTP exchange is recorded and normalized by `canonicalize_exchange()` (lowercasing header keys, stripping transport-default headers such as `User-Agent`, `Host`, `Connection`, `Accept-Encoding`, `Accept`, `Content-Type`, and `Content-Length`, parsing query strings and cookies, and canonicalizing JSON bodies). `evaluate_trace_equivalence()` then compares the reference and candidate traces step-by-step, pinpointing the exact `first_divergence_step` and `divergence_field` (`method`, `path`, `headers`, `cookies`, `body_json`, or `trace_length`). Finally, `check_ast_anticheat()` parses the candidate AST to reject any residual `import requests` or `from requests import ...` statements (`CHEAT_RESIDUAL_REQUESTS`).

### 3.2 Five Orthogonal Experimental Arms (`P1..P4` Verified)

We evaluate five arms across all 30 workflows:
1. **`Arm 1` (`arm1_codemod`) — Deterministic AST/Syntax Codemod**: Replaces `import requests` with `import httpx` and `requests.Session()` with `httpx.Client()` inside `workflow_steps.py` without cross-file state bridging.
2. **`Arm 2` (`arm2_target_only`) — Gemma 4 + Target File Only**: Provides `workflow_steps.py` and a static one-line `requests` ↔ `httpx` cheat-sheet (`0` repository context functions; mean prompt `497.7` tokens).
3. **`Arm 3` (`arm3_embedding_topk`) — Gemma 4 + 256-d Embedding Top-K Retrieval**: Injects the top-ranked repository function by 256-d embedding cosine similarity to `workflow_steps.py` (`distractors.py:preview_http_headers`; mean prompt `637.7` tokens).
4. **`Arm 4` (`arm4_callgraph_slice`) — Gemma 4 + StateBridge Call-Graph State Slice (Ours)**: Extracts the 1–2 hop static call-graph slice from `session_setup.py` (`prepare_client_session`, the family state-bridge helper, `finalize_step_contract`, and module-level state constants; mean prompt `799.9` tokens).
5. **`Arm 5` (`arm5_unrelated_slice`) — Gemma 4 + Equal-Token Unrelated Slice (Negative Control)**: Injects `distractors.py:compute_window_digest()`, matched in character/token budget to `Arm 4` (mean prompt `702.7` tokens, verified by PBT invariant `P4`) while sharing `0` call-graph edges with `execute_step`.

All LLM arms were executed against real **`google/models/gemma-4-31b-it`** (31B dense instruction-tuned) and **`google/models/gemma-4-26b-a4b-it`** (26B total / 4B active Mixture-of-Experts instruction-tuned) with `temperature=0.0`, separating `<thought>...</thought>` reasoning blocks from extracted code fences (`240` real generations total).

---

## 4. Empirical Results on Gemma 4

### 4.1 Overall 5-Arm Solve Rates and Statistical Significance

Table 2 summarizes the live loopback HTTP execution outcomes across all **270 runs** (replayed end-to-end in **8.507 seconds** via `scripts/build_ssot.py --verify-replay`).

| Experimental Arm | `Gemma-4-31B-IT` Solve Rate | `Gemma-4-31B-IT` Outcomes (`PASS` / `WIRE_DIV` / `ERR`) | `Gemma-4-26B-A4B-IT` Solve Rate | `Gemma-4-26B-A4B-IT` Outcomes (`PASS` / `WIRE_DIV` / `ERR`) | Mean Prompt Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Arm 1`: Deterministic AST Codemod** | **0.0%** (`0/30`) | `0` / `17` / `13` | **0.0%** (`0/30`) | `0` / `17` / `13` | `0.0` |
| **`Arm 2`: Target File Only** | **13.3%** (`4/30`) | `4` / `26` / `0` | **13.3%** (`4/30`) | `4` / `26` / `0` | `497.7` |
| **`Arm 3`: 256-d Embedding Top-K** | **13.3%** (`4/30`) | `4` / `26` / `0` | **13.3%** (`4/30`) | `4` / `26` / `0` | `637.7` |
| **`Arm 5`: Equal-Token Unrelated Slice** | **16.7%** (`5/30`) | `5` / `25` / `0` | **13.3%** (`4/30`) | `4` / `26` / `0` | `702.7` |
| **`Arm 4`: StateBridge Call-Graph Slice (Ours)** | **90.0%** (`27/30`) | `27` / `3` / `0` | **96.7%** (`29/30`) | `29` / `1` / `0` | `799.9` |

**Key Finding 1 — Call-Graph State Slicing (`Arm 4`) Outperforms Single-File and Embedding RAG by +76.7% to +83.3%:**
On `Gemma-4-31B-IT`, `StateBridge` (`Arm 4`) improves solve rate from **13.3%** (`4/30`) in `Arm 2` and **13.3%** (`4/30`) in `Arm 3` to **90.0%** (`27/30`), winning `23` discordant workflow pairs against `0` losses (exact two-sided McNemar $p = 2.4e-07$, 10,000-draw paired bootstrap 95% CI `[60.0%, 90.0%]`). On `Gemma-4-26B-A4B-IT`, `StateBridge` (`Arm 4`) jumps from **13.3%** (`4/30`) in `Arm 2` and `Arm 3` to **96.7%** (`29/30`), winning `25` discordant pairs against `0` losses (exact McNemar $p = 6e-08$, 95% CI `[70.0%, 96.7%]`).

**Key Finding 2 — Ruling Out Prompt-Length Confounds (`Arm 4` vs `Arm 5`):**
Comparing `Arm 4` against the equal-token unrelated slice (`Arm 5`) confirms that adding repository tokens without call-graph connectivity yields no meaningful gain (`16.7%` on `31B-IT`, `13.3%` on `26B-A4B-IT`). `Arm 4` beats `Arm 5` by **+73.3%** on `Gemma-4-31B-IT` ($p = 2.98e-06$, 95% CI `[53.3%, 90.0%]`) and **+83.3%** on `Gemma-4-26B-A4B-IT` ($p = 6e-08$, 95% CI `[70.0%, 96.7%]`).

### 4.2 Per-Family Breakdown and Dense (`31B-IT`) vs MoE (`26B-A4B-IT`) Reasoning Telemetry

| Divergence Family (`6` Workflows Each) | `Arm 1` Codemod | `31B-IT` `Arm 2`/`3`/`5` | `31B-IT` `Arm 4` (`StateBridge`) | `26B-A4B-IT` `Arm 2`/`3`/`5` | `26B-A4B-IT` `Arm 4` (`StateBridge`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`F1_REDIRECT_FOLLOW_DEFAULT`** | `0/6` (0.0%) | `1/6` / `1/6` / `1/6` | **`6/6` (100.0%)** | `1/6` / `1/6` / `1/6` | **`6/6` (100.0%)** |
| **`F2_CROSS_ORIGIN_AUTH_STRIP`** | `0/6` (0.0%) | `1/6` / `1/6` / `1/6` | **`6/6` (100.0%)** | `1/6` / `1/6` / `1/6` | **`6/6` (100.0%)** |
| **`F3_COOKIE_JAR_SCOPING`** | `0/6` (0.0%) | `0/6` / `0/6` / `1/6` | **`5/6` (83.3%)** | `0/6` / `0/6` / `0/6` | **`5/6` (83.3%)** |
| **`F4_STREAM_CONTEXT_LIFECYCLE`** | `0/6` (0.0%) | `1/6` / `1/6` / `1/6` | **`6/6` (100.0%)** | `1/6` / `1/6` / `1/6` | **`6/6` (100.0%)** |
| **`F5_CLIENT_INIT_AND_STATUS_ERROR`** | `0/6` (0.0%) | `1/6` / `1/6` / `1/6` | **`4/6` (66.7%)** | `1/6` / `1/6` / `1/6` | **`6/6` (100.0%)** |

Comparing the two Gemma 4 architectures reveals a striking efficiency-accuracy trade-off:
- **MoE `Gemma-4-26B-A4B-IT` (4B active parameters) matches or exceeds Dense `31B-IT` (`96.7%` vs `90.0%`) by investing more internal `<thought>` reasoning**: Across `Arm 4`, `Gemma-4-26B-A4B-IT` generates a mean of **`14820.8`** `<thought>` characters (`258.8` completion tokens) compared to **`8396.0`** `<thought>` characters (`234.7` completion tokens) on `Gemma-4-31B-IT`.
- **Where the remaining errors occur**: Without `StateBridge` (`Arms 2, 3, 5`), **86.7%** (`26/30`) of runs fail with `STEP_K_WIRE_DIVERGENCE`—the generated code is syntactically valid Python that imports `httpx`, yet silently diverges at Step 0 or Step 1 on the wire (e.g., failing to follow `302` redirects, leaking or dropping `Authorization` headers across origins, or failing to extract scoped cookies). With `StateBridge` (`Arm 4`), wire divergences drop from `26/30` to `3/30` on `31B-IT` and `1/30` on `26B-A4B-IT`.

---

## 5. Related Works & Citations

1. **Gemma 4 Technical Report (Google DeepMind, 2026)**: Introduces the Gemma 4 open-weight model family, including the dense `Gemma-4-31B-IT` and sparse Mixture-of-Experts `Gemma-4-26B-A4B-IT` architectures with native `<thought>` reasoning traces. Our work provides the first controlled evaluation of both architectures on multi-step stateful library migration.
2. **CodeMigrationBench (2026) & PyMigBench (2023)**: Benchmarks for real-world Python dependency and library migrations. While PyMigBench catalogs syntax-level code changes and CodeMigrationBench evaluates repository test pass rates, `StateBridge-Bench` isolates *stateful false-friend APIs* using a deterministic loopback HTTP state machine that pinpoints exact wire-level step divergences.
3. **RepoTransBench (2024) & GitChameleon 2.0 (2025)**: Evaluate repository-level code translation and version-conditioned API generation. Our `TwinAPI` audit complements these benchmarks by demonstrating that high embedding similarity (`0.873` cosine) actively misleads Top-K vector retrieval (`Arm 3`) during sibling-library migration.
4. **RFC 9110 (HTTP Semantics) & RFC 6265 (HTTP State Management Mechanism)**: Define the redirect status semantics (`301`/`302`/`303`), cross-origin `Authorization` header stripping requirements, and `Cookie`/`Set-Cookie` domain scoping rules that underpin Divergence Families `F1`, `F2`, and `F3`.

---

## 6. Reproduction & Open-Source Artifacts

All artifacts are self-contained and reproducible offline in `< 15 seconds` without external network calls or API keys:
- **One-Command SSOT Replay & Verification**:
  ```bash
  uv run --with requests==2.34.2 --with httpx==0.28.1 python3 scripts/build_ssot.py --verify-replay
  uv run --with requests==2.34.2 --with httpx==0.28.1 --with pytest python3 -m pytest -v
  python3 scripts/verify_ssot_consistency.py
  ```
- **Deliverable Map**:
  - `skills/state-bridge/SKILL.md` & `skills/state-bridge/scripts/statebridge_core.py`: Reusable Agent Skill, 30-workflow benchmark, loopback HTTP state-machine sandbox, and `P1..P4` PBT suite.
  - `artifacts/metrics_ssot.json`, `artifacts/runs.jsonl`, `artifacts/raw_llm_generations.jsonl`, `artifacts/twinapi_audit.json`, `artifacts/twinapi_embeddings_256d.json`: Complete frozen generations and live execution receipts.
  - `paper/paper.pdf` & `paper/paper.md`: Full arXiv-style technical paper with vector/high-DPI figures.
  - `notebooks/reproduce_statebridge.ipynb`: Executed Kaggle Notebook reproducing all 270 runs, tables, and McNemar/bootstrap tests.
  - `explorer/index.html`: Zero-CDN interactive HTML Explorer with 1-vs-1 code diffs, First-Divergence wire inspector, and URL hash deep-linking.
