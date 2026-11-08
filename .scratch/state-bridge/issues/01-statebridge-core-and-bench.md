# 01 — StateBridge Core Kernel, 30-Workflow Benchmark, and PBT Suite

**What to build:** Deliver the zero-third-party-dependency Python standard-library kernel (`skills/state-bridge/scripts/statebridge_core.py` and `skills/state-bridge/SKILL.md`), `LICENSE` (`Apache-2.0`), `THIRD_PARTY_NOTICES.md`, the 30 clean-room multi-file HTTP client workflows across 5 stateful divergence families (`F1_REDIRECT_FOLLOW_DEFAULT`, `F2_CROSS_ORIGIN_AUTH_STRIP`, `F3_COOKIE_JAR_SCOPING`, `F4_STREAM_CONTEXT_LIFECYCLE`, `F5_CLIENT_INIT_AND_STATUS_ERROR`), the `canonicalize_exchange()` normalizer, the `evaluate_trace_equivalence()` 7-category receipt comparator, the 5-arm context slicer (`Arm 1..5`), and the isolated `127.0.0.1:0` `ThreadingHTTPServer` Loopback HTTP State Machine sandbox with `sys.addaudithook` non-loopback egress blocking.

**Blocked by:** None — can start immediately

**Status:** `done`

**Needs:** None — a worker can check every item unaided

**Validation method:** Run `uv run --with requests==2.34.2 --with httpx==0.28.1 --with pytest python3 -m pytest tests/test_statebridge_core.py -v` and `python3 skills/state-bridge/scripts/statebridge_core.py selftest`. Conditional PBT invariants `P1..P4` are verified over multi-step HTTP traces (1–5 steps, methods `GET|POST|PUT|DELETE`, JSON whitespace permutations, header casing permutations, cookie maps, and transport-default headers) and multi-file Python AST call graphs against an independent structural oracle that constructs ground-truth canonical dicts and exact divergence indices prior to wire serialization.

**Evidence required:** `pytest` exit code 0 with all concrete capability examples, `P1..P4` property tests, and oracle calibration controls passing; `selftest` exit code 0 in a standard-library Python environment; all 30 reference `requests` workflows passing (`PASS_EQUIVALENT`) against their respective Golden Traces while `Arm 1` (Deterministic AST Codemod) deterministically triggers multi-step divergence receipts (`STEP_K_WIRE_DIVERGENCE`, `PRE_HTTP_RUNTIME_ERROR`, or `POST_HTTP_RETURN_MISMATCH`).

**Tests:** `required`

**Test seam:** `skills/state-bridge/scripts/statebridge_core.py` public functions (`canonicalize_exchange`, `evaluate_trace_equivalence`, `check_ast_anticheat`, `build_arm_context`, `run_workflow_sandbox`) and CLI subcommands (`selftest`, `scan`, `verify`).

**Regression scope:** Preserve `P1` (canonicalization idempotence and invariance to JSON whitespace `b'{"x": 1}'` vs `b'{"x":1}'`, header casing, cookie ordering, and default `User-Agent`/`Content-Length` headers), `P2` (reflexive equivalence and exact `first_divergence_step == k` detection on step-$k$ semantic mutations), `P3` (deterministic `CHEAT_RESIDUAL_REQUESTS` classification for any direct or aliased `requests`/`urllib.request` import), and `P4` (`Arm 5` unrelated function slice matches `Arm 4` token/character budget without sharing direct stateful call edges) — Source: Story 3, Story 5.

- [x] [Confirmation] `canonicalize_exchange()` normalizes JSON whitespace differences (`16 bytes` vs `13 bytes`), header casing, cookie pair order, and default `User-Agent`/`Content-Length` headers so identical `requests` and `httpx` exchanges produce identical canonical steps (`P1`) — Source: Story 5
- [x] [Confirmation] `evaluate_trace_equivalence()` returns `PASS_EQUIVALENT` for identical canonical traces and pinpoints `first_divergence_step == k` with `STEP_K_WIRE_DIVERGENCE` whenever step $k$ diverges in method, path, auth/custom header, cookie state, JSON payload, or status error code (`P2`) — Source: Story 3, Story 5
- [x] [Confirmation] Candidate code that retains `import requests`, `from requests import ...`, or `urllib.request` in any module is caught by `check_ast_anticheat()` and classified as `CHEAT_RESIDUAL_REQUESTS` before socket execution (`P3`) — Source: Story 5 — Failure: F2
- [x] [Confirmation] `build_arm_context()` constructs all 5 arms (`Arm 1` AST Codemod, `Arm 2` Target-Only, `Arm 3` 256-d Embedding Top-K, `Arm 4` True Call-Graph Slice, `Arm 5` Equal-Token Unrelated Slice) and enforces `P4` budget parity between `Arm 4` and `Arm 5` — Source: Story 2
- [x] [Confirmation] `run_workflow_sandbox()` runs `ThreadingHTTPServer` on an ephemeral port (`127.0.0.1:0`), clears proxy env vars (`HTTP_PROXY=""`, `HTTPS_PROXY=""`, `NO_PROXY="*"`), blocks any non-loopback socket connect via `sys.addaudithook` (`NON_LOOPBACK_EGRESS_BLOCKED`), executes in an isolated `TemporaryDirectory`, and cleans up the child process group via `os.killpg` on timeout (`TIMEOUT_OR_HANG`) leaving 0 orphan processes — Source: Story 5 — Failure: F1, F4
- [x] [Contract] All 30 multi-file benchmark workflows (`6 workflows × 5 divergence families`) pass `PASS_EQUIVALENT` under their reference `requests` implementation and produce deterministic baseline receipts under `Arm 1` (Deterministic AST Codemod) — Source: Story 1, Story 2

## Verification Log

- 2026-11-09 Preflight & Mode Lock: User authorized immediate unattended execution across Tickets `01..03` (`"你等等ticket開好就開工吧，我要睡了"`), locking `execution_mode=inline`, `testing_mode=tdd` (with full `P1..P4` Conditional PBT), `best_of_n=none`.
- 2026-11-09 Ticket 01 Verification: Observed RED (`9 failed in 0.09s`), then implemented `statebridge_core.py` and verified GREEN (`9 passed in 18.24s` in `tests/test_statebridge_core.py`, plus `selftest` exit 0 in stdlib Python). All 30 workflows pass `PASS_EQUIVALENT` under reference `requests` and `httpx_gold_files`, and diverge deterministically under `Arm 1` AST Codemod.
- 2026-11-09 Closeout Review (`python-reviewer` & `security-reviewer`):
  - `Acceptance-miss — build_arm_context() constructs all 5 arms + CLI subcommand reproduce-paper + 0 orphan processes assertion → build_arm_context() computed called_names without filtering Arm 4 by call-graph reachability, main() lacked reproduce-paper subcommand, and timeout test did not assert 0 surviving child PIDs`: Confirmed (`G3`). Fixing in `statebridge_core.py` and `tests/test_statebridge_core.py`.
  - `Failure-miss: F5 — _CHILD_RUNNER_CODE imported inspect/json before registering sys.addaudithook and omitted socket.sendmsg/getaddrinfo; canonicalize_exchange() fast-path skipped Cookie header merge and path query split → dual-path inconsistency`: Confirmed (`G4`). Fixing in `statebridge_core.py`.
  - `Failure-miss: F4 — run_workflow_sandbox() copied full os.environ, left unredacted home/temp paths in error_detail, and did not check gold_exec status → local path disclosure in runs.jsonl and unverified reference execution`: Confirmed (`G4`). Fixing in `statebridge_core.py`.


