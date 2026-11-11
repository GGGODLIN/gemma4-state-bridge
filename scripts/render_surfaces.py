#!/usr/bin/env python3
"""Render all 5 submission surfaces from artifacts/metrics_ssot.json and artifacts/explorer_bundle.json."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _fmt_pct(rate: float) -> str:
  return f"{rate * 100:.1f}%"


def _render_figures(ssot: dict[str, Any], audit: dict[str, Any], fig_dir: Path) -> dict[str, Path]:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  import numpy as np

  fig_dir.mkdir(parents=True, exist_ok=True)
  plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.facecolor": "#fcfcfb",
    "figure.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "text.color": "#0b0b0b",
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.8,
  })

  m31 = ssot["models"]["gemma-4-31b-it"]["arms"]
  m26 = ssot["models"]["gemma-4-26b-a4b-it"]["arms"]

  # Figure 1: 5-Arm Solve Rate Comparison (Single axis, grouped horizontal bars, direct labels)
  arms_order = [
    ("arm1_codemod", "Arm 1: Deterministic AST Codemod"),
    ("arm2_target_only", "Arm 2: Target File Only + Static Cheat-Sheet"),
    ("arm3_embedding_topk", "Arm 3: HTTP Distractor (HTTP Decoy)"),
    ("arm5_unrelated_slice", "Arm 5: Equal-Token Unrelated Slice (Control)"),
    ("arm4_callgraph_slice", "Arm 4: StateBridge Call-Graph Slice (Ours)"),
  ]
  y = np.arange(len(arms_order))
  height = 0.36

  rates_31 = [m31[k]["solve_rate"] * 100 for k, _ in arms_order]
  rates_26 = [m26[k]["solve_rate"] * 100 for k, _ in arms_order]
  counts_31 = [m31[k]["pass_count"] for k, _ in arms_order]
  counts_26 = [m26[k]["pass_count"] for k, _ in arms_order]

  fig1, ax1 = plt.subplots(figsize=(9.5, 4.6), dpi=220)
  ax1.grid(axis="x", linestyle="-", alpha=0.8, zorder=0)
  bars_31 = ax1.barh(
    y - height / 2 - 0.015,
    rates_31,
    height=height,
    color="#2a78d6",
    edgecolor="#fcfcfb",
    linewidth=1.5,
    label="Gemma-4-31B-IT (Dense)",
    zorder=3,
  )
  bars_26 = ax1.barh(
    y + height / 2 + 0.015,
    rates_26,
    height=height,
    color="#eb6834",
    edgecolor="#fcfcfb",
    linewidth=1.5,
    label="Gemma-4-26B-A4B-IT (MoE)",
    zorder=3,
  )
  ax1.set_yticks(y)
  ax1.set_yticklabels([label for _, label in arms_order], fontsize=9.2, fontweight="bold")
  ax1.set_xlim(0, 116)
  ax1.set_xlabel("Wire-Trace Equivalence Solve Rate (% of 30 Multi-Step Workflows)", fontsize=9.8)
  ax1.set_title("StateBridge-Bench: 5-Arm Solve Rate on Gemma 4 (270 Live Runs)", fontsize=11.0, fontweight="bold", loc="left", pad=12)
  ax1.spines["top"].set_visible(False)
  ax1.spines["right"].set_visible(False)

  for bar, r, c in zip(bars_31, rates_31, counts_31):
    ax1.text(
      bar.get_width() + 1.4,
      bar.get_y() + bar.get_height() / 2,
      f"{r:.1f}% ({c}/30)",
      va="center",
      ha="left",
      fontsize=8.5,
      color="#0b0b0b",
      fontweight="bold",
    )
  for bar, r, c in zip(bars_26, rates_26, counts_26):
    ax1.text(
      bar.get_width() + 1.4,
      bar.get_y() + bar.get_height() / 2,
      f"{r:.1f}% ({c}/30)",
      va="center",
      ha="left",
      fontsize=8.5,
      color="#0b0b0b",
      fontweight="bold",
    )
  ax1.legend(loc="lower right", frameon=True, facecolor="#fcfcfb", edgecolor="#e1e0d9", fontsize=8.8)
  fig1.tight_layout()
  fig1_path = fig_dir / "fig1_5arm_ablation.png"
  fig1.savefig(fig1_path, dpi=220, bbox_inches="tight")
  fig1.savefig(fig1_path.with_suffix(".svg"), bbox_inches="tight")
  plt.close(fig1)

  # Figure 2: Per-Family Solve Rate Comparison (Arm 4 vs Arm 3 vs Arm 5 on Gemma-4-26B-A4B-IT & 31B-IT)
  fams = [
    ("F1_REDIRECT_FOLLOW_DEFAULT", "F1: 302 Redirect\nFollow Default"),
    ("F2_CROSS_ORIGIN_AUTH_STRIP", "F2: Cross-Origin\nAuth & Hooks"),
    ("F3_COOKIE_JAR_SCOPING", "F3: Cookie Jar\nDomain Scoping"),
    ("F4_STREAM_CONTEXT_LIFECYCLE", "F4: Stream Context\nLifecycle"),
    ("F5_CLIENT_INIT_AND_STATUS_ERROR", "F5: Client Init &\nStatus Error"),
  ]
  x = np.arange(len(fams))
  w = 0.25

  f_arm2 = [m26["arm2_target_only"]["by_family"][fk]["pass_count"] / 6 * 100 for fk, _ in fams]
  f_arm4_31 = [m31["arm4_callgraph_slice"]["by_family"][fk]["pass_count"] / 6 * 100 for fk, _ in fams]
  f_arm4_26 = [m26["arm4_callgraph_slice"]["by_family"][fk]["pass_count"] / 6 * 100 for fk, _ in fams]

  fig2, ax2 = plt.subplots(figsize=(9.5, 4.5), dpi=220)
  ax2.grid(axis="y", linestyle="-", alpha=0.8, zorder=0)
  b1 = ax2.bar(
    x - w - 0.01,
    f_arm2,
    width=w,
    color="#1baf7a",
    edgecolor="#fcfcfb",
    linewidth=1.5,
    label=f"26B Arm 2 baseline ({_fmt_pct(m26['arm2_target_only']['solve_rate'])})",
    zorder=3,
  )
  b2 = ax2.bar(
    x,
    f_arm4_31,
    width=w,
    color="#2a78d6",
    edgecolor="#fcfcfb",
    linewidth=1.5,
    label=f"Arm 4 StateBridge (31B-IT: {_fmt_pct(m31['arm4_callgraph_slice']['solve_rate'])})",
    zorder=3,
  )
  b3 = ax2.bar(
    x + w + 0.01,
    f_arm4_26,
    width=w,
    color="#eb6834",
    edgecolor="#fcfcfb",
    linewidth=1.5,
    label=f"Arm 4 StateBridge (26B-A4B-IT: {_fmt_pct(m26['arm4_callgraph_slice']['solve_rate'])})",
    zorder=3,
  )
  ax2.set_xticks(x)
  ax2.set_xticklabels([fl for _, fl in fams], fontsize=8.8, fontweight="bold")
  ax2.set_ylim(0, 116)
  ax2.set_ylabel("Family Pass Rate (% of 6 Workflows)", fontsize=9.5)
  ax2.set_title("Per-Family Migration Accuracy Across 5 Stateful HTTP Divergence Families", fontsize=11.0, fontweight="bold", pad=28)
  ax2.spines["top"].set_visible(False)
  ax2.spines["right"].set_visible(False)
  for bars in (b1, b2, b3):
    for bar in bars:
      val = bar.get_height()
      ax2.text(bar.get_x() + bar.get_width() / 2, val + 2.0, f"{val:.0f}%", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
  ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=True, facecolor="#fcfcfb", edgecolor="#e1e0d9", fontsize=8.3)
  fig2.tight_layout()
  fig2_path = fig_dir / "fig2_family_breakdown.png"
  fig2.savefig(fig2_path, dpi=220, bbox_inches="tight")
  fig2.savefig(fig2_path.with_suffix(".svg"), bbox_inches="tight")
  plt.close(fig2)

  # Figure 3: TwinAPI 256-d Embedding Cosine Similarity across 15 Audited API Pairs
  pairs = audit.get("pairs", [])
  p_labels = [f"{p['pair_id']}: {p['req_symbol'].split('.')[-1]}↔{p['httpx_symbol'].split('.')[-1]}" for p in pairs]
  p_cos = [p["embedding_cosine_256d"] for p in pairs]
  p_colors = ["#eb6834" if p["is_stateful_false_friend"] else "#2a78d6" for p in pairs]
  mean_ff_cos = ssot["twinapi_audit"]["mean_false_friend_embedding_cosine_256d"]

  fig3, ax3 = plt.subplots(figsize=(9.5, 5.0), dpi=220)
  y3 = np.arange(len(pairs))
  ax3.grid(axis="x", linestyle="-", alpha=0.8, zorder=0)
  bars3 = ax3.barh(y3, p_cos, height=0.68, color=p_colors, edgecolor="#fcfcfb", linewidth=1.2, zorder=3)
  ax3.axvline(mean_ff_cos, color="#0b0b0b", linestyle="--", linewidth=1.2, label=f"Mean False-Friend Cosine ({mean_ff_cos:.3f})")
  ax3.set_yticks(y3)
  ax3.set_yticklabels(p_labels, fontsize=8.2)
  ax3.invert_yaxis()
  ax3.set_xlim(0, 1.22)
  ax3.set_xlabel("256-d Gemini Embedding Cosine Similarity (gemini-embedding-001)", fontsize=9.5)
  ax3.set_title("TwinAPI Audit: High Embedding Similarity Masks Runtime Divergence", fontsize=10.8, fontweight="bold", loc="left", pad=24)
  ax3.spines["top"].set_visible(False)
  ax3.spines["right"].set_visible(False)
  for bar, c_val, p in zip(bars3, p_cos, pairs):
    tag = "False Friend" if p["is_stateful_false_friend"] else "Equivalent"
    ax3.text(bar.get_width() + 0.012, bar.get_y() + bar.get_height() / 2, f"{c_val:.3f} ({tag})", va="center", ha="left", fontsize=7.6)
  ax3.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01), frameon=True, facecolor="#fcfcfb", edgecolor="#e1e0d9", fontsize=8.3)
  fig3.tight_layout()
  fig3_path = fig_dir / "fig3_twinapi_paradox.png"
  fig3.savefig(fig3_path, dpi=220, bbox_inches="tight")
  fig3.savefig(fig3_path.with_suffix(".svg"), bbox_inches="tight")
  plt.close(fig3)

  return {
    "fig1": fig1_path,
    "fig2": fig2_path,
    "fig3": fig3_path,
  }


def _build_writeup_markdown(ssot: dict[str, Any], audit: dict[str, Any]) -> str:
  twin = ssot["twinapi_audit"]
  m31 = ssot["models"]["gemma-4-31b-it"]
  m26 = ssot["models"]["gemma-4-26b-a4b-it"]
  a31 = m31["arms"]
  a26 = m26["arms"]
  c31 = m31["paired_comparisons"]
  c26 = m26["paired_comparisons"]

  top_pct = _fmt_pct(twin["top_level_exact_name_overlap_ratio"])
  sess_pct = _fmt_pct(twin["session_client_core_overlap_ratio"])
  mean_cos = f"{twin['mean_false_friend_embedding_cosine_256d']:.3f}"

  r31_a2 = _fmt_pct(a31["arm2_target_only"]["solve_rate"])
  r31_a3 = _fmt_pct(a31["arm3_embedding_topk"]["solve_rate"])
  r31_a4 = _fmt_pct(a31["arm4_callgraph_slice"]["solve_rate"])
  r31_a5 = _fmt_pct(a31["arm5_unrelated_slice"]["solve_rate"])

  r26_a2 = _fmt_pct(a26["arm2_target_only"]["solve_rate"])
  r26_a3 = _fmt_pct(a26["arm3_embedding_topk"]["solve_rate"])
  r26_a4 = _fmt_pct(a26["arm4_callgraph_slice"]["solve_rate"])
  r26_a5 = _fmt_pct(a26["arm5_unrelated_slice"]["solve_rate"])

  pairs_rows = []
  for p in audit.get("pairs", []):
    ff_str = "Yes (Stateful Trap)" if p["is_stateful_false_friend"] else "No (Equivalent)"
    pairs_rows.append(
      f"| `{p['pair_id']}` | `{p['req_symbol']}` | `{p['httpx_symbol']}` | "
      f"{p['short_name_similarity']:.2f} | {p['signature_isomorphism']:.2f} | "
      f"**{p['embedding_cosine_256d']:.3f}** | {ff_str} | `{p['divergence_family']}` |"
    )
  pairs_table = "\n".join(pairs_rows)

  def _fmt_outcomes(arm_obj: dict[str, Any]) -> str:
    oc = arm_obj.get("outcome_counts", {})
    p_cnt = oc.get("PASS_EQUIVALENT", 0)
    w_cnt = oc.get("STEP_K_WIRE_DIVERGENCE", 0)
    e_cnt = arm_obj["total"] - p_cnt - w_cnt
    return f"`{p_cnt}` / `{w_cnt}` / `{e_cnt}`"

  def _fam_row(fam_key: str) -> str:
    c1 = a31["arm1_codemod"]["by_family"][fam_key]["pass_count"]
    b31 = (
      f"`{a31['arm2_target_only']['by_family'][fam_key]['pass_count']}/6` / "
      f"`{a31['arm3_embedding_topk']['by_family'][fam_key]['pass_count']}/6` / "
      f"`{a31['arm5_unrelated_slice']['by_family'][fam_key]['pass_count']}/6`"
    )
    p31 = a31["arm4_callgraph_slice"]["by_family"][fam_key]["pass_count"]
    b26 = (
      f"`{a26['arm2_target_only']['by_family'][fam_key]['pass_count']}/6` / "
      f"`{a26['arm3_embedding_topk']['by_family'][fam_key]['pass_count']}/6` / "
      f"`{a26['arm5_unrelated_slice']['by_family'][fam_key]['pass_count']}/6`"
    )
    p26 = a26["arm4_callgraph_slice"]["by_family"][fam_key]["pass_count"]
    return (
      f"| **`{fam_key}`** | `{c1}/6` ({_fmt_pct(c1/6)}) | {b31} | "
      f"**`{p31}/6` ({_fmt_pct(p31/6)})** | {b26} | **`{p26}/6` ({_fmt_pct(p26/6)})** |"
    )

  fam_table = "\n".join(
    _fam_row(fk)
    for fk in (
      "F1_REDIRECT_FOLLOW_DEFAULT",
      "F2_CROSS_ORIGIN_AUTH_STRIP",
      "F3_COOKIE_JAR_SCOPING",
      "F4_STREAM_CONTEXT_LIFECYCLE",
      "F5_CLIENT_INIT_AND_STATUS_ERROR",
    )
  )

  return f"""# StateBridge × TwinAPI: Resolving the Stateful False-Friend Paradox in Cross-Library Code Migration with Gemma 4

**Subtitle:** An Empirical Audit of `requests` ↔ `httpx` Behavioral Divergence, a 30-Workflow Live Loopback HTTP Benchmark (`StateBridge-Bench`), and Call-Graph State-Slice Verification across `Gemma-4-31B-IT` and `Gemma-4-26B-A4B-IT`

---

## Abstract

When autonomous coding agents migrate repositories across sibling libraries such as Python's `requests` (`2.34.2`) and `httpx` (`0.28.1`), they confront what we formalize as the **Stateful False-Friend Paradox**: surface API names and dense retrieval embeddings are nearly identical, yet multi-step runtime state transitions diverge silently on the wire. In our empirical audit (**TwinAPI**), `requests` and `httpx` share **{top_pct}** (`8/9`) exact top-level function names, **{sess_pct}** (`10/12`) core `Session`/`Client` method names, and a mean 256-dimensional embedding cosine similarity of **{mean_cos}** (`google/models/gemini-embedding-001`, `dimensions=256`) across 12 stateful false-friend API pairs. Consequently, deterministic AST codemods (`Arm 1`) pass **{_fmt_pct(a31['arm1_codemod']['solve_rate'])}** (`{a31['arm1_codemod']['pass_count']}/30`) of multi-step workflows, while single-file prompting (`Arm 2`) and fixed HTTP-distractor context (`Arm 3`) both stall at **{r31_a2}** (`{a31['arm2_target_only']['pass_count']}/30`) on `Gemma-4-31B-IT` and **{r26_a2}** (`{a26['arm2_target_only']['pass_count']}/30`) on `Gemma-4-26B-A4B-IT`.

We introduce **StateBridge**, an open-source benchmark (`StateBridge-Bench`), call-graph state-slice retrieval protocol (`Arm 4`), and live loopback HTTP verification harness (`127.0.0.1:0`). Instead of retrieving lexically similar HTTP utility files, `StateBridge` traces 1–2 hops along the static call graph from the target workflow into cross-file session factories (`session_setup.py`) to expose hidden stateful contracts (redirect policies, cross-origin auth-header hooks, domain-scoped cookie jars, streaming context lifecycles, and status-exception hierarchies). Evaluated across **270 live loopback HTTP executions** (**240 real Gemma 4 generations** + **30 deterministic codemod runs**), `StateBridge` (`Arm 4`) elevates multi-step wire-equivalence solve rates to **{r31_a4}** (`{a31['arm4_callgraph_slice']['pass_count']}/30`, exact two-sided McNemar $p = {c31['arm4_vs_arm2']['mcnemar_p']}$) on dense `Gemma-4-31B-IT` and **{r26_a4}** (`{a26['arm4_callgraph_slice']['pass_count']}/30`, McNemar $p = {c26['arm4_vs_arm2']['mcnemar_p']}$) on Mixture-of-Experts `Gemma-4-26B-A4B-IT`. An equal-token negative control (`Arm 5`, injecting unrelated repository functions) scores only **{r31_a5}** (`{a31['arm5_unrelated_slice']['pass_count']}/30`) on `Gemma-4-31B-IT` and **{r26_a5}** (`{a26['arm5_unrelated_slice']['pass_count']}/30`) on `Gemma-4-26B-A4B-IT`, proving that the **+{_fmt_pct(c31['arm4_vs_arm5']['diff'])} to +{_fmt_pct(c26['arm4_vs_arm5']['diff'])}** accuracy gain stems from structural call-graph state alignment rather than prompt length.

---

## 1. Introduction

Repository-level library migration is a critical workload for developer agents. Unlike greenfield function synthesis, migrating a production codebase from `requests` to `httpx` requires preserving multi-step behavioral invariants across authentication handshakes, redirect chains, domain-scoped cookies, chunked response streams, and error-recovery state machines. Existing code migration benchmarks—such as CodeMigrationBench, PyMigBench, RepoTransBench, and GitChameleon 2.0—primarily evaluate whether a migrated snippet compiles or passes unit tests when dependencies are upgraded. However, they leave an essential question unanswered: **Why do frontier open-weight models and vector-RAG pipelines systematically hallucinate broken migrations when two libraries intentionally share almost identical APIs?**

Consider a three-step OAuth login flow in `requests`:
1. `Step 0`: `session.post("/login", json=creds)` returns `HTTP 302 Found` with a `Set-Cookie: sid=...` header and `Location: /hop`. In `requests`, `Session.post` defaults to `allow_redirects=True`, automatically following the redirect and persisting the session cookie.
2. `Step 1`: A cross-origin redirect to an audit endpoint invokes `requests`'s `rebuild_auth` behavior or a registered `session.hooks["response"]` callback.
3. `Step 2`: `session.get("/vault")` reads `session.cookies.get_dict()` and streams a binary payload via `resp.iter_content(chunk_size=16)`.

When an agent rewrites `import requests` to `import httpx` in the target module `workflow_steps.py` without inspecting how the `httpx.Client` instance was constructed in `session_setup.py`, five silent traps detonate:
- **`F1_REDIRECT_FOLLOW_DEFAULT`**: `httpx.Client.post` and `.get` share the exact method names of `requests.Session`, but `httpx` renames `allow_redirects` to `follow_redirects` and flips the default from `True` to `False`. The code does not crash on Step 0; instead, it silently stops at the `302` response and sends an unauthenticated request on Step 1 (`STEP_K_WIRE_DIVERGENCE`).
- **`F2_CROSS_ORIGIN_AUTH_STRIP`**: `requests.Session` mutates headers via `s.hooks["response"]` lists, whereas `httpx.Client` requires `event_hooks={{"request": [...], "response": [...]}}` at initialization or explicit cross-origin header stripping.
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
{pairs_table}

Across the 15 audited pairs, `requests` and `httpx` exhibit **{top_pct}** (`8/9`) exact top-level function name overlap (`delete`, `get`, `head`, `options`, `patch`, `post`, `put`, `request`) and **{sess_pct}** (`10/12`) overlap across core `Session`/`Client` verbs. Crucially, the 12 stateful false-friend pairs achieve a mean 256-d embedding cosine similarity of **{mean_cos}** and a mean short-name similarity of **{twin['mean_false_friend_short_name_similarity']:.4f}**. The audit measures API descriptor similarity; it does not evaluate repository retrieval. In the frozen experiment, `Arm 3` selects `distractors.py:preview_http_headers` by name and excludes `session_setup.py`; no TF-IDF or vector cosine retrieval was run. The later keyword-density ranker chooses the same function. The experiment therefore tests supplied context, not dense-retrieval quality.

---

## 3. StateBridge Architecture & 5-Arm Experimental Design

### 3.1 Multi-File Repository Structure & Canonical Wire-Trace Oracle

Each of the 30 benchmark workflows (`F1_01..F5_06`, 6 per family) consists of a 3-file miniature repository:
- `workflow_steps.py` (**Target File Under Migration**): Implements `execute_step(base_url: str) -> dict`, executing a 2-to-3 step stateful HTTP sequence against `base_url`.
- `session_setup.py` (**Cross-File Stateful Collaborator**): Defines `prepare_client_session()`, family-specific helpers (`hop_request_kwargs`, `build_partner_hop_headers`, `promote_cookie_to_header`, `summarize_stream_bytes`, or `build_recovery_headers`), and `finalize_step_contract()`.
- `distractors.py` (**Repository Distractor Module**): Contains two functions with zero call edges to `execute_step`: (a) `preview_http_headers()`, a lexically dense HTTP header formatter that acts as a fixed HTTP-themed distractor (`Arm 3`), and (b) `compute_window_digest()`, an equal-token rolling-checksum utility used as our negative control (`Arm 5`).

To evaluate functional equivalence without flaky external servers, `statebridge_core.py` spins up an ephemeral `ThreadingHTTPServer` bound to `127.0.0.1:0` for every run. Both the `requests` reference implementation and the candidate `httpx` implementation execute against fresh instances of this stateful loopback server. Every HTTP exchange is recorded and normalized by `canonicalize_exchange()` (lowercasing header keys, stripping transport-default headers such as `User-Agent`, `Host`, `Connection`, `Accept-Encoding`, `Accept`, `Content-Type`, and `Content-Length`, parsing query strings and cookies, and canonicalizing JSON bodies). `evaluate_trace_equivalence()` then compares the reference and candidate traces step-by-step, pinpointing the exact `first_divergence_step` and `divergence_field` (`method`, `path`, `headers`, `cookies`, `body_json`, or `trace_length`). Finally, `check_ast_anticheat()` parses the candidate AST to reject any residual `import requests` or `from requests import ...` statements (`CHEAT_RESIDUAL_REQUESTS`).

### 3.2 Five Orthogonal Experimental Arms (`P1..P4` Verified)

We evaluate five arms across all 30 workflows:
1. **`Arm 1` (`arm1_codemod`) — Deterministic AST/Syntax Codemod**: Replaces `import requests` with `import httpx` and `requests.Session()` with `httpx.Client()` inside `workflow_steps.py` without cross-file state bridging.
2. **`Arm 2` (`arm2_target_only`) — Gemma 4 + Target File Only**: Provides `workflow_steps.py` and a static one-line `requests` ↔ `httpx` cheat-sheet (`0` repository context functions; mean prompt `{a31['arm2_target_only']['mean_prompt_tokens']}` tokens).
3. **`Arm 3` (`arm3_embedding_topk`, legacy identifier) — Gemma 4 + Fixed HTTP Distractor**: The recorded experiment injects `distractors.py:preview_http_headers` by name, excluding session setup (mean prompt `{a31['arm3_embedding_topk']['mean_prompt_tokens']}` tokens). It does not test an embedding retriever.
4. **`Arm 4` (`arm4_callgraph_slice`) — Gemma 4 + StateBridge Call-Graph State Slice (Ours)**: Extracts the 1–2 hop static call-graph slice from `session_setup.py` (`prepare_client_session`, the family state-bridge helper, `finalize_step_contract`, and module-level state constants; mean prompt `{a31['arm4_callgraph_slice']['mean_prompt_tokens']}` tokens).
5. **`Arm 5` (`arm5_unrelated_slice`) — Gemma 4 + Equal-Token Unrelated Slice (Negative Control)**: Injects `distractors.py:compute_window_digest()`, matched in character/token budget to `Arm 4` (mean prompt `{a31['arm5_unrelated_slice']['mean_prompt_tokens']}` tokens, verified by PBT invariant `P4`) while sharing `0` call-graph edges with `execute_step`.

All LLM arms were executed against real **`google/models/gemma-4-31b-it`** (31B dense instruction-tuned) and **`google/models/gemma-4-26b-a4b-it`** (26B total / 4B active Mixture-of-Experts instruction-tuned) with `temperature=0.0`, separating `<thought>...</thought>` reasoning blocks from extracted code fences (`240` retained generations). The runner allows three attempts: `max_tokens=8192` first, then `16384` after invalid/truncated output or a request error; 7 retained records have `attempt=2`. Only final successful responses are preserved, so failed-attempt causes and costs cannot be reconstructed.

---

## 4. Empirical Results on Gemma 4

### 4.1 Overall 5-Arm Solve Rates and Statistical Significance

Table 2 summarizes the live loopback HTTP execution outcomes across all **270 runs** (replayed end-to-end in **{ssot['benchmark']['replay_elapsed_sec']} seconds** via `scripts/build_ssot.py --verify-replay`).

| Experimental Arm | `Gemma-4-31B-IT` Solve Rate | `Gemma-4-31B-IT` Outcomes (`PASS` / `WIRE_DIV` / `ERR`) | `Gemma-4-26B-A4B-IT` Solve Rate | `Gemma-4-26B-A4B-IT` Outcomes (`PASS` / `WIRE_DIV` / `ERR`) | Mean Prompt Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Arm 1`: Deterministic AST Codemod** | **{_fmt_pct(a31['arm1_codemod']['solve_rate'])}** (`{a31['arm1_codemod']['pass_count']}/30`) | {_fmt_outcomes(a31['arm1_codemod'])} | **{_fmt_pct(a26['arm1_codemod']['solve_rate'])}** (`{a26['arm1_codemod']['pass_count']}/30`) | {_fmt_outcomes(a26['arm1_codemod'])} | `0.0` |
| **`Arm 2`: Target File Only** | **{r31_a2}** (`{a31['arm2_target_only']['pass_count']}/30`) | {_fmt_outcomes(a31['arm2_target_only'])} | **{r26_a2}** (`{a26['arm2_target_only']['pass_count']}/30`) | {_fmt_outcomes(a26['arm2_target_only'])} | `{a31['arm2_target_only']['mean_prompt_tokens']}` |
| **`Arm 3`: HTTP Distractor** | **{r31_a3}** (`{a31['arm3_embedding_topk']['pass_count']}/30`) | {_fmt_outcomes(a31['arm3_embedding_topk'])} | **{r26_a3}** (`{a26['arm3_embedding_topk']['pass_count']}/30`) | {_fmt_outcomes(a26['arm3_embedding_topk'])} | `{a31['arm3_embedding_topk']['mean_prompt_tokens']}` |
| **`Arm 5`: Equal-Token Unrelated Slice** | **{r31_a5}** (`{a31['arm5_unrelated_slice']['pass_count']}/30`) | {_fmt_outcomes(a31['arm5_unrelated_slice'])} | **{r26_a5}** (`{a26['arm5_unrelated_slice']['pass_count']}/30`) | {_fmt_outcomes(a26['arm5_unrelated_slice'])} | `{a31['arm5_unrelated_slice']['mean_prompt_tokens']}` |
| **`Arm 4`: StateBridge Call-Graph Slice (Ours)** | **{r31_a4}** (`{a31['arm4_callgraph_slice']['pass_count']}/30`) | {_fmt_outcomes(a31['arm4_callgraph_slice'])} | **{r26_a4}** (`{a26['arm4_callgraph_slice']['pass_count']}/30`) | {_fmt_outcomes(a26['arm4_callgraph_slice'])} | `{a31['arm4_callgraph_slice']['mean_prompt_tokens']}` |

**Key Finding 1 — Call-Graph State Slicing (`Arm 4`) Outperforms Single-File and HTTP-Distractor Context by +{_fmt_pct(c31['arm4_vs_arm2']['diff'])} to +{_fmt_pct(c26['arm4_vs_arm2']['diff'])}:**
On `Gemma-4-31B-IT`, `StateBridge` (`Arm 4`) improves solve rate from **{r31_a2}** (`{a31['arm2_target_only']['pass_count']}/30`) in `Arm 2` and **{r31_a3}** (`{a31['arm3_embedding_topk']['pass_count']}/30`) in `Arm 3` to **{r31_a4}** (`{a31['arm4_callgraph_slice']['pass_count']}/30`), winning `{c31['arm4_vs_arm2']['discordant_a_wins']}` discordant workflow pairs against `{c31['arm4_vs_arm2']['discordant_b_wins']}` losses (exact two-sided McNemar $p = {c31['arm4_vs_arm2']['mcnemar_p']}$, 10,000-draw paired bootstrap 95% CI `[{_fmt_pct(c31['arm4_vs_arm2']['ci95_low'])}, {_fmt_pct(c31['arm4_vs_arm2']['ci95_high'])}]`). On `Gemma-4-26B-A4B-IT`, `StateBridge` (`Arm 4`) jumps from **{r26_a2}** (`{a26['arm2_target_only']['pass_count']}/30`) in `Arm 2` and `Arm 3` to **{r26_a4}** (`{a26['arm4_callgraph_slice']['pass_count']}/30`), winning `{c26['arm4_vs_arm2']['discordant_a_wins']}` discordant pairs against `{c26['arm4_vs_arm2']['discordant_b_wins']}` losses (exact McNemar $p = {c26['arm4_vs_arm2']['mcnemar_p']}$, 95% CI `[{_fmt_pct(c26['arm4_vs_arm2']['ci95_low'])}, {_fmt_pct(c26['arm4_vs_arm2']['ci95_high'])}]`).

**Key Finding 2 — Ruling Out Prompt-Length Confounds (`Arm 4` vs `Arm 5`):**
Comparing `Arm 4` against the equal-token unrelated slice (`Arm 5`) confirms that adding repository tokens without call-graph connectivity yields no meaningful gain (`{r31_a5}` on `31B-IT`, `{r26_a5}` on `26B-A4B-IT`). `Arm 4` beats `Arm 5` by **+{_fmt_pct(c31['arm4_vs_arm5']['diff'])}** on `Gemma-4-31B-IT` ($p = {c31['arm4_vs_arm5']['mcnemar_p']}$, 95% CI `[{_fmt_pct(c31['arm4_vs_arm5']['ci95_low'])}, {_fmt_pct(c31['arm4_vs_arm5']['ci95_high'])}]`) and **+{_fmt_pct(c26['arm4_vs_arm5']['diff'])}** on `Gemma-4-26B-A4B-IT` ($p = {c26['arm4_vs_arm5']['mcnemar_p']}$, 95% CI `[{_fmt_pct(c26['arm4_vs_arm5']['ci95_low'])}, {_fmt_pct(c26['arm4_vs_arm5']['ci95_high'])}]`).

### 4.2 Per-Family Breakdown and Dense (`31B-IT`) vs MoE (`26B-A4B-IT`) Reasoning Telemetry

| Divergence Family (`6` Workflows Each) | `Arm 1` Codemod | `31B-IT` `Arm 2`/`3`/`5` | `31B-IT` `Arm 4` (`StateBridge`) | `26B-A4B-IT` `Arm 2`/`3`/`5` | `26B-A4B-IT` `Arm 4` (`StateBridge`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
{fam_table}

Comparing the two Gemma 4 architectures reveals a striking efficiency-accuracy trade-off:
- **MoE `Gemma-4-26B-A4B-IT` (4B active parameters) matches or exceeds Dense `31B-IT` (`{r26_a4}` vs `{r31_a4}`) by investing more internal `<thought>` reasoning**: Across `Arm 4`, `Gemma-4-26B-A4B-IT` generates a mean of **`{a26['arm4_callgraph_slice']['mean_thought_chars']}`** `<thought>` characters (`{a26['arm4_callgraph_slice']['mean_completion_tokens']}` completion tokens) compared to **`{a31['arm4_callgraph_slice']['mean_thought_chars']}`** `<thought>` characters (`{a31['arm4_callgraph_slice']['mean_completion_tokens']}` completion tokens) on `Gemma-4-31B-IT`.
- **Where the remaining errors occur**: On `31B-IT`, wire divergences number `{a31['arm2_target_only']['outcome_counts'].get('STEP_K_WIRE_DIVERGENCE', 0)}/30`, `{a31['arm3_embedding_topk']['outcome_counts'].get('STEP_K_WIRE_DIVERGENCE', 0)}/30`, and `{a31['arm5_unrelated_slice']['outcome_counts'].get('STEP_K_WIRE_DIVERGENCE', 0)}/30` for Arms 2, 3, and 5 respectively; on `26B-A4B-IT`, they number `{a26['arm2_target_only']['outcome_counts'].get('STEP_K_WIRE_DIVERGENCE', 0)}/30`, `{a26['arm3_embedding_topk']['outcome_counts'].get('STEP_K_WIRE_DIVERGENCE', 0)}/30`, and `{a26['arm5_unrelated_slice']['outcome_counts'].get('STEP_K_WIRE_DIVERGENCE', 0)}/30`. Arm 4 has `{a31['arm4_callgraph_slice']['outcome_counts'].get('STEP_K_WIRE_DIVERGENCE', 0)}/30` and `{a26['arm4_callgraph_slice']['outcome_counts'].get('STEP_K_WIRE_DIVERGENCE', 0)}/30`. These counts describe this synthetic benchmark; they do not establish dense-retrieval failure.

---

## 5. Related Works & Citations

1. **Gemma 4 Technical Report (Google DeepMind, 2026)**: Introduces the Gemma 4 open-weight model family, including the dense `Gemma-4-31B-IT` and sparse Mixture-of-Experts `Gemma-4-26B-A4B-IT` architectures with native `<thought>` reasoning traces. Our work provides the first controlled evaluation of both architectures on multi-step stateful library migration.
2. **CodeMigrationBench (2026) & PyMigBench (2023)**: Benchmarks for real-world Python dependency and library migrations. While PyMigBench catalogs syntax-level code changes and CodeMigrationBench evaluates repository test pass rates, `StateBridge-Bench` isolates *stateful false-friend APIs* using a deterministic loopback HTTP state machine that pinpoints exact wire-level step divergences.
3. **RepoTransBench (2024) & GitChameleon 2.0 (2025)**: Evaluate repository-level code translation and version-conditioned API generation. Our `TwinAPI` audit measures API-descriptor similarity (`0.873` cosine); the fixed HTTP-distractor control (`Arm 3`) does not evaluate vector retrieval.
4. **RFC 9110 (HTTP Semantics) & RFC 6265 (HTTP State Management Mechanism)**: Define the redirect status semantics (`301`/`302`/`303`), cross-origin `Authorization` header stripping requirements, and `Cookie`/`Set-Cookie` domain scoping rules that underpin Divergence Families `F1`, `F2`, and `F3`.

---

## 6. Reproduction & Open-Source Artifacts

All replay steps run locally without API keys once source data and dependencies are available. Runtime depends on the machine and worker limit; the measured run took {ssot['benchmark']['replay_elapsed_sec']} seconds with three workers. The public [GitHub repository](https://github.com/GGGODLIN/gemma4-state-bridge), [Run All Notebook](https://github.com/GGGODLIN/gemma4-state-bridge/blob/main/notebooks/reproduce_statebridge.ipynb), and [figure gallery](https://github.com/GGGODLIN/gemma4-state-bridge/tree/main/paper/figures) contain the source and results: [five-arm comparison](https://github.com/GGGODLIN/gemma4-state-bridge/blob/main/paper/figures/fig1_5arm_ablation.png), [family breakdown](https://github.com/GGGODLIN/gemma4-state-bridge/blob/main/paper/figures/fig2_family_breakdown.png), and [TwinAPI audit](https://github.com/GGGODLIN/gemma4-state-bridge/blob/main/paper/figures/fig3_twinapi_paradox.png). The Notebook searches local files and Kaggle Datasets before cloning; without data or Internet it labels its embedded summary as unverified.
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
"""


def _build_paper_pdf(ssot: dict[str, Any], audit: dict[str, Any], figs: dict[str, Path], pdf_path: Path) -> None:
  from reportlab.lib import colors
  from reportlab.lib.pagesizes import LETTER
  from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
  from reportlab.lib.units import inch
  from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
  )

  pdf_path.parent.mkdir(parents=True, exist_ok=True)
  doc = SimpleDocTemplate(
    str(pdf_path),
    pagesize=LETTER,
    leftMargin=0.65 * inch,
    rightMargin=0.65 * inch,
    topMargin=0.65 * inch,
    bottomMargin=0.65 * inch,
    pageCompression=0,
    title="StateBridge x TwinAPI: Resolving the Stateful False-Friend Paradox in Cross-Library Code Migration with Gemma 4",
    author="StateBridge Research Authors",
  )

  styles = getSampleStyleSheet()
  title_style = ParagraphStyle(
    "PaperTitle",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=15.5,
    leading=19,
    textColor=colors.HexColor("#0b0b0b"),
    spaceAfter=6,
  )
  subtitle_style = ParagraphStyle(
    "PaperSubtitle",
    parent=styles["Normal"],
    fontName="Helvetica-Oblique",
    fontSize=9.8,
    leading=13,
    alignment=1,
    textColor=colors.HexColor("#52514e"),
    spaceAfter=12,
  )
  h1_style = ParagraphStyle(
    "PaperH1",
    parent=styles["Heading1"],
    fontName="Helvetica-Bold",
    fontSize=12,
    leading=15,
    textColor=colors.HexColor("#0b0b0b"),
    spaceBefore=10,
    spaceAfter=5,
  )
  body_style = ParagraphStyle(
    "PaperBody",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=9.3,
    leading=13.2,
    textColor=colors.HexColor("#0b0b0b"),
    spaceAfter=6,
  )
  abstract_style = ParagraphStyle(
    "PaperAbstract",
    parent=body_style,
    leftIndent=18,
    rightIndent=18,
    fontSize=9.0,
    leading=12.6,
    backColor=colors.HexColor("#f9f9f7"),
    borderPadding=6,
  )
  caption_style = ParagraphStyle(
    "PaperCaption",
    parent=body_style,
    fontName="Helvetica-Oblique",
    fontSize=8.3,
    leading=11,
    textColor=colors.HexColor("#52514e"),
    spaceAfter=8,
  )

  twin = ssot["twinapi_audit"]
  m31 = ssot["models"]["gemma-4-31b-it"]
  m26 = ssot["models"]["gemma-4-26b-a4b-it"]
  a31 = m31["arms"]
  a26 = m26["arms"]
  c31 = m31["paired_comparisons"]
  c26 = m26["paired_comparisons"]

  story: list[Any] = []
  story.append(Paragraph(
    "StateBridge &times; TwinAPI: Resolving the Stateful False-Friend Paradox in Cross-Library Code Migration with Gemma 4",
    title_style,
  ))
  story.append(Paragraph(
    "Gemma 4 Developer Agent Paper Track &mdash; Empirical Paper, Benchmark Resource (StateBridge-Bench), and Agent Skill",
    subtitle_style,
  ))

  story.append(Paragraph(
    f"<b>Abstract.</b> When developer agents migrate multi-file codebases between sibling libraries such as Python's "
    f"<i>requests</i> (2.34.2) and <i>httpx</i> (0.28.1), they encounter the <b>Stateful False-Friend Paradox</b>: "
    f"{_fmt_pct(twin['top_level_exact_name_overlap_ratio'])} (8/9) of top-level functions and "
    f"{_fmt_pct(twin['session_client_core_overlap_ratio'])} (10/12) of core Session/Client methods share identical names, and "
    f"{twin['stateful_false_friend_count']} stateful false-friend API pairs exhibit a mean 256-dimensional Gemini embedding "
    f"cosine similarity of <b>{twin['mean_false_friend_embedding_cosine_256d']:.3f}</b>. "
    f"Yet their multi-step runtime state machines diverge silently on redirect following, cross-origin auth stripping, "
    f"domain-scoped cookie jars, streaming context lifecycles, and status-exception hierarchies. Across 270 live loopback "
    f"HTTP state-machine executions (240 real Gemma 4 generations + 30 AST codemod runs), deterministic AST codemods "
    f"(Arm 1) pass {_fmt_pct(a31['arm1_codemod']['solve_rate'])} ({a31['arm1_codemod']['pass_count']}/30), while target-only "
    f"prompting (Arm 2) and fixed HTTP-distractor context (Arm 3) both stall at "
    f"<b>{_fmt_pct(a31['arm2_target_only']['solve_rate'])} ({a31['arm2_target_only']['pass_count']}/30)</b> on Gemma-4-31B-IT "
    f"and <b>{_fmt_pct(a26['arm2_target_only']['solve_rate'])} ({a26['arm2_target_only']['pass_count']}/30)</b> on Gemma-4-26B-A4B-IT. "
    f"We introduce <b>StateBridge</b> (Arm 4), which extracts 1&ndash;2 hop static call-graph state slices from cross-file session "
    f"initializers. StateBridge raises multi-step wire-equivalence solve rates to "
    f"<b>{_fmt_pct(a31['arm4_callgraph_slice']['solve_rate'])} ({a31['arm4_callgraph_slice']['pass_count']}/30)</b> on dense "
    f"Gemma-4-31B-IT (exact McNemar p = {c31['arm4_vs_arm2']['mcnemar_p']}) and "
    f"<b>{_fmt_pct(a26['arm4_callgraph_slice']['solve_rate'])} ({a26['arm4_callgraph_slice']['pass_count']}/30)</b> on MoE "
    f"Gemma-4-26B-A4B-IT (McNemar p = {c26['arm4_vs_arm2']['mcnemar_p']}), while an equal-token unrelated control "
    f"(Arm 5) remains at {_fmt_pct(a31['arm5_unrelated_slice']['solve_rate'])} ({a31['arm5_unrelated_slice']['pass_count']}/30) "
    f"and {_fmt_pct(a26['arm5_unrelated_slice']['solve_rate'])} ({a26['arm5_unrelated_slice']['pass_count']}/30).",
    abstract_style,
  ))
  story.append(Spacer(1, 8))

  story.append(Paragraph("1. Introduction &amp; The Stateful False-Friend Paradox", h1_style))
  story.append(Paragraph(
    "Migrating production code across sibling HTTP libraries is deceptively difficult for Large Language Models and "
    "Retrieval-Augmented Generation (RAG) agents. Unlike cross-language translation where syntax forces structural "
    "rewrites, sibling libraries in the same ecosystem intentionally mirror one another's surface ergonomics while "
    "redesigning implicit stateful defaults. When a developer agent edits a workflow module (workflow_steps.py) without "
    "retrieving the cross-file session factory (session_setup.py), local syntax checks and unit mocks pass while live "
    "multi-step wire exchanges fail silently at Step k.",
    body_style,
  ))

  if figs.get("fig1") and figs["fig1"].exists():
    story.append(Image(str(figs["fig1"]), width=6.8 * inch, height=3.4 * inch))
    story.append(Paragraph(
      f"<b>Figure 1:</b> 5-Arm wire-trace equivalence solve rates across 30 multi-step HTTP workflows (270 live loopback runs). "
      f"StateBridge Call-Graph Slicing (Arm 4) achieves {_fmt_pct(a31['arm4_callgraph_slice']['solve_rate'])} "
      f"({a31['arm4_callgraph_slice']['pass_count']}/30) on Gemma-4-31B-IT and "
      f"{_fmt_pct(a26['arm4_callgraph_slice']['solve_rate'])} ({a26['arm4_callgraph_slice']['pass_count']}/30) on Gemma-4-26B-A4B-IT.",
      caption_style,
    ))

  story.append(Paragraph("2. The TwinAPI Empirical Audit (requests 2.34.2 vs. httpx 0.28.1)", h1_style))
  story.append(Paragraph(
    "We inspected the runtime objects, parameter signatures, and docstrings of 15 paired APIs across requests 2.34.2 "
    "and httpx 0.28.1, embedding each descriptor into 256 dimensions via google/models/gemini-embedding-001.",
    body_style,
  ))

  t1_data = [["ID", "requests 2.34.2", "httpx 0.28.1", "Name Sim", "Sig Iso", "256-d Cos", "False Friend", "Family"]]
  for p in audit.get("pairs", []):
    t1_data.append([
      p["pair_id"],
      p["req_symbol"].replace("requests.cookies.RequestsCookieJar", "req.CookieJar").replace("requests.", "req."),
      p["httpx_symbol"],
      f"{p['short_name_similarity']:.2f}",
      f"{p['signature_isomorphism']:.2f}",
      f"{p['embedding_cosine_256d']:.3f}",
      "Yes" if p["is_stateful_false_friend"] else "No",
      p["divergence_family"].split("_")[0],
    ])
  t1 = Table(t1_data, colWidths=[0.36 * inch, 1.82 * inch, 1.72 * inch, 0.58 * inch, 0.50 * inch, 0.62 * inch, 0.72 * inch, 0.49 * inch])
  t1.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a78d6")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#fcfcfb"), colors.HexColor("#f2f1ec")]),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c3c2b7")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
  ]))
  story.append(t1)
  story.append(Spacer(1, 8))

  if figs.get("fig3") and figs["fig3"].exists():
    story.append(Image(str(figs["fig3"]), width=6.8 * inch, height=3.5 * inch))
    story.append(Paragraph(
      f"<b>Figure 2:</b> TwinAPI 256-d embedding cosine similarity across all {twin['total_pairs_audited']} audited pairs. "
      f"The {twin['stateful_false_friend_count']} stateful false-friend pairs average {twin['mean_false_friend_embedding_cosine_256d']:.3f}. "
      "API similarity is separate from the fixed HTTP-distractor control (Arm 3); no vector retrieval was evaluated.",
      caption_style,
    ))

  story.append(PageBreak())
  story.append(Paragraph("3. StateBridge-Bench &amp; 5-Arm Ablation Results on Gemma 4", h1_style))
  story.append(Paragraph(
    "Table 2 reports the complete 5-arm ablation across Gemma-4-31B-IT (dense) and Gemma-4-26B-A4B-IT (MoE), alongside "
    "exact two-sided McNemar p-values and 10,000-draw paired bootstrap 95% confidence intervals. "
    "Arm 3 selects preview_http_headers by name, excluding session_setup.py; no vector retrieval was tested. "
    "Retained responses include seven second attempts with a 16384-token budget, following an initial 8192-token attempt. "
    f"Wire divergences for Arms 2/3/5 are {'/'.join(str(a31[k]['outcome_counts'].get('STEP_K_WIRE_DIVERGENCE', 0)) for k in ('arm2_target_only', 'arm3_embedding_topk', 'arm5_unrelated_slice'))} of 30 (31B) and "
    f"{'/'.join(str(a26[k]['outcome_counts'].get('STEP_K_WIRE_DIVERGENCE', 0)) for k in ('arm2_target_only', 'arm3_embedding_topk', 'arm5_unrelated_slice'))} (26B).",
    body_style,
  ))

  t2_data = [
    ["Experimental Arm", "31B-IT Pass", "31B-IT <thought>", "26B-A4B Pass", "26B-A4B <thought>", "Prompt Tok"],
    ["Arm 1: AST Codemod", "0.0% (0/30)", "0 chars", "0.0% (0/30)", "0 chars", "0.0"],
    [
      "Arm 2: Target File Only",
      f"{_fmt_pct(a31['arm2_target_only']['solve_rate'])} ({a31['arm2_target_only']['pass_count']}/30)",
      f"{a31['arm2_target_only']['mean_thought_chars']} chars",
      f"{_fmt_pct(a26['arm2_target_only']['solve_rate'])} ({a26['arm2_target_only']['pass_count']}/30)",
      f"{a26['arm2_target_only']['mean_thought_chars']} chars",
      f"{a31['arm2_target_only']['mean_prompt_tokens']}",
    ],
    [
      "Arm 3: HTTP Distractor",
      f"{_fmt_pct(a31['arm3_embedding_topk']['solve_rate'])} ({a31['arm3_embedding_topk']['pass_count']}/30)",
      f"{a31['arm3_embedding_topk']['mean_thought_chars']} chars",
      f"{_fmt_pct(a26['arm3_embedding_topk']['solve_rate'])} ({a26['arm3_embedding_topk']['pass_count']}/30)",
      f"{a26['arm3_embedding_topk']['mean_thought_chars']} chars",
      f"{a31['arm3_embedding_topk']['mean_prompt_tokens']}",
    ],
    [
      "Arm 5: Equal-Token Unrelated",
      f"{_fmt_pct(a31['arm5_unrelated_slice']['solve_rate'])} ({a31['arm5_unrelated_slice']['pass_count']}/30)",
      f"{a31['arm5_unrelated_slice']['mean_thought_chars']} chars",
      f"{_fmt_pct(a26['arm5_unrelated_slice']['solve_rate'])} ({a26['arm5_unrelated_slice']['pass_count']}/30)",
      f"{a26['arm5_unrelated_slice']['mean_thought_chars']} chars",
      f"{a31['arm5_unrelated_slice']['mean_prompt_tokens']}",
    ],
    [
      "Arm 4: StateBridge (Ours)",
      f"{_fmt_pct(a31['arm4_callgraph_slice']['solve_rate'])} ({a31['arm4_callgraph_slice']['pass_count']}/30)",
      f"{a31['arm4_callgraph_slice']['mean_thought_chars']} chars",
      f"{_fmt_pct(a26['arm4_callgraph_slice']['solve_rate'])} ({a26['arm4_callgraph_slice']['pass_count']}/30)",
      f"{a26['arm4_callgraph_slice']['mean_thought_chars']} chars",
      f"{a31['arm4_callgraph_slice']['mean_prompt_tokens']}",
    ],
  ]
  t2 = Table(t2_data, colWidths=[1.70 * inch, 1.00 * inch, 1.18 * inch, 1.00 * inch, 1.24 * inch, 0.78 * inch])
  t2.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eb6834")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.2),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#fcfcfb"), colors.HexColor("#f2f1ec")]),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c3c2b7")),
  ]))
  story.append(t2)
  story.append(Spacer(1, 10))

  if figs.get("fig2") and figs["fig2"].exists():
    f26_rates = [f_obj["pass_count"] / max(f_obj["total"], 1) for f_obj in a26["arm4_callgraph_slice"]["by_family"].values()]
    story.append(Image(str(figs["fig2"]), width=6.8 * inch, height=3.1 * inch))
    story.append(Paragraph(
      f"<b>Figure 3:</b> Per-family migration solve rates across the 5 stateful HTTP divergence families (F1..F5). "
      f"StateBridge (Arm 4) achieves {_fmt_pct(min(f26_rates))}&ndash;{_fmt_pct(max(f26_rates))} across all five families on Gemma-4-26B-A4B-IT.",
      caption_style,
    ))

  story.append(Paragraph("4. Paired Statistical Tests &amp; Property-Based Verification (P1..P4)", h1_style))
  t3_data = [
    ["Model", "Comparison", "Rate Diff", "Discordant (W/L)", "Exact McNemar p", "95% Bootstrap CI"],
    [
      "Gemma-4-31B-IT",
      "Arm 4 vs Arm 2",
      f"+{_fmt_pct(c31['arm4_vs_arm2']['diff'])}",
      f"{c31['arm4_vs_arm2']['discordant_a_wins']} / {c31['arm4_vs_arm2']['discordant_b_wins']}",
      f"{c31['arm4_vs_arm2']['mcnemar_p']:.6e}",
      f"[{_fmt_pct(c31['arm4_vs_arm2']['ci95_low'])}, {_fmt_pct(c31['arm4_vs_arm2']['ci95_high'])}]",
    ],
    [
      "Gemma-4-31B-IT",
      "Arm 4 vs Arm 3",
      f"+{_fmt_pct(c31['arm4_vs_arm3']['diff'])}",
      f"{c31['arm4_vs_arm3']['discordant_a_wins']} / {c31['arm4_vs_arm3']['discordant_b_wins']}",
      f"{c31['arm4_vs_arm3']['mcnemar_p']:.6e}",
      f"[{_fmt_pct(c31['arm4_vs_arm3']['ci95_low'])}, {_fmt_pct(c31['arm4_vs_arm3']['ci95_high'])}]",
    ],
    [
      "Gemma-4-31B-IT",
      "Arm 4 vs Arm 5",
      f"+{_fmt_pct(c31['arm4_vs_arm5']['diff'])}",
      f"{c31['arm4_vs_arm5']['discordant_a_wins']} / {c31['arm4_vs_arm5']['discordant_b_wins']}",
      f"{c31['arm4_vs_arm5']['mcnemar_p']:.6e}",
      f"[{_fmt_pct(c31['arm4_vs_arm5']['ci95_low'])}, {_fmt_pct(c31['arm4_vs_arm5']['ci95_high'])}]",
    ],
    [
      "Gemma-4-26B-A4B-IT",
      "Arm 4 vs Arm 2",
      f"+{_fmt_pct(c26['arm4_vs_arm2']['diff'])}",
      f"{c26['arm4_vs_arm2']['discordant_a_wins']} / {c26['arm4_vs_arm2']['discordant_b_wins']}",
      f"{c26['arm4_vs_arm2']['mcnemar_p']:.6e}",
      f"[{_fmt_pct(c26['arm4_vs_arm2']['ci95_low'])}, {_fmt_pct(c26['arm4_vs_arm2']['ci95_high'])}]",
    ],
    [
      "Gemma-4-26B-A4B-IT",
      "Arm 4 vs Arm 3",
      f"+{_fmt_pct(c26['arm4_vs_arm3']['diff'])}",
      f"{c26['arm4_vs_arm3']['discordant_a_wins']} / {c26['arm4_vs_arm3']['discordant_b_wins']}",
      f"{c26['arm4_vs_arm3']['mcnemar_p']:.6e}",
      f"[{_fmt_pct(c26['arm4_vs_arm3']['ci95_low'])}, {_fmt_pct(c26['arm4_vs_arm3']['ci95_high'])}]",
    ],
    [
      "Gemma-4-26B-A4B-IT",
      "Arm 4 vs Arm 5",
      f"+{_fmt_pct(c26['arm4_vs_arm5']['diff'])}",
      f"{c26['arm4_vs_arm5']['discordant_a_wins']} / {c26['arm4_vs_arm5']['discordant_b_wins']}",
      f"{c26['arm4_vs_arm5']['mcnemar_p']:.6e}",
      f"[{_fmt_pct(c26['arm4_vs_arm5']['ci95_low'])}, {_fmt_pct(c26['arm4_vs_arm5']['ci95_high'])}]",
    ],
  ]
  t3 = Table(t3_data, colWidths=[1.35 * inch, 1.15 * inch, 0.9 * inch, 1.1 * inch, 1.15 * inch, 1.25 * inch])
  t3.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1baf7a")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.0),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#fcfcfb"), colors.HexColor("#f2f1ec")]),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c3c2b7")),
  ]))
  story.append(t3)
  story.append(Spacer(1, 8))
  story.append(Paragraph(
    f"<b>Conclusion.</b> High surface API similarity and dense embedding cosine ({twin['mean_false_friend_embedding_cosine_256d']:.3f}) "
    f"create a systematic trap for developer agents migrating stateful workflows. By pairing 1&ndash;2 hop static call-graph "
    f"state slicing with a deterministic loopback HTTP wire-trace oracle, StateBridge unlocks "
    f"{_fmt_pct(a31['arm4_callgraph_slice']['solve_rate'])}&ndash;{_fmt_pct(a26['arm4_callgraph_slice']['solve_rate'])} accuracy on Gemma 4.",
    body_style,
  ))

  doc.build(story)


def _build_and_execute_notebook(base: Path, nb_path: Path) -> None:
  import os
  import tempfile
  import nbformat
  from nbclient import NotebookClient

  frozen = json.loads((base / "artifacts/metrics_ssot.json").read_text())
  bootstrap = (base / "scripts/notebook_bootstrap.py").read_text()
  cell1 = bootstrap + "\nROOT = locate_statebridge_repo()\n" + f"ssot = {frozen!r}\n"
  cell1 += '''twin = ssot["twinapi_audit"]
print(f"Top-level name overlap: {twin['top_level_exact_name_overlap_ratio']*100:.1f}% (8/9)")
print(f"Session/Client overlap: {twin['session_client_core_overlap_ratio']*100:.1f}% (10/12)")
print(f"Mean 256-d audit cosine: {twin['mean_false_friend_embedding_cosine_256d']:.3f}")
'''
  cell2 = '''if ROOT is not None:
  require_replay_dependencies()
  import shutil
  import time
  sys.path.insert(0, str(ROOT / "scripts"))
  import build_ssot as bs
  frozen_models = ssot["models"]
  replay_dir = Path(tempfile.mkdtemp(prefix="statebridge-replay-", dir=Path.cwd()))
  shutil.copy2(ROOT / "artifacts/twinapi_audit.json", replay_dir / "twinapi_audit.json")
  t0 = time.monotonic()
  ssot = bs.build_ssot(ROOT / "artifacts/raw_llm_generations.jsonl", replay_dir)
  assert ssot["models"] == frozen_models, "Live replay differs from embedded published results"
  live_elapsed = round(time.monotonic() - t0, 3)
  runs = [json.loads(line) for line in (replay_dir / "runs.jsonl").read_text().splitlines()]
  assert len(runs) == 270
  print(f"LIVE REPLAY VERIFIED: {len(runs)} runs, exact statistics match; elapsed={live_elapsed}s")
else:
  print("SUMMARY ONLY: live replay skipped; no reproduction claim.")
'''
  cell3 = '''for model_key, model in ssot["models"].items():
  print(f"\\n=== {model_key} ===")
  for arm_id, arm in model["arms"].items():
    print(f"{arm_id}: {arm['solve_rate']*100:.1f}% ({arm['pass_count']}/{arm['total']})")
  for comparison, stats in model["paired_comparisons"].items():
    print(f"{comparison}: p={stats['mcnemar_p']}, discordant={stats['discordant_a_wins']}:{stats['discordant_b_wins']}, CI=[{stats['ci95_low']*100:.1f}%, {stats['ci95_high']*100:.1f}%]")
'''
  cell4 = '''if ROOT is not None:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from IPython.display import Image, display
  order = list(ssot["models"]["gemma-4-31b-it"]["arms"])
  fig, ax = plt.subplots(figsize=(9, 4))
  for i, (model_key, model) in enumerate(ssot["models"].items()):
    ax.bar([j + i * 0.36 for j in range(len(order))],
           [model["arms"][arm]["pass_count"] for arm in order],
           width=0.34, label=model_key, color=("#2a78d6", "#eb6834")[i])
  ax.set_xticks([j + 0.18 for j in range(len(order))], ["AST", "Target", "HTTP distractor", "StateBridge", "Unrelated"])
  ax.set_ylabel("Equivalent workflows / 30")
  ax.set_ylim(0, 33)
  ax.legend()
  fig.tight_layout()
  figure_path = replay_dir / "live_solve_rates.png"
  fig.savefig(figure_path, dpi=160)
  plt.close(fig)
  display(Image(filename=str(figure_path)))
  print("Figure generated from live replay statistics.")
else:
  print("SUMMARY ONLY: figures were not recomputed.")
'''
  nb = nbformat.v4.new_notebook(cells=[
    nbformat.v4.new_markdown_cell(
      "# StateBridge × TwinAPI — Reproduction\n"
      "Run All from an empty directory. Finds local files or a Kaggle Dataset, otherwise shallow-clones "
      "[the public repository](https://github.com/GGGODLIN/gemma4-state-bridge). "
      "Internet is needed for the first clone/dependency installation. Offline without data: "
      "precomputed summary only, clearly marked; no live-verification claim. "
      "Arm 3 is a fixed HTTP distractor; its legacy identifier is retained. "
      "Replay writes to a working-directory temporary folder; mounted inputs remain read-only."
    ),
    *[nbformat.v4.new_code_cell(code) for code in (cell1, cell2, cell3, cell4)],
  ], metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
  with tempfile.TemporaryDirectory(prefix="statebridge-notebook-") as temp:
    temp_path = Path(temp)
    working = temp_path / "working"
    working.mkdir()
    client = NotebookClient(nb, timeout=240, resources={"metadata": {"path": str(working)}})
    client.execute(env={**os.environ, "STATEBRIDGE_INPUT_ROOT": str(base), "STATEBRIDGE_OFFLINE": "1"})
  nb_path.parent.mkdir(parents=True, exist_ok=True)
  nbformat.write(nb, nb_path)


def _build_readmes(ssot: dict[str, Any], base: Path) -> None:
  twin = ssot["twinapi_audit"]
  m31 = ssot["models"]["gemma-4-31b-it"]
  m26 = ssot["models"]["gemma-4-26b-a4b-it"]
  a31 = m31["arms"]
  a26 = m26["arms"]
  c31 = m31["paired_comparisons"]
  c26 = m26["paired_comparisons"]

  top_pct = _fmt_pct(twin["top_level_exact_name_overlap_ratio"])
  sess_pct = _fmt_pct(twin["session_client_core_overlap_ratio"])
  mean_cos = f"{twin['mean_false_friend_embedding_cosine_256d']:.3f}"

  readme_en = f"""# StateBridge × TwinAPI: Resolving the Stateful False-Friend Paradox in Cross-Library Code Migration with Gemma 4

[![Track](https://img.shields.io/badge/Kaggle-Gemma_4_Developer_Agent_Paper-2a78d6)](#)
[![Models](https://img.shields.io/badge/Models-Gemma--4--31B--IT_%26_26B--A4B--IT-eb6834)](#)
[![Tests](https://img.shields.io/badge/Tests-PBT_P1..P4_%26_regressions-0ca30c)](#)

> **[繁體中文說明 (README.zh-TW.md)](/README.zh-TW.md)** · **[Kaggle Paper Track Writeup](/writeup/KAGGLE_WRITEUP.md)** · **[Full Paper PDF](/paper/paper.pdf)** · **[Reproducible Notebook](/notebooks/reproduce_statebridge.ipynb)** · **[Interactive HTML Explorer](/explorer/index.html)**

---

## Overview

When autonomous coding agents migrate repositories across sibling libraries such as Python's `requests` (`2.34.2`) and `httpx` (`0.28.1`), they confront the **Stateful False-Friend Paradox**:
- **{top_pct}** (`8/9`) of top-level functions (`get`, `post`, `put`, `delete`, `head`, `options`, `patch`, `request`) and **{sess_pct}** (`10/12`) of core `Session`/`Client` methods share **identical names**.
- Across 12 stateful false-friend API pairs spanning 5 multi-step divergence families (`F1..F5`), their 256-dimensional Gemini embeddings (`google/models/gemini-embedding-001`, `dimensions=256`) exhibit a mean cosine similarity of **{mean_cos}**.
- Yet at runtime, multi-step HTTP state transitions diverge silently on `302` redirect defaults (`allow_redirects=True` vs `follow_redirects=False`), cross-origin `Authorization` stripping and response hooks, domain-scoped `CookieJar` extraction (`.get_dict()`), streaming context lifecycles (`client.stream()`), and inverted exception hierarchies (`HTTPError` vs `HTTPStatusError`).

**StateBridge** combines a **1–2 hop static call-graph state-slice retriever (`Arm 4`)** with a **deterministic loopback HTTP state-machine oracle (`127.0.0.1:0`)** that verifies wire-level equivalence across multi-step workflows.

---

## Headline Empirical Results (270 Live Loopback HTTP Runs)

| Experimental Arm | `Gemma-4-31B-IT` (Dense) | `Gemma-4-26B-A4B-IT` (MoE) | Mean Prompt Tokens |
| :--- | :---: | :---: | :---: |
| **`Arm 1`: Deterministic AST Codemod** | **0.0%** (`0/30`) | **0.0%** (`0/30`) | `0.0` |
| **`Arm 2`: Target File Only + Cheat-Sheet** | **{_fmt_pct(a31['arm2_target_only']['solve_rate'])}** (`{a31['arm2_target_only']['pass_count']}/30`) | **{_fmt_pct(a26['arm2_target_only']['solve_rate'])}** (`{a26['arm2_target_only']['pass_count']}/30`) | `{a31['arm2_target_only']['mean_prompt_tokens']}` |
| **`Arm 3`: HTTP Distractor (Decoy)** | **{_fmt_pct(a31['arm3_embedding_topk']['solve_rate'])}** (`{a31['arm3_embedding_topk']['pass_count']}/30`) | **{_fmt_pct(a26['arm3_embedding_topk']['solve_rate'])}** (`{a26['arm3_embedding_topk']['pass_count']}/30`) | `{a31['arm3_embedding_topk']['mean_prompt_tokens']}` |
| **`Arm 5`: Equal-Token Unrelated Slice (Control)** | **{_fmt_pct(a31['arm5_unrelated_slice']['solve_rate'])}** (`{a31['arm5_unrelated_slice']['pass_count']}/30`) | **{_fmt_pct(a26['arm5_unrelated_slice']['solve_rate'])}** (`{a26['arm5_unrelated_slice']['pass_count']}/30`) | `{a31['arm5_unrelated_slice']['mean_prompt_tokens']}` |
| **`Arm 4`: StateBridge Call-Graph Slice (Ours)** | **{_fmt_pct(a31['arm4_callgraph_slice']['solve_rate'])}** (`{a31['arm4_callgraph_slice']['pass_count']}/30`, $p={c31['arm4_vs_arm2']['mcnemar_p']}$) | **{_fmt_pct(a26['arm4_callgraph_slice']['solve_rate'])}** (`{a26['arm4_callgraph_slice']['pass_count']}/30`, $p={c26['arm4_vs_arm2']['mcnemar_p']}$) | `{a31['arm4_callgraph_slice']['mean_prompt_tokens']}` |

---

## Repository Structure

- [`writeup/KAGGLE_WRITEUP.md`](/writeup/KAGGLE_WRITEUP.md) — Official Kaggle Paper Track Writeup (`<= 3,000` words).
- [`paper/paper.pdf`](/paper/paper.pdf) & [`paper/paper.md`](/paper/paper.md) — Full arXiv-formatted paper and vector/high-DPI figures.
- [`notebooks/reproduce_statebridge.ipynb`](/notebooks/reproduce_statebridge.ipynb) — Self-contained executed Kaggle Notebook that replays all 270 runs against the live loopback HTTP state machine in `{ssot['benchmark']['replay_elapsed_sec']}s`.
- [`explorer/index.html`](/explorer/index.html) — Single-file offline interactive HTML Explorer (`TwinAPI` map, 1-vs-1 diff split view, First-Divergence step inspector, `#case=<id>&arm=<arm>&model=<model>` deep-linking). Download the repository and open this file directly in a browser; no server, backend, CDN, or build is required.
- [`skills/state-bridge/SKILL.md`](/skills/state-bridge/SKILL.md) & [`skills/state-bridge/scripts/statebridge_core.py`](/skills/state-bridge/scripts/statebridge_core.py) — Reusable Agent Skill, 30-workflow `StateBridge-Bench` suite, loopback HTTP oracle, and `P1..P4` PBT invariants.
- [`artifacts/`](/artifacts) — Frozen 240 real Gemma 4 generations (`raw_llm_generations.jsonl`), 270 live execution receipts (`runs.jsonl`), 256-d embeddings (`twinapi_embeddings_256d.json`), and single source of truth metrics (`metrics_ssot.json`).

---

## Quickstart (Local Replay; Runtime Depends on Hardware)

Arm 3 used a fixed HTTP distractor selected by name, excluding session setup; its legacy `arm3_embedding_topk` key does not mean vector retrieval was evaluated. The Notebook finds local files or a mounted `/kaggle/input/<dataset>/` repository, then tries a shallow clone. Internet is needed for the initial clone and missing dependencies. Without source data and Internet it displays a marked precomputed summary, without claiming live verification.

```bash
# 1. Replay all 270 candidate programs against the live 127.0.0.1:0 Loopback HTTP State Machine
uv run --with requests==2.34.2 --with httpx==0.28.1 python3 scripts/build_ssot.py --verify-replay

# 2. Run pytest (PBT P1..P4, SSOT, and Notebook bootstrap regressions)
uv run --with requests==2.34.2 --with httpx==0.28.1 --with pytest python3 -m pytest -v

# 3. Verify 100% SSOT numerical consistency across all 5 submission surfaces
python3 scripts/verify_ssot_consistency.py
```
"""

  readme_zh = f"""# StateBridge × TwinAPI：以呼叫圖狀態切片與本地迴路狀態機破解 Gemma 4 跨套件遷移的「狀態型假朋友悖論」

> **[English README (README.md)](/README.md)** · **[Kaggle 論文賽道報告](/writeup/KAGGLE_WRITEUP.md)** · **[完整論文 PDF](/paper/paper.pdf)** · **[可重現 Notebook](/notebooks/reproduce_statebridge.ipynb)** · **[互動式 HTML 演練室](/explorer/index.html)**

---

## 核心發現（The Stateful False-Friend Paradox）

當開發者 Agent 將專案由 `requests` (`2.34.2`) 遷移至 `httpx` (`0.28.1`) 時，會撞上**「狀態型假朋友悖論（Stateful False-Friend Paradox）」**：
- 兩套函式庫有 **{top_pct}** (`8/9`) 的頂層函式與 **{sess_pct}** (`10/12`) 的 `Session`/`Client` 核心方法**名稱完全相同**；
- 12 組具狀態分歧的對應 API 在 256 維 Gemini 向量空間（`google/models/gemini-embedding-001`）中的平均餘弦相似度高達 **{mean_cos}**；
- 然而在多步 HTTP 執行期（`302` 轉址預設值、跨網域 `Authorization` 標頭剝除、Cookie 網域萃取 `.get_dict()`、串流 Context 生命週期、例外繼承階層反轉），底層狀態機卻在無聲中分道揚鑣。

**StateBridge** 結合 **1–2 跳靜態呼叫圖狀態切片（`Arm 4`）** 與 **本地迴路 HTTP 狀態機沙箱（`127.0.0.1:0`）**，在 **270 次真實沙箱執行**（240 次真實 `Gemma-4-31B-IT` 與 `Gemma-4-26B-A4B-IT` 推論 ＋ 30 次 AST Codemod）中取得顯著突破：

| 實驗組別 (Arm) | `Gemma-4-31B-IT` (Dense) 通過率 | `Gemma-4-26B-A4B-IT` (MoE) 通過率 | 平均輸入 Tokens |
| :--- | :---: | :---: | :---: |
| **`Arm 1`：決定性 AST Codemod** | **0.0%** (`0/30`) | **0.0%** (`0/30`) | `0.0` |
| **`Arm 2`：單一目標檔 + 靜態對照表** | **{_fmt_pct(a31['arm2_target_only']['solve_rate'])}** (`{a31['arm2_target_only']['pass_count']}/30`) | **{_fmt_pct(a26['arm2_target_only']['solve_rate'])}** (`{a26['arm2_target_only']['pass_count']}/30`) | `{a31['arm2_target_only']['mean_prompt_tokens']}` |
| **`Arm 3`：固定 HTTP 干擾函式** | **{_fmt_pct(a31['arm3_embedding_topk']['solve_rate'])}** (`{a31['arm3_embedding_topk']['pass_count']}/30`) | **{_fmt_pct(a26['arm3_embedding_topk']['solve_rate'])}** (`{a26['arm3_embedding_topk']['pass_count']}/30`) | `{a31['arm3_embedding_topk']['mean_prompt_tokens']}` |
| **`Arm 5`：等預算無關函式切片（對照組）** | **{_fmt_pct(a31['arm5_unrelated_slice']['solve_rate'])}** (`{a31['arm5_unrelated_slice']['pass_count']}/30`) | **{_fmt_pct(a26['arm5_unrelated_slice']['solve_rate'])}** (`{a26['arm5_unrelated_slice']['pass_count']}/30`) | `{a31['arm5_unrelated_slice']['mean_prompt_tokens']}` |
| **`Arm 4`：StateBridge 呼叫圖狀態切片** | **{_fmt_pct(a31['arm4_callgraph_slice']['solve_rate'])}** (`{a31['arm4_callgraph_slice']['pass_count']}/30`, $p={c31['arm4_vs_arm2']['mcnemar_p']}$) | **{_fmt_pct(a26['arm4_callgraph_slice']['solve_rate'])}** (`{a26['arm4_callgraph_slice']['pass_count']}/30`, $p={c26['arm4_vs_arm2']['mcnemar_p']}$) | `{a31['arm4_callgraph_slice']['mean_prompt_tokens']}` |

---

## 離線重現指令（時間依硬體而異）

下載 repo 後，直接用瀏覽器開啟 [Explorer](/explorer/index.html)，不需要後端、伺服器或建置。Arm 3 是固定選取的 HTTP 干擾函式對照組；`arm3_embedding_topk` 為沿用的識別名稱，實驗未測試向量檢索。

Notebook 先尋找本機 repo 或 `/kaggle/input/<dataset>/` 掛載內容，找不到才從公開 repo clone。第一次下載原碼與安裝依賴需要網路；沒有資料且離線時，只展示明確標示的預先計算摘要，不宣稱通過重現驗收。

```bash
uv run --with requests==2.34.2 --with httpx==0.28.1 python3 scripts/build_ssot.py --verify-replay
uv run --with requests==2.34.2 --with httpx==0.28.1 --with pytest python3 -m pytest -v
python3 scripts/verify_ssot_consistency.py
```
"""

  (base / "README.md").write_text(readme_en, encoding="utf-8")
  (base / "README.zh-TW.md").write_text(readme_zh, encoding="utf-8")


def render_all_surfaces(root_dir: Path | None = None) -> dict[str, Any]:
  """Render Writeup, Paper MD/PDF, Executed Notebook, HTML Explorer, and READMEs from SSOT."""
  base = root_dir or ROOT
  ssot_path = base / "artifacts" / "metrics_ssot.json"
  audit_path = base / "artifacts" / "twinapi_audit.json"
  bundle_path = base / "artifacts" / "explorer_bundle.json"

  ssot = json.loads(ssot_path.read_text(encoding="utf-8"))
  audit = json.loads(audit_path.read_text(encoding="utf-8"))
  bundle_raw = bundle_path.read_text(encoding="utf-8")

  # 1. Figures
  figs = _render_figures(ssot, audit, base / "paper" / "figures")

  # 2. Kaggle Writeup & Paper Markdown
  writeup_md = _build_writeup_markdown(ssot, audit)
  writeup_path = base / "writeup" / "KAGGLE_WRITEUP.md"
  writeup_path.parent.mkdir(parents=True, exist_ok=True)
  writeup_path.write_text(writeup_md, encoding="utf-8")

  paper_md_path = base / "paper" / "paper.md"
  paper_md_path.parent.mkdir(parents=True, exist_ok=True)
  paper_md_path.write_text(writeup_md, encoding="utf-8")

  # 3. Paper PDF
  paper_pdf_path = base / "paper" / "paper.pdf"
  _build_paper_pdf(ssot, audit, figs, paper_pdf_path)

  # 4. Executed Jupyter Notebook
  nb_path = base / "notebooks" / "reproduce_statebridge.ipynb"
  _build_and_execute_notebook(base, nb_path)

  # 5. Interactive HTML Explorer
  tpl_path = base / "scripts" / "explorer_template.html"
  if not tpl_path.exists():
    tpl_path = ROOT / "scripts" / "explorer_template.html"
  tpl_html = tpl_path.read_text(encoding="utf-8")
  bundle_obj = json.loads(bundle_raw)
  for case_item in bundle_obj.get("cases", []):
    for run_item in case_item.get("runs", {}).values():
      code_lines = [
        ln for ln in (run_item.get("migrated_code") or "").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
      ]
      run_item["migrated_code"] = "\n".join(code_lines)
  compact_bundle = json.dumps(bundle_obj, separators=(",", ":"), ensure_ascii=False)
  compact_bundle = compact_bundle.replace('"pass_count":', '"pass_count": ')
  exp_path = base / "explorer" / "index.html"
  exp_path.parent.mkdir(parents=True, exist_ok=True)
  exp_path.write_text(tpl_html.replace("__EXPLORER_BUNDLE_JSON__", compact_bundle), encoding="utf-8")

  # 6. README.md and README.zh-TW.md
  _build_readmes(ssot, base)

  return {
    "status": "ok",
    "writeup_words": len(writeup_md.split()),
    "pdf_bytes": paper_pdf_path.stat().st_size,
    "explorer_bytes": exp_path.stat().st_size,
  }


if __name__ == "__main__":
  summary = render_all_surfaces()
  print(json.dumps(summary, indent=2))

