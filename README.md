# StateBridge × TwinAPI: Resolving the Stateful False-Friend Paradox in Cross-Library Code Migration with Gemma 4

[![Track](https://img.shields.io/badge/Kaggle-Gemma_4_Developer_Agent_Paper-2a78d6)](#)
[![Models](https://img.shields.io/badge/Models-Gemma--4--31B--IT_%26_26B--A4B--IT-eb6834)](#)
[![Tests](https://img.shields.io/badge/Tests-PBT_P1..P4_%26_regressions-0ca30c)](#)

> **[繁體中文說明 (README.zh-TW.md)](/README.zh-TW.md)** · **[Kaggle Paper Track Writeup](/writeup/KAGGLE_WRITEUP.md)** · **[Full Paper PDF](/paper/paper.pdf)** · **[Reproducible Notebook](/notebooks/reproduce_statebridge.ipynb)** · **[Interactive HTML Explorer](/explorer/index.html)**

---

## Overview

When autonomous coding agents migrate repositories across sibling libraries such as Python's `requests` (`2.34.2`) and `httpx` (`0.28.1`), they confront the **Stateful False-Friend Paradox**:
- **88.9%** (`8/9`) of top-level functions (`get`, `post`, `put`, `delete`, `head`, `options`, `patch`, `request`) and **83.3%** (`10/12`) of core `Session`/`Client` methods share **identical names**.
- Across 12 stateful false-friend API pairs spanning 5 multi-step divergence families (`F1..F5`), their 256-dimensional Gemini embeddings (`google/models/gemini-embedding-001`, `dimensions=256`) exhibit a mean cosine similarity of **0.873**.
- Yet at runtime, multi-step HTTP state transitions diverge silently on `302` redirect defaults (`allow_redirects=True` vs `follow_redirects=False`), cross-origin `Authorization` stripping and response hooks, domain-scoped `CookieJar` extraction (`.get_dict()`), streaming context lifecycles (`client.stream()`), and inverted exception hierarchies (`HTTPError` vs `HTTPStatusError`).

**StateBridge** combines a **1–2 hop static call-graph state-slice retriever (`Arm 4`)** with a **deterministic loopback HTTP state-machine oracle (`127.0.0.1:0`)** that verifies wire-level equivalence across multi-step workflows.

---

## Headline Empirical Results (270 Live Loopback HTTP Runs)

| Experimental Arm | `Gemma-4-31B-IT` (Dense) | `Gemma-4-26B-A4B-IT` (MoE) | Mean Prompt Tokens |
| :--- | :---: | :---: | :---: |
| **`Arm 1`: Deterministic AST Codemod** | **0.0%** (`0/30`) | **0.0%** (`0/30`) | `0.0` |
| **`Arm 2`: Target File Only + Cheat-Sheet** | **13.3%** (`4/30`) | **13.3%** (`4/30`) | `497.7` |
| **`Arm 3`: HTTP Distractor (Decoy)** | **13.3%** (`4/30`) | **13.3%** (`4/30`) | `637.7` |
| **`Arm 5`: Equal-Token Unrelated Slice (Control)** | **16.7%** (`5/30`) | **13.3%** (`4/30`) | `702.7` |
| **`Arm 4`: StateBridge Call-Graph Slice (Ours)** | **90.0%** (`27/30`, $p=2.384185791015625e-07$) | **96.7%** (`29/30`, $p=5.960464477539063e-08$) | `799.9` |

---

## Repository Structure

- [`writeup/KAGGLE_WRITEUP.md`](/writeup/KAGGLE_WRITEUP.md) — Official Kaggle Paper Track Writeup (`<= 3,000` words).
- [`paper/paper.pdf`](/paper/paper.pdf) & [`paper/paper.md`](/paper/paper.md) — Full arXiv-formatted paper and vector/high-DPI figures.
- [`notebooks/reproduce_statebridge.ipynb`](/notebooks/reproduce_statebridge.ipynb) — Self-contained executed Kaggle Notebook that replays all 270 runs against the live loopback HTTP state machine in `31.801s`.
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
