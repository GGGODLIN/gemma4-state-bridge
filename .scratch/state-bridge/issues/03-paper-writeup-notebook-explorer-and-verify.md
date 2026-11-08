# 03 — Paper PDF, Kaggle Writeup, Executed Notebook, Interactive Explorer, and SSOT Gate

**What to build:** Implement `scripts/render_surfaces.py` and `scripts/verify_ssot_consistency.py` to compile all submission deliverables directly from `artifacts/metrics_ssot.json` and `artifacts/explorer_bundle.json`:
1. `writeup/KAGGLE_WRITEUP.md` (`<= 3,000` words, containing Title & Subtitle, Abstract, Introduction, Stateful False-Friend Audit, StateBridge Architecture & 5-Arm Experimental Results, Related Works & Citations, and Reproduction Guide),
2. `paper/paper.md` and arXiv-formatted `paper/paper.pdf` (with vector/high-DPI figures generated from `metrics_ssot.json`),
3. `notebooks/reproduce_statebridge.ipynb` (self-contained Kaggle Notebook that replays all 270 runs live against the `127.0.0.1:0` Loopback HTTP State Machine in `< 15 seconds` and recomputes all tables and statistical tests, executed with populated cell outputs),
4. `explorer/index.html` (single-file self-contained interactive HTML Explorer featuring the `TwinAPI` Stateful False-Friend Map, 1-vs-1 Diff Split View, auto-collapsed matching HTTP steps, highlighted `First-Divergence Step`, `Arm 4` vs `Arm 3`/`Arm 5` call-graph toggle, and `#case=<id>&arm=<arm>&model=<model>` URL hash deep-linking), and
5. `README.md` and `README.zh-TW.md`.

**Blocked by:** 02 — TwinAPI Audit, Real Dual-Model Gemma 4 Matrix, and SSOT Compiler

**Status:** `ready-for-agent`

**Needs:** None — a worker can check every item unaided

**Validation method:** Run `uv run --with requests==2.34.2 --with httpx==0.28.1 --with pytest python3 -m pytest tests/test_verify_ssot_consistency.py -v` and `python3 scripts/verify_ssot_consistency.py`.

**Evidence required:** `verify_ssot_consistency.py` exits 0 confirming: (1) `writeup/KAGGLE_WRITEUP.md` has all required Kaggle Paper Track sections and word count `<= 3,000` words; (2) `paper/paper.pdf` exists, is non-empty, and has valid PDF header `%PDF-`; (3) `notebooks/reproduce_statebridge.ipynb` has executed cell outputs matching `metrics_ssot.json`; (4) `explorer/index.html` embeds `explorer_bundle.json` with zero external CDN dependencies needed for core rendering; and (5) every solve rate, count, and $p$-value across Writeup, Paper, Notebook, Explorer, and README matches `artifacts/metrics_ssot.json` 100%.

**Tests:** `required`

**Test seam:** `scripts/render_surfaces.py` and `scripts/verify_ssot_consistency.py`.

**Regression scope:** Ensure that any discrepancy between `artifacts/metrics_ssot.json` and any downstream deliverable (`writeup/KAGGLE_WRITEUP.md`, `paper/paper.md`, `notebooks/reproduce_statebridge.ipynb`, `explorer/index.html`, `README.md`), or any Writeup word count exceeding `3,000` words, causes `verify_ssot_consistency.py` to fail with a non-zero exit code and pinpoint the mismatched surface — Source: Story 1, Story 4, Story 6.

- [ ] [Confirmation] `writeup/KAGGLE_WRITEUP.md` contains Title & Subtitle, Abstract, Introduction, Research Description (Methods + Experiments), Related Works & Citations, and is verified to be `<= 3,000` words — Source: Story 2, Story 6 — Failure: F6
- [ ] [Confirmation] `notebooks/reproduce_statebridge.ipynb` executes cleanly from top to bottom in `< 15 seconds` without internet or API keys and outputs the exact 5-arm ablation tables and McNemar / Bootstrap statistics from `artifacts/metrics_ssot.json` — Source: Story 1, Story 6
- [ ] [Confirmation] `explorer/index.html` is a self-contained single-file HTML application driven by `artifacts/explorer_bundle.json` with `TwinAPI` Stateful False-Friend Map, 1-vs-1 Diff Split View, auto-collapsed identical HTTP steps, highlighted `First-Divergence Step`, and `#case=<id>&arm=<arm>&model=<model>` URL hash deep-linking — Source: Story 4
- [ ] [Confirmation] `scripts/verify_ssot_consistency.py` checks all 5 surfaces (`writeup/KAGGLE_WRITEUP.md`, `paper/paper.md` + `paper/paper.pdf`, `notebooks/reproduce_statebridge.ipynb`, `explorer/index.html`, `README.md`) against `artifacts/metrics_ssot.json`, failing on any drifted number or missing artifact (verified with a positive mutation control test in `tests/test_verify_ssot_consistency.py`) — Source: Story 6 — Failure: F5
