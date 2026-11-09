# `gemma4-state-bridge` Post-Implementation Review (`review-implement`)

## Run 1
### Input
- target: `/Users/linhancheng/Desktop/projects/gemma4-state-bridge`
- base_sha: `e435f827643868adc69b91ce821dcab6c820250e`
- head_sha: `3c1b70d37a79c572c89ed24378f98c36eb35543f`
- spec: `/Users/linhancheng/Desktop/projects/gemma4-state-bridge/.scratch/state-bridge/spec.md`
- tickets:
  - `/Users/linhancheng/Desktop/projects/gemma4-state-bridge/.scratch/state-bridge/issues/01-statebridge-core-and-bench.md`
  - `/Users/linhancheng/Desktop/projects/gemma4-state-bridge/.scratch/state-bridge/issues/02-twinapi-audit-and-gemma4-experiments.md`
  - `/Users/linhancheng/Desktop/projects/gemma4-state-bridge/.scratch/state-bridge/issues/03-paper-writeup-notebook-explorer-and-verify.md`
- raw_session_paths:
  - `/Users/linhancheng/.claude/projects/-Users-linhancheng-Desktop-projects/d3e41a3e-d10a-4193-a28b-0ca37ca8a618.jsonl` (single root session `d3e41a3e-d10a-4193-a28b-0ca37ca8a618`, 0 parent handoffs)
- selected_axes: `all` (`scope`, `yagni`)
- selected_seats: `opus/scope`, `opus/yagni`
- logical_models: `opus`, `opus`
- resolved_models: `google/gemini-4-argon:high`, `google/gemini-4-argon:high`
- started_at: `2026-11-09T06:22:00+08:00`
- session_count: `1`
- total_raw_bytes: `196696413`
- elapsed_time: `366.5s`
- token_use: `590147` (`310328` scope + `279819` yagni)
### Axis status
- scope: completed
- yagni: completed
### Events
- `2026-11-09T06:22:00+08:00` | `start` | source=`auto` (signals: `ticket_count=3 >= 3`, `confirmed_core_findings >= 3`; prior dispositions: none / 查無前輪) | dispatched `opus/scope` and `opus/yagni` against `e435f827643868adc69b91ce821dcab6c820250e..3c1b70d37a79c572c89ed24378f98c36eb35543f`.
- `2026-11-09T06:26:00+08:00` | `seat-return` | `opus/yagni` (`a627825802cacb4c6`, `google/gemini-4-argon:high`) completed with 3 findings (`Y1..Y3`).
- `2026-11-09T06:28:10+08:00` | `seat-return` | `opus/scope` (`a16a4646a3f145d28`, `google/gemini-4-argon:high`) completed with 5 findings (`S1..S5`).
### Findings
- **S1 (`opus/scope`, `CONFIRMED`)**: `skills/state-bridge/scripts/statebridge_core.py:1453-1460` (`reproduce-paper` CLI subcommand) only reads existing `artifacts/metrics_ssot.json` and prints `data["benchmark"]`, instead of replaying the 30 workflows / 270 runs against the live Loopback HTTP State Machine in `< 15s` and printing the 5-arm ablation tables, McNemar $p$-values, and Bootstrap 95% CIs (`spec.md:16, 20, 29`; session `d3e41a3e-d10a-4193-a28b-0ca37ca8a618.jsonl:22799` quote `"可以"`).
- **S2 (`opus/scope`, `CONFIRMED`)**: `skills/state-bridge/scripts/statebridge_core.py:997-1010` (`build_arm_context`) selects `arm3_embedding_topk` via substring match `"preview_http_headers" in k` rather than scoring against `artifacts/twinapi_embeddings_256d.json`, and ends `arm4_callgraph_slice` filter with `or "session_setup.py:" in k` which short-circuits `called_names` call-graph reachability (`spec.md:13-14, 30`; session `d3e41a3e-d10a-4193-a28b-0ca37ca8a618.jsonl:22776` quote `"1a, 2a, 3a"`).
- **S3 (`opus/scope`, `CONFIRMED`)**: `scripts/render_surfaces.py:607, 673, 741-743` (`_build_paper_pdf`) hardcodes `"90.0%&ndash;93.3%"` in the PDF Conclusion paragraph while `metrics_ssot.json` has `96.7%` (`29/30`), and `scripts/verify_ssot_consistency.py:123-130` only checks `%PDF-` header and file size without verifying PDF text numbers (`spec.md:25, 36`; `03-paper-writeup-notebook-explorer-and-verify.md:18, 29`).
- **S4 (`opus/scope`, `PLAUSIBLE`)**: `scripts/run_gemma4_matrix.py:148-149` used `max_tokens=16384` (and `temperature=0.2` on retry) for 6 long-`<thought>` prompts while `spec.md:33` and `decisions/0001:3` stated `temperature=0.0, max_tokens=8192` (`spec.md:33` vs `spec.md:70` and `choices.md:C5`).
- **S5 (`opus/scope`, `PLAUSIBLE`)**: `scripts/explorer_template.html:1054-1074` left pane of the 1-vs-1 Diff Split View only offers static reference tabs (`httpx_gold_source`, `requests_source`, `session_setup_httpx`) without an `Arm 4 (StateBridge)` tab to compare `Arm 4` directly against `Arm 3`/`Arm 5` side-by-side (`spec.md:23, 35`; session `d3e41a3e-d10a-4193-a28b-0ca37ca8a618.jsonl:22776` quote `"1a, 2a, 3a"`).
- **Y1 (`opus/yagni`)**: `scripts/render_surfaces.py:862-909, 1053-1054` defines a 48-line `_fallback_explorer_template()` that is never reached because `scripts/explorer_template.html` is committed in the repo (`Story 1..6`, `T03-A3..A4`).
- **Y2 (`opus/yagni`)**: `skills/state-bridge/scripts/statebridge_core.py:40-48, 999-1002, 1060, 1143-1144` defines unreferenced `OUTCOME_CATEGORIES`, dead `or emb_bias < 0`, and write-only `self.runtime_state` (`Story 1..6`, `T01-A1..A6`).
- **Y3 (`opus/yagni`)**: `scripts/verify_ssot_consistency.py:174-175` hardcodes a check for `"5.96e-7"` that is redundant with the dynamic SSOT `p26_str` / `p31_str` checks (`Story 6`, `T03-A4`).
### Main decisions
- **S1 (`CONFIRMED`)**: Verified against `statebridge_core.py:1453-1460`. Accepted as repair obligation: update `reproduce-paper` CLI subcommand to invoke `build_ssot.build_ssot()` live replay (`< 15s`) and print full 5-Arm ablation tables, McNemar $p$-values, and Bootstrap 95% CIs.
- **S2 (`CONFIRMED`)**: Verified against `statebridge_core.py:997-1010`. Accepted as repair obligation: load `artifacts/twinapi_embeddings_256d.json` in `arm3_embedding_topk` to rank candidate functions by 256-d cosine similarity against `requests.Session` descriptors, and remove the tautological `or "session_setup.py:" in k` from `arm4_callgraph_slice`, replacing it with true 1–2 hop AST symbol reachability from `called_names` via shared module-level state constants in `session_setup.py`.
- **S3 (`CONFIRMED`)**: Verified against `render_surfaces.py:607, 673, 741-743` and `verify_ssot_consistency.py:123-130`. Accepted as repair obligation: replace hardcoded strings in `_build_paper_pdf()` with dynamic `ssot` formatting (`90.0%–96.7%`) and extract uncompressed/FlateDecode PDF text streams in `verify_ssot_consistency.py` to enforce `expected_tokens` and `_PCT_RE` allowlist on `paper/paper.pdf`.
- **S4 (`PLAUSIBLE`)**: Keep `temperature=0.0` in `run_gemma4_matrix.py` and synchronize `spec.md:33`, `decisions/0001:3`, and `02-twinapi-audit-and-gemma4-experiments.md:3` to document `max_tokens=8192 (auto-escalating to 16384 on finish_reason="length" for verbose <thought> traces)`; also surface `C5` in the final user handback per `auto-choice`.
- **S5 (`PLAUSIBLE`)**: Accepted as repair obligation within approved scope: add an `"Arm 4 (StateBridge)"` comparison tab (`arm4_migrated_code`) to the left pane of the 1-vs-1 Diff Split View in `scripts/explorer_template.html` (and default to it when inspecting non-Arm-4 runs) so readers can compare `Arm 4` vs `Arm 3`/`Arm 5` directly side-by-side as well as against `Gold httpx` / `Original requests` / `session_setup.py`.
- **Y1 (`accept`)**: Verified via `command grep` that `_fallback_explorer_template()` has 0 callers when `scripts/explorer_template.html` exists. Accepted as repair obligation: delete `_fallback_explorer_template()` and read `scripts/explorer_template.html` directly.
- **Y2 (`accept`)**: Verified via `command grep` that `OUTCOME_CATEGORIES`, `or emb_bias < 0`, and `self.runtime_state` have 0 readers. Accepted as repair obligation: remove all three.
- **Y3 (`accept`)**: Verified `scripts/verify_ssot_consistency.py:174-175`. Accepted as repair obligation: remove the hardcoded `"5.96e-7"` check and rely on dynamic `p31_str`/`p26_str` and `_PCT_RE` checks.
### Dispositions
- disposition: S1 | scope | main=CONFIRMED | user=n/a | `statebridge_core.py reproduce-paper` 改為實際執行 270 筆 Loopback HTTP 重播並印出 5-Arm 消融表與 McNemar/Bootstrap 統計
- disposition: S2 | scope | main=CONFIRMED | user=n/a | `build_arm_context` 的 `Arm 3` 接上 `twinapi_embeddings_256d.json` 餘弦排序，並移除 `Arm 4` 尾端恆真的 `or "session_setup.py:" in k` 改用 1–2 跳 AST 符號可達性
- disposition: S3 | scope | main=CONFIRMED | user=n/a | `render_surfaces.py` PDF 結論段改由 `ssot` 動態帶入 `90.0%–96.7%`，並於 `verify_ssot_consistency.py` 解壓檢查 `paper/paper.pdf` 內文數值
- disposition: S4 | scope | main=PLAUSIBLE | user=n/a | `run_gemma4_matrix.py` 鎖定 `temperature=0.0` 並同步更新 `spec.md:33` 與 `decisions/0001:3` 記載 `max_tokens=8192→16384` 長 `<thought>` 防截斷規則，同步於 handback 呈現 `C5`
- disposition: S5 | scope | main=PLAUSIBLE | user=n/a | `explorer_template.html` 1-vs-1 Diff Split View 左欄新增 `Arm 4 (StateBridge)` 對照分頁，支援 `Arm 4` vs `Arm 3`/`Arm 5` 直接並排比對
- disposition: Y1 | yagni | main=accept | user=n/a | 刪除 `scripts/render_surfaces.py` 未使用的 48 行 `_fallback_explorer_template()`
- disposition: Y2 | yagni | main=accept | user=n/a | 刪除 `statebridge_core.py` 未被讀取的 `OUTCOME_CATEGORIES`、`or emb_bias < 0` 與 `self.runtime_state`
- disposition: Y3 | yagni | main=accept | user=n/a | 刪除 `scripts/verify_ssot_consistency.py` 硬編碼之 `"5.96e-7"` 檢查
### Repair obligations
- `S1`: Update `skills/state-bridge/scripts/statebridge_core.py` (`reproduce-paper` CLI subcommand) to execute `build_ssot.build_ssot()` and output the full 5-Arm ablation tables, McNemar $p$-values, and Bootstrap 95% CIs.
- `S2`: Update `skills/state-bridge/scripts/statebridge_core.py` (`build_arm_context`) so `arm3_embedding_topk` computes 256-d cosine similarity from `twinapi_embeddings_256d.json` and `arm4_callgraph_slice` filters `setup_funcs` via 1–2 hop AST reachability from `called_names` without `or "session_setup.py:" in k`.
- `S3`: Update `scripts/render_surfaces.py` (`_build_paper_pdf`) to dynamically interpolate all percentages/cosines from `ssot`, and update `scripts/verify_ssot_consistency.py` to decompress FlateDecode streams in `paper/paper.pdf` and verify its text against `expected_tokens` and `allowed_pcts`.
- `S4`: Lock `temperature=0.0` in `scripts/run_gemma4_matrix.py` and synchronize `spec.md:33`, `decisions/0001:3`, and `02-twinapi-audit-and-gemma4-experiments.md:3`.
- `S5`: Add `Arm 4 (StateBridge)` tab (`arm4_code`) to the left pane of the 1-vs-1 Diff Split View in `scripts/explorer_template.html`.
- `Y1`: Remove `_fallback_explorer_template()` from `scripts/render_surfaces.py`.
- `Y2`: Remove `OUTCOME_CATEGORIES`, `or emb_bias < 0`, and `self.runtime_state` from `skills/state-bridge/scripts/statebridge_core.py`.
- `Y3`: Remove `if "5.96e-7" in e_text:` from `scripts/verify_ssot_consistency.py`.
### Targeted rechecks
- `S1`: `pass` — verified at `56cfcba999abea8f87ecd64718af9b2700bf5e6b` (`statebridge_core.py reproduce-paper` executes `build_ssot.build_ssot()` live replay in `8.507s < 15s` and outputs `benchmark`, `twinapi_audit`, and `models` 5-arm ablation tables + McNemar/Bootstrap statistics).
- `S2`: `pass` — verified at `56cfcba999abea8f87ecd64718af9b2700bf5e6b` (`arm3_embedding_topk` loads `twinapi_embeddings_256d.json` and ranks candidate functions; `arm4_callgraph_slice` filters `setup_funcs` via `called_names | module_consts` 1–2 hop AST reachability with 0 matches for `session_setup.py:" in k`).
- `S3`: `pass` — verified at `56cfcba999abea8f87ecd64718af9b2700bf5e6b` (`render_surfaces.py` dynamically formats `90.0%–96.7%` in `paper.pdf` Conclusion; `verify_ssot_consistency.py` extracts `BT..ET` text from `paper.pdf` and passes all 13 `expected_tokens` and `_PCT_RE` allowlist checks).
- `S4`: `pass` — verified at `56cfcba999abea8f87ecd64718af9b2700bf5e6b` (`run_gemma4_matrix.py` uses `temperature=0.0` and `max_tokens=8192 if attempt == 1 else 16384`, synchronized across `spec.md:33`, `decisions/0001:3`, and `02-twinapi-audit-and-gemma4-experiments.md:3`).
- `S5`: `pass` — verified at `56cfcba999abea8f87ecd64718af9b2700bf5e6b` (`explorer_template.html` and `explorer/index.html` include the `Arm 4 (StateBridge)` tab `data-reftab="arm4_migrated_code"` in the 1-vs-1 Diff Split View).
- `Y1`: `pass` — verified at `56cfcba999abea8f87ecd64718af9b2700bf5e6b` (`_fallback_explorer_template` removed, `command grep` count = 0).
- `Y2`: `pass` — verified at `56cfcba999abea8f87ecd64718af9b2700bf5e6b` (`OUTCOME_CATEGORIES` and `runtime_state` removed, `command grep` count = 0).
- `Y3`: `pass` — verified at `56cfcba999abea8f87ecd64718af9b2700bf5e6b` (`5.96e-7` check removed from `verify_ssot_consistency.py`, `command grep` count = 0).
- Final suite against `56cfcba999abea8f87ecd64718af9b2700bf5e6b`: `17 passed in 19.54s` (`pytest /Users/linhancheng/Desktop/projects/gemma4-state-bridge/tests/ -v`).
### Summary
- Run status: PASS


