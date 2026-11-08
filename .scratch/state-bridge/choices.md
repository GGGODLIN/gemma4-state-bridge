# `gemma4-state-bridge` 實作決定帳本（auto-choice）

## 八類掃描總覽（Pass `e435f827..HEAD`，2026-11-09）

1. **資料形狀與格式**：有（列 C1、C2）
2. **存放位置與命名**：無（所有產出物路徑皆由 `.scratch/state-bridge/spec.md` 與 Tickets 01–03 明文指定）
3. **API 契約與錯誤行為**：有（列 C3）
4. **新增相依套件**：有（列 C4）
5. **並行與效能取捨**：有（列 C5）
6. **範圍解讀**：有（列 C6）
7. **測試設定改動**：無（未放寬任何斷言、未 skip 測試、未 mock 受測 seam）
8. **會被照抄的模式**：有（列 C7）

---

## 逐條決定明細

### C1 — `explorer/index.html` 內嵌 JSON bundle 壓縮與移除註解行
- **何時**：Ticket 03（`scripts/render_surfaces.py:988-1000`）
- **決定**：壓縮內嵌於 `explorer/index.html` 的 `window.__EXPLORER_BUNDLE__` JSON 並剝除 `migrated_code` 內的純 `#` 註解行。
  - **觸發事件**：`artifacts/explorer_bundle.json` 以 2 空白縮排儲存時達 513 KB，直接原樣塞入 `explorer/index.html` 會使單檔膨脹至 554 KB。
  - **現在怎麼做**：`artifacts/explorer_bundle.json` 磁碟檔維持易讀的 2 空白縮排，而注入 `explorer/index.html` 時改用 `separators=(",", ":")` 緊湊序列化並移除 `migrated_code` 中的純註解行（保留完整可執行程式碼），將 `explorer/index.html` 壓至 424 KB。
  - **沒選的做法會怎樣**：若不壓縮，`explorer/index.html` 體積增加約 130 KB 純空白與重複註解；若連 `artifacts/explorer_bundle.json` 也壓成單行，人類與外部腳本用文字編輯器檢視 JSON 檔時會難以閱讀。
- **規格缺口**：`spec.md:35` 規定 `explorer/index.html` 由 `artifacts/explorer_bundle.json` 驅動且單檔自足，但未規定內嵌至 HTML `<script>` 時是否應緊湊化。
- **影響範圍**：僅影響 `explorer/index.html` 的檔案大小與 1-vs-1 Code View 是否顯示模型輸出的 `#` 註解行；不影響 `artifacts/runs.jsonl` 或 `artifacts/raw_llm_generations.jsonl` 的原始紀錄。
- **判定**：`sound` — 同時兼顧 `artifacts/explorer_bundle.json` 的可讀性與 `explorer/index.html` 的單檔載入效率，且保留全部可執行程式碼。
- **把握度**：高

### C2 — `artifacts/runs.jsonl` 之 `error_detail` 本機路徑去識別化（`<sandbox>` 與 `~`）
- **何時**：Closeout Review 修補（`skills/state-bridge/scripts/statebridge_core.py:1229-1239`、`scripts/verify_ssot_consistency.py:72-77`）
- **決定**：在寫入 `error_detail` 前將子程序 `stderr` 中的暫存目錄路徑與使用者家目錄替換為 `<sandbox>` 與 `~`。
  - **觸發事件**：`security-reviewer` 指出子程序 Traceback 含 `/Users/linhancheng/.cache/uv/...` 與 `/var/folders/...`，直接寫入 `artifacts/runs.jsonl` 會在公開提交物中暴露宿主機帳號與快取路徑。
  - **現在怎麼做**：透過 `_sanitize_stderr(stderr, tmp_path)` 將 `tmp_path`、`/var/folders/.../statebridge_run_*` 替換為 `<sandbox>`，並將 `str(Path.home())` 替換為 `~`，同時在 `verify_ssot_consistency.py` 強制檢查 `runs.jsonl` 0 筆家目錄字串。
  - **沒選的做法會怎樣**：若不清洗，公開 GitHub repo 與 Kaggle 附件中的 `artifacts/runs.jsonl` 會殘留 59 筆個人電腦絕對路徑，違反全域 `CLAUDE.md`「機器特定路徑不寫進 commit 的 doc」規範。
- **規格缺口**：`spec.md:34` 要求記錄 `error_detail`，但未明文規定 Traceback 中的宿主機絕對路徑遮罩格式。
- **影響範圍**：所有讀取 `artifacts/runs.jsonl` 之 `error_detail` 的除錯工具與論文讀者。
- **判定**：`sound` — 保留完整例外型別、行號與套件相對路徑，同時消除個人機器路徑外洩。
- **把握度**：高

### C3 — `run_workflow_sandbox()` 於參考基準 `gold_exec` 非 `ok` 時直接拋出 `RuntimeError`
- **何時**：Closeout Review 修補（`skills/state-bridge/scripts/statebridge_core.py:1357-1358`）
- **決定**：當 `requests` 參考實作（`gold_exec`）在沙箱中非預期失敗或逾時時，立即拋出 `RuntimeError` 中止該筆比對，而非繼續與 `cand_exec` 比對。
  - **觸發事件**：`python-reviewer` 指出若宿主機瞬間高負載導致 `gold_exec` 逾時，`gold_exec["trace"]` 會變空列表，繼續往下跑會把正常的候選程式誤判為 `STEP_K_WIRE_DIVERGENCE`。
  - **現在怎麼做**：在 `_execute_against_loopback` 執行完 `gold_exec` 後立即檢查 `if gold_exec["status"] != "ok": raise RuntimeError(...)`。
  - **沒選的做法會怎樣**：若靜默繼續比對，基準環境偶發抖動會被偽裝成候選模型的分歧錯誤，污染實驗數據。
- **規格缺口**：`spec.md:32` 定義了候選程式的 7 種互斥分類收據，但未定義當「參考基準本身執行失敗」時應如何處理。
- **影響範圍**：`statebridge_core.run_workflow_sandbox()` 與 `scripts/build_ssot.py`。
- **判定**：`sound` — 參考基準必須 100% 成功（`test_statebridge_core.py:245` 已驗證 30 個案例皆通過），基準異常時 fail-fast 才能防止髒數據落盤。
- **把握度**：高

### C4 — 以 `reportlab` 與 `matplotlib` 離線編譯 `paper/paper.pdf` 與三張出版級圖表
- **何時**：Ticket 03（`scripts/render_surfaces.py:18-184, 360-677`）
- **決定**：在 `scripts/render_surfaces.py` 執行期透過 `uv run --with matplotlib --with reportlab --with numpy` 產生 220 DPI 圖表並直接排版輸出 `paper/paper.pdf`。
  - **觸發事件**：本機環境未安裝 `pdflatex`、`typst`、`pandoc` 或 `weasyprint`（`which typst pandoc pdflatex weasyprint` 皆無輸出），但 Ticket 03 要求產出內嵌圖表的 `paper/paper.pdf`。
  - **現在怎麼做**：核心 `skills/state-bridge/scripts/statebridge_core.py` 維持 100% Python 標準庫零外部依賴；僅在文件編譯腳本 `scripts/render_surfaces.py` 使用 `matplotlib` 與 `reportlab` 生成 `paper/figures/*.png` 與 `paper/paper.pdf`。
  - **沒選的做法會怎樣**：若安裝龐大的 MacTeX / TeXLive 套件需耗時數十分鐘並改動系統環境；若只交 Markdown 則違反 Ticket 03 的 `paper/paper.pdf` 驗收條件。
- **規格缺口**：`spec.md:36` 要求產出 `paper/paper.pdf`，但未指定本機無 LaTeX/Typst 時的 PDF 渲染引擎。
- **影響範圍**：僅限 `scripts/render_surfaces.py` 重新渲染 PDF 時的 `uv run` 套件快取，不影響 `statebridge_core.py` 或 `build_ssot.py`。
- **判定**：`sound` — 純 Python wheel 於 50ms 內由 `uv` 快取載入、完全不污染全域系統環境，且經視覺巡檢確認 3 頁 PDF 排版無溢位。
- **把握度**：高

### C5 — `run_gemma4_matrix.py` 將 `max_tokens` 由 `8192` 調升至 `16384`（重試 `temperature=0.2`）
- **何時**：Ticket 02 實跑與 Closeout 修補（`scripts/run_gemma4_matrix.py:124-131`）
- **決定**：將對 Magpie 發送的 `Gemma-4-31B-IT` 與 `Gemma-4-26B-A4B-IT` 推論請求輸出上限由 `8192` 提高至 `16384` tokens，且當 `attempt > 1` 重試時將 `temperature` 設為 `0.2`。
  - **觸發事件**：在第一輪 `max_tokens=8192` 實跑中，`Gemma-4-26B-A4B-IT` 在 `Arm 4` 的平均 `<thought>` 長度達 `14,724.5` 字元，有 6/240 筆因推理鏈較長而在吐完程式碼前撞上 `finish_reason="length"`（其中 `F2_05` 在程式碼圍欄中途截斷）。
  - **現在怎麼做**：依 `model-onboard`「輸出預算：遇到 length／MAX_TOKENS 或空正文而仍有 reasoning，先核輸出預算和轉送路徑，再判模型能力」規範，將 `max_tokens` 設為 `16384`，使 240/240 筆推論全數獲得完整閉合的程式碼圍欄（`F2_05` 重跑後 `Arm 4` 通過率由 `28/30` 升至 `29/30`）。
  - **沒選的做法會怎樣**：若死守 `max_tokens=8192`，會把 MoE 模型投入更多 `<thought>` 推理 tokens 的特性誤判為 `PARSE_OR_SYNTAX_ERROR`（模型能力失敗），違反 `model-onboard` 紀律。
- **規格缺口**：`spec.md:33` 初稿預估 `max_tokens=8192` 足夠容納 `<thought>` 與程式碼，未預見 `Gemma-4-26B-A4B-IT` 在跨檔案呼叫圖切片下的 `<thought>` 長度可超過 30,000 字元（~8,000 tokens）。
- **影響範圍**：`scripts/run_gemma4_matrix.py` 與 `artifacts/raw_llm_generations.jsonl` 中該 6 筆長推理案例。
- **判定**：`needs-user` — 調升輸出 token 上限屬於實驗參數微調（雖然符合 `model-onboard` 反截斷規則且已在規格與 Verification Log 揭露）。
  - **暫定做法**：採用 `max_tokens=16384` 跑完之 240/240 無截斷生成結果作為 `metrics_ssot.json` 正本，並在 `spec.md` 與 `02-twinapi-audit-and-gemma4-experiments.md` 明文記錄此發現。
  - **撤回方式**：若想改回嚴格限制 `8192` tokens 並將超長推理計為失敗，可於 `run_gemma4_matrix.py` 將 `max_tokens` 改回 `8192`、對 `total_tokens > 8192` 的列標記 `truncated=True`，再執行 `python3 scripts/build_ssot.py && python3 scripts/render_surfaces.py`。
- **把握度**：中

### C6 — `build_ssot.py` 於模型與統計結果不變時保留既有 `replay_elapsed_sec`
- **何時**：Closeout Review 修補（`scripts/build_ssot.py:310-318`）
- **決定**：當重跑 `build_ssot.py` 時，若實測耗時 `< 15.0s` 且重算出來的 `models` 統計字典與磁碟上既有的 `metrics_ssot.json` 完全一致，則保留既有 `replay_elapsed_sec` 數值，不因每次 CPU 幾毫秒的浮動而改寫 JSON。
  - **觸發事件**：`python-reviewer` 指出 Ticket 02 Regression scope 要求重複執行 `build_ssot.py` 產出完全一致的 `metrics_ssot.json`，若每次覆寫動態秒數，執行 `build_ssot.py --verify-replay` 會讓 `metrics_ssot.json` 與已渲染的論文秒數產生微小漂移。
  - **現在怎麼做**：確認當次重播 `< 15.0s` 且 `prev_ssot.get("models") == models_summary` 時，沿用已記錄的 `replay_elapsed_sec`（並於 Notebook Cell 2 同時印出當次實測秒數與基準秒數）。
  - **沒選的做法會怎樣**：每次跑測試或 `--verify-replay` 都會造成 `git status` 出現 `metrics_ssot.json` 與 `explorer_bundle.json` 的單行秒數 dirty diff。
- **規格缺口**：`spec.md:16` 要求記錄重播時間 `< 15s`，同時 Ticket 02 要求重跑產出 deterministic identical `metrics_ssot.json`。
- **影響範圍**：`scripts/build_ssot.py`、`artifacts/metrics_ssot.json` 與 `artifacts/explorer_bundle.json`。
- **判定**：`sound` — 同時滿足「實測驗證 `< 15s`」與「重跑產出位元組完全一致的 SSOT 檔案」兩項要求。
- **把握度**：高

### C7 — 跨五表面之百分比白名單防漂移模式（`_allowed_percentages`）
- **何時**：Ticket 03 與 Closeout 修補（`scripts/verify_ssot_consistency.py:18-39, 98-188`）
- **決定**：從 `metrics_ssot.json` 動態推導所有合法百分比集合（`_allowed_percentages(ssot)`），並以正規表示式 `\b\d+\.\d+%` 掃描 Writeup、Paper、Notebook、Explorer、README 全部五個表面，只要出現任何不在集合內的百分比即判定失敗。
  - **觸發事件**：單純檢查「某個正確數字有沒有出現（`in text`）」時，若同一數字在文中出現 4 次、其中 1 次被改錯成 `99.9%`，`in text` 仍會通過。
  - **現在怎麼做**：除了正向檢查 `expected_tokens` 必出現外，反向掃描所有 `\b\d+\.\d+%` 並強制每一筆都必須屬於 `metrics_ssot.json` 推導出的合法集合。
  - **沒選的做法會怎樣**：正控制突變測試（將文中一處百分比改為 `99.9%`）會漏抓，無法保證五個交付表面 100% 零數值漂移。
- **規格缺口**：`spec.md:36` 要求 `verify_ssot_consistency.py` 確保所有表面與 `metrics_ssot.json` 100% 一致，但未規定防部分突變的演算法。
- **影響範圍**：`scripts/verify_ssot_consistency.py` 與未來新增至論文或 README 的任何小數百分比。
- **判定**：`sound` — 徹底堵死「改錯其中一個表格格位但全文仍有同名數值」的漏網之魚，並已由 `test_verify_ssot_consistency.py` 正控制突變測試驗證。
- **把握度**：高
