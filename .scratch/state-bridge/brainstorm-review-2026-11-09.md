# Brainstorm Review Report: `StateBridge` (`requests → httpx` Stateful Call-Graph Migration & Verification)

- **Review Date**: 2026-11-09
- **Target Path**: `/Users/linhancheng/Desktop/projects/gemma4-state-bridge`
- **Competition**: Kaggle `Google — The Gemma 4 Developer Agent Paper Track` (Deadline: 2026-11-12 23:59 UTC, Prize Pool: US$35,000, License: Apache 2.0)

---

## 1. Anchored Hypothesis（補齊背景與邊界後的構想假設）

- **Observed Pain**:
  - `requests`（`2.34.2`）與 `httpx`（`0.28.1`）在公開 API 名稱上高達 83.3%～88.9% 完全同名（表面與 Embedding 相似度極高，形成「語義假朋友 Stateful False Friends」），但在跨函式多步狀態互動（1. `allow_redirects=True` vs `follow_redirects=False` 預設與參數改名；2. 301/302/303 跨網域跳轉時 `Authorization` 與自訂標頭剝離／保留與 POST-to-GET 轉換；3. 跨請求 Cookie 持久化與路徑隔離；4. `stream=True` vs `Client.stream()` 上下文生命週期與 `.read()`/`.iter_bytes()` 消費時機；5. `Client.__init__` 與單次請求參數位置搬移及 `HTTPError` vs `HTTPStatusError` 例外階層）上存在嚴重分歧。
  - 當客戶端將 Session/Client 建立、認證注入、請求發送與回應解析拆散在多個跨檔案函式時，單看目標函式的 AST Codemod、局部 LLM 或 Embedding 相似檢索看不見上下游狀態契約，導致單步看似合法、多步執行卻靜默分歧。
- **Core Mechanism & Experiment**:
  - 打造單一 Python 標準庫核心 `skills/state-bridge/scripts/statebridge_core.py`（兼作 Google ADK 離線 Skill、跨檔案 AST 狀態呼叫圖切片器、確定性 AST Codemod、以及本機 `127.0.0.1:0` 可重播 Loopback HTTP 狀態機驗證器）。
  - 在 `StateBridge-Bench`（跨檔案多步 HTTP 工作流程基準）上，以真實 `Gemma-4-31B-IT` 與 `Gemma-4-26B-A4B-IT` 對比 5 個正交實驗組（`Arm 1: 確定性 AST Codemod`、`Arm 2: Gemma 4 + 僅目標函式`、`Arm 3: Gemma 4 + Repo 內 256 維 Embedding Top-K 相似函式`、`Arm 4: Gemma 4 + StateBridge 真實跨函式呼叫圖子圖切片`、`Arm 5: Gemma 4 + 等 Token 無關函式負對照組`），量測多步行為保持率與首次分歧點（First-Divergence Step）。

---

## 2. 6-Lens Reviewer Status

| Seat | Reviewer Lens | Status | Highest Severity | Key Contribution |
| --- | --- | --- | --- | --- |
| **Seat 1** | Contrarian / YAGNI & Codebase Reuse | `completed` | `Blocker` | 實測證實 `requests` 與 `httpx` 頂層公開函式 88.9% 同名、`Session`/`Client` 方法 83.3% 同名；抓出原版 `TwinAPI` 公開 API 配對會被 `Edit Distance = 0` 打平，以及原版 `Arm 5` 原始碼內含呼叫名會洩漏圖邊。 |
| **Seat 2** | Cost, Quota & Token Economics | `completed` | `Blocker` | 實測抓出 `google/models/gemma-4-*` 預設強制輸出 `<thought>`（佔可見程式碼 6.75x～12.5x，小 `max_tokens` 會使程式碼歸零），且 Magpie `google` 供應商 429 會以 `restKey()="google"` 連坐鎖死主 session。 |
| **Seat 3** | UX & Developer Ergonomics | `completed` | `Blocker` | 提出「雙軌分離驗證（Frozen Real-Run Artifact Replay）」讓評審免 API Key、免 60GB GPU 在 15 秒內實跑 Loopback HTTP 狀態機重現全論文表格，並將 `TwinAPI` 重新定位為「Stateful False-Friends（高相似度假朋友）診斷層」。 |
| **Seat 4** | Migration, Compatibility & Rollback Safety | `completed` | `Blocker` | 實測 `sys.addaudithook` 攔截非 `127.0.0.1` 外網連線、指出 `httpx 0.28.1` 移除 `proxies=` 參數與預設 `trust_env=True` 風險，並要求 Ephemeral Port `127.0.0.1:0` + Process Group `killpg` + `THIRD_PARTY_NOTICES.md`。 |
| **Seat 5** | Observability & 3 AM Debuggability | `completed` | `Blocker` | 實測抓出 `requests` 與 `httpx` 在 `json=` 序列化空白（`16 bytes` vs `13 bytes`）、預設 `User-Agent` 與例外文字上的天然差異，設計 `canonicalize_exchange()` 與 7 類互斥 `outcome_category`（含防偷留 `import requests` 作弊）。 |
| **Seat 6** | Long-Term Tech Debt & Coupling | `completed` | `Blocker` | 設計單一真值來源（SSOT）編譯管線：以 `statebridge_core.py` 為唯一核心，由 `build_ssot.py` 產出 `metrics_ssot.json` 與 `explorer_bundle.json`，自動渲染 Writeup、LaTeX、Notebook 與單檔 HTML Explorer 並跑 `verify_ssot_consistency.py`。 |

---

## 3. Existing Reuse Candidates（Seat 1 & Seat 2 查獲的本機現成資源）

1. **Python 標準庫 `ast` + `http.server.ThreadingHTTPServer` + `http.cookies` + `urllib.parse`**：
   - 100% 零外部依賴即可完成跨檔案呼叫子圖抽取、`Arm 1` AST Codemod 與本機 Loopback HTTP 狀態機，直接讓 `skills/state-bridge/scripts/statebridge_core.py` 同時作為 ADK Skill 與 Benchmark 核心引擎，免寫兩套程式碼。
2. **本機 `uv` 快取環境（`requests==2.34.2`、`httpx==0.28.1` on Python 3.11/3.14）**：
   - 已在本機快取就緒，可直接在離線模式（`NO_PROXY="*"`）下秒級執行受測遷移程式碼。
3. **本機 `:8317` Magpie 上的 `google/models/gemma-4-31b-it`、`google/models/gemma-4-26b-a4b-it` 與 `google/models/gemini-embedding-001`（支援 `{"dimensions": 256}`）**：
   - 無總額度上限，只需控制 `max_tokens >= 8192` 與 `min_interval = 2.0s`（~15-20 RPM）即可在 15 分鐘內零 429 跑完完整實驗矩陣。

---

## 4. Prioritized Objections & Contested Trade-offs

### A. `[contested: Seat 1 vs Seat 3]` `TwinAPI`（跨套件對齊層）與 `Arm 5`（負對照組）的定位取捨
- **Seat 1（減法席）發現**：`requests` 與 `httpx` 的公開方法有 83.3%～88.9% 完全同名，若把 `TwinAPI` 做成「用 256 維 Embedding 找對應的 `httpx` 公開 API 名稱」，會被 `Edit Distance = 0` 直接打平；且原版 `Arm 5（同節點文字但打亂圖邊）` 因為呼叫關係就寫在函式原始碼內，打亂外部邊標籤無法遮蔽真實呼叫。
- **Seat 3（體驗席）建議**：把「公開 API 名稱與 256 維 Embedding 高度相似（Cosine > 0.85），但跨函式多步狀態行為卻分歧」當成論文開篇的第一個量化發現（**The Stateful False-Friend Paradox**），用來證明為什麼單靠 API 對齊或 Repo 內 Embedding 檢索（`Arm 3`）救不了跨函式遷移；同時將 `Arm 5` 改為 **「等 Token 預算的同專案無關函式切片（Equal-Token Unrelated Function Slice）」**（或在跨檔案抽象介面下遮蔽具體實作綁定），乾淨證明是「順著狀態呼叫邊抽出的上下游函式（`Arm 4`）」有效，而非 Context 長度變長有效！

### B. `[Blocker — Must Fix in Spec]` HTTP 封包正規化（`canonicalize_exchange`）與 7 類互斥判分收據（Seat 5）
- **風險**：`requests` 送 `json={"x": 1}` 預設帶空白（`16 bytes`），`httpx` 不帶空白（`13 bytes`），且預設 `User-Agent` 與 `raise_for_status()` 例外字串不同；若直接比對 Raw Wire 封包會 100% 假陽性失敗；若模型偷偷保留 `import requests` 則會作弊過關。
- **強制護欄**：
  1. `canonicalize_exchange()`：JSON body 先 `json.loads` 比對結構、Header keys 轉小寫並剔除預設 `user-agent`/`host`/`connection`/`accept-encoding`/`content-length`、`Cookie` 解析為 `dict`、例外只比對錯誤類別與 `status_code`。
  2. 每筆執行輸出 7 類互斥 `outcome_category`：`PASS_EQUIVALENT`、`CHEAT_RESIDUAL_REQUESTS`（AST 抓到殘留 `requests`/`urllib`）、`PARSE_OR_SYNTAX_ERROR`、`PRE_HTTP_RUNTIME_ERROR`、`STEP_K_WIRE_DIVERGENCE`、`POST_HTTP_RETURN_MISMATCH`、`TIMEOUT_OR_HANG`。

### C. `[Blocker — Must Fix in Spec]` `<thought>` 推理 Token 預算與防 Magpie 全域 429 連坐節流（Seat 2）
- **風險**：`google/models/gemma-4-31b-it` 與 `26b-a4b-it` 預設輸出 `<thought>...</thought>`（佔可見程式碼 6.75x～12.5x），小 `max_tokens` 會導致程式碼被截斷為空；且併發過高觸發 429 會讓 Magpie 將整個 `google` provider 鎖死 60 秒。
- **強制護欄**：
  1. 請求固定設 `max_tokens = 8192`、`temperature = 0.0`，提取程式碼前先剝離 `<thought>[\s\S]*?</thought>` 並將思考長度獨立記帳。
  2. 嚴格節流 `max_concurrency = 2`、`min_interval = 2.0s`（控制在 15–20 RPM 內），每跑完一發立即 `os.fsync()` 寫入 `artifacts/raw_llm_generations.jsonl` 支援斷點續跑。

### D. `[Blocker — Must Fix in Spec]` 雙軌分離驗證（評審 15 秒免 Key 重現）與單一真值來源 SSOT（Seat 3, Seat 4, Seat 6）
- **強制護欄**：
  1. **雙軌分離**：作者先跑完真實 `Gemma-4-31B-IT` 與 `26B-A4B-IT` 並封存於 `artifacts/raw_llm_generations.jsonl`；評審跑 Notebook 或 CLI `reproduce-paper` 時，預設直接讀取封存的模型生成程式碼，**在評審本機即時啟動 `ThreadingHTTPServer` (`127.0.0.1:0`) 實跑所有多步 HTTP 測試（`< 15 秒`、免 API Key、免 GPU）**，當場算出所有表格與統計檢定。
  2. **零外網與零孤兒程序**：受測子程序掛載 `sys.addaudithook` 阻斷一切非 `127.0.0.1` 連線、清空 `HTTP_PROXY`/`HTTPS_PROXY`、設 `start_new_session=True` + `2.0s` 逾時 `os.killpg`。
  3. **SSOT 自動渲染**：所有數字由 `artifacts/metrics_ssot.json` 唯一產出並自動編譯進 Writeup、LaTeX、Notebook 與單檔自包含 HTML Explorer（由 `verify_ssot_consistency.py` 驗收）。

---

## 5. Key Decisions Needed Before `/to-spec`

1. **`TwinAPI` 與 5 個實驗組（Arms）的精確收斂方式**（見聊天室待拍板問題 1）。
2. **視覺化 Explorer 的交付形式**（見聊天室待拍板問題 2）。
3. **基準任務規模與配額節流策略**（見聊天室待拍板問題 3）。
