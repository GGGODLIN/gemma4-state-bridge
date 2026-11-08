# StateBridge × TwinAPI：以呼叫圖狀態切片與本地迴路狀態機破解 Gemma 4 跨套件遷移的「狀態型假朋友悖論」

> **[English README (README.md)](/README.md)** · **[Kaggle 論文賽道報告](/writeup/KAGGLE_WRITEUP.md)** · **[完整論文 PDF](/paper/paper.pdf)** · **[可重現 Notebook](/notebooks/reproduce_statebridge.ipynb)** · **[互動式 HTML 演練室](/explorer/index.html)**

---

## 核心發現（The Stateful False-Friend Paradox）

當開發者 Agent 將專案由 `requests` (`2.34.2`) 遷移至 `httpx` (`0.28.1`) 時，會撞上**「狀態型假朋友悖論（Stateful False-Friend Paradox）」**：
- 兩套函式庫有 **88.9%** (`8/9`) 的頂層函式與 **83.3%** (`10/12`) 的 `Session`/`Client` 核心方法**名稱完全相同**；
- 12 組具狀態分歧的對應 API 在 256 維 Gemini 向量空間（`google/models/gemini-embedding-001`）中的平均餘弦相似度高達 **0.873**；
- 然而在多步 HTTP 執行期（`302` 轉址預設值、跨網域 `Authorization` 標頭剝除、Cookie 網域萃取 `.get_dict()`、串流 Context 生命週期、例外繼承階層反轉），底層狀態機卻在無聲中分道揚鑣。

**StateBridge** 結合 **1–2 跳靜態呼叫圖狀態切片（`Arm 4`）** 與 **本地迴路 HTTP 狀態機沙箱（`127.0.0.1:0`）**，在 **270 次真實沙箱執行**（240 次真實 `Gemma-4-31B-IT` 與 `Gemma-4-26B-A4B-IT` 推論 ＋ 30 次 AST Codemod）中取得顯著突破：

| 實驗組別 (Arm) | `Gemma-4-31B-IT` (Dense) 通過率 | `Gemma-4-26B-A4B-IT` (MoE) 通過率 | 平均輸入 Tokens |
| :--- | :---: | :---: | :---: |
| **`Arm 1`：決定性 AST Codemod** | **0.0%** (`0/30`) | **0.0%** (`0/30`) | `0.0` |
| **`Arm 2`：單一目標檔 + 靜態對照表** | **13.3%** (`4/30`) | **13.3%** (`4/30`) | `497.7` |
| **`Arm 3`：256 維 Embedding Top-K** | **13.3%** (`4/30`) | **13.3%** (`4/30`) | `637.7` |
| **`Arm 5`：等預算無關函式切片（對照組）** | **16.7%** (`5/30`) | **13.3%** (`4/30`) | `702.7` |
| **`Arm 4`：StateBridge 呼叫圖狀態切片** | **90.0%** (`27/30`, $p=2.4e-07$) | **96.7%** (`29/30`, $p=6e-08$) | `799.9` |

---

## 15 秒離線重現指令

```bash
uv run --with requests==2.34.2 --with httpx==0.28.1 python3 scripts/build_ssot.py --verify-replay
uv run --with requests==2.34.2 --with httpx==0.28.1 --with pytest python3 -m pytest -v
python3 scripts/verify_ssot_consistency.py
```
