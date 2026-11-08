#!/usr/bin/env python3
"""Real Dual-Model Gemma 4 (31B-IT & 26B-A4B-IT) Inference Matrix Runner."""

from __future__ import annotations
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = ROOT / "skills" / "state-bridge" / "scripts"
if str(SKILL_SCRIPTS) not in sys.path:
  sys.path.insert(0, str(SKILL_SCRIPTS))

import statebridge_core as sb  # noqa: E402

ARTIFACTS_DIR = ROOT / "artifacts"
DEFAULT_OUT_JSONL = ARTIFACTS_DIR / "raw_llm_generations.jsonl"

MODELS = (
  "google/models/gemma-4-31b-it",
  "google/models/gemma-4-26b-a4b-it",
)
LLM_ARMS = (
  "arm2_target_only",
  "arm3_embedding_topk",
  "arm4_callgraph_slice",
  "arm5_unrelated_slice",
)

_THOUGHT_CLOSED_RE = re.compile(r"<thought>([\s\S]*?)</thought>", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n([\s\S]*?)```", re.IGNORECASE)


def _is_valid_python(code: str) -> bool:
  if not code or not code.strip():
    return False
  try:
    ast.parse(code)
    return True
  except SyntaxError:
    return False


def extract_code_and_thought(raw_content: str, finish_reason: str = "stop") -> dict[str, Any]:
  """Strip <thought>...</thought> blocks and extract migrated Python code from fence."""
  text = raw_content or ""
  thoughts = _THOUGHT_CLOSED_RE.findall(text)
  thought_chars = sum(len(t) for t in thoughts)

  low = text.lower()
  has_open_thought = low.count("<thought>") > low.count("</thought>")
  if has_open_thought:
    return {
      "thought_chars": len(text),
      "visible_chars": 0,
      "extracted_code": "",
      "truncated": True,
    }

  stripped = _THOUGHT_CLOSED_RE.sub("", text).strip()
  fences = _CODE_FENCE_RE.findall(stripped)
  if fences:
    code = fences[-1].strip() + "\n"
  elif "```" in stripped:
    # Unclosed markdown code fence
    code = ""
  elif "def execute_step" in stripped and "import httpx" in stripped and finish_reason != "length":
    code = stripped + "\n"
  else:
    code = ""

  if finish_reason == "length" and not fences:
    code = ""

  truncated = (finish_reason == "length" and not _is_valid_python(code)) or (not code)
  return {
    "thought_chars": thought_chars,
    "visible_chars": len(stripped),
    "extracted_code": code,
    "truncated": bool(truncated),
  }


def load_completed_keys(jsonl_path: Path) -> set[tuple[str, str, str]]:
  """Load already completed (workflow_id, arm_id, model_id) keys from JSONL checkpoint."""
  done: set[tuple[str, str, str]] = set()
  if not jsonl_path.exists():
    return done
  for line in jsonl_path.read_text(encoding="utf-8").splitlines():
    line_s = line.strip()
    if not line_s:
      continue
    try:
      row = json.loads(line_s)
      code = str(row.get("extracted_code") or "")
      if _is_valid_python(code) and not row.get("truncated"):
        done.add((str(row["workflow_id"]), str(row["arm_id"]), str(row["model_id"])))
    except Exception:
      continue
  return done


def _load_magpie_key() -> str:
  env_path = Path.home() / ".cli-proxy-api" / "keys.env"
  if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
      if line.startswith("CLIPROXY_KEY_ADMIN="):
        return line.split("=", 1)[1].strip().strip("'\"")
  return "probe"


class _RateLimiter:
  def __init__(self, min_interval_sec: float = 2.0):
    self.min_interval_sec = min_interval_sec
    self.lock = threading.Lock()
    self.last_call = 0.0

  def wait_turn(self) -> None:
    with self.lock:
      now = time.monotonic()
      wait = self.min_interval_sec - (now - self.last_call)
      if wait > 0:
        time.sleep(wait)
      self.last_call = time.monotonic()


def _call_gemma4_single(
  model_id: str,
  prompt: str,
  api_key: str,
  limiter: _RateLimiter,
  max_attempts: int = 3,
) -> dict[str, Any]:
  last_err = ""
  for attempt in range(1, max_attempts + 1):
    payload = json.dumps({
      "model": model_id,
      "messages": [{"role": "user", "content": prompt}],
      "temperature": 0.0 if attempt == 1 else 0.2,
      "max_tokens": 16384,
    }).encode("utf-8")
    limiter.wait_turn()
    t0 = time.monotonic()
    req = urllib.request.Request(
      "http://127.0.0.1:8317/v1/chat/completions",
      data=payload,
      headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
      },
      method="POST",
    )
    try:
      with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
      elapsed = round(time.monotonic() - t0, 3)
      choice = (data.get("choices") or [{}])[0]
      content = (choice.get("message") or {}).get("content") or ""
      finish_reason = choice.get("finish_reason") or "stop"
      usage = data.get("usage") or {}
      parsed = extract_code_and_thought(content, finish_reason=finish_reason)
      if _is_valid_python(parsed["extracted_code"]) and not parsed["truncated"]:
        comp_tok = int(usage.get("completion_tokens", 0))
        details = usage.get("completion_tokens_details") or {}
        reas_tok = int(details.get("reasoning_tokens") or round(parsed["thought_chars"] / 4.0))
        vis_tok = max(0, comp_tok - int(details.get("reasoning_tokens") or 0)) or int(round(parsed["visible_chars"] / 4.0))
        return {
          "status": "ok",
          "elapsed_sec": elapsed,
          "finish_reason": finish_reason,
          "raw_content": content,
          "extracted_code": parsed["extracted_code"],
          "thought_chars": parsed["thought_chars"],
          "visible_chars": parsed["visible_chars"],
          "reasoning_tokens": reas_tok,
          "visible_completion_tokens": vis_tok,
          "truncated": False,
          "prompt_tokens": usage.get("prompt_tokens", 0),
          "completion_tokens": comp_tok,
          "total_tokens": usage.get("total_tokens", 0),
          "attempt": attempt,
        }
      last_err = f"truncated_or_empty (finish_reason={finish_reason})"
    except urllib.error.HTTPError as exc:
      err_body = exc.read().decode("utf-8", errors="replace")[:300]
      last_err = f"HTTP {exc.code}: {err_body}"
      time.sleep(3.0 * attempt)
    except Exception as exc:
      last_err = str(exc)
      time.sleep(2.0 * attempt)

  return {
    "status": "error",
    "elapsed_sec": 0.0,
    "finish_reason": "error",
    "raw_content": "",
    "extracted_code": "",
    "thought_chars": 0,
    "visible_chars": 0,
    "truncated": True,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "error": last_err,
  }


def _compact_jsonl(jsonl_path: Path, keep_only_ok: bool = False) -> None:
  if not jsonl_path.exists():
    return
  by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
  for line in jsonl_path.read_text(encoding="utf-8").splitlines():
    line_s = line.strip()
    if not line_s:
      continue
    try:
      row = json.loads(line_s)
      key = (str(row["workflow_id"]), str(row["arm_id"]), str(row["model_id"]))
      if row.get("raw_content"):
        re_parsed = extract_code_and_thought(str(row.get("raw_content") or ""), str(row.get("finish_reason") or "stop"))
        row["extracted_code"] = re_parsed["extracted_code"]
        row["thought_chars"] = re_parsed["thought_chars"]
        row["visible_chars"] = re_parsed["visible_chars"]
        row["truncated"] = re_parsed["truncated"]
      row.setdefault("reasoning_tokens", int(round(int(row.get("thought_chars") or 0) / 4.0)))
      row.setdefault("visible_completion_tokens", int(row.get("completion_tokens") or round(int(row.get("visible_chars") or 0) / 4.0)))
      is_ok = _is_valid_python(str(row.get("extracted_code") or "")) and not row.get("truncated")
      if keep_only_ok and not is_ok:
        continue
      if key not in by_key or is_ok:
        by_key[key] = row
    except Exception:
      continue
  jsonl_path.write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in by_key.values()) + ("\n" if by_key else ""),
    encoding="utf-8",
  )


def run_matrix(out_jsonl: Path | None = None, min_interval_sec: float = 2.0) -> dict[str, Any]:
  target_path = out_jsonl or DEFAULT_OUT_JSONL
  target_path.parent.mkdir(parents=True, exist_ok=True)
  _compact_jsonl(target_path, keep_only_ok=True)

  completed = load_completed_keys(target_path)
  workflows = sb.get_benchmark_workflows()
  api_key = _load_magpie_key()
  limiter = _RateLimiter(min_interval_sec=min_interval_sec)
  write_lock = threading.Lock()

  tasks: list[tuple[dict[str, Any], str, str]] = []
  for wf in workflows:
    for arm_id in LLM_ARMS:
      for model_id in MODELS:
        key = (wf["id"], arm_id, model_id)
        if key not in completed:
          tasks.append((wf, arm_id, model_id))

  print(json.dumps({
    "event": "matrix_start",
    "already_completed": len(completed),
    "remaining_tasks": len(tasks),
    "total_expected": len(workflows) * len(LLM_ARMS) * len(MODELS),
  }), flush=True)

  if not tasks:
    return {"completed": len(completed), "newly_run": 0}

  newly_run = 0
  with open(target_path, "a", encoding="utf-8") as f_out:
    def _worker(item: tuple[dict[str, Any], str, str]) -> dict[str, Any]:
      wf, arm_id, model_id = item
      ctx = sb.build_arm_context(wf, arm_id)
      res = _call_gemma4_single(model_id, ctx["prompt"], api_key, limiter)
      row = {
        "workflow_id": wf["id"],
        "family": wf["family"],
        "arm_id": arm_id,
        "model_id": model_id,
        "context_functions": ctx["context_functions"],
        "context_chars": ctx["context_chars"],
        "prompt_chars": len(ctx["prompt"]),
        **res,
      }
      with write_lock:
        f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
        f_out.flush()
        os.fsync(f_out.fileno())
      return row

    with ThreadPoolExecutor(max_workers=2) as pool:
      futs = [pool.submit(_worker, t) for t in tasks]
      for fut in as_completed(futs):
        r = fut.result()
        newly_run += 1
        if newly_run % 10 == 0 or newly_run == len(tasks):
          print(json.dumps({
            "event": "matrix_progress",
            "newly_run": newly_run,
            "of_remaining": len(tasks),
            "last_workflow": r["workflow_id"],
            "last_arm": r["arm_id"],
            "last_model": r["model_id"],
            "status": r["status"],
          }), flush=True)

  _compact_jsonl(target_path, keep_only_ok=False)
  return {"completed": len(completed) + newly_run, "newly_run": newly_run}


if __name__ == "__main__":
  summary = run_matrix()
  print(json.dumps(summary))
