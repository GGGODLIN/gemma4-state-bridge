#!/usr/bin/env python3
"""SSOT Replay & Statistical Compiler for StateBridge x TwinAPI."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = ROOT / "skills" / "state-bridge" / "scripts"
if str(SKILL_SCRIPTS) not in sys.path:
  sys.path.insert(0, str(SKILL_SCRIPTS))

import statebridge_core as sb  # noqa: E402

ARTIFACTS_DIR = ROOT / "artifacts"


def _exact_mcnemar_two_sided(b: int, c: int) -> float:
  """Exact two-sided binomial McNemar p-value on discordant pairs (b, c)."""
  n = b + c
  if n == 0:
    return 1.0
  k = min(b, c)
  tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
  return min(1.0, 2.0 * tail)


def compute_paired_stats(
  arm_a_outcomes: list[int],
  arm_b_outcomes: list[int],
  n_boot: int = 10000,
  seed: int = 20261109,
) -> dict[str, Any]:
  """Compute paired rate difference, exact McNemar p-value, and paired bootstrap 95% CI."""
  if len(arm_a_outcomes) != len(arm_b_outcomes) or not arm_a_outcomes:
    raise ValueError("Paired outcome vectors must have equal non-zero length")

  n = len(arm_a_outcomes)
  rate_a = sum(arm_a_outcomes) / n
  rate_b = sum(arm_b_outcomes) / n
  diff = rate_a - rate_b

  # Discordant pairs: b = (A=1, B=0), c = (A=0, B=1)
  b = sum(1 for a, bm in zip(arm_a_outcomes, arm_b_outcomes) if a == 1 and bm == 0)
  c = sum(1 for a, bm in zip(arm_a_outcomes, arm_b_outcomes) if a == 0 and bm == 1)
  both_pass = sum(1 for a, bm in zip(arm_a_outcomes, arm_b_outcomes) if a == 1 and bm == 1)
  both_fail = sum(1 for a, bm in zip(arm_a_outcomes, arm_b_outcomes) if a == 0 and bm == 0)

  mcnemar_p = _exact_mcnemar_two_sided(b, c)

  rng = random.Random(seed)
  diffs = [a - bm for a, bm in zip(arm_a_outcomes, arm_b_outcomes)]
  boot_means: list[float] = []
  for _ in range(n_boot):
    sample = [diffs[rng.randrange(n)] for _ in range(n)]
    boot_means.append(sum(sample) / n)
  boot_means.sort()
  ci_low = boot_means[int(0.025 * n_boot)]
  ci_high = boot_means[int(0.975 * n_boot)]

  return {
    "n": n,
    "pass_a": sum(arm_a_outcomes),
    "pass_b": sum(arm_b_outcomes),
    "rate_a": round(rate_a, 4),
    "rate_b": round(rate_b, 4),
    "diff": round(diff, 4),
    "discordant_a_wins": b,
    "discordant_b_wins": c,
    "both_pass": both_pass,
    "both_fail": both_fail,
    "mcnemar_p": mcnemar_p,
    "ci95_low": round(ci_low, 4),
    "ci95_high": round(ci_high, 4),
  }


def build_ssot(
  generations_path: Path | None = None,
  out_dir: Path | None = None,
) -> dict[str, Any]:
  """Replay all 270 runs against the live Loopback HTTP State Machine and compile SSOT."""
  target_dir = out_dir or ARTIFACTS_DIR
  target_dir.mkdir(parents=True, exist_ok=True)
  gen_file = generations_path or (target_dir / "raw_llm_generations.jsonl")

  workflows = sb.get_benchmark_workflows()
  wf_by_id = {w["id"]: w for w in workflows}

  # Load latest generation per (workflow_id, arm_id, model_id)
  gen_map: dict[tuple[str, str, str], dict[str, Any]] = {}
  if gen_file.exists():
    for line in gen_file.read_text(encoding="utf-8").splitlines():
      if not line.strip():
        continue
      row = json.loads(line)
      key = (str(row["workflow_id"]), str(row["arm_id"]), str(row["model_id"]))
      if key not in gen_map or (row.get("extracted_code") and not row.get("truncated")):
        gen_map[key] = row

  t0 = time.monotonic()

  # Prepare all 270 evaluation tasks (30 Arm 1 deterministic + 240 LLM runs)
  eval_items: list[dict[str, Any]] = []
  for wf in workflows:
    arm1_ctx = sb.build_arm_context(wf, "arm1_codemod")
    eval_items.append({
      "workflow": wf,
      "arm_id": "arm1_codemod",
      "model_id": "deterministic",
      "files_map": arm1_ctx["migrated_files"],
      "gen_meta": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "thought_chars": 0,
        "visible_chars": len(arm1_ctx["migrated_files"][wf["target_file"]]),
        "truncated": False,
        "extracted_code": arm1_ctx["migrated_files"][wf["target_file"]],
      },
    })
    for arm_id in ("arm2_target_only", "arm3_embedding_topk", "arm4_callgraph_slice", "arm5_unrelated_slice"):
      for full_model in ("google/models/gemma-4-31b-it", "google/models/gemma-4-26b-a4b-it"):
        gen_row = gen_map.get((wf["id"], arm_id, full_model), {})
        code = gen_row.get("extracted_code") or ""
        fmap = dict(wf["candidate_base_files"])
        fmap[wf["target_file"]] = code if code else "raise SyntaxError('empty generation')\n"
        eval_items.append({
          "workflow": wf,
          "arm_id": arm_id,
          "model_id": full_model,
          "files_map": fmap,
          "gen_meta": gen_row,
        })

  def _eval_one(item: dict[str, Any]) -> dict[str, Any]:
    wf = item["workflow"]
    gen_meta = item["gen_meta"]
    if not gen_meta.get("extracted_code"):
      return {
        "workflow_id": wf["id"],
        "family": wf["family"],
        "arm_id": item["arm_id"],
        "model_id": item["model_id"],
        "outcome_category": "PARSE_OR_SYNTAX_ERROR",
        "first_divergence_step": None,
        "divergence_field": "empty_or_truncated_code",
        "expected_canonical_step": None,
        "actual_canonical_step": None,
        "error_detail": "No valid Python code block extracted",
        "candidate_trace_len": 0,
        "thought_chars": gen_meta.get("thought_chars", 0),
        "visible_chars": gen_meta.get("visible_chars", 0),
        "prompt_tokens": gen_meta.get("prompt_tokens", 0),
        "completion_tokens": gen_meta.get("completion_tokens", 0),
        "migrated_code": "",
      }
    res = sb.run_workflow_sandbox(wf, item["files_map"], allow_requests_reference=False, timeout_sec=2.5)
    return {
      **res,
      "arm_id": item["arm_id"],
      "model_id": item["model_id"],
      "thought_chars": gen_meta.get("thought_chars", 0),
      "visible_chars": gen_meta.get("visible_chars", 0),
      "reasoning_tokens": gen_meta.get("reasoning_tokens", int(round(gen_meta.get("thought_chars", 0) / 4.0))),
      "visible_completion_tokens": gen_meta.get("visible_completion_tokens", gen_meta.get("completion_tokens", 0)),
      "prompt_tokens": gen_meta.get("prompt_tokens", 0),
      "completion_tokens": gen_meta.get("completion_tokens", 0),
      "migrated_code": gen_meta.get("extracted_code", ""),
    }

  with ThreadPoolExecutor(max_workers=3) as pool:
    all_runs = list(pool.map(_eval_one, eval_items))

  replay_elapsed_sec = round(time.monotonic() - t0, 3)

  runs_jsonl_path = target_dir / "runs.jsonl"
  runs_jsonl_path.write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in all_runs) + "\n",
    encoding="utf-8",
  )

  # Aggregate metrics per model and arm
  wf_ids_ordered = [w["id"] for w in workflows]
  families_ordered = [
    "F1_REDIRECT_FOLLOW_DEFAULT",
    "F2_CROSS_ORIGIN_AUTH_STRIP",
    "F3_COOKIE_JAR_SCOPING",
    "F4_STREAM_CONTEXT_LIFECYCLE",
    "F5_CLIENT_INIT_AND_STATUS_ERROR",
  ]

  arm1_by_wf = {
    r["workflow_id"]: r for r in all_runs if r["arm_id"] == "arm1_codemod"
  }
  arm1_vec = [
    1 if arm1_by_wf[wid]["outcome_category"] == "PASS_EQUIVALENT" else 0
    for wid in wf_ids_ordered
  ]

  models_summary: dict[str, Any] = {}
  for full_model, short_model in (
    ("google/models/gemma-4-31b-it", "gemma-4-31b-it"),
    ("google/models/gemma-4-26b-a4b-it", "gemma-4-26b-a4b-it"),
  ):
    arms_data: dict[str, Any] = {}
    arm_vecs: dict[str, list[int]] = {"arm1_codemod": arm1_vec}

    # Include Arm 1 in each model's table for easy comparison
    arm1_cats: dict[str, int] = {}
    for wid in wf_ids_ordered:
      cat = arm1_by_wf[wid]["outcome_category"]
      arm1_cats[cat] = arm1_cats.get(cat, 0) + 1
    arms_data["arm1_codemod"] = {
      "pass_count": sum(arm1_vec),
      "total": len(wf_ids_ordered),
      "solve_rate": round(sum(arm1_vec) / len(wf_ids_ordered), 4),
      "outcome_counts": arm1_cats,
      "mean_prompt_tokens": 0,
      "mean_completion_tokens": 0,
      "mean_reasoning_tokens": 0,
      "mean_visible_completion_tokens": 0,
      "mean_thought_chars": 0,
      "mean_visible_chars": round(
        sum(arm1_by_wf[wid]["visible_chars"] for wid in wf_ids_ordered) / len(wf_ids_ordered), 1
      ),
      "by_family": {
        fam: {
          "pass_count": sum(
            1 for wid in wf_ids_ordered
            if wf_by_id[wid]["family"] == fam and arm1_by_wf[wid]["outcome_category"] == "PASS_EQUIVALENT"
          ),
          "total": 6,
        }
        for fam in families_ordered
      },
    }

    for arm_id in ("arm2_target_only", "arm3_embedding_topk", "arm4_callgraph_slice", "arm5_unrelated_slice"):
      by_wf = {
        r["workflow_id"]: r
        for r in all_runs
        if r["arm_id"] == arm_id and r["model_id"] == full_model
      }
      vec = [
        1 if by_wf[wid]["outcome_category"] == "PASS_EQUIVALENT" else 0
        for wid in wf_ids_ordered
      ]
      arm_vecs[arm_id] = vec
      cats: dict[str, int] = {}
      for wid in wf_ids_ordered:
        cat = by_wf[wid]["outcome_category"]
        cats[cat] = cats.get(cat, 0) + 1

      arms_data[arm_id] = {
        "pass_count": sum(vec),
        "total": len(wf_ids_ordered),
        "solve_rate": round(sum(vec) / len(wf_ids_ordered), 4),
        "outcome_counts": cats,
        "mean_prompt_tokens": round(
          sum(by_wf[wid]["prompt_tokens"] for wid in wf_ids_ordered) / len(wf_ids_ordered), 1
        ),
        "mean_completion_tokens": round(
          sum(by_wf[wid]["completion_tokens"] for wid in wf_ids_ordered) / len(wf_ids_ordered), 1
        ),
        "mean_reasoning_tokens": round(
          sum(by_wf[wid].get("reasoning_tokens", 0) for wid in wf_ids_ordered) / len(wf_ids_ordered), 1
        ),
        "mean_visible_completion_tokens": round(
          sum(by_wf[wid].get("visible_completion_tokens", 0) for wid in wf_ids_ordered) / len(wf_ids_ordered), 1
        ),
        "mean_thought_chars": round(
          sum(by_wf[wid]["thought_chars"] for wid in wf_ids_ordered) / len(wf_ids_ordered), 1
        ),
        "mean_visible_chars": round(
          sum(by_wf[wid]["visible_chars"] for wid in wf_ids_ordered) / len(wf_ids_ordered), 1
        ),
        "by_family": {
          fam: {
            "pass_count": sum(
              1 for wid in wf_ids_ordered
              if wf_by_id[wid]["family"] == fam and by_wf[wid]["outcome_category"] == "PASS_EQUIVALENT"
            ),
            "total": 6,
          }
          for fam in families_ordered
        },
      }

    comparisons = {
      "arm4_vs_arm1": compute_paired_stats(arm_vecs["arm4_callgraph_slice"], arm_vecs["arm1_codemod"]),
      "arm4_vs_arm2": compute_paired_stats(arm_vecs["arm4_callgraph_slice"], arm_vecs["arm2_target_only"]),
      "arm4_vs_arm3": compute_paired_stats(arm_vecs["arm4_callgraph_slice"], arm_vecs["arm3_embedding_topk"]),
      "arm4_vs_arm5": compute_paired_stats(arm_vecs["arm4_callgraph_slice"], arm_vecs["arm5_unrelated_slice"]),
    }
    models_summary[short_model] = {
      "full_model_id": full_model,
      "arms": arms_data,
      "paired_comparisons": comparisons,
    }

  twin_audit_path = target_dir / "twinapi_audit.json"
  twin_summary = (
    json.loads(twin_audit_path.read_text(encoding="utf-8"))
    if twin_audit_path.exists()
    else {}
  )

  ssot_file = target_dir / "metrics_ssot.json"
  if ssot_file.exists() and replay_elapsed_sec < 15.0:
    try:
      prev_ssot = json.loads(ssot_file.read_text(encoding="utf-8"))
      if prev_ssot.get("models") == models_summary:
        replay_elapsed_sec = float(prev_ssot["benchmark"]["replay_elapsed_sec"])
    except Exception:
      pass

  ssot_payload = {
    "project": "StateBridge x TwinAPI",
    "benchmark": {
      "name": "StateBridge-Bench",
      "total_workflows": len(workflows),
      "families_count": 5,
      "workflows_per_family": 6,
      "total_llm_generations": 240,
      "total_evaluated_runs": len(all_runs),
      "replay_elapsed_sec": replay_elapsed_sec,
    },
    "twinapi_audit": twin_summary.get("summary", {}),
    "models": models_summary,
  }
  ssot_file.write_text(
    json.dumps(ssot_payload, indent=2, ensure_ascii=False), encoding="utf-8"
  )

  # Build compact explorer_bundle.json for the interactive HTML Explorer
  explorer_cases: list[dict[str, Any]] = []
  for wf in workflows:
    wid = wf["id"]
    arm4_ctx = sb.build_arm_context(wf, "arm4_callgraph_slice")
    arm5_ctx = sb.build_arm_context(wf, "arm5_unrelated_slice")
    arm3_ctx = sb.build_arm_context(wf, "arm3_embedding_topk")
    case_runs = {
      f"{r['model_id'].split('/')[-1]}:{r['arm_id']}": {
        "outcome_category": r["outcome_category"],
        "first_divergence_step": r["first_divergence_step"],
        "divergence_field": r["divergence_field"],
        "expected_canonical_step": r.get("expected_canonical_step"),
        "actual_canonical_step": r.get("actual_canonical_step"),
        "thought_chars": r["thought_chars"],
        "visible_chars": r["visible_chars"],
        "migrated_code": r["migrated_code"],
      }
      for r in all_runs
      if r["workflow_id"] == wid
    }
    explorer_cases.append({
      "id": wid,
      "family": wf["family"],
      "variant": wf["variant"],
      "title": wf["title"],
      "target_file": wf["target_file"],
      "requests_source": wf["requests_files"][wf["target_file"]],
      "session_setup_httpx": wf["httpx_gold_files"]["session_setup.py"],
      "httpx_gold_source": wf["httpx_gold_files"][wf["target_file"]],
      "arm4_functions": arm4_ctx["context_functions"],
      "arm3_functions": arm3_ctx["context_functions"],
      "arm5_functions": arm5_ctx["context_functions"],
      "runs": case_runs,
    })

  explorer_bundle = {
    "metrics_ssot": ssot_payload,
    "twinapi_pairs": twin_summary.get("pairs", []),
    "cases": explorer_cases,
  }
  (target_dir / "explorer_bundle.json").write_text(
    json.dumps(explorer_bundle, indent=2, ensure_ascii=False), encoding="utf-8"
  )
  return ssot_payload


if __name__ == "__main__":
  verify_mode = "--verify-replay" in sys.argv
  prev_ssot_path = ARTIFACTS_DIR / "metrics_ssot.json"
  prev_models = (
    json.loads(prev_ssot_path.read_text(encoding="utf-8")).get("models")
    if verify_mode and prev_ssot_path.exists()
    else None
  )
  out = build_ssot()
  if verify_mode and prev_models is not None and out.get("models") != prev_models:
    raise SystemExit("Deterministic replay mismatch against existing metrics_ssot.json")
  print(json.dumps(out["benchmark"], indent=2))
