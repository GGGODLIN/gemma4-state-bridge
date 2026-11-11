"""Acceptance test suite for Ticket 02 Seam 2 (TwinAPI Audit, Gemma 4 Runner, and SSOT Compiler)."""

from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
SKILL_SCRIPTS_DIR = ROOT / "skills" / "state-bridge" / "scripts"
for p in (SCRIPTS_DIR, SKILL_SCRIPTS_DIR):
  if str(p) not in sys.path:
    sys.path.insert(0, str(p))

import build_ssot as ssot_mod  # noqa: E402
import build_twinapi_audit as twin_mod  # noqa: E402
import run_gemma4_matrix as runner_mod  # noqa: E402


def test_extract_code_and_thought_strips_thought_and_handles_malformed():
  sample_good = (
    "<thought>\nWe need to migrate requests.Session() to httpx.Client().\n"
    "In requests we used import requests, let's replace it.\n</thought>\n"
    "Here is the migrated file:\n"
    "```python\nimport httpx\n\ndef execute_step(base_url: str) -> dict:\n  return {'ok': True}\n```\n"
  )
  parsed = runner_mod.extract_code_and_thought(sample_good, finish_reason="stop")
  assert parsed["truncated"] is False
  assert parsed["thought_chars"] > 20
  assert "import requests" not in parsed["extracted_code"]
  assert "import httpx" in parsed["extracted_code"]

  # Unclosed <thought> or finish_reason == 'length' with no/unclosed code block must flag truncated
  sample_truncated = "<thought>Still thinking about redirects and cookies..."
  parsed_trunc = runner_mod.extract_code_and_thought(sample_truncated, finish_reason="length")
  assert parsed_trunc["truncated"] is True
  assert parsed_trunc["extracted_code"] == ""

  sample_mid_fence = (
    "<thought>Done thinking.</thought>\n"
    "```python\nimport httpx\ndef execute_step(base_url: str) -> dict:\n  r = session.get("
  )
  parsed_mid = runner_mod.extract_code_and_thought(sample_mid_fence, finish_reason="length")
  assert parsed_mid["truncated"] is True
  assert parsed_mid["extracted_code"] == ""

  assert twin_mod._levenshtein_similarity("post", "post") == 1.0
  assert twin_mod._cosine([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_checkpoint_loader_skips_completed_entries(tmp_path: Path):
  ckpt = tmp_path / "raw_llm_generations.jsonl"
  row = {
    "workflow_id": "F1_01",
    "arm_id": "arm4_callgraph_slice",
    "model_id": "google/models/gemma-4-31b-it",
    "extracted_code": "import httpx\n",
    "truncated": False,
  }
  ckpt.write_text(json.dumps(row) + "\n", encoding="utf-8")
  done = runner_mod.load_completed_keys(ckpt)
  assert ("F1_01", "arm4_callgraph_slice", "google/models/gemma-4-31b-it") in done


def test_compute_paired_stats_mcnemar_and_bootstrap():
  # Arm 4 passes 26/30, Arm 2 passes 8/30
  arm4 = [1] * 26 + [0] * 4
  arm2 = [1] * 8 + [0] * 22
  stats = ssot_mod.compute_paired_stats(arm4, arm2)
  assert stats["n"] == 30
  assert stats["rate_a"] > stats["rate_b"]
  assert stats["diff"] == round(26 / 30 - 8 / 30, 4)
  assert stats["mcnemar_p"] < 0.001
  assert stats["ci95_low"] > 0.25


def test_exact_mcnemar_preserves_small_nonzero_probabilities():
  assert ssot_mod._exact_mcnemar_two_sided(29, 0) == 2 / 2**29
  assert ssot_mod._exact_mcnemar_two_sided(0, 29) == 2 / 2**29
  assert ssot_mod._exact_mcnemar_two_sided(0, 0) == 1.0


def test_twinapi_audit_and_full_270_run_ssot_artifacts_exist_and_validate():
  emb_path = ROOT / "artifacts" / "twinapi_embeddings_256d.json"
  audit_path = ROOT / "artifacts" / "twinapi_audit.json"
  gen_path = ROOT / "artifacts" / "raw_llm_generations.jsonl"
  runs_path = ROOT / "artifacts" / "runs.jsonl"
  ssot_path = ROOT / "artifacts" / "metrics_ssot.json"
  bundle_path = ROOT / "artifacts" / "explorer_bundle.json"

  assert emb_path.exists(), "twinapi_embeddings_256d.json missing"
  assert audit_path.exists(), "twinapi_audit.json missing"
  assert gen_path.exists(), "raw_llm_generations.jsonl missing"
  assert runs_path.exists(), "runs.jsonl missing"
  assert ssot_path.exists(), "metrics_ssot.json missing"
  assert bundle_path.exists(), "explorer_bundle.json missing"

  emb_data = json.loads(emb_path.read_text(encoding="utf-8"))
  assert emb_data["dimensions"] == 256
  assert len(emb_data["vectors"]) >= 24

  audit_data = json.loads(audit_path.read_text(encoding="utf-8"))
  assert audit_data["summary"]["top_level_exact_name_overlap_ratio"] >= 0.85
  assert len(audit_data["pairs"]) >= 15

  gen_lines = [json.loads(line) for line in gen_path.read_text(encoding="utf-8").splitlines() if line.strip()]
  assert len(gen_lines) == 240

  run_lines = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
  assert len(run_lines) == 270

  ssot = json.loads(ssot_path.read_text(encoding="utf-8"))
  assert ssot["benchmark"]["total_workflows"] == 30
  assert ssot["benchmark"]["total_evaluated_runs"] == 270
  for model_key in ("gemma-4-31b-it", "gemma-4-26b-a4b-it"):
    m_arms = ssot["models"][model_key]["arms"]
    assert m_arms["arm4_callgraph_slice"]["pass_count"] > m_arms["arm2_target_only"]["pass_count"]
    assert m_arms["arm4_callgraph_slice"]["pass_count"] > m_arms["arm3_embedding_topk"]["pass_count"]
    assert m_arms["arm4_callgraph_slice"]["pass_count"] > m_arms["arm5_unrelated_slice"]["pass_count"]
