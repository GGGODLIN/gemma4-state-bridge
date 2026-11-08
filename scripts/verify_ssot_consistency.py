#!/usr/bin/env python3
"""SSOT Consistency Gate across all 5 submission surfaces."""

from __future__ import annotations
import json
from pathlib import Path
import re
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_PCT_RE = re.compile(r"\b\d+\.\d+%")


def _fmt_pct(rate: float) -> str:
  return f"{rate * 100:.1f}%"


def _allowed_percentages(ssot: dict[str, Any]) -> set[str]:
  twin = ssot["twinapi_audit"]
  allowed: set[str] = {
    "0.0%",
    "100.0%",
    _fmt_pct(twin["top_level_exact_name_overlap_ratio"]),
    _fmt_pct(twin["session_client_core_overlap_ratio"]),
  }
  for m_obj in ssot["models"].values():
    for arm_obj in m_obj["arms"].values():
      allowed.add(_fmt_pct(arm_obj["solve_rate"]))
      allowed.add(_fmt_pct(1.0 - arm_obj["solve_rate"]))
      for fam_obj in arm_obj.get("by_family", {}).values():
        allowed.add(_fmt_pct(fam_obj["pass_count"] / max(fam_obj["total"], 1)))
    for cmp_obj in m_obj["paired_comparisons"].values():
      allowed.add(_fmt_pct(cmp_obj["rate_a"]))
      allowed.add(_fmt_pct(cmp_obj["rate_b"]))
      allowed.add(_fmt_pct(cmp_obj["diff"]))
      allowed.add(_fmt_pct(cmp_obj["ci95_low"]))
      allowed.add(_fmt_pct(cmp_obj["ci95_high"]))
  return allowed


def verify_all_surfaces(root_dir: Path | None = None) -> dict[str, Any]:
  """Verify Writeup, Paper MD/PDF, Notebook, Explorer, and READMEs against metrics_ssot.json."""
  base = root_dir or ROOT
  errors: list[str] = []

  ssot_path = base / "artifacts" / "metrics_ssot.json"
  if not ssot_path.exists():
    return {"ok": False, "errors": ["Missing artifacts/metrics_ssot.json"], "word_count": 0}

  ssot = json.loads(ssot_path.read_text(encoding="utf-8"))
  twin = ssot["twinapi_audit"]
  m31 = ssot["models"]["gemma-4-31b-it"]
  m26 = ssot["models"]["gemma-4-26b-a4b-it"]
  allowed_pcts = _allowed_percentages(ssot)

  p31_str = str(m31["paired_comparisons"]["arm4_vs_arm2"]["mcnemar_p"])
  p26_str = str(m26["paired_comparisons"]["arm4_vs_arm2"]["mcnemar_p"])

  expected_tokens = [
    _fmt_pct(twin["top_level_exact_name_overlap_ratio"]),       # 88.9%
    _fmt_pct(twin["session_client_core_overlap_ratio"]),        # 83.3%
    f"{twin['mean_false_friend_embedding_cosine_256d']:.3f}",   # 0.873
    _fmt_pct(m31["arms"]["arm4_callgraph_slice"]["solve_rate"]),  # 90.0%
    f"{m31['arms']['arm4_callgraph_slice']['pass_count']}/30",    # 27/30
    _fmt_pct(m26["arms"]["arm4_callgraph_slice"]["solve_rate"]),  # 96.7%
    f"{m26['arms']['arm4_callgraph_slice']['pass_count']}/30",    # 29/30
    _fmt_pct(m31["arms"]["arm2_target_only"]["solve_rate"]),      # 13.3%
    f"{m31['arms']['arm2_target_only']['pass_count']}/30",        # 4/30
    _fmt_pct(m31["arms"]["arm5_unrelated_slice"]["solve_rate"]),  # 16.7%
    f"{m31['arms']['arm5_unrelated_slice']['pass_count']}/30",    # 5/30
    p31_str,
    p26_str,
  ]

  # 0. Verify runs.jsonl contains zero host path leaks if present
  runs_path = base / "artifacts" / "runs.jsonl"
  if runs_path.exists():
    runs_raw = runs_path.read_text(encoding="utf-8")
    if str(Path.home()) in runs_raw or "/var/folders/" in runs_raw:
      errors.append("artifacts/runs.jsonl contains unredacted host filesystem paths")

  # 1. Verify writeup/KAGGLE_WRITEUP.md
  writeup_path = base / "writeup" / "KAGGLE_WRITEUP.md"
  word_count = 0
  if not writeup_path.exists():
    errors.append("Missing writeup/KAGGLE_WRITEUP.md")
  else:
    w_text = writeup_path.read_text(encoding="utf-8")
    word_count = len(w_text.split())
    if word_count > 3000 or word_count < 1200:
      errors.append(f"writeup/KAGGLE_WRITEUP.md word_count={word_count} outside [1200, 3000]")
    for sec in (
      "## Abstract",
      "## 1. Introduction",
      "## 2. The TwinAPI Stateful False-Friend Audit",
      "## 3. StateBridge Architecture & 5-Arm Experimental Design",
      "## 4. Empirical Results on Gemma 4",
      "## 5. Related Works & Citations",
    ):
      if sec not in w_text:
        errors.append(f"writeup/KAGGLE_WRITEUP.md missing section {sec}")
    for tok in expected_tokens:
      if tok not in w_text:
        errors.append(f"writeup/KAGGLE_WRITEUP.md missing SSOT token '{tok}'")
    for found_pct in _PCT_RE.findall(w_text):
      if found_pct not in allowed_pcts:
        errors.append(f"writeup/KAGGLE_WRITEUP.md contains non-SSOT drifted percentage '{found_pct}'")

  # 2. Verify paper/paper.md and paper/paper.pdf
  paper_md_path = base / "paper" / "paper.md"
  paper_pdf_path = base / "paper" / "paper.pdf"
  if not paper_md_path.exists():
    errors.append("Missing paper/paper.md")
  else:
    p_text = paper_md_path.read_text(encoding="utf-8")
    for tok in expected_tokens:
      if tok not in p_text:
        errors.append(f"paper/paper.md missing SSOT token '{tok}'")
    for found_pct in _PCT_RE.findall(p_text):
      if found_pct not in allowed_pcts:
        errors.append(f"paper/paper.md contains non-SSOT drifted percentage '{found_pct}'")

  if not paper_pdf_path.exists():
    errors.append("Missing paper/paper.pdf")
  else:
    pdf_bytes = paper_pdf_path.read_bytes()
    if not pdf_bytes.startswith(b"%PDF-"):
      errors.append("paper/paper.pdf missing %PDF- header")
    if len(pdf_bytes) < 25000:
      errors.append(f"paper/paper.pdf too small ({len(pdf_bytes)} bytes)")

  # 3. Verify notebooks/reproduce_statebridge.ipynb
  nb_path = base / "notebooks" / "reproduce_statebridge.ipynb"
  if not nb_path.exists():
    errors.append("Missing notebooks/reproduce_statebridge.ipynb")
  else:
    nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
    code_cells = [c for c in nb_data.get("cells", []) if c.get("cell_type") == "code"]
    if len(code_cells) < 3:
      errors.append("notebooks/reproduce_statebridge.ipynb has fewer than 3 code cells")
    out_text = "\n".join(
      "".join(o.get("text", []))
      for c in code_cells
      for o in c.get("outputs", [])
    )
    for tok in expected_tokens:
      if tok not in out_text:
        errors.append(f"notebooks/reproduce_statebridge.ipynb executed output missing SSOT token '{tok}'")
    for found_pct in _PCT_RE.findall(out_text):
      if found_pct not in allowed_pcts:
        errors.append(f"notebooks/reproduce_statebridge.ipynb contains non-SSOT drifted percentage '{found_pct}'")

  # 4. Verify explorer/index.html
  exp_path = base / "explorer" / "index.html"
  if not exp_path.exists():
    errors.append("Missing explorer/index.html")
  else:
    e_text = exp_path.read_text(encoding="utf-8")
    if "cdn." in e_text.lower():
      errors.append("explorer/index.html contains forbidden external CDN reference")
    for req_str in (
      "StateBridge",
      "TwinAPI",
      "First-Divergence",
      "window.location.hash",
      "F1_REDIRECT_FOLLOW_DEFAULT",
      f'"pass_count": {m26["arms"]["arm4_callgraph_slice"]["pass_count"]}',
      f'"pass_count": {m31["arms"]["arm4_callgraph_slice"]["pass_count"]}',
      p26_str,
    ):
      if req_str not in e_text:
        errors.append(f"explorer/index.html missing required string '{req_str}'")
    if "5.96e-7" in e_text:
      errors.append("explorer/index.html contains stale McNemar p-value 5.96e-7")
    for found_pct in _PCT_RE.findall(e_text):
      if found_pct not in allowed_pcts:
        errors.append(f"explorer/index.html contains non-SSOT drifted percentage '{found_pct}'")

  # 5. Verify README.md and README.zh-TW.md
  for r_name in ("README.md", "README.zh-TW.md"):
    r_path = base / r_name
    if not r_path.exists():
      errors.append(f"Missing {r_name}")
    else:
      r_text = r_path.read_text(encoding="utf-8")
      for tok in expected_tokens:
        if tok not in r_text:
          errors.append(f"{r_name} missing SSOT token '{tok}'")
      for found_pct in _PCT_RE.findall(r_text):
        if found_pct not in allowed_pcts:
          errors.append(f"{r_name} contains non-SSOT drifted percentage '{found_pct}'")

  return {
    "ok": len(errors) == 0,
    "errors": errors,
    "word_count": word_count,
    "checked_surfaces": [
      "writeup/KAGGLE_WRITEUP.md",
      "paper/paper.md",
      "paper/paper.pdf",
      "notebooks/reproduce_statebridge.ipynb",
      "explorer/index.html",
      "README.md",
      "README.zh-TW.md",
    ],
  }


if __name__ == "__main__":
  res = verify_all_surfaces()
  print(json.dumps(res, indent=2, ensure_ascii=False))
  if not res.get("ok"):
    raise SystemExit(1)

