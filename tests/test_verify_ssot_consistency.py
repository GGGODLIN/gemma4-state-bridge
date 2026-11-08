"""Acceptance test suite for Ticket 03 Seam 3 (render_surfaces.py & verify_ssot_consistency.py)."""

from __future__ import annotations
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPTS_DIR))

import render_surfaces as render_mod  # noqa: E402
import verify_ssot_consistency as verify_mod  # noqa: E402


def test_kaggle_writeup_structure_and_word_count_cap():
  writeup_path = ROOT / "writeup" / "KAGGLE_WRITEUP.md"
  assert writeup_path.exists(), "writeup/KAGGLE_WRITEUP.md must exist"
  text = writeup_path.read_text(encoding="utf-8")
  words = text.split()
  assert 1200 <= len(words) <= 3000, f"Writeup word count {len(words)} must be in [1200, 3000]"
  for section in (
    "## Abstract",
    "## 1. Introduction",
    "## 2. The TwinAPI Stateful False-Friend Audit",
    "## 3. StateBridge Architecture & 5-Arm Experimental Design",
    "## 4. Empirical Results on Gemma 4",
    "## 5. Related Works & Citations",
  ):
    assert section in text, f"Missing required section: {section}"


def test_paper_pdf_and_executed_notebook_exist_and_match_ssot():
  pdf_path = ROOT / "paper" / "paper.pdf"
  md_path = ROOT / "paper" / "paper.md"
  nb_path = ROOT / "notebooks" / "reproduce_statebridge.ipynb"

  assert md_path.exists(), "paper/paper.md missing"
  assert pdf_path.exists(), "paper/paper.pdf missing"
  raw_pdf = pdf_path.read_bytes()
  assert raw_pdf.startswith(b"%PDF-"), "paper/paper.pdf is not a valid PDF"
  assert len(raw_pdf) > 25000, "paper/paper.pdf should contain real rendered content and figures"

  assert nb_path.exists(), "notebooks/reproduce_statebridge.ipynb missing"
  nb = json.loads(nb_path.read_text(encoding="utf-8"))
  code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
  assert len(code_cells) >= 3
  outputs_text = "\n".join(
    "".join(o.get("text", []))
    for c in code_cells
    for o in c.get("outputs", [])
  )
  assert "gemma-4-31b-it" in outputs_text
  assert "gemma-4-26b-a4b-it" in outputs_text
  assert "arm4_callgraph_slice" in outputs_text


def test_interactive_explorer_is_self_contained_and_supports_hash_routing():
  exp_path = ROOT / "explorer" / "index.html"
  assert exp_path.exists(), "explorer/index.html missing"
  html = exp_path.read_text(encoding="utf-8")
  assert "StateBridge" in html
  assert "TwinAPI" in html
  assert "First-Divergence" in html
  assert "window.location.hash" in html
  assert "F1_REDIRECT_FOLLOW_DEFAULT" in html
  assert "cdn." not in html.lower(), "Explorer must be zero-CDN offline self-contained"


def test_verify_ssot_consistency_passes_and_catches_mutations(tmp_path: Path):
  report = verify_mod.verify_all_surfaces(ROOT)
  assert report["ok"] is True, f"SSOT verification failed: {report.get('errors')}"
  assert report["word_count"] <= 3000

  # Positive mutation control (Failure: F5 & F6):
  # Copy surfaces into tmp_path and mutate a key solve rate in KAGGLE_WRITEUP.md;
  # verify_all_surfaces must fail and pinpoint writeup/KAGGLE_WRITEUP.md.
  for rel in (
    "artifacts/metrics_ssot.json",
    "artifacts/explorer_bundle.json",
    "writeup/KAGGLE_WRITEUP.md",
    "paper/paper.md",
    "paper/paper.pdf",
    "notebooks/reproduce_statebridge.ipynb",
    "explorer/index.html",
    "README.md",
    "README.zh-TW.md",
  ):
    src = ROOT / rel
    dst = tmp_path / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

  mut_writeup = tmp_path / "writeup" / "KAGGLE_WRITEUP.md"
  orig_text = mut_writeup.read_text(encoding="utf-8")
  # Mutate 93.3% (28/30) to 99.9%
  assert "93.3%" in orig_text
  mut_writeup.write_text(orig_text.replace("93.3%", "99.9%", 1), encoding="utf-8")

  bad_report = verify_mod.verify_all_surfaces(tmp_path)
  assert bad_report["ok"] is False
  assert any("KAGGLE_WRITEUP.md" in err for err in bad_report["errors"])

  # Also test word-count cap violation (> 3000 words)
  mut_writeup.write_text(orig_text + "\n" + ("extraword " * 2500), encoding="utf-8")
  wc_report = verify_mod.verify_all_surfaces(tmp_path)
  assert wc_report["ok"] is False
  assert any("word_count" in err for err in wc_report["errors"])
