"""Notebook 啟動路徑與離線標示的回歸測試。"""

import runpy
from pathlib import Path


BOOTSTRAP = Path(__file__).resolve().parents[1] / "scripts/notebook_bootstrap.py"
REQUIRED = (
  "artifacts/raw_llm_generations.jsonl",
  "artifacts/metrics_ssot.json",
  "artifacts/twinapi_audit.json",
  "scripts/build_ssot.py",
  "skills/state-bridge/scripts/statebridge_core.py",
)


def make_source(root):
  for name in REQUIRED:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
  return root


def test_nested_dataset_source_is_found_from_empty_working_directory(tmp_path, monkeypatch):
  work = tmp_path / "working"
  work.mkdir()
  inputs = tmp_path / "input"
  source = make_source(inputs / "dataset" / "repo")
  monkeypatch.chdir(work)
  monkeypatch.setenv("STATEBRIDGE_INPUT_ROOT", str(inputs))
  monkeypatch.setenv("STATEBRIDGE_OFFLINE", "1")
  bootstrap = runpy.run_path(str(BOOTSTRAP))
  assert bootstrap["locate_statebridge_repo"]() == source
  assert list(work.iterdir()) == []


def test_offline_missing_data_reports_summary_only(tmp_path, monkeypatch, capsys):
  monkeypatch.chdir(tmp_path)
  monkeypatch.setenv("STATEBRIDGE_INPUT_ROOT", str(tmp_path / "missing"))
  monkeypatch.setenv("STATEBRIDGE_OFFLINE", "1")
  bootstrap = runpy.run_path(str(BOOTSTRAP))
  assert bootstrap["locate_statebridge_repo"]() is None
  output = capsys.readouterr().out
  assert "SUMMARY ONLY" in output
  assert "NOT been recomputed" in output
  assert "/kaggle/input/<dataset>/" in output


def test_failed_clone_reports_summary_only(tmp_path, monkeypatch, capsys):
  monkeypatch.chdir(tmp_path)
  monkeypatch.setenv("STATEBRIDGE_INPUT_ROOT", str(tmp_path / "missing"))
  monkeypatch.setenv("STATEBRIDGE_OFFLINE", "0")
  monkeypatch.setenv("STATEBRIDGE_REPO_URL", str(tmp_path / "missing.git"))
  bootstrap = runpy.run_path(str(BOOTSTRAP))
  assert bootstrap["locate_statebridge_repo"]() is None
  output = capsys.readouterr().out
  assert "Clone unavailable" in output
  assert "SUMMARY ONLY" in output
