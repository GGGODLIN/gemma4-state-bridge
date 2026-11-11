"""Notebook 自足啟動；本檔原碼會嵌入 Notebook，不依賴本機匯入路徑。"""

import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

PUBLIC_REPO = "https://github.com/GGGODLIN/gemma4-state-bridge.git"


def is_statebridge_repo(path):
  return all((path / item).is_file() for item in (
    "artifacts/raw_llm_generations.jsonl",
    "artifacts/metrics_ssot.json",
    "artifacts/twinapi_audit.json",
    "scripts/build_ssot.py",
    "skills/state-bridge/scripts/statebridge_core.py",
  ))


def locate_statebridge_repo():
  cwd = Path.cwd()
  for parent in (cwd, *cwd.parents):
    for candidate in (parent, parent / "gemma4-state-bridge"):
      if is_statebridge_repo(candidate):
        print("Source: local repository")
        return candidate
  input_root = Path(os.environ.get("STATEBRIDGE_INPUT_ROOT", "/kaggle/input"))
  if input_root.exists():
    for raw in sorted(input_root.rglob("raw_llm_generations.jsonl")):
      candidate = raw.parent.parent
      if is_statebridge_repo(candidate):
        print("Source: mounted Kaggle Dataset")
        return candidate
  if os.environ.get("STATEBRIDGE_OFFLINE") != "1":
    destination = Path(tempfile.mkdtemp(prefix="statebridge-source-", dir=cwd))
    repo_url = os.environ.get("STATEBRIDGE_REPO_URL", PUBLIC_REPO)
    try:
      subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(destination)],
        check=True, capture_output=True, text=True, timeout=90,
      )
      if is_statebridge_repo(destination):
        print("Source: shallow Git clone")
        return destination
    except (OSError, subprocess.SubprocessError) as exc:
      print(f"Clone unavailable ({type(exc).__name__}).")
  print("SUMMARY ONLY: precomputed results; statistics and figures have NOT been recomputed.")
  print("Enable Kaggle Internet, or mount a Dataset containing the repository under /kaggle/input/<dataset>/.")
  return None


def require_replay_dependencies():
  required = {"requests": "2.34.2", "httpx": "0.28.1"}
  missing = []
  for package, version in required.items():
    try:
      installed = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
      installed = None
    if installed != version:
      missing.append(f"{package}=={version}")
  try:
    import matplotlib
  except ImportError:
    missing.append("matplotlib")
  if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *missing], check=True)
  print("Replay dependencies: requests==2.34.2, httpx==0.28.1, matplotlib")
