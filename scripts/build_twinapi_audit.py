#!/usr/bin/env python3
"""TwinAPI Stateful False-Friend Audit & 256-d Embedding Catalog Builder."""

from __future__ import annotations
import inspect
import json
import math
import urllib.request
from pathlib import Path
from typing import Any

import httpx
import requests

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"

TWIN_PAIRS_SPEC = [
  {
    "pair_id": "P01",
    "req_symbol": "requests.Session.post",
    "httpx_symbol": "httpx.Client.post",
    "req_obj": requests.Session.post,
    "httpx_obj": httpx.Client.post,
    "is_stateful_false_friend": True,
    "divergence_family": "F1_REDIRECT_FOLLOW_DEFAULT",
    "divergence_mechanism": "requests defaults to allow_redirects=True; httpx renames kwarg to follow_redirects and defaults to False.",
  },
  {
    "pair_id": "P02",
    "req_symbol": "requests.Session.get",
    "httpx_symbol": "httpx.Client.get",
    "req_obj": requests.Session.get,
    "httpx_obj": httpx.Client.get,
    "is_stateful_false_friend": True,
    "divergence_family": "F1_REDIRECT_FOLLOW_DEFAULT",
    "divergence_mechanism": "Identical method name (.get), but 301/302 redirect following is disabled by default in httpx.",
  },
  {
    "pair_id": "P03",
    "req_symbol": "requests.Session.request",
    "httpx_symbol": "httpx.Client.request",
    "req_obj": requests.Session.request,
    "httpx_obj": httpx.Client.request,
    "is_stateful_false_friend": True,
    "divergence_family": "F2_CROSS_ORIGIN_AUTH_STRIP",
    "divergence_mechanism": "requests Session.request accepts hooks= and allow_redirects=; httpx moves event_hooks to Client.__init__.",
  },
  {
    "pair_id": "P04",
    "req_symbol": "requests.Session",
    "httpx_symbol": "httpx.Client",
    "req_obj": requests.Session,
    "httpx_obj": httpx.Client,
    "is_stateful_false_friend": True,
    "divergence_family": "F2_CROSS_ORIGIN_AUTH_STRIP",
    "divergence_mechanism": "requests.Session mutates headers/hooks in-place via s.hooks['response']; httpx.Client uses event_hooks and immutable URL/Header normalization.",
  },
  {
    "pair_id": "P05",
    "req_symbol": "requests.cookies.RequestsCookieJar.get_dict",
    "httpx_symbol": "httpx.Cookies",
    "req_obj": requests.cookies.RequestsCookieJar.get_dict,
    "httpx_obj": httpx.Cookies,
    "is_stateful_false_friend": True,
    "divergence_family": "F3_COOKIE_JAR_SCOPING",
    "divergence_mechanism": "requests provides session.cookies.get_dict(); httpx.Cookies has no .get_dict() method and requires dict(client.cookies) or jar iteration.",
  },
  {
    "pair_id": "P06",
    "req_symbol": "requests.Session.put",
    "httpx_symbol": "httpx.Client.put",
    "req_obj": requests.Session.put,
    "httpx_obj": httpx.Client.put,
    "is_stateful_false_friend": True,
    "divergence_family": "F3_COOKIE_JAR_SCOPING",
    "divergence_mechanism": "Cookie persistence and response hook mutation timing differ before vs after response return.",
  },
  {
    "pair_id": "P07",
    "req_symbol": "requests.Response.iter_content",
    "httpx_symbol": "httpx.Response.iter_bytes",
    "req_obj": requests.Response.iter_content,
    "httpx_obj": httpx.Response.iter_bytes,
    "is_stateful_false_friend": True,
    "divergence_family": "F4_STREAM_CONTEXT_LIFECYCLE",
    "divergence_mechanism": "requests uses session.get(url, stream=True) + resp.iter_content(); httpx requires `with client.stream('GET', url) as resp: resp.iter_bytes()`.",
  },
  {
    "pair_id": "P08",
    "req_symbol": "requests.Response.close",
    "httpx_symbol": "httpx.Response.close",
    "req_obj": requests.Response.close,
    "httpx_obj": httpx.Response.close,
    "is_stateful_false_friend": True,
    "divergence_family": "F4_STREAM_CONTEXT_LIFECYCLE",
    "divergence_mechanism": "Identical method name (.close), but httpx streaming responses must be consumed/closed inside client.stream() context manager.",
  },
  {
    "pair_id": "P09",
    "req_symbol": "requests.Response.raise_for_status",
    "httpx_symbol": "httpx.Response.raise_for_status",
    "req_obj": requests.Response.raise_for_status,
    "httpx_obj": httpx.Response.raise_for_status,
    "is_stateful_false_friend": True,
    "divergence_family": "F5_CLIENT_INIT_AND_STATUS_ERROR",
    "divergence_mechanism": "Identical method name, but raises requests.exceptions.HTTPError vs httpx.HTTPStatusError with distinct hierarchy and message format.",
  },
  {
    "pair_id": "P10",
    "req_symbol": "requests.exceptions.HTTPError",
    "httpx_symbol": "httpx.HTTPStatusError",
    "req_obj": requests.exceptions.HTTPError,
    "httpx_obj": httpx.HTTPStatusError,
    "is_stateful_false_friend": True,
    "divergence_family": "F5_CLIENT_INIT_AND_STATUS_ERROR",
    "divergence_mechanism": "In httpx, HTTPError is the base class for all errors while HTTPStatusError is the status-code subclass (inverting requests' naming!).",
  },
  {
    "pair_id": "P11",
    "req_symbol": "requests.Session.delete",
    "httpx_symbol": "httpx.Client.delete",
    "req_obj": requests.Session.delete,
    "httpx_obj": httpx.Client.delete,
    "is_stateful_false_friend": True,
    "divergence_family": "F1_REDIRECT_FOLLOW_DEFAULT",
    "divergence_mechanism": "Identical short name (.delete), diverges on redirect following default and body parameter support.",
  },
  {
    "pair_id": "P12",
    "req_symbol": "requests.Session.patch",
    "httpx_symbol": "httpx.Client.patch",
    "req_obj": requests.Session.patch,
    "httpx_obj": httpx.Client.patch,
    "is_stateful_false_friend": True,
    "divergence_family": "F1_REDIRECT_FOLLOW_DEFAULT",
    "divergence_mechanism": "Identical short name (.patch), diverges on follow_redirects default and JSON whitespace serialization.",
  },
  {
    "pair_id": "P13",
    "req_symbol": "requests.Session.head",
    "httpx_symbol": "httpx.Client.head",
    "req_obj": requests.Session.head,
    "httpx_obj": httpx.Client.head,
    "is_stateful_false_friend": False,
    "divergence_family": "NONE",
    "divergence_mechanism": "Both default to not following redirects on HEAD unless explicitly enabled.",
  },
  {
    "pair_id": "P14",
    "req_symbol": "requests.Session.options",
    "httpx_symbol": "httpx.Client.options",
    "req_obj": requests.Session.options,
    "httpx_obj": httpx.Client.options,
    "is_stateful_false_friend": False,
    "divergence_family": "NONE",
    "divergence_mechanism": "Direct single-step OPTIONS call behaves equivalently when no redirect hop occurs.",
  },
  {
    "pair_id": "P15",
    "req_symbol": "requests.Response.json",
    "httpx_symbol": "httpx.Response.json",
    "req_obj": requests.Response.json,
    "httpx_obj": httpx.Response.json,
    "is_stateful_false_friend": False,
    "divergence_family": "NONE",
    "divergence_mechanism": "Both decode UTF-8 JSON response bodies via standard json.loads semantics.",
  },
]


def _levenshtein_similarity(a: str, b: str) -> float:
  if a == b:
    return 1.0
  la, lb = len(a), len(b)
  dp = list(range(lb + 1))
  for i in range(1, la + 1):
    prev = dp[0]
    dp[0] = i
    for j in range(1, lb + 1):
      temp = dp[j]
      cost = 0 if a[i - 1] == b[j - 1] else 1
      dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
      prev = temp
  dist = dp[lb]
  return round(1.0 - (dist / max(la, lb, 1)), 4)


def _signature_shape(obj: Any) -> dict[str, Any]:
  try:
    sig = inspect.signature(obj)
    kinds = [str(p.kind.name) for p in sig.parameters.values()]
    return {"arity": len(kinds), "kinds": kinds, "masked_sig": f"({', '.join(f'arg{i}:{k}' for i, k in enumerate(kinds))})"}
  except Exception:
    return {"arity": 0, "kinds": [], "masked_sig": "()"}


def _cosine(u: list[float], v: list[float]) -> float:
  dot = math.fsum(a * b for a, b in zip(u, v))
  nu = math.sqrt(math.fsum(a * a for a in u))
  nv = math.sqrt(math.fsum(b * b for b in v))
  if nu == 0.0 or nv == 0.0:
    return 0.0
  return round(dot / (nu * nv), 4)


def _load_magpie_key() -> str:
  env_path = Path.home() / ".cli-proxy-api" / "keys.env"
  if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
      if line.startswith("CLIPROXY_KEY_ADMIN="):
        return line.split("=", 1)[1].strip().strip("'\"")
  return "probe"


def _fetch_256d_embeddings(texts: list[str]) -> list[list[float]]:
  key = _load_magpie_key()
  payload = json.dumps({
    "model": "google/models/gemini-embedding-001",
    "input": texts,
    "dimensions": 256,
  }).encode("utf-8")
  req = urllib.request.Request(
    "http://127.0.0.1:8317/v1/embeddings",
    data=payload,
    headers={
      "Authorization": f"Bearer {key}",
      "Content-Type": "application/json",
    },
    method="POST",
  )
  with urllib.request.urlopen(req, timeout=30) as resp:
    data = json.loads(resp.read().decode("utf-8"))
  raw_rows = data["data"]
  rows = [
    r for _, r in sorted(enumerate(raw_rows), key=lambda iv: iv[1].get("index", iv[0]))
  ]
  return [[round(float(x), 6) for x in r["embedding"]] for r in rows]


def build_twinapi_catalog(out_dir: Path | None = None) -> dict[str, Any]:
  target_dir = out_dir or ARTIFACTS_DIR
  target_dir.mkdir(parents=True, exist_ok=True)

  # 1. Measure public symbol overlap between requests 2.34.2 and httpx 0.28.1
  req_top_funcs = {
    name for name in getattr(requests, "__all__", [])
    if callable(getattr(requests, name, None)) and not inspect.isclass(getattr(requests, name, None))
  }
  httpx_top_all = set(getattr(httpx, "__all__", dir(httpx)))
  top_overlap = sorted(req_top_funcs & httpx_top_all)
  top_overlap_ratio = round(len(top_overlap) / max(len(req_top_funcs), 1), 4)

  req_session_methods = {
    m for m in dir(requests.Session)
    if not m.startswith("_") and callable(getattr(requests.Session, m, None))
  }
  httpx_client_methods = {
    m for m in dir(httpx.Client)
    if not m.startswith("_") and callable(getattr(httpx.Client, m, None))
  }
  core_http_verbs = {"get", "post", "put", "delete", "head", "options", "patch", "request", "send", "close", "prepare_request", "stream"}
  session_client_overlap = sorted((req_session_methods & httpx_client_overlap if False else req_session_methods & httpx_client_methods) & core_http_verbs)
  session_client_overlap_ratio = round(len(session_client_overlap) / len(core_http_verbs), 4)

  # 2. Build descriptor strings and fetch real 256-d embeddings
  symbols_ordered: list[str] = []
  descriptors: list[str] = []
  sig_shapes: dict[str, dict[str, Any]] = {}

  for item in TWIN_PAIRS_SPEC:
    for sym_key, obj_key in (("req_symbol", "req_obj"), ("httpx_symbol", "httpx_obj")):
      sym = item[sym_key]
      obj = item[obj_key]
      shape = _signature_shape(obj)
      sig_shapes[sym] = shape
      doc_first = (inspect.getdoc(obj) or "").splitlines()[0] if inspect.getdoc(obj) else ""
      short_name = sym.split(".")[-1]
      desc = f"{short_name} {shape['masked_sig']} : {doc_first[:120]}"
      symbols_ordered.append(sym)
      descriptors.append(desc)

  vectors_list = _fetch_256d_embeddings(descriptors)
  vectors_map = {sym: vec for sym, vec in zip(symbols_ordered, vectors_list)}

  # 3. Score all 15 pairs
  pairs_report: list[dict[str, Any]] = []
  false_friend_cosines: list[float] = []
  false_friend_name_sims: list[float] = []

  for item in TWIN_PAIRS_SPEC:
    rs = item["req_symbol"]
    hs = item["httpx_symbol"]
    r_short = rs.split(".")[-1].lower()
    h_short = hs.split(".")[-1].lower()
    name_sim = _levenshtein_similarity(r_short, h_short)
    r_shape = sig_shapes[rs]
    h_shape = sig_shapes[hs]
    sig_iso = 1.0 if r_shape["kinds"] == h_shape["kinds"] else round(
      1.0 / (1.0 + abs(r_shape["arity"] - h_shape["arity"])), 4
    )
    emb_cos = _cosine(vectors_map[rs], vectors_map[hs])
    if item["is_stateful_false_friend"]:
      false_friend_cosines.append(emb_cos)
      false_friend_name_sims.append(name_sim)

    pairs_report.append({
      "pair_id": item["pair_id"],
      "req_symbol": rs,
      "httpx_symbol": hs,
      "short_name_similarity": name_sim,
      "signature_isomorphism": sig_iso,
      "embedding_cosine_256d": emb_cos,
      "is_stateful_false_friend": item["is_stateful_false_friend"],
      "divergence_family": item["divergence_family"],
      "divergence_mechanism": item["divergence_mechanism"],
    })

  emb_payload = {
    "model": "google/models/gemini-embedding-001",
    "dimensions": 256,
    "requests_version": requests.__version__,
    "httpx_version": httpx.__version__,
    "vectors": vectors_map,
  }
  (target_dir / "twinapi_embeddings_256d.json").write_text(
    json.dumps(emb_payload, indent=2), encoding="utf-8"
  )

  audit_payload = {
    "requests_version": requests.__version__,
    "httpx_version": httpx.__version__,
    "summary": {
      "requests_top_level_functions": sorted(req_top_funcs),
      "top_level_exact_name_overlap": top_overlap,
      "top_level_exact_name_overlap_ratio": top_overlap_ratio,
      "session_client_core_overlap": session_client_overlap,
      "session_client_core_overlap_ratio": session_client_overlap_ratio,
      "stateful_false_friend_count": sum(1 for p in pairs_report if p["is_stateful_false_friend"]),
      "total_pairs_audited": len(pairs_report),
      "mean_false_friend_embedding_cosine_256d": round(
        sum(false_friend_cosines) / len(false_friend_cosines), 4
      ),
      "mean_false_friend_short_name_similarity": round(
        sum(false_friend_name_sims) / len(false_friend_name_sims), 4
      ),
    },
    "pairs": pairs_report,
  }
  (target_dir / "twinapi_audit.json").write_text(
    json.dumps(audit_payload, indent=2), encoding="utf-8"
  )
  return audit_payload


if __name__ == "__main__":
  res = build_twinapi_catalog()
  print(json.dumps(res["summary"], indent=2))
