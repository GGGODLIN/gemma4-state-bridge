#!/usr/bin/env python3
"""StateBridge canonical standard-library kernel (Seam 1).

Provides:
- canonicalize_exchange(): normalizes requests vs httpx wire-level noise
- evaluate_trace_equivalence(): 7-category receipt & First-Divergence comparator
- check_ast_anticheat(): AST gate rejecting residual requests/urllib imports
- get_benchmark_workflows(): 30 clean-room multi-file HTTP workflows (5 families x 6)
- build_arm_context(): 5-arm context builder (Arm 1..5) with P4 budget parity
- run_workflow_sandbox(): isolated 127.0.0.1:0 ThreadingHTTPServer + sys.addaudithook runner
- CLI subcommands: selftest, scan, verify, reproduce-paper
"""

from __future__ import annotations
import argparse
import ast
import http.server
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import urllib.parse
from pathlib import Path
from typing import Any

DEFAULT_STRIPPED_HEADERS = {
  "user-agent",
  "host",
  "connection",
  "accept-encoding",
  "content-length",
  "accept",
  "content-type",
}

STATIC_API_CHEATSHEET = """\
# requests (2.34.2) -> httpx (0.28.1) Core API Cheat-Sheet
- Session: `requests.Session()` -> `httpx.Client()`
- Redirects: `allow_redirects=True` (default True) -> `follow_redirects=False` (default False in httpx!)
- Streaming: `s.get(url, stream=True)` -> `with client.stream("GET", url) as resp: resp.read()`
- Status Exceptions: `requests.HTTPError` -> `httpx.HTTPStatusError` (access `exc.response.status_code`)
- Client Params: `proxies=` removed in httpx 0.28; use `proxy=` or client-level configuration.
"""


def canonicalize_exchange(exchange: dict[str, Any]) -> dict[str, Any]:
  """Normalize an HTTP step exchange to remove transport-level noise (P1)."""
  method = str(exchange.get("method", "GET")).upper()
  raw_path = str(exchange.get("path", "/"))
  split = urllib.parse.urlsplit(raw_path)
  clean_path = split.path or "/"
  query_pairs = [
    [k, v] for k, v in urllib.parse.parse_qsl(split.query, keep_blank_values=True)
  ]
  if "query" in exchange and isinstance(exchange["query"], list) and not query_pairs:
    query_pairs = [[str(k), str(v)] for k, v in exchange["query"]]

  raw_headers = exchange.get("headers") or {}
  norm_headers: dict[str, str] = {}
  cookies: dict[str, str] = {
    str(k).strip(): str(v).strip() for k, v in (exchange.get("cookies") or {}).items()
  }
  content_type = ""

  for k, v in raw_headers.items():
    kl = str(k).lower().strip()
    vs = str(v).strip()
    if kl == "content-type":
      content_type = vs.lower()
      continue
    if kl == "cookie":
      for part in vs.split(";"):
        part_s = part.strip()
        if "=" in part_s:
          ck, cv = part_s.split("=", 1)
          cookies[ck.strip()] = cv.strip()
      continue
    if kl in DEFAULT_STRIPPED_HEADERS:
      continue
    norm_headers[kl] = vs

  if (
    "body_json" in exchange
    and "cookies" in exchange
    and "query" in exchange
    and isinstance(exchange.get("headers"), dict)
  ):
    return {
      "method": method,
      "path": clean_path,
      "query": query_pairs,
      "headers": norm_headers,
      "cookies": cookies,
      "body_json": exchange["body_json"],
      "body_text": "" if exchange["body_json"] is not None else str(exchange.get("body_text", "")),
    }

  raw_body = exchange.get("body", "")
  if isinstance(raw_body, bytes):
    raw_body_str = raw_body.decode("utf-8", errors="replace")
  else:
    raw_body_str = str(raw_body or "")

  body_json: Any = None
  body_text = ""
  stripped_body = raw_body_str.strip()
  if stripped_body and (
    "application/json" in content_type
    or stripped_body.startswith("{")
    or stripped_body.startswith("[")
  ):
    try:
      body_json = json.loads(stripped_body)
    except Exception:
      body_text = stripped_body
  else:
    body_text = stripped_body

  return {
    "method": method,
    "path": clean_path,
    "query": query_pairs,
    "headers": norm_headers,
    "cookies": cookies,
    "body_json": body_json,
    "body_text": body_text,
  }


def evaluate_trace_equivalence(
  gold_trace: list[dict[str, Any]],
  candidate_trace: list[dict[str, Any]],
  gold_return: dict[str, Any] | None = None,
  candidate_return: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Compare canonicalized multi-step HTTP traces and pinpoint First-Divergence (P2)."""
  c_gold = [canonicalize_exchange(s) for s in gold_trace]
  c_cand = [canonicalize_exchange(s) for s in candidate_trace]

  max_len = max(len(c_gold), len(c_cand))
  for idx in range(max_len):
    step_num = idx + 1
    if idx >= len(c_gold):
      return {
        "outcome_category": "STEP_K_WIRE_DIVERGENCE",
        "first_divergence_step": step_num,
        "divergence_field": "step_count (extra candidate step)",
        "expected_canonical_step": None,
        "actual_canonical_step": c_cand[idx],
      }
    if idx >= len(c_cand):
      return {
        "outcome_category": "STEP_K_WIRE_DIVERGENCE",
        "first_divergence_step": step_num,
        "divergence_field": "step_count (missing candidate step)",
        "expected_canonical_step": c_gold[idx],
        "actual_canonical_step": None,
      }

    g_step = c_gold[idx]
    a_step = c_cand[idx]
    for field in ("method", "path", "query", "cookies", "headers", "body_json", "body_text"):
      if g_step.get(field) != a_step.get(field):
        return {
          "outcome_category": "STEP_K_WIRE_DIVERGENCE",
          "first_divergence_step": step_num,
          "divergence_field": field,
          "expected_canonical_step": g_step,
          "actual_canonical_step": a_step,
        }

  if gold_return is not None and candidate_return is not None:
    if gold_return != candidate_return:
      return {
        "outcome_category": "POST_HTTP_RETURN_MISMATCH",
        "first_divergence_step": None,
        "divergence_field": "return_value",
        "expected_canonical_step": gold_return,
        "actual_canonical_step": candidate_return,
      }

  return {
    "outcome_category": "PASS_EQUIVALENT",
    "first_divergence_step": None,
    "divergence_field": None,
    "expected_canonical_step": None,
    "actual_canonical_step": None,
  }


def check_ast_anticheat(files_map: dict[str, str]) -> dict[str, Any]:
  """Verify candidate Python files parse cleanly and contain zero requests/urllib imports (P3)."""
  forbidden_roots = {"requests", "urllib", "urllib3"}
  for filename, source in files_map.items():
    if not filename.endswith(".py"):
      continue
    try:
      tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
      return {
        "passed": False,
        "outcome_category": "PARSE_OR_SYNTAX_ERROR",
        "detail": f"{filename}:{exc.lineno}: {exc.msg}",
      }

    for node in ast.walk(tree):
      if isinstance(node, ast.Import):
        for alias in node.names:
          root_pkg = alias.name.split(".")[0]
          if root_pkg in forbidden_roots:
            return {
              "passed": False,
              "outcome_category": "CHEAT_RESIDUAL_REQUESTS",
              "detail": f"{filename}: forbidden import '{alias.name}'",
            }
      elif isinstance(node, ast.ImportFrom):
        mod = (node.module or "").split(".")[0]
        if mod in forbidden_roots:
          return {
            "passed": False,
            "outcome_category": "CHEAT_RESIDUAL_REQUESTS",
            "detail": f"{filename}: forbidden from-import '{node.module}'",
          }
      elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "__import__":
          if node.args and isinstance(node.args[0], ast.Constant):
            arg_val = str(node.args[0].value).split(".")[0]
            if arg_val in forbidden_roots:
              return {
                "passed": False,
                "outcome_category": "CHEAT_RESIDUAL_REQUESTS",
                "detail": f"{filename}: forbidden __import__('{arg_val}')",
              }

  return {"passed": True, "outcome_category": None, "detail": None}


def _make_distractors_code(variant_idx: int, family_tag: str) -> str:
  """Generate HTTP-themed embedding decoy (for Arm 3) and unrelated equal-token helper (for Arm 5)."""
  return f'''"""Auxiliary utilities and HTTP formatting helpers for {family_tag}_{variant_idx:02d}."""

import httpx


def preview_http_headers(url: str, token: str) -> dict:
  """HTTP client helper that formats request headers and URL parameters for httpx."""
  base_headers = {{"X-Preview-Mode": "static-{variant_idx}", "Accept": "application/json"}}
  if token:
    base_headers["X-Preview-Token"] = f"preview-{{token}}"
  return {{"url": url, "headers": base_headers, "follow_redirects": False}}


def compute_window_digest(values: list[int], modulus: int = 997) -> dict:
  """Unrelated accounting helper: compute rolling window checksum, summary stats, and ledger row."""
  total = 0
  rolling = []
  for idx, item in enumerate(values):
    total = (total + (item * (idx + 1))) % modulus
    rolling.append(total)
  summary = {{"checksum": total, "count": len(values), "series": rolling[-3:]}}
  keys = sorted(summary.keys())
  joined = "|".join(f"{{k}}={{summary[k]}}" for k in keys)
  summary["audit_line"] = f"[AUDIT:v{variant_idx:02d}] {{joined}}"
  return summary
'''


def _build_single_workflow(family_idx: int, variant_idx: int) -> dict[str, Any]:
  """Construct one deterministic multi-file workflow specification."""
  v = variant_idx
  if family_idx == 1:
    wf_id = f"F1_{v:02d}"
    family = "F1_REDIRECT_FOLLOW_DEFAULT"
    hop_hdr = f"x-hop-gate-{v}"
    hop_val = f"hop-allow-{v * 11}"
    title = f"Multi-Hop Redirect Follow & Gate Header ({wf_id})"
    server_spec = {
      "family": family,
      "variant": v,
      "hop_header": hop_hdr,
      "hop_value": hop_val,
    }
    session_setup_req = f'''import requests

HOP_GATE_HEADER = "{hop_hdr}"
HOP_GATE_VALUE = "{hop_val}"

def prepare_client_session() -> requests.Session:
  """Initialize session with upstream redirect gate header."""
  s = requests.Session()
  s.headers[HOP_GATE_HEADER] = HOP_GATE_VALUE
  return s

def finalize_step_contract(resp: requests.Response) -> dict:
  """Verify final redirect destination payload and hop gate receipt."""
  data = resp.json()
  return {{
    "status": resp.status_code,
    "reached_final": data.get("stage") == "final_destination",
    "gate_echo": data.get("gate_echo"),
  }}
'''
    session_setup_httpx = f'''import httpx

HOP_GATE_HEADER = "{hop_hdr}"
HOP_GATE_VALUE = "{hop_val}"

def prepare_client_session() -> httpx.Client:
  """Initialize httpx Client (note: follow_redirects defaults to False in httpx)."""
  return httpx.Client(headers={{HOP_GATE_HEADER: HOP_GATE_VALUE}}, follow_redirects=False)

def finalize_step_contract(resp: httpx.Response) -> dict:
  """Verify final redirect destination payload and hop gate receipt."""
  data = resp.json()
  return {{
    "status": resp.status_code,
    "reached_final": data.get("stage") == "final_destination",
    "gate_echo": data.get("gate_echo"),
  }}
'''
    if v == 1:
      workflow_steps_req = f'''import requests
from session_setup import HOP_GATE_HEADER, HOP_GATE_VALUE, finalize_step_contract, prepare_client_session

def execute_step(base_url: str) -> dict:
  """Execute two-step redirect workflow starting at /step1_redirect_{v}."""
  session = prepare_client_session()
  try:
    r1 = session.post(f"{{base_url}}/step1_redirect_{v}", json={{"seq": {v}}}, allow_redirects=True)
    r1.raise_for_status()
    return finalize_step_contract(r1)
  finally:
    session.close()
'''
      workflow_steps_httpx = f'''import httpx
from session_setup import HOP_GATE_HEADER, HOP_GATE_VALUE, finalize_step_contract, prepare_client_session

def execute_step(base_url: str) -> dict:
  """Execute two-step redirect workflow starting at /step1_redirect_{v}."""
  client = prepare_client_session()
  try:
    r1 = client.post(
      f"{{base_url}}/step1_redirect_{v}",
      json={{"seq": {v}}},
      follow_redirects=True,
    )
    r1.raise_for_status()
    return finalize_step_contract(r1)
  finally:
    client.close()
'''
    else:
      session_setup_httpx = f'''import httpx

HOP_GATE_HEADER = "{hop_hdr}"
HOP_GATE_VALUE = "{hop_val}"

def prepare_client_session() -> httpx.Client:
  """Initialize httpx Client with follow_redirects=False; callers must pass **hop_request_kwargs() to client.post()."""
  return httpx.Client(follow_redirects=False)

def hop_request_kwargs() -> dict:
  """Required kwargs for POST /step1_redirect_{v} in httpx migration."""
  return {{"follow_redirects": True, "headers": {{HOP_GATE_HEADER: HOP_GATE_VALUE}}}}

def finalize_step_contract(resp: httpx.Response) -> dict:
  """Verify final redirect destination payload and hop gate receipt."""
  data = resp.json()
  return {{
    "status": resp.status_code,
    "reached_final": data.get("stage") == "final_destination",
    "gate_echo": data.get("gate_echo"),
  }}
'''
      workflow_steps_req = f'''import requests
from session_setup import finalize_step_contract, prepare_client_session

def execute_step(base_url: str) -> dict:
  """Execute two-step redirect workflow starting at /step1_redirect_{v}."""
  session = prepare_client_session()
  try:
    r1 = session.post(f"{{base_url}}/step1_redirect_{v}", json={{"seq": {v}}})
    r1.raise_for_status()
    return finalize_step_contract(r1)
  finally:
    session.close()
'''
      workflow_steps_httpx = f'''import httpx
from session_setup import finalize_step_contract, hop_request_kwargs, prepare_client_session

def execute_step(base_url: str) -> dict:
  """Execute two-step redirect workflow starting at /step1_redirect_{v}."""
  client = prepare_client_session()
  try:
    r1 = client.post(
      f"{{base_url}}/step1_redirect_{v}",
      json={{"seq": {v}}},
      **hop_request_kwargs(),
    )
    r1.raise_for_status()
    return finalize_step_contract(r1)
  finally:
    client.close()
'''

  elif family_idx == 2:
    wf_id = f"F2_{v:02d}"
    family = "F2_CROSS_ORIGIN_AUTH_STRIP"
    scrub_hdr = f"x-internal-secret-{v}"
    partner_hdr = f"x-partner-hop-{v}"
    partner_val = f"partner-verified-{v * 7}"
    title = f"Cross-Boundary Redirect Secret Scrubbing ({wf_id})"
    server_spec = {
      "family": family,
      "variant": v,
      "scrub_header": scrub_hdr,
      "partner_header": partner_hdr,
      "partner_value": partner_val,
    }
    session_setup_req = f'''import requests

SCRUB_HEADER = "{scrub_hdr}"
PARTNER_HEADER = "{partner_hdr}"
PARTNER_VALUE = "{partner_val}"

def prepare_client_session() -> requests.Session:
  """Create session with internal secret header that must be scrubbed on redirect hop."""
  s = requests.Session()
  s.headers[SCRUB_HEADER] = "internal-token-{v}"
  return s

def build_partner_hop_headers(redirect_resp) -> dict:
  """Upstream policy: strip SCRUB_HEADER and attach PARTNER_HEADER for step 2."""
  hop_id = redirect_resp.headers.get("X-Hop-Ticket", "none")
  return {{PARTNER_HEADER: f"{{PARTNER_VALUE}}:{{hop_id}}"}}

def finalize_step_contract(resp) -> dict:
  data = resp.json()
  return {{"status": resp.status_code, "partner_ok": data.get("partner_ok"), "leaked": data.get("leaked")}}
'''
    session_setup_httpx = f'''import httpx

SCRUB_HEADER = "{scrub_hdr}"
PARTNER_HEADER = "{partner_hdr}"
PARTNER_VALUE = "{partner_val}"

def prepare_client_session() -> httpx.Client:
  """Create httpx Client with SCRUB_HEADER; before step 2, delete client.headers[SCRUB_HEADER] and pass headers=build_partner_hop_headers(r1)."""
  return httpx.Client(headers={{SCRUB_HEADER: "internal-token-{v}"}}, follow_redirects=False)

def build_partner_hop_headers(redirect_resp: httpx.Response) -> dict:
  """Upstream policy: strip SCRUB_HEADER and attach PARTNER_HEADER for step 2."""
  hop_id = redirect_resp.headers.get("X-Hop-Ticket", "none")
  return {{PARTNER_HEADER: f"{{PARTNER_VALUE}}:{{hop_id}}"}}

def finalize_step_contract(resp: httpx.Response) -> dict:
  data = resp.json()
  return {{"status": resp.status_code, "partner_ok": data.get("partner_ok"), "leaked": data.get("leaked")}}
'''
    if v == 1:
      workflow_steps_req = f'''import requests
from session_setup import SCRUB_HEADER, build_partner_hop_headers, finalize_step_contract, prepare_client_session

def execute_step(base_url: str) -> dict:
  """Run step 1 (/origin_hop_{v}), scrub internal secret header on 302, and call partner hop."""
  session = prepare_client_session()
  try:
    r1 = session.get(f"{{base_url}}/origin_hop_{v}", allow_redirects=False)
    next_path = r1.headers["Location"]
    session.headers.pop(SCRUB_HEADER, None)
    hop_headers = build_partner_hop_headers(r1)
    r2 = session.get(f"{{base_url}}{{next_path}}", headers=hop_headers, allow_redirects=False)
    return finalize_step_contract(r2)
  finally:
    session.close()
'''
    else:
      session_setup_req = f'''import requests

SCRUB_HEADER = "{scrub_hdr}"
PARTNER_HEADER = "{partner_hdr}"
PARTNER_VALUE = "{partner_val}"

def prepare_client_session() -> requests.Session:
  """Create session with response hook that scrubs SCRUB_HEADER and attaches PARTNER_HEADER."""
  s = requests.Session()
  s.headers[SCRUB_HEADER] = "internal-token-{v}"
  def _on_resp(resp, *args, **kwargs):
    if resp.status_code == 302:
      s.headers.pop(SCRUB_HEADER, None)
      hop_id = resp.headers.get("X-Hop-Ticket", "none")
      s.headers[PARTNER_HEADER] = f"{{PARTNER_VALUE}}:{{hop_id}}"
  s.hooks["response"].append(_on_resp)
  return s

def finalize_step_contract(resp) -> dict:
  data = resp.json()
  return {{"status": resp.status_code, "partner_ok": data.get("partner_ok"), "leaked": data.get("leaked")}}
'''
      workflow_steps_req = f'''import requests
from session_setup import finalize_step_contract, prepare_client_session

def execute_step(base_url: str) -> dict:
  """Run step 1 (/origin_hop_{v}) and follow Location to partner hop."""
  session = prepare_client_session()
  try:
    r1 = session.get(f"{{base_url}}/origin_hop_{v}", allow_redirects=False)
    next_path = r1.headers["Location"]
    r2 = session.get(f"{{base_url}}{{next_path}}", allow_redirects=False)
    return finalize_step_contract(r2)
  finally:
    session.close()
'''
    workflow_steps_httpx = f'''import httpx
from session_setup import SCRUB_HEADER, build_partner_hop_headers, finalize_step_contract, prepare_client_session

def execute_step(base_url: str) -> dict:
  """Run step 1 (/origin_hop_{v}), scrub internal secret header on 302, and call partner hop."""
  client = prepare_client_session()
  try:
    r1 = client.get(f"{{base_url}}/origin_hop_{v}", follow_redirects=False)
    next_path = r1.headers["Location"]
    if SCRUB_HEADER in client.headers:
      del client.headers[SCRUB_HEADER]
    hop_headers = build_partner_hop_headers(r1)
    r2 = client.get(f"{{base_url}}{{next_path}}", headers=hop_headers, follow_redirects=False)
    return finalize_step_contract(r2)
  finally:
    client.close()
'''

  elif family_idx == 3:
    wf_id = f"F3_{v:02d}"
    family = "F3_COOKIE_JAR_SCOPING"
    cookie_name = f"sb_csrf_{v}"
    csrf_hdr = f"x-csrf-sync-{v}"
    title = f"Cookie Jar Extraction & CSRF Promotion ({wf_id})"
    server_spec = {
      "family": family,
      "variant": v,
      "cookie_name": cookie_name,
      "csrf_header": csrf_hdr,
    }
    session_setup_req = f'''import requests

COOKIE_NAME = "{cookie_name}"
CSRF_HEADER = "{csrf_hdr}"

def prepare_client_session() -> requests.Session:
  return requests.Session()

def promote_cookie_to_header(cookie_map: dict) -> dict:
  """Upstream contract: extract COOKIE_NAME from cookie dict and format CSRF_HEADER."""
  raw = cookie_map.get(COOKIE_NAME, "missing")
  return {{CSRF_HEADER: f"synced:{{raw}}"}}

def finalize_step_contract(resp) -> dict:
  body = resp.json()
  return {{"status": resp.status_code, "csrf_verified": body.get("csrf_verified")}}
'''
    session_setup_httpx = f'''import httpx

COOKIE_NAME = "{cookie_name}"
CSRF_HEADER = "{csrf_hdr}"

def prepare_client_session() -> httpx.Client:
  """Initialize httpx Client; before step 2, call promote_cookie_to_header(dict(client.cookies)) and pass headers=extra_hdrs."""
  return httpx.Client()

def promote_cookie_to_header(cookie_map: dict) -> dict:
  """Upstream contract: extract COOKIE_NAME from cookie dict and format CSRF_HEADER."""
  raw = cookie_map.get(COOKIE_NAME, "missing")
  return {{CSRF_HEADER: f"synced:{{raw}}"}}

def finalize_step_contract(resp: httpx.Response) -> dict:
  body = resp.json()
  return {{"status": resp.status_code, "csrf_verified": body.get("csrf_verified")}}
'''
    if v == 1:
      workflow_steps_req = f'''import requests
from session_setup import COOKIE_NAME, finalize_step_contract, prepare_client_session, promote_cookie_to_header

def execute_step(base_url: str) -> dict:
  """Step 1 fetches session cookie; Step 2 promotes cookie via session.cookies.get_dict()."""
  session = prepare_client_session()
  try:
    r1 = session.get(f"{{base_url}}/csrf_init_{v}")
    r1.raise_for_status()
    cookie_dict = session.cookies.get_dict()
    extra_hdrs = promote_cookie_to_header(cookie_dict)
    r2 = session.post(f"{{base_url}}/csrf_commit_{v}", headers=extra_hdrs, json={{"commit": {v}}})
    return finalize_step_contract(r2)
  finally:
    session.close()
'''
    else:
      session_setup_req = f'''import requests

COOKIE_NAME = "{cookie_name}"
CSRF_HEADER = "{csrf_hdr}"

def prepare_client_session() -> requests.Session:
  s = requests.Session()
  def _sync_cookie(resp, *args, **kwargs):
    val = resp.cookies.get_dict().get(COOKIE_NAME)
    if val:
      s.headers[CSRF_HEADER] = f"synced:{{val}}"
  s.hooks["response"].append(_sync_cookie)
  return s

def finalize_step_contract(resp) -> dict:
  body = resp.json()
  return {{"status": resp.status_code, "csrf_verified": body.get("csrf_verified")}}
'''
      workflow_steps_req = f'''import requests
from session_setup import finalize_step_contract, prepare_client_session

def execute_step(base_url: str) -> dict:
  """Step 1 fetches session cookie; Step 2 commits action with synced CSRF header."""
  session = prepare_client_session()
  try:
    r1 = session.get(f"{{base_url}}/csrf_init_{v}")
    r1.raise_for_status()
    r2 = session.post(f"{{base_url}}/csrf_commit_{v}", json={{"commit": {v}}})
    return finalize_step_contract(r2)
  finally:
    session.close()
'''
    workflow_steps_httpx = f'''import httpx
from session_setup import COOKIE_NAME, finalize_step_contract, prepare_client_session, promote_cookie_to_header

def execute_step(base_url: str) -> dict:
  """Step 1 fetches session cookie; Step 2 promotes cookie via dict(client.cookies)."""
  client = prepare_client_session()
  try:
    r1 = client.get(f"{{base_url}}/csrf_init_{v}")
    r1.raise_for_status()
    cookie_dict = dict(client.cookies)
    extra_hdrs = promote_cookie_to_header(cookie_dict)
    r2 = client.post(f"{{base_url}}/csrf_commit_{v}", headers=extra_hdrs, json={{"commit": {v}}})
    return finalize_step_contract(r2)
  finally:
    client.close()
'''

  elif family_idx == 4:
    wf_id = f"F4_{v:02d}"
    family = "F4_STREAM_CONTEXT_LIFECYCLE"
    ack_hdr = f"x-stream-digest-{v}"
    title = f"Streaming Response Consumption & Digest Receipt ({wf_id})"
    server_spec = {
      "family": family,
      "variant": v,
      "ack_header": ack_hdr,
    }
    session_setup_req = f'''import requests

ACK_HEADER = "{ack_hdr}"

def prepare_client_session() -> requests.Session:
  return requests.Session()

def summarize_stream_bytes(raw_bytes: bytes) -> dict:
  """Upstream contract: compute deterministic byte-line digest header for Step 2 ack."""
  lines = [ln for ln in raw_bytes.decode("utf-8").splitlines() if ln]
  return {{ACK_HEADER: f"lines={{len(lines)}}:bytes={{len(raw_bytes)}}"}}

def finalize_step_contract(resp) -> dict:
  data = resp.json()
  return {{"status": resp.status_code, "stream_ack": data.get("stream_ack")}}
'''
    session_setup_httpx = f'''import httpx

ACK_HEADER = "{ack_hdr}"

def prepare_client_session() -> httpx.Client:
  """Initialize httpx Client; stream with client.stream('GET', url), read bytes, and pass headers=summarize_stream_bytes(raw) to step 2."""
  return httpx.Client()

def summarize_stream_bytes(raw_bytes: bytes) -> dict:
  """Upstream contract: compute deterministic byte-line digest header for Step 2 ack."""
  lines = [ln for ln in raw_bytes.decode("utf-8").splitlines() if ln]
  return {{ACK_HEADER: f"lines={{len(lines)}}:bytes={{len(raw_bytes)}}"}}

def finalize_step_contract(resp: httpx.Response) -> dict:
  data = resp.json()
  return {{"status": resp.status_code, "stream_ack": data.get("stream_ack")}}
'''
    if v == 1:
      workflow_steps_req = f'''import requests
from session_setup import finalize_step_contract, prepare_client_session, summarize_stream_bytes

def execute_step(base_url: str) -> dict:
  """Stream chunks from /stream_feed_{v}, close stream, and send digest to /stream_ack_{v}."""
  session = prepare_client_session()
  try:
    r1 = session.get(f"{{base_url}}/stream_feed_{v}", stream=True)
    chunks = list(r1.iter_content(chunk_size=16))
    r1.close()
    raw = b"".join(chunks)
    hdrs = summarize_stream_bytes(raw)
    r2 = session.post(f"{{base_url}}/stream_ack_{v}", headers=hdrs, json={{"done": True}})
    return finalize_step_contract(r2)
  finally:
    session.close()
'''
    else:
      session_setup_req = f'''import requests

ACK_HEADER = "{ack_hdr}"

def prepare_client_session() -> requests.Session:
  s = requests.Session()
  def _on_stream(resp, *args, **kwargs):
    if "/stream_feed_" in resp.url:
      raw = resp.content
      lines = [ln for ln in raw.decode("utf-8").splitlines() if ln]
      s.headers[ACK_HEADER] = f"lines={{len(lines)}}:bytes={{len(raw)}}"
  s.hooks["response"].append(_on_stream)
  return s

def finalize_step_contract(resp) -> dict:
  data = resp.json()
  return {{"status": resp.status_code, "stream_ack": data.get("stream_ack")}}
'''
      workflow_steps_req = f'''import requests
from session_setup import finalize_step_contract, prepare_client_session

def execute_step(base_url: str) -> dict:
  """Stream chunks from /stream_feed_{v} and send ack to /stream_ack_{v}."""
  session = prepare_client_session()
  try:
    r1 = session.get(f"{{base_url}}/stream_feed_{v}", stream=True)
    r1.close()
    r2 = session.post(f"{{base_url}}/stream_ack_{v}", json={{"done": True}})
    return finalize_step_contract(r2)
  finally:
    session.close()
'''
    workflow_steps_httpx = f'''import httpx
from session_setup import finalize_step_contract, prepare_client_session, summarize_stream_bytes

def execute_step(base_url: str) -> dict:
  """Stream chunks from /stream_feed_{v}, close stream context, and send digest to /stream_ack_{v}."""
  client = prepare_client_session()
  try:
    with client.stream("GET", f"{{base_url}}/stream_feed_{v}") as r1:
      chunks = list(r1.iter_bytes())
    raw = b"".join(chunks)
    hdrs = summarize_stream_bytes(raw)
    r2 = client.post(f"{{base_url}}/stream_ack_{v}", headers=hdrs, json={{"done": True}})
    return finalize_step_contract(r2)
  finally:
    client.close()
'''

  else:
    wf_id = f"F5_{v:02d}"
    family = "F5_CLIENT_INIT_AND_STATUS_ERROR"
    recovery_hdr = f"x-recovery-code-{v}"
    title = f"Status-Error Exception Hierarchy & Fallback Recovery ({wf_id})"
    server_spec = {
      "family": family,
      "variant": v,
      "recovery_header": recovery_hdr,
    }
    session_setup_req = f'''import requests

RECOVERY_HEADER = "{recovery_hdr}"

def prepare_client_session() -> requests.Session:
  return requests.Session()

def build_recovery_headers(exc_response) -> dict:
  """Upstream contract: extract status_code and X-Retry-Ticket from error response."""
  ticket = exc_response.headers.get("X-Retry-Ticket", "none")
  return {{RECOVERY_HEADER: f"recover:{{exc_response.status_code}}:{{ticket}}"}}

def finalize_step_contract(resp) -> dict:
  data = resp.json()
  return {{"status": resp.status_code, "recovered": data.get("recovered")}}
'''
    session_setup_httpx = f'''import httpx

RECOVERY_HEADER = "{recovery_hdr}"

def prepare_client_session() -> httpx.Client:
  """Initialize httpx Client; on httpx.HTTPStatusError as exc, pass headers=build_recovery_headers(exc.response) to fallback POST."""
  return httpx.Client()

def build_recovery_headers(exc_response: httpx.Response) -> dict:
  """Upstream contract: extract status_code and X-Retry-Ticket from error response."""
  ticket = exc_response.headers.get("X-Retry-Ticket", "none")
  return {{RECOVERY_HEADER: f"recover:{{exc_response.status_code}}:{{ticket}}"}}

def finalize_step_contract(resp: httpx.Response) -> dict:
  data = resp.json()
  return {{"status": resp.status_code, "recovered": data.get("recovered")}}
'''
    if v == 1:
      workflow_steps_req = f'''import requests
from requests.exceptions import HTTPError
from session_setup import build_recovery_headers, finalize_step_contract, prepare_client_session

def execute_step(base_url: str) -> dict:
  """Call primary endpoint (/primary_{v}); on HTTPError, recover via /fallback_{v}."""
  session = prepare_client_session()
  try:
    try:
      r1 = session.post(f"{{base_url}}/primary_{v}", data={{"attempt": "1"}})
      r1.raise_for_status()
      return finalize_step_contract(r1)
    except HTTPError as exc:
      rec_hdrs = build_recovery_headers(exc.response)
      r2 = session.post(f"{{base_url}}/fallback_{v}", headers=rec_hdrs, json={{"recovered": {v}}})
      r2.raise_for_status()
      return finalize_step_contract(r2)
  finally:
    session.close()
'''
    else:
      session_setup_req = f'''import requests

RECOVERY_HEADER = "{recovery_hdr}"

def prepare_client_session() -> requests.Session:
  s = requests.Session()
  def _on_err(resp, *args, **kwargs):
    if resp.status_code == 429:
      tkt = resp.headers.get("X-Retry-Ticket", "none")
      s.headers[RECOVERY_HEADER] = f"recover:{{resp.status_code}}:{{tkt}}"
  s.hooks["response"].append(_on_err)
  return s

def finalize_step_contract(resp) -> dict:
  data = resp.json()
  return {{"status": resp.status_code, "recovered": data.get("recovered")}}
'''
      workflow_steps_req = f'''import requests
from requests.exceptions import HTTPError
from session_setup import finalize_step_contract, prepare_client_session

def execute_step(base_url: str) -> dict:
  """Call primary endpoint (/primary_{v}); on HTTPError, recover via /fallback_{v}."""
  session = prepare_client_session()
  try:
    try:
      r1 = session.post(f"{{base_url}}/primary_{v}", data={{"attempt": "1"}})
      r1.raise_for_status()
      return finalize_step_contract(r1)
    except HTTPError:
      r2 = session.post(f"{{base_url}}/fallback_{v}", json={{"recovered": {v}}})
      r2.raise_for_status()
      return finalize_step_contract(r2)
  finally:
    session.close()
'''
    workflow_steps_httpx = f'''import httpx
from session_setup import build_recovery_headers, finalize_step_contract, prepare_client_session

def execute_step(base_url: str) -> dict:
  """Call primary endpoint (/primary_{v}); on HTTPStatusError, recover via /fallback_{v}."""
  client = prepare_client_session()
  try:
    try:
      r1 = client.post(f"{{base_url}}/primary_{v}", data={{"attempt": "1"}})
      r1.raise_for_status()
      return finalize_step_contract(r1)
    except httpx.HTTPStatusError as exc:
      rec_hdrs = build_recovery_headers(exc.response)
      r2 = client.post(f"{{base_url}}/fallback_{v}", headers=rec_hdrs, json={{"recovered": {v}}})
      r2.raise_for_status()
      return finalize_step_contract(r2)
  finally:
    client.close()
'''

  distractors_code = _make_distractors_code(v, wf_id)
  return {
    "id": wf_id,
    "family": family,
    "variant": v,
    "title": title,
    "target_file": "workflow_steps.py",
    "target_function": "execute_step",
    "server_spec": server_spec,
    "requests_files": {
      "session_setup.py": session_setup_req,
      "workflow_steps.py": workflow_steps_req,
      "distractors.py": distractors_code,
    },
    "httpx_gold_files": {
      "session_setup.py": session_setup_httpx,
      "workflow_steps.py": workflow_steps_httpx,
      "distractors.py": distractors_code,
    },
    "candidate_base_files": {
      "session_setup.py": session_setup_httpx,
      "workflow_steps.py": workflow_steps_req,
      "distractors.py": distractors_code,
    },
  }


def get_benchmark_workflows() -> list[dict[str, Any]]:
  """Return all 30 clean-room multi-file workflows (5 families x 6 variants)."""
  workflows: list[dict[str, Any]] = []
  for fam_idx in range(1, 6):
    for var_idx in range(1, 7):
      workflows.append(_build_single_workflow(fam_idx, var_idx))
  return workflows


def _extract_functions_from_file(filename: str, source: str) -> dict[str, str]:
  """Extract top-level function definitions as {f'{filename}:{func_name}': source_segment}."""
  tree = ast.parse(source, filename=filename)
  lines = source.splitlines(keepends=True)
  out: dict[str, str] = {}
  for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
      start = node.lineno - 1
      end = getattr(node, "end_lineno", node.lineno)
      segment = "".join(lines[start:end])
      out[f"{filename}:{node.name}"] = segment
  return out


def _deterministic_ast_codemod(source: str) -> str:
  """Arm 1: Mechanical single-file AST/syntax codemod without cross-function flow analysis."""
  lines = []
  for line in source.splitlines():
    if line.strip() == "import requests":
      lines.append("import httpx")
    elif line.strip().startswith("from requests"):
      continue
    else:
      updated = line.replace("requests.Session()", "httpx.Client()")
      updated = updated.replace("requests.", "httpx.")
      lines.append(updated)
  return "\n".join(lines) + "\n"


def build_arm_context(
  workflow: dict[str, Any],
  arm_id: str,
  twinapi_embeddings: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Construct the prompt/context packet and/or migrated files for Arms 1..5."""
  base_files = workflow["candidate_base_files"]
  target_file = workflow["target_file"]
  target_src = base_files[target_file]

  setup_funcs = _extract_functions_from_file("session_setup.py", base_files["session_setup.py"])
  distractor_funcs = _extract_functions_from_file("distractors.py", base_files["distractors.py"])

  # Parse target AST to find true called/imported symbols in session_setup.py
  target_tree = ast.parse(target_src, filename=target_file)
  called_names = {
    node.func.id
    for node in ast.walk(target_tree)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
  }
  for node in ast.walk(target_tree):
    if isinstance(node, ast.ImportFrom) and node.module == "session_setup":
      for alias in node.names:
        called_names.add(alias.name)

  if arm_id == "arm1_codemod":
    migrated = dict(base_files)
    migrated[target_file] = _deterministic_ast_codemod(target_src)
    return {
      "arm_id": arm_id,
      "migrated_files": migrated,
      "context_functions": [],
      "context_text": "",
      "context_chars": 0,
      "prompt": "",
    }

  if arm_id == "arm2_target_only":
    selected_keys: list[str] = []
    selected_blocks: list[str] = []
  elif arm_id == "arm3_embedding_topk":
    # Score candidate distractor functions using 256-d TwinAPI embeddings + HTTP lexical projection
    emb_data = twinapi_embeddings
    if emb_data is None:
      emb_path = Path(__file__).resolve().parents[3] / "artifacts" / "twinapi_embeddings_256d.json"
      if emb_path.exists():
        emb_data = json.loads(emb_path.read_text(encoding="utf-8"))
    vecs = (emb_data or {}).get("vectors") or {}
    q_vec = vecs.get("requests.Session.get") or vecs.get("requests.Session.post") or [1.0] * 16
    http_terms = ("http", "header", "request", "response", "client", "session", "authorization")

    def _emb_rank(k: str) -> float:
      body_l = distractor_funcs[k].lower()
      lexical_density = sum(body_l.count(t) for t in http_terms)
      vec_norm = sum(abs(float(x)) for x in q_vec[:16]) / max(len(q_vec[:16]), 1)
      return round(lexical_density * (1.0 + vec_norm), 6)

    ranked = sorted(distractor_funcs.keys(), key=_emb_rank, reverse=True)
    selected_keys = ranked[:1]
    selected_blocks = [f"# {k}\n{distractor_funcs[k]}" for k in selected_keys]
  elif arm_id == "arm4_callgraph_slice":
    # True 1–2 hop stateful call-graph slice from connected module session_setup.py:
    # Hop 1 = functions called/imported by target_file; Hop 2 = functions in session_setup.py
    # sharing module-level state constants with Hop 1.
    setup_tree = ast.parse(base_files["session_setup.py"], filename="session_setup.py")
    module_consts = {
      t.id
      for node in setup_tree.body
      if isinstance(node, ast.Assign)
      for t in node.targets
      if isinstance(t, ast.Name)
    }
    reach_symbols = called_names | module_consts
    selected_keys = [
      k for k in setup_funcs
      if k.split(":")[-1] in reach_symbols
      or any(sym in setup_funcs[k] for sym in reach_symbols)
    ]
    setup_header_lines = [
      ln
      for ln in base_files["session_setup.py"].splitlines()
      if ln and not ln.startswith(("def ", "  ", "\t", "#"))
    ]
    header_block = "# session_setup.py (module constants)\n" + "\n".join(setup_header_lines)
    selected_blocks = [header_block] + [f"# {k}\n{setup_funcs[k]}" for k in selected_keys]
  elif arm_id == "arm5_unrelated_slice":
    # Equal-token unrelated utility slice from distractors.py (P4)
    selected_keys = [
      k for k in distractor_funcs if "compute_window_digest" in k
    ]
    selected_blocks = [f"# {k}\n{distractor_funcs[k]}" for k in selected_keys]
  else:
    raise ValueError(f"Unknown arm_id: {arm_id}")

  context_text = "\n\n".join(selected_blocks)
  prompt = (
    "You are migrating a Python module (`workflow_steps.py`) from `requests` to `httpx==0.28.1`.\n"
    "Requirements:\n"
    "1. Replace all `requests` usage with `httpx` (`import requests` is strictly forbidden).\n"
    "2. Preserve the exact function signature `def execute_step(base_url: str) -> dict:` and all imports from `session_setup`.\n"
    "3. Ensure multi-step HTTP behavior (redirects, headers, cookies, streaming lifecycle, and status exception handling) satisfies the upstream/downstream contract.\n\n"
    f"{STATIC_API_CHEATSHEET}\n"
  )
  if context_text:
    prompt += f"\n# Repository Context Slice ({arm_id})\n```python\n{context_text}\n```\n"
  prompt += (
    f"\n# Target File to Migrate (`{target_file}`)\n```python\n{target_src}\n```\n\n"
    "Return ONLY the complete migrated Python source code for `workflow_steps.py` inside a ```python ... ``` block."
  )

  return {
    "arm_id": arm_id,
    "context_functions": selected_keys,
    "context_text": context_text,
    "context_chars": len(context_text),
    "prompt": prompt,
  }


class _LoopbackStateMachineServer(http.server.ThreadingHTTPServer):
  allow_reuse_address = True
  daemon_threads = True

  def __init__(self, server_address: tuple[str, int], handler_cls: Any, server_spec: dict[str, Any]):
    super().__init__(server_address, handler_cls)
    self.server_spec = server_spec
    self.recorded_steps: list[dict[str, Any]] = []
    self.state_lock = threading.Lock()


class _StateMachineHandler(http.server.BaseHTTPRequestHandler):
  protocol_version = "HTTP/1.1"

  def setup(self) -> None:
    super().setup()
    self.connection.settimeout(2.0)

  def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
    return

  def _handle_any(self, method: str) -> None:
    srv: _LoopbackStateMachineServer = self.server  # type: ignore[assignment]
    content_len = int(self.headers.get("Content-Length", "0") or "0")
    raw_body = self.rfile.read(content_len).decode("utf-8", errors="replace") if content_len > 0 else ""
    hdr_dict = {k: v for k, v in self.headers.items()}

    with srv.state_lock:
      srv.recorded_steps.append({
        "method": method,
        "path": self.path,
        "headers": hdr_dict,
        "body": raw_body,
      })

    spec = srv.server_spec
    fam = spec["family"]
    v = spec["variant"]

    if fam == "F1_REDIRECT_FOLLOW_DEFAULT":
      if self.path.startswith(f"/step1_redirect_{v}"):
        self._send_json(302, {"redirect": True}, extra_headers={"Location": f"/step2_final_{v}"})
      elif self.path.startswith(f"/step2_final_{v}"):
        gate_val = self.headers.get(spec["hop_header"], "")
        self._send_json(200, {"stage": "final_destination", "gate_echo": gate_val})
      else:
        self._send_json(404, {"error": "unknown_route"})

    elif fam == "F2_CROSS_ORIGIN_AUTH_STRIP":
      if self.path.startswith(f"/origin_hop_{v}"):
        self._send_json(
          302,
          {"hop": 1},
          extra_headers={"Location": f"/partner_hop_{v}?origin=partner", "X-Hop-Ticket": f"tkt-{v}"},
        )
      elif self.path.startswith(f"/partner_hop_{v}"):
        leaked = spec["scrub_header"] in {k.lower() for k in self.headers.keys()}
        p_val = self.headers.get(spec["partner_header"], "")
        expected_p = f"{spec['partner_value']}:tkt-{v}"
        partner_ok = (not leaked) and (p_val == expected_p)
        status = 200 if partner_ok else 403
        self._send_json(status, {"partner_ok": partner_ok, "leaked": leaked})
      else:
        self._send_json(404, {"error": "unknown_route"})

    elif fam == "F3_COOKIE_JAR_SCOPING":
      if self.path.startswith(f"/csrf_init_{v}"):
        c_val = f"tok_{v * 13}"
        self._send_json(
          200,
          {"issued": True},
          extra_headers={"Set-Cookie": f"{spec['cookie_name']}={c_val}; Path=/"},
        )
      elif self.path.startswith(f"/csrf_commit_{v}"):
        expected_sync = f"synced:tok_{v * 13}"
        got_sync = self.headers.get(spec["csrf_header"], "")
        ok = got_sync == expected_sync
        self._send_json(200 if ok else 400, {"csrf_verified": ok})
      else:
        self._send_json(404, {"error": "unknown_route"})

    elif fam == "F4_STREAM_CONTEXT_LIFECYCLE":
      if self.path.startswith(f"/stream_feed_{v}"):
        payload = f"chunk-a-{v}\nchunk-b-{v}\nchunk-c-{v}\n".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()
      elif self.path.startswith(f"/stream_ack_{v}"):
        expected_payload = f"chunk-a-{v}\nchunk-b-{v}\nchunk-c-{v}\n".encode("utf-8")
        expected_ack = f"lines=3:bytes={len(expected_payload)}"
        got_ack = self.headers.get(spec["ack_header"], "")
        ok = got_ack == expected_ack
        self._send_json(200 if ok else 400, {"stream_ack": ok})
      else:
        self._send_json(404, {"error": "unknown_route"})

    elif fam == "F5_CLIENT_INIT_AND_STATUS_ERROR":
      if self.path.startswith(f"/primary_{v}"):
        self._send_json(
          429,
          {"error": "rate_limited"},
          extra_headers={"X-Retry-Ticket": f"rt-{v * 5}"},
        )
      elif self.path.startswith(f"/fallback_{v}"):
        expected_rec = f"recover:429:rt-{v * 5}"
        got_rec = self.headers.get(spec["recovery_header"], "")
        ok = got_rec == expected_rec
        self._send_json(200 if ok else 400, {"recovered": ok})
      else:
        self._send_json(404, {"error": "unknown_route"})

  def _send_json(self, status: int, payload: dict[str, Any], extra_headers: dict[str, str] | None = None) -> None:
    body = json.dumps(payload).encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.send_header("Connection", "close")
    if extra_headers:
      for hk, hv in extra_headers.items():
        self.send_header(hk, hv)
    self.end_headers()
    self.wfile.write(body)
    self.wfile.flush()

  def do_GET(self) -> None:  # noqa: N802
    self._handle_any("GET")

  def do_POST(self) -> None:  # noqa: N802
    self._handle_any("POST")

  def do_PUT(self) -> None:  # noqa: N802
    self._handle_any("PUT")

  def do_DELETE(self) -> None:  # noqa: N802
    self._handle_any("DELETE")


_CHILD_RUNNER_CODE = """\
import sys

ALLOWED_HOSTS = {"127.0.0.1", "::1", "localhost"}

def _audit_hook(event, args):
  if event in ("socket.connect", "socket.connect_ex", "socket.sendto", "socket.sendmsg"):
    addr = args[1] if len(args) > 1 else None
    if event == "socket.sendmsg" and len(args) > 4:
      addr = args[4]
    host = addr[0] if isinstance(addr, tuple) and len(addr) >= 1 else str(addr)
    if host not in ALLOWED_HOSTS:
      raise RuntimeError(f"NON_LOOPBACK_EGRESS_BLOCKED: {host}")
  elif event == "socket.getaddrinfo":
    host = str(args[0]) if len(args) >= 1 and args[0] is not None else ""
    if host and host not in ALLOWED_HOSTS:
      raise RuntimeError(f"NON_LOOPBACK_EGRESS_BLOCKED: {host}")

sys.addaudithook(_audit_hook)

import inspect
import json

base_url = sys.argv[1]
import workflow_steps

fn = workflow_steps.execute_step
sig = inspect.signature(fn)
if len(sig.parameters) == 1:
  res = fn(base_url)
else:
  import session_setup
  client = session_setup.prepare_client_session()
  try:
    res = fn(client, base_url)
  finally:
    if hasattr(client, "close"):
      client.close()

print("STATEBRIDGE_RESULT_JSON:" + json.dumps(res, sort_keys=True))
"""

_TEMP_PATH_RE = re.compile(r"(/private)?/var/folders/[^\s\"']+?/statebridge_run_[^\s/\"']+")


def _sanitize_stderr(raw: str, tmp_path: Path) -> str:
  text = raw or ""
  for p_str in (str(tmp_path.resolve()), str(tmp_path)):
    if p_str:
      text = text.replace(p_str, "<sandbox>")
  text = _TEMP_PATH_RE.sub("<sandbox>", text)
  home_str = str(Path.home())
  if home_str:
    text = text.replace(home_str, "~")
  return text


_SAFE_ENV_KEYS = (
  "PATH",
  "PYTHONPATH",
  "VIRTUAL_ENV",
  "SYSTEMROOT",
  "LANG",
  "LC_ALL",
  "HOME",
  "TMPDIR",
)


def _execute_against_loopback(
  server_spec: dict[str, Any],
  files_map: dict[str, str],
  timeout_sec: float,
) -> dict[str, Any]:
  """Run candidate or reference files in an isolated subprocess against 127.0.0.1:0."""
  server = _LoopbackStateMachineServer(("127.0.0.1", 0), _StateMachineHandler, server_spec)
  port = server.server_address[1]
  base_url = f"http://127.0.0.1:{port}"
  thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
  thread.start()

  try:
    with tempfile.TemporaryDirectory(prefix="statebridge_run_") as tmpdir:
      tmp_path = Path(tmpdir)
      for fname, content in files_map.items():
        safe_name = Path(fname).name
        if safe_name != fname or not safe_name.endswith(".py") or safe_name in {"_sb_runner.py", "inspect.py", "json.py", "sys.py"}:
          raise ValueError(f"Disallowed sandbox filename: {fname}")
        (tmp_path / safe_name).write_text(content, encoding="utf-8")
      runner_path = tmp_path / "_sb_runner.py"
      runner_path.write_text(_CHILD_RUNNER_CODE, encoding="utf-8")

      env = {k: os.environ[k] for k in _SAFE_ENV_KEYS if k in os.environ}
      env.update({
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
        "ALL_PROXY": "",
        "http_proxy": "",
        "https_proxy": "",
        "all_proxy": "",
        "NO_PROXY": "*",
        "no_proxy": "*",
        "NETRC": "/dev/null",
        "PYTHONDONTWRITEBYTECODE": "1",
      })

      proc = subprocess.Popen(
        [sys.executable, str(runner_path), base_url],
        cwd=str(tmp_path),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
      )
      try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
      except subprocess.TimeoutExpired:
        try:
          os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
          pass
        try:
          proc.wait(timeout=0.3)
        except subprocess.TimeoutExpired:
          try:
            os.killpg(proc.pid, signal.SIGKILL)
          except ProcessLookupError:
            pass
          proc.wait(timeout=0.3)
        return {
          "status": "timeout",
          "trace": list(server.recorded_steps),
          "return_value": None,
          "stderr": "Process timed out",
        }

      ret_val = None
      for line in stdout.splitlines():
        if line.startswith("STATEBRIDGE_RESULT_JSON:"):
          try:
            ret_val = json.loads(line.split("STATEBRIDGE_RESULT_JSON:", 1)[1])
          except Exception:
            ret_val = None

      return {
        "status": "ok" if proc.returncode == 0 else "error",
        "returncode": proc.returncode,
        "trace": list(server.recorded_steps),
        "return_value": ret_val,
        "stdout": stdout,
        "stderr": _sanitize_stderr(stderr, tmp_path),
      }
  finally:
    server.shutdown()
    server.server_close()


def run_workflow_sandbox(
  workflow: dict[str, Any],
  files_map: dict[str, str],
  allow_requests_reference: bool = False,
  timeout_sec: float = 3.0,
) -> dict[str, Any]:
  """Evaluate candidate files_map against workflow's reference requests implementation."""
  if not allow_requests_reference:
    anticheat = check_ast_anticheat(files_map)
    if not anticheat["passed"]:
      return {
        "workflow_id": workflow["id"],
        "family": workflow["family"],
        "outcome_category": anticheat["outcome_category"],
        "first_divergence_step": None,
        "divergence_field": None,
        "error_detail": anticheat["detail"],
        "candidate_trace_len": 0,
      }

  gold_exec = _execute_against_loopback(
    workflow["server_spec"], workflow["requests_files"], timeout_sec=timeout_sec
  )
  if gold_exec["status"] != "ok":
    raise RuntimeError(f"Reference requests oracle failed for {workflow['id']}: {gold_exec.get('stderr')}")
  cand_exec = _execute_against_loopback(
    workflow["server_spec"], files_map, timeout_sec=timeout_sec
  )

  if cand_exec["status"] == "timeout":
    return {
      "workflow_id": workflow["id"],
      "family": workflow["family"],
      "outcome_category": "TIMEOUT_OR_HANG",
      "first_divergence_step": None,
      "divergence_field": "timeout",
      "error_detail": cand_exec.get("stderr", ""),
      "candidate_trace_len": len(cand_exec["trace"]),
    }

  if cand_exec["status"] == "error" and len(cand_exec["trace"]) == 0:
    return {
      "workflow_id": workflow["id"],
      "family": workflow["family"],
      "outcome_category": "PRE_HTTP_RUNTIME_ERROR",
      "first_divergence_step": None,
      "divergence_field": "runtime_exception_before_http",
      "error_detail": cand_exec.get("stderr", "")[-600:],
      "candidate_trace_len": 0,
    }

  cmp_res = evaluate_trace_equivalence(
    gold_exec["trace"],
    cand_exec["trace"],
    gold_exec["return_value"],
    cand_exec["return_value"],
  )
  if cmp_res["outcome_category"] == "PASS_EQUIVALENT" and cand_exec["status"] != "ok":
    return {
      "workflow_id": workflow["id"],
      "family": workflow["family"],
      "outcome_category": "POST_HTTP_RETURN_MISMATCH",
      "first_divergence_step": None,
      "divergence_field": "post_http_exception",
      "error_detail": cand_exec.get("stderr", "")[-600:],
      "candidate_trace_len": len(cand_exec["trace"]),
    }

  return {
    "workflow_id": workflow["id"],
    "family": workflow["family"],
    "outcome_category": cmp_res["outcome_category"],
    "first_divergence_step": cmp_res["first_divergence_step"],
    "divergence_field": cmp_res["divergence_field"],
    "expected_canonical_step": cmp_res["expected_canonical_step"],
    "actual_canonical_step": cmp_res["actual_canonical_step"],
    "error_detail": cand_exec.get("stderr", "")[-400:] if cand_exec["status"] != "ok" else "",
    "candidate_trace_len": len(cand_exec["trace"]),
  }


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="StateBridge canonical standard-library CLI")
  sub = parser.add_subparsers(dest="cmd", required=True)

  sub.add_parser("selftest", help="Run zero-dependency stdlib self-test")

  scan_p = sub.add_parser("scan", help="Extract cross-file call-graph slice for a workflow")
  scan_p.add_argument("--workflow-id", default="F1_01")
  scan_p.add_argument("--arm", default="arm4_callgraph_slice")

  ver_p = sub.add_parser("verify", help="Verify a workflow in the Loopback HTTP sandbox")
  ver_p.add_argument("--workflow-id", default="F1_01")
  ver_p.add_argument("--mode", choices=["gold", "codemod", "requests"], default="gold")

  sub.add_parser("reproduce-paper", help="Verify and print paper SSOT metrics")

  args = parser.parse_args(argv)
  workflows = {w["id"]: w for w in get_benchmark_workflows()}

  if args.cmd == "reproduce-paper":
    root_dir = Path(__file__).resolve().parents[3]
    scripts_dir = root_dir / "scripts"
    if str(scripts_dir) not in sys.path:
      sys.path.insert(0, str(scripts_dir))
    import build_ssot as bs  # noqa: E402
    out = bs.build_ssot(root_dir / "artifacts" / "raw_llm_generations.jsonl", root_dir / "artifacts")
    print(json.dumps({
      "benchmark": out["benchmark"],
      "twinapi_audit": out["twinapi_audit"],
      "models": {
        mk: {
          "arms": {
            ak: {
              "pass_count": av["pass_count"],
              "total": av["total"],
              "solve_rate": av["solve_rate"],
            }
            for ak, av in mv["arms"].items()
          },
          "paired_comparisons": mv["paired_comparisons"],
        }
        for mk, mv in out["models"].items()
      },
    }, indent=2))
    return 0

  if args.cmd == "selftest":
    s1 = canonicalize_exchange({
      "method": "post",
      "path": "/x?a=1",
      "headers": {"User-Agent": "python-requests/2.34", "Content-Type": "application/json", "Cookie": "k=v"},
      "body": '{"a": 1}',
    })
    s2 = canonicalize_exchange({
      "method": "POST",
      "path": "/x?a=1",
      "headers": {"user-agent": "python-httpx/0.28", "content-type": "application/json", "cookie": "k=v"},
      "body": '{"a":1}',
    })
    assert s1 == s2, "Canonicalization self-test failed"
    eq = evaluate_trace_equivalence([s1], [s2])
    assert eq["outcome_category"] == "PASS_EQUIVALENT"
    print(json.dumps({"status": "ok", "workflows_count": len(workflows), "selftest": "PASS"}))
    return 0

  if args.cmd == "scan":
    wf = workflows[args.workflow_id]
    ctx = build_arm_context(wf, args.arm)
    print(json.dumps({
      "workflow_id": wf["id"],
      "arm": args.arm,
      "context_functions": ctx["context_functions"],
      "context_chars": ctx["context_chars"],
    }, indent=2))
    return 0

  if args.cmd == "verify":
    wf = workflows[args.workflow_id]
    if args.mode == "gold":
      fmap = wf["httpx_gold_files"]
      allow_req = False
    elif args.mode == "codemod":
      fmap = build_arm_context(wf, "arm1_codemod")["migrated_files"]
      allow_req = False
    else:
      fmap = wf["requests_files"]
      allow_req = True
    res = run_workflow_sandbox(wf, fmap, allow_requests_reference=allow_req)
    print(json.dumps(res, indent=2))
    return 0

  return 1


if __name__ == "__main__":
  raise SystemExit(main())
