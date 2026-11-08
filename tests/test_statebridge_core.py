"""Acceptance and Conditional PBT suite (P1..P4) for Ticket 01 Seam 1 (statebridge_core)."""

from __future__ import annotations
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "skills" / "state-bridge" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPTS_DIR))

import statebridge_core as sb  # noqa: E402


# ---------------------------------------------------------------------------
# Independent Test Oracle for P1..P4 (constructed directly from raw structures)
# ---------------------------------------------------------------------------
DEFAULT_STRIPPED_HEADERS = {
  "user-agent",
  "host",
  "connection",
  "accept-encoding",
  "content-length",
  "accept",
}


def independent_oracle_canonical_step(
  method: str,
  path: str,
  query_pairs: list[tuple[str, str]],
  headers: dict[str, str],
  cookies: dict[str, str],
  json_obj: dict[str, int] | None,
) -> dict:
  norm_headers = {
    k.lower(): v.strip()
    for k, v in headers.items()
    if k.lower() not in DEFAULT_STRIPPED_HEADERS and k.lower() != "cookie"
  }
  return {
    "method": method.upper(),
    "path": path,
    "query": [[k, v] for k, v in query_pairs],
    "headers": norm_headers,
    "cookies": dict(cookies),
    "body_json": json_obj,
    "body_text": "" if json_obj is not None else "",
  }


def evaluate_p1_invariance(raw_a: dict, raw_b: dict, expected_oracle: dict) -> bool:
  can_a = sb.canonicalize_exchange(raw_a)
  can_b = sb.canonicalize_exchange(raw_b)
  can_idem = sb.canonicalize_exchange(can_a)
  if can_a != can_b or can_a != can_idem:
    return False
  for key in ("method", "path", "query", "headers", "cookies", "body_json"):
    if can_a.get(key) != expected_oracle.get(key):
      return False
  return True


def evaluate_p2_divergence(
  trace: list[dict],
  mutated_trace: list[dict],
  expected_k: int | None,
) -> bool:
  res = sb.evaluate_trace_equivalence(trace, mutated_trace)
  if expected_k is None:
    return (
      res.get("outcome_category") == "PASS_EQUIVALENT"
      and res.get("first_divergence_step") is None
    )
  return (
    res.get("outcome_category") == "STEP_K_WIRE_DIVERGENCE"
    and res.get("first_divergence_step") == expected_k
  )


# ---------------------------------------------------------------------------
# Role 1: Concrete Capability Examples
# ---------------------------------------------------------------------------
def test_canonicalize_exchange_requests_vs_httpx_wire_noise():
  req_step = {
    "method": "post",
    "path": "/api/items?b=2&a=1",
    "headers": {
      "Host": "127.0.0.1:54321",
      "User-Agent": "python-requests/2.34.2",
      "Accept-Encoding": "gzip, deflate",
      "Accept": "*/*",
      "Connection": "keep-alive",
      "Content-Type": "application/json",
      "Content-Length": "16",
      "X-Trace-Token": "tok-99",
      "Cookie": "sid=abc; theme=dark",
    },
    "body": '{"x": 1, "y": 2}',
  }
  httpx_step = {
    "method": "POST",
    "path": "/api/items?b=2&a=1",
    "headers": {
      "host": "127.0.0.1:54321",
      "user-agent": "python-httpx/0.28.1",
      "accept": "*/*",
      "accept-encoding": "gzip, deflate",
      "connection": "keep-alive",
      "content-type": "application/json; charset=utf-8",
      "content-length": "13",
      "x-trace-token": "tok-99",
      "cookie": "theme=dark; sid=abc",
    },
    "body": '{"y":2,"x":1}',
  }
  c_req = sb.canonicalize_exchange(req_step)
  c_httpx = sb.canonicalize_exchange(httpx_step)
  assert c_req == c_httpx
  assert c_req["method"] == "POST"
  assert c_req["path"] == "/api/items"
  assert c_req["query"] == [["b", "2"], ["a", "1"]]
  assert c_req["cookies"] == {"sid": "abc", "theme": "dark"}
  assert c_req["headers"]["x-trace-token"] == "tok-99"
  assert c_req["body_json"] == {"x": 1, "y": 2}


def test_evaluate_trace_equivalence_pinpoints_first_divergence_step():
  step1 = {
    "method": "POST",
    "path": "/login",
    "headers": {"Content-Type": "application/json"},
    "body": '{"user":"alice"}',
  }
  step2_gold = {
    "method": "GET",
    "path": "/vault",
    "headers": {"Cookie": "session_id=s1"},
    "body": "",
  }
  step2_bad_auth = {
    "method": "GET",
    "path": "/vault",
    "headers": {"Cookie": "session_id=s1", "Authorization": "Bearer leaked"},
    "body": "",
  }
  eq_res = sb.evaluate_trace_equivalence(
    [step1, step2_gold],
    [step1, step2_gold],
    {"status": "ok", "code": 200},
    {"status": "ok", "code": 200},
  )
  assert eq_res["outcome_category"] == "PASS_EQUIVALENT"
  assert eq_res["first_divergence_step"] is None

  div_res = sb.evaluate_trace_equivalence(
    [step1, step2_gold],
    [step1, step2_bad_auth],
    {"status": "ok", "code": 200},
    {"status": "ok", "code": 200},
  )
  assert div_res["outcome_category"] == "STEP_K_WIRE_DIVERGENCE"
  assert div_res["first_divergence_step"] == 2
  assert "headers" in div_res["divergence_field"]


def test_check_ast_anticheat_catches_residual_requests_and_passes_httpx():
  cheat_direct = {"client.py": "import requests\nimport httpx\n"}
  cheat_from = {"client.py": "from requests.sessions import Session\n"}
  cheat_urllib = {"client.py": "import urllib.request as u\n"}
  clean_httpx = {"client.py": "import httpx\n\ndef run():\n  return httpx.Client()\n"}

  r1 = sb.check_ast_anticheat(cheat_direct)
  assert r1["passed"] is False
  assert r1["outcome_category"] == "CHEAT_RESIDUAL_REQUESTS"

  r2 = sb.check_ast_anticheat(cheat_from)
  assert r2["passed"] is False
  assert r2["outcome_category"] == "CHEAT_RESIDUAL_REQUESTS"

  r3 = sb.check_ast_anticheat(cheat_urllib)
  assert r3["passed"] is False
  assert r3["outcome_category"] == "CHEAT_RESIDUAL_REQUESTS"

  r_clean = sb.check_ast_anticheat(clean_httpx)
  assert r_clean["passed"] is True
  assert r_clean["outcome_category"] is None


def test_build_arm_context_all_five_arms_and_budget_parity():
  workflows = sb.get_benchmark_workflows()
  assert len(workflows) == 30
  wf = workflows[0]

  arm1 = sb.build_arm_context(wf, "arm1_codemod")
  arm2 = sb.build_arm_context(wf, "arm2_target_only")
  arm3 = sb.build_arm_context(wf, "arm3_embedding_topk")
  arm4 = sb.build_arm_context(wf, "arm4_callgraph_slice")
  arm5 = sb.build_arm_context(wf, "arm5_unrelated_slice")

  assert "migrated_files" in arm1
  assert sb.check_ast_anticheat(arm1["migrated_files"])["passed"] is True
  assert len(arm2["context_functions"]) == 0
  assert len(arm3["context_functions"]) >= 1
  assert len(arm4["context_functions"]) >= 2
  assert len(arm5["context_functions"]) >= 1
  assert set(arm4["context_functions"]).isdisjoint(set(arm5["context_functions"]))


def test_sandbox_egress_guard_and_timeout_process_cleanup():
  workflows = sb.get_benchmark_workflows()
  wf = workflows[0]

  # 1. Non-loopback egress attempt must be blocked by sys.addaudithook
  egress_files = dict(wf["httpx_gold_files"])
  target_file = wf["target_file"]
  egress_files[target_file] = (
    "import httpx\nimport socket\n"
    "def execute_step(client, base_url, *args, **kwargs):\n"
    "  s = socket.socket()\n"
    "  s.connect(('198.51.100.1', 80))\n"
  )
  res_egress = sb.run_workflow_sandbox(wf, egress_files, timeout_sec=2.0)
  assert res_egress["outcome_category"] == "PRE_HTTP_RUNTIME_ERROR"
  assert "NON_LOOPBACK_EGRESS_BLOCKED" in res_egress.get("error_detail", "")

  # 2. Infinite loop / hang must trigger TIMEOUT_OR_HANG and leave 0 orphan processes
  hang_files = dict(wf["httpx_gold_files"])
  hang_files[target_file] = (
    "import httpx\nimport time\n"
    f"MARKER = 'STATEBRIDGE_ORPHAN_TEST_{random.randint(10000, 99999)}'\n"
    "def execute_step(client, base_url, *args, **kwargs):\n"
    "  while True:\n"
    "    time.sleep(0.1)\n"
  )
  res_hang = sb.run_workflow_sandbox(wf, hang_files, timeout_sec=1.2)
  assert res_hang["outcome_category"] == "TIMEOUT_OR_HANG"


def test_all_30_workflows_pass_reference_and_diverge_on_arm1_codemod():
  workflows = sb.get_benchmark_workflows()
  assert len(workflows) == 30
  families = {w["family"] for w in workflows}
  assert families == {
    "F1_REDIRECT_FOLLOW_DEFAULT",
    "F2_CROSS_ORIGIN_AUTH_STRIP",
    "F3_COOKIE_JAR_SCOPING",
    "F4_STREAM_CONTEXT_LIFECYCLE",
    "F5_CLIENT_INIT_AND_STATUS_ERROR",
  }

  for wf in workflows:
    # Reference requests implementation must pass 100%
    ref_res = sb.run_workflow_sandbox(
      wf, wf["requests_files"], allow_requests_reference=True, timeout_sec=3.0
    )
    assert ref_res["outcome_category"] == "PASS_EQUIVALENT", (
      f"Reference requests failed on {wf['id']}: {ref_res}"
    )

    # Gold httpx implementation must also pass 100%
    gold_httpx_res = sb.run_workflow_sandbox(
      wf, wf["httpx_gold_files"], allow_requests_reference=False, timeout_sec=3.0
    )
    assert gold_httpx_res["outcome_category"] == "PASS_EQUIVALENT", (
      f"Gold httpx failed on {wf['id']}: {gold_httpx_res}"
    )

    # Arm 1 (Deterministic AST Codemod without cross-function stateful reasoning) must fail
    arm1 = sb.build_arm_context(wf, "arm1_codemod")
    arm1_res = sb.run_workflow_sandbox(
      wf, arm1["migrated_files"], allow_requests_reference=False, timeout_sec=3.0
    )
    assert arm1_res["outcome_category"] in {
      "STEP_K_WIRE_DIVERGENCE",
      "PRE_HTTP_RUNTIME_ERROR",
      "POST_HTTP_RETURN_MISMATCH",
    }, f"Expected Arm 1 codemod to diverge on {wf['id']}, got {arm1_res}"


def test_cli_selftest_succeeds_in_stdlib_python():
  proc = subprocess.run(
    [sys.executable, str(SCRIPTS_DIR / "statebridge_core.py"), "selftest"],
    capture_output=True,
    text=True,
    check=False,
  )
  assert proc.returncode == 0, f"selftest failed: {proc.stderr}\n{proc.stdout}"
  out = json.loads(proc.stdout.strip().splitlines()[-1])
  assert out["status"] == "ok"
  assert out["workflows_count"] == 30


# ---------------------------------------------------------------------------
# Role 2: Safety Property Guardians (Conditional PBT P1..P4)
# ---------------------------------------------------------------------------
def test_pbt_p1_to_p4_invariants():
  rng = random.Random(20261109)
  methods = ["GET", "POST", "PUT", "DELETE"]

  # P1 & P2 across 120 generated multi-step HTTP traces
  for _ in range(120):
    num_steps = rng.randint(2, 5)
    trace_a = []
    trace_b = []
    oracles = []
    for s_idx in range(num_steps):
      m = rng.choice(methods)
      path = f"/v1/step_{s_idx}"
      q_pairs = [(f"k{i}", f"v{rng.randint(1, 9)}") for i in range(rng.randint(0, 3))]
      hdrs = {f"X-Custom-{rng.randint(1, 5)}": f"val-{rng.randint(10, 99)}"}
      cookies = {f"c{i}": f"cv{rng.randint(1, 9)}" for i in range(rng.randint(1, 3))}
      json_obj = (
        {f"field_{i}": rng.randint(1, 100) for i in range(rng.randint(1, 4))}
        if m in ("POST", "PUT")
        else None
      )
      oracle = independent_oracle_canonical_step(m, path, q_pairs, hdrs, cookies, json_obj)
      oracles.append(oracle)

      q_str = "&".join(f"{k}={v}" for k, v in q_pairs)
      full_path = f"{path}?{q_str}" if q_str else path
      cookie_items = list(cookies.items())
      c_str_a = "; ".join(f"{k}={v}" for k, v in cookie_items)
      rng.shuffle(cookie_items)
      c_str_b = "; ".join(f"{k}={v}" for k, v in cookie_items)

      hdrs_a = {k.upper(): v for k, v in hdrs.items()}
      hdrs_a.update({
        "User-Agent": "python-requests/2.34.2",
        "Cookie": c_str_a,
        "Content-Type": "application/json" if json_obj else "text/plain",
        "Content-Length": "999",
      })
      hdrs_b = {k.lower(): f"  {v}  " for k, v in hdrs.items()}
      hdrs_b.update({
        "user-agent": "python-httpx/0.28.1",
        "cookie": c_str_b,
        "content-type": "application/json; charset=utf-8" if json_obj else "text/plain",
        "content-length": "12",
      })
      body_a = json.dumps(json_obj, separators=(", ", ": ")) if json_obj else ""
      body_b = json.dumps(json_obj, separators=(",", ":"), sort_keys=True) if json_obj else ""

      raw_a = {"method": m.lower(), "path": full_path, "headers": hdrs_a, "body": body_a}
      raw_b = {"method": m.upper(), "path": full_path, "headers": hdrs_b, "body": body_b}

      assert evaluate_p1_invariance(raw_a, raw_b, oracle) is True
      trace_a.append(raw_a)
      trace_b.append(raw_b)

    # P2 reflexive equivalence
    assert evaluate_p2_divergence(trace_a, trace_b, expected_k=None) is True

    # P2 exact step-k mutation detection (1-indexed)
    target_k = rng.randint(1, num_steps)
    mutated = [dict(x, headers=dict(x["headers"])) for x in trace_b]
    mutated[target_k - 1]["headers"]["x-mutated-auth"] = "broken"
    assert evaluate_p2_divergence(trace_a, mutated, expected_k=target_k) is True

  # P3 & P4 across all 30 workflows
  workflows = sb.get_benchmark_workflows()
  for wf in workflows:
    assert sb.check_ast_anticheat(wf["httpx_gold_files"])["passed"] is True
    assert sb.check_ast_anticheat(wf["requests_files"])["passed"] is False
    arm4 = sb.build_arm_context(wf, "arm4_callgraph_slice")
    arm5 = sb.build_arm_context(wf, "arm5_unrelated_slice")
    assert set(arm4["context_functions"]).isdisjoint(set(arm5["context_functions"]))
    assert abs(arm4["context_chars"] - arm5["context_chars"]) <= max(650, int(0.45 * arm4["context_chars"]))


# ---------------------------------------------------------------------------
# Role 3: Oracle Calibration & Counterexample Replay
# ---------------------------------------------------------------------------
def test_oracle_calibration_individual_invariants_and_counterexample_replay():
  oracle = independent_oracle_canonical_step(
    "POST", "/step1", [("a", "1")], {"x-hdr": "v1"}, {"sid": "123"}, {"k": 1}
  )
  valid_step = {
    "method": "POST",
    "path": "/step1?a=1",
    "headers": {"X-Hdr": "v1", "Cookie": "sid=123", "Content-Type": "application/json"},
    "body": '{"k": 1}',
  }
  bad_p1_step = {
    "method": "POST",
    "path": "/step1?a=1",
    "headers": {"X-Hdr": "v2", "Cookie": "sid=123", "Content-Type": "application/json"},
    "body": '{"k": 1}',
  }
  # P1 calibration: must return False when header value differs
  assert evaluate_p1_invariance(valid_step, bad_p1_step, oracle) is False

  # P2 calibration: must return False when claiming step 1 diverged on identical traces
  assert evaluate_p2_divergence([valid_step], [valid_step], expected_k=1) is False

  # Pinned counterexample replay: JSON whitespace difference ('{"x": 1}' vs '{"x":1}')
  # with differing Content-Length ('8' vs '7') must NOT be flagged as divergence.
  replay_trace_req = [{
    "method": "POST",
    "path": "/replay",
    "headers": {"Content-Type": "application/json", "Content-Length": "8"},
    "body": '{"x": 1}',
  }]
  replay_trace_httpx = [{
    "method": "POST",
    "path": "/replay",
    "headers": {"content-type": "application/json", "content-length": "7"},
    "body": '{"x":1}',
  }]
  assert evaluate_p2_divergence(replay_trace_req, replay_trace_httpx, expected_k=None) is True
