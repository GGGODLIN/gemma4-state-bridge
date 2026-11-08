---
name: state-bridge
description: Cross-function stateful call-graph slicer and deterministic Loopback HTTP state-machine verifier for migrating Python HTTP client workflows (requests to httpx). Use when migrating requests.Session or multi-file HTTP client code to httpx.Client, diagnosing 301/302/303 redirect or cookie persistence regressions, or verifying multi-step protocol equivalence offline. Do not use for single-line non-HTTP string formatting or external live internet scraping.
---

# StateBridge ADK Skill (`state-bridge`)

`state-bridge` extracts cross-function stateful call-graph slices (`Session` / `Client` initialization, authentication/header injection, multi-hop redirect handling, cookie jar scoping, and streaming response lifecycle) and verifies migrated `httpx` workflows against a deterministic local Loopback HTTP State Machine (`127.0.0.1:0`) using only the Python standard library.

## Commands (Offline Python Standard Library)

1. **Self-test the canonical kernel and 30 benchmark workflows**:
   ```bash
   python3 skills/state-bridge/scripts/statebridge_core.py selftest
   ```
2. **Scan a multi-file directory or workflow for cross-function stateful call-graph slices**:
   ```bash
   python3 skills/state-bridge/scripts/statebridge_core.py scan --workflow-id F1_01
   ```
3. **Verify a candidate migration against the deterministic Loopback HTTP State Machine**:
   ```bash
   python3 skills/state-bridge/scripts/statebridge_core.py verify --workflow-id F1_01 --mode gold
   ```

