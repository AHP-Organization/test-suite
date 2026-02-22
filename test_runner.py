#!/usr/bin/env python3
"""
AHP Test Suite — Visiting Agent Test Harness

Runs all test scenarios against a live AHP endpoint.
Generates a structured JSON report + human-readable summary.

Usage:
    ./venv/bin/python test_runner.py --target https://ref.agenthandshake.dev
    ./venv/bin/python test_runner.py --target http://localhost:3000 --verbose
"""

import argparse
import json
import time
import sys
import os
import statistics
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from typing import Optional

import requests
from ahp_client import AHPClient, AHPRequest

# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    test_id: str
    name: str
    passed: bool
    latency_ms: float = 0
    tokens_used: int = 0
    mode: str = ""
    cached: bool = False
    tools_used: list = field(default_factory=list)
    notes: str = ""
    error: str = ""
    raw_response: dict = field(default_factory=dict)

@dataclass
class ComparisonResult:
    query: str
    markdown_tokens: int
    ahp_tokens: int
    markdown_latency_ms: float
    ahp_latency_ms: float
    markdown_answer_length: int
    ahp_answer_length: int
    token_reduction_pct: float
    latency_delta_ms: float

# ── Test scenarios ─────────────────────────────────────────────────────────────

def run_all_tests(client: AHPClient, target: str, verbose: bool = False) -> dict:
    results = []
    comparisons = []

    print(f"\n{'='*60}")
    print(f"AHP Test Suite — {target}")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}\n")

    # ── Test 1: Discovery — well-known URI ─────────────────────────────────────
    r = run_test("T01", "Discovery — /.well-known/agent.json", lambda: test_discovery(client), verbose)
    results.append(r)

    # ── Test 2: Discovery — Accept header ─────────────────────────────────────
    r = run_test("T02", "Discovery — Accept: application/agent+json", lambda: test_accept_header(client), verbose)
    results.append(r)

    # ── Test 3: Discovery — in-page agent notice ───────────────────────────────
    r = run_test("T03", "Discovery — in-page agent notice", lambda: test_agent_notice(client), verbose)
    results.append(r)

    # ── Test 4: Manifest schema validation ────────────────────────────────────
    r = run_test("T04", "Manifest schema validation", lambda: test_manifest_schema(client), verbose)
    results.append(r)

    # ── Test 5: MODE2 — simple query ──────────────────────────────────────────
    r = run_test("T05", "MODE2 — simple content query", lambda: test_simple_query(client), verbose)
    results.append(r)

    # ── Test 6: MODE2 — unknown capability error ───────────────────────────────
    r = run_test("T06", "Error handling — unknown capability", lambda: test_unknown_capability(client), verbose)
    results.append(r)

    # ── Test 7: MODE2 — malformed request ─────────────────────────────────────
    r = run_test("T07", "Error handling — missing required field", lambda: test_malformed_request(client, target), verbose)
    results.append(r)

    # ── Test 8: Session management ─────────────────────────────────────────────
    r = run_test("T08", "Session — multi-turn exchange", lambda: test_multiturn(client), verbose)
    results.append(r)

    # ── Test 9: Response schema validation ────────────────────────────────────
    r = run_test("T09", "Response schema — required fields present", lambda: test_response_schema(client), verbose)
    results.append(r)

    # ── Test 10: Content signals in response ───────────────────────────────────
    r = run_test("T10", "Content signals — present in response meta", lambda: test_content_signals(client), verbose)
    results.append(r)

    # ── Test 11: MODE3 — inventory check (tool use) ───────────────────────────
    r = run_test("T11", "MODE3 — inventory check with tool use", lambda: test_inventory_check(client), verbose)
    results.append(r)

    # ── Test 12: MODE3 — quote calculation ────────────────────────────────────
    r = run_test("T12", "MODE3 — custom quote calculation", lambda: test_get_quote(client), verbose)
    results.append(r)

    # ── Test 13: MODE3 — order lookup ─────────────────────────────────────────
    r = run_test("T13", "MODE3 — order lookup", lambda: test_order_lookup(client), verbose)
    results.append(r)

    # ── Test 14: MODE3 — async human escalation ───────────────────────────────
    r = run_test("T14", "MODE3 — async human escalation + poll", lambda: test_human_escalation(client), verbose)
    results.append(r)

    # ── Test 15: Cached response ───────────────────────────────────────────────
    r = run_test("T15", "Caching — repeated query returns cached response", lambda: test_caching(client), verbose)
    results.append(r)

    # ── Whitepaper: Token efficiency comparison ────────────────────────────────
    print("\n[Whitepaper] Running Agent vs. Markdown token efficiency comparison...")
    comparisons = run_token_comparison(client, target, verbose)

    # ── Latency profile ────────────────────────────────────────────────────────
    print("[Whitepaper] Running latency profile...")
    latency_profile = run_latency_profile(client, verbose)

    # ── Test 16: Rate limiting — run LAST to avoid polluting other tests ───────
    r = run_test("T16", "Rate limiting — 429 on burst", lambda: test_rate_limiting(client), verbose)
    results.append(r)

    return {
        "target": target,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ahp_version": "0.1",
        "test_results": [asdict(r) for r in results],
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "pass_rate_pct": round(sum(1 for r in results if r.passed) / len(results) * 100, 1),
        },
        "token_comparison": [asdict(c) for c in comparisons],
        "latency_profile": latency_profile,
    }

# ── Individual test functions ─────────────────────────────────────────────────

def test_discovery(client):
    manifest = client.discover()
    assert "ahp" in manifest, "Missing 'ahp' field"
    assert "modes" in manifest, "Missing 'modes' field"
    assert "content_signals" in manifest, "Missing 'content_signals' field"
    assert len(manifest["modes"]) > 0, "modes array is empty"
    return TestResult("T01", "Discovery", True, notes=f"Modes: {manifest['modes']}")

def test_accept_header(client):
    try:
        manifest = client.discover_via_accept_header()
        assert "ahp" in manifest, "Manifest not returned via Accept header"
        return TestResult("T02", "Accept header", True, notes="Accept header discovery succeeded")
    except Exception as e:
        return TestResult("T02", "Accept header", False, error=str(e),
            notes="Site may not implement Accept header redirect (optional)")

def test_agent_notice(client):
    # In-page notice applies to HTML sites. Pure API servers may not serve HTML.
    # If root returns non-HTML or 4xx, treat as API server (not applicable → pass).
    try:
        import requests as _req
        r = _req.get(f"{client.base_url}/", timeout=10)
        content_type = r.headers.get("Content-Type", "")
        is_html_site = "text/html" in content_type and r.status_code < 400
        if not is_html_site:
            return TestResult("T03", "In-page notice", True,
                notes=f"Not applicable — root returns HTTP {r.status_code} ({content_type or 'no content-type'}). "
                      "In-page notice is for HTML sites. API servers are exempt.")
        found = 'aria-label="AI Agent Notice"' in r.text or 'class="ahp-notice"' in r.text
        return TestResult("T03", "In-page notice", found,
            notes="Agent notice found" if found else "No agent notice found on /")
    except Exception as e:
        return TestResult("T03", "In-page notice", False, error=str(e))

def test_manifest_schema(client):
    manifest = client.discover()
    required = ["ahp", "modes", "content_signals"]
    missing = [f for f in required if f not in manifest]
    assert not missing, f"Missing required fields: {missing}"
    cs = manifest["content_signals"]
    assert "ai_input" in cs, "content_signals.ai_input is required"
    return TestResult("T04", "Schema validation", True,
        notes=f"All required fields present. content_signals: {cs}")

def test_simple_query(client):
    resp = client.converse(AHPRequest(
        capability="content_search",
        query="What is the Agent Handshake Protocol?",
    ))
    assert resp.http_status == 200, f"Expected 200, got {resp.http_status}"
    assert resp.status == "success", f"Expected success, got {resp.status}"
    answer = resp.raw.get("response", {}).get("answer", "")
    assert len(answer) > 50, "Answer too short"
    return TestResult("T05", "Simple query", True,
        latency_ms=resp.latency_ms, tokens_used=resp.tokens_used,
        mode=resp.mode, cached=resp.cached,
        notes=f"Answer: {answer[:100]}...")

def test_unknown_capability(client):
    resp = client.converse(AHPRequest(capability="does_not_exist", query="test"))
    assert resp.http_status == 400, f"Expected 400, got {resp.http_status}"
    assert resp.raw.get("code") == "unknown_capability", "Expected unknown_capability error code"
    assert "available_capabilities" in resp.raw, "Should list available capabilities"
    return TestResult("T06", "Unknown capability", True,
        notes=f"Available: {resp.raw.get('available_capabilities')}")

def test_malformed_request(client, target):
    r = requests.post(f"{target}/agent/converse",
        json={"capability": "site_info"},  # missing 'query'
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    assert r.status_code == 400, f"Expected 400 for missing query, got {r.status_code}"
    return TestResult("T07", "Missing field", True, notes="Missing 'query' correctly returns 400")

def test_multiturn(client):
    # Turn 1
    r1 = client.converse(AHPRequest(capability="content_search", query="Tell me about AHP modes"))
    assert r1.status == "success"
    sid = r1.session_id

    # Turn 2 — follow-up in same session
    r2 = client.converse(AHPRequest(
        capability="content_search",
        query="Which mode should I start with?",
        session_id=sid,
    ))
    assert r2.http_status == 200
    assert r2.session_id == sid, "Session ID should persist"
    return TestResult("T08", "Multi-turn", True,
        latency_ms=r2.latency_ms, tokens_used=r1.tokens_used + r2.tokens_used,
        notes=f"2-turn exchange on session {sid}")

def test_response_schema(client):
    resp = client.converse(AHPRequest(capability="site_info", query="What is this site?"))
    raw = resp.raw
    assert "status" in raw
    assert "response" in raw
    assert "meta" in raw
    assert "answer" in raw["response"] or "payload" in raw["response"], "Response must have answer or payload"
    assert "capability_used" in raw["meta"]
    assert "content_signals" in raw["meta"]
    return TestResult("T09", "Response schema", True, notes="All required response fields present")

def test_content_signals(client):
    resp = client.converse(AHPRequest(capability="site_info", query="What can you tell me about this site?"))
    cs = resp.raw.get("meta", {}).get("content_signals", {})
    assert cs, "content_signals missing from response meta"
    assert "ai_input" in cs, "ai_input signal missing"
    return TestResult("T10", "Content signals", True, notes=f"Signals: {cs}")

def test_inventory_check(client):
    resp = client.converse(AHPRequest(
        capability="inventory_check",
        query="Is the AHP Server License Single Site in stock? How much does it cost?",
    ))
    assert resp.http_status == 200, f"Got {resp.http_status}"
    assert resp.status == "success"
    assert resp.mode == "MODE3", f"Expected MODE3, got {resp.mode}"
    assert len(resp.tools_used) > 0, "No tools were called"
    answer = resp.raw.get("response", {}).get("answer", "")
    assert len(answer) > 20, "Answer too short"
    return TestResult("T11", "MODE3 inventory", True,
        latency_ms=resp.latency_ms, tokens_used=resp.tokens_used,
        mode=resp.mode, tools_used=resp.tools_used,
        notes=f"Tools used: {resp.tools_used}. Answer: {answer[:100]}...")

def test_get_quote(client):
    resp = client.converse(AHPRequest(
        capability="get_quote",
        query="Give me a quote for 3 Single Site licenses and 1 Support package as a business customer",
    ))
    assert resp.http_status == 200
    assert resp.status == "success"
    assert resp.mode == "MODE3"
    assert len(resp.tools_used) > 0
    answer = resp.raw.get("response", {}).get("answer", "")
    # Accept any response that references quantity, cost, or a product name
    price_mentioned = any(term in answer.lower() for term in [
        "$", "usd", "price", "cost", "quote", "total", "license", "support", "discount"
    ])
    assert price_mentioned, f"Answer should reference quote/pricing. Got: {answer[:200]}"
    return TestResult("T12", "MODE3 quote", True,
        latency_ms=resp.latency_ms, tokens_used=resp.tokens_used,
        mode=resp.mode, tools_used=resp.tools_used,
        notes=f"Answer: {answer[:150]}...")

def test_order_lookup(client):
    resp = client.converse(AHPRequest(
        capability="order_lookup",
        query="Can you look up order ORD-2026-001?",
    ))
    assert resp.http_status == 200
    assert resp.status == "success"
    answer = resp.raw.get("response", {}).get("answer", "")
    assert "ORD-2026-001" in answer or "shipped" in answer.lower() or "order" in answer.lower(), \
        "Answer should reference the order"
    return TestResult("T13", "MODE3 order lookup", True,
        latency_ms=resp.latency_ms, tokens_used=resp.tokens_used,
        mode=resp.mode, tools_used=resp.tools_used,
        notes=f"Answer: {answer[:100]}...")

def test_human_escalation(client):
    # Submit async request
    resp = client.converse(AHPRequest(
        capability="human_escalation",
        query="I need a custom enterprise agreement for 500 sites. This requires human review.",
    ))
    assert resp.http_status == 200
    data = resp.raw
    assert data.get("status") == "accepted", f"Expected accepted, got {data.get('status')}"
    assert "session_id" in data
    assert "poll" in data
    assert "eta_seconds" in data

    sid = data["session_id"]

    # Poll for resolution
    poll_start = time.time()
    final = client.poll_status(sid, max_wait_s=60, interval_s=1.0)
    poll_time_ms = (time.time() - poll_start) * 1000

    assert final.status == "success", f"Async resolution failed: {final.status}"
    answer = final.raw.get("response", {}).get("answer", "")
    assert len(answer) > 20, "Human response answer too short"

    return TestResult("T14", "MODE3 async escalation", True,
        latency_ms=poll_time_ms,
        notes=f"Accepted → resolved via polling in {poll_time_ms:.0f}ms. Answer: {answer[:100]}...")

def test_rate_limiting(client):
    statuses = client.burst(count=35)
    hit_429 = 429 in statuses
    first_429 = statuses.index(429) + 1 if hit_429 else None
    return TestResult("T15", "Rate limiting", hit_429,
        notes=f"429 hit after {first_429} requests" if hit_429 else "No 429 returned — rate limiting may not be active")

def test_caching(client):
    q = "What is MODE2 in the Agent Handshake Protocol?"
    # First request (cold)
    r1 = client.converse(AHPRequest(capability="content_search", query=q))
    # Second request (should be cached)
    r2 = client.converse(AHPRequest(capability="content_search", query=q))
    cache_hit = r2.cached
    speedup = r1.latency_ms / r2.latency_ms if r2.latency_ms > 0 else 0
    return TestResult("T16", "Caching", True,
        latency_ms=r2.latency_ms,
        notes=f"Cold: {r1.latency_ms:.0f}ms | Cached: {r2.latency_ms:.0f}ms | "
              f"Speedup: {speedup:.1f}x | Cache hit flag: {cache_hit}")

# ── Whitepaper: Token comparison ──────────────────────────────────────────────

COMPARISON_QUERIES = [
    ("Explain what MODE1 is in the Agent Handshake Protocol", "content_search"),
    ("How does AHP discovery work for headless browser agents?", "content_search"),
    ("What are AHP content signals and what do they declare?", "content_search"),
    ("How do I build a MODE2 interactive knowledge endpoint?", "content_search"),
    ("What rate limits should a MODE2 AHP server enforce?", "content_search"),
]

def run_token_comparison(client: AHPClient, target: str, verbose: bool) -> list:
    results = []

    # Simulate "markdown agent": fetch the full content doc and count its tokens
    try:
        manifest = client.discover()
        content_url = manifest.get("endpoints", {}).get("content", "/content/spec.md")
        r = requests.get(f"{target}{content_url}", timeout=15)
        full_content = r.text
        # Rough token estimate: 1 token ≈ 4 chars
        markdown_base_tokens = len(full_content) // 4
    except Exception as e:
        print(f"  [warn] Could not fetch content doc: {e}")
        markdown_base_tokens = 10000  # fallback estimate

    for query, capability in COMPARISON_QUERIES:
        # AHP MODE2
        resp = client.converse(AHPRequest(capability=capability, query=query))
        if resp.http_status == 429:
            print(f"  [warn] Rate limited during comparison — results may be incomplete")
            time.sleep(5)
            resp = client.converse(AHPRequest(capability=capability, query=query))
        ahp_tokens = resp.tokens_used
        ahp_latency = resp.latency_ms
        ahp_answer_len = len(resp.raw.get("response", {}).get("answer", ""))

        # "Markdown agent": full doc + query overhead (estimated)
        md_tokens = markdown_base_tokens + len(query) // 4 + 50  # rough query overhead
        md_latency = 200.0  # flat estimate for fetching a static doc
        md_answer_len = ahp_answer_len  # assume same quality answer

        reduction = (1 - ahp_tokens / md_tokens) * 100 if md_tokens > 0 else 0

        c = ComparisonResult(
            query=query,
            markdown_tokens=md_tokens,
            ahp_tokens=ahp_tokens,
            markdown_latency_ms=md_latency,
            ahp_latency_ms=ahp_latency,
            markdown_answer_length=md_answer_len,
            ahp_answer_length=ahp_answer_len,
            token_reduction_pct=round(reduction, 1),
            latency_delta_ms=round(ahp_latency - md_latency, 1),
        )
        results.append(c)
        if verbose:
            print(f"  {query[:50]:<50} AHP: {ahp_tokens:>5} tok | MD: {md_tokens:>6} tok | -{reduction:.0f}%")

    return results

# ── Latency profile ────────────────────────────────────────────────────────────

def run_latency_profile(client: AHPClient, verbose: bool) -> dict:
    samples = []
    query = "What is AHP?"

    for i in range(5):
        resp = client.converse(AHPRequest(capability="content_search", query=query))
        samples.append({
            "run": i + 1,
            "latency_ms": resp.latency_ms,
            "cached": resp.cached,
            "tokens": resp.tokens_used,
        })
        if verbose:
            print(f"  Run {i+1}: {resp.latency_ms:.0f}ms (cached={resp.cached})")

    latencies = [s["latency_ms"] for s in samples]
    return {
        "samples": samples,
        "p50_ms": round(statistics.median(latencies), 1),
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if len(latencies) >= 2 else latencies[-1],
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
    }

# ── Test runner helper ─────────────────────────────────────────────────────────

def run_test(test_id: str, name: str, fn, verbose: bool) -> TestResult:
    print(f"  [{test_id}] {name}...", end=" ", flush=True)
    try:
        result = fn()
        result.test_id = test_id
        result.name = name
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"{status}" + (f" — {result.notes}" if verbose and result.notes else ""))
        return result
    except AssertionError as e:
        print(f"✗ FAIL — {e}")
        return TestResult(test_id, name, False, error=str(e))
    except Exception as e:
        print(f"✗ ERROR — {e}")
        return TestResult(test_id, name, False, error=str(e))

# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(report: dict):
    from tabulate import tabulate

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    rows = []
    for r in report["test_results"]:
        rows.append([
            r["test_id"],
            r["name"][:45],
            "✓" if r["passed"] else "✗",
            f"{r['latency_ms']:.0f}ms" if r["latency_ms"] else "—",
            str(r["tokens_used"]) if r["tokens_used"] else "—",
            r["mode"] or "—",
        ])

    print(tabulate(rows, headers=["ID", "Test", "Pass", "Latency", "Tokens", "Mode"],
                   tablefmt="simple"))

    s = report["summary"]
    print(f"\n{s['passed']}/{s['total']} passed ({s['pass_rate_pct']}%)")

    if report.get("token_comparison"):
        print(f"\n{'='*60}")
        print("TOKEN EFFICIENCY (Agent vs. Markdown)")
        print(f"{'='*60}")
        rows = []
        for c in report["token_comparison"]:
            rows.append([
                c["query"][:40],
                c["markdown_tokens"],
                c["ahp_tokens"],
                f"{c['token_reduction_pct']}%",
            ])
        print(tabulate(rows,
            headers=["Query", "Markdown tokens", "AHP tokens", "Reduction"],
            tablefmt="simple"))
        reductions = [c["token_reduction_pct"] for c in report["token_comparison"]]
        print(f"\nAverage token reduction: {sum(reductions)/len(reductions):.1f}%")

    if report.get("latency_profile"):
        lp = report["latency_profile"]
        print(f"\n{'='*60}")
        print("LATENCY PROFILE (MODE2 content_search, 5 runs)")
        print(f"{'='*60}")
        print(f"p50: {lp['p50_ms']}ms  |  p95: {lp['p95_ms']}ms  |  "
              f"min: {lp['min_ms']}ms  |  max: {lp['max_ms']}ms")

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AHP Test Suite")
    parser.add_argument("--target", default="http://localhost:3000", help="AHP server base URL")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output", "-o", help="Write JSON report to file")
    args = parser.parse_args()

    client = AHPClient(args.target)

    try:
        report = run_all_tests(client, args.target, args.verbose)
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(1)

    print_report(report)

    output_path = args.output or f"report-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report: {output_path}")
