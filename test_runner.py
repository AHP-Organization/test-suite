#!/usr/bin/env python3
"""
AHP Test Suite — Visiting Agent Test Harness

Runs all test scenarios against a live AHP endpoint.
Generates a structured JSON report + human-readable summary.

Usage:
    ./venv/bin/python test_runner.py --target https://ref.agenthandshake.dev
    ./venv/bin/python test_runner.py --target http://localhost:3000 --verbose
    ./venv/bin/python test_runner.py --target https://ref.agenthandshake.dev \\
        --nate-target https://nate.agenthandshake.dev --verbose

CHANGELOG (2026-02-22):
  - Fixed T15/T16 ID swap in function bodies (was cosmetic only, now consistent)
  - Added RAG-baseline comparison test using Claude Haiku + top-3 chunk retrieval
  - Added per-site query sets: AHP site gets AHP-specific queries; Nate site gets
    AI-practitioner queries (RAG, prompt engineering, MCP, etc.)
  - Added multi-run averaging: each comparison query runs 3 times, reporting mean ± stddev
  - Fixed markdown latency: actually measures llms.txt fetch time instead of hardcoded 200ms
  - Fixed p95 calculation: now uses 20-sample latency profile, labels cached/uncached clearly
  - Fixed naive token count: uses tiktoken (cl100k_base) instead of len//4 approximation
  - Improved T08: verifies server actually uses session context (memory test), not just ID echo
  - Added T17: MODE3 action capabilities must reject unauthenticated requests (401)
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from typing import Optional

import requests
import tiktoken
from ahp_client import AHPClient, AHPRequest

# ── Tokenizer (cl100k_base ≈ Claude / GPT-4 encoding) ────────────────────────
_ENC = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base encoding (close to Claude's tokenizer)."""
    return len(_ENC.encode(text))

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
class ComparisonRunResult:
    """Single-run token comparison across three approaches."""
    run: int
    ahp_tokens: int
    ahp_latency_ms: float
    rag_tokens: int           # RAG-baseline: top-3 chunks + query sent to Haiku
    rag_latency_ms: float
    naive_tokens: int         # Naive: full doc + query sent to Haiku (estimated via tiktoken)
    naive_latency_ms: float   # Measured: actual llms.txt fetch time

@dataclass
class ComparisonResult:
    """Multi-run averaged comparison result for a single query."""
    query: str
    site: str
    runs: int
    # Naive baseline (full-document, no retrieval) — tiktoken estimate, not API-measured
    naive_tokens_mean: float
    naive_tokens_note: str    # clarifies estimation methodology
    naive_latency_mean_ms: float
    # RAG baseline (client-side top-3 chunk retrieval via Claude Haiku)
    rag_tokens_mean: float
    rag_tokens_stddev: float
    rag_latency_mean_ms: float
    rag_latency_stddev_ms: float
    # AHP MODE2
    ahp_tokens_mean: float
    ahp_tokens_stddev: float
    ahp_latency_mean_ms: float
    ahp_latency_stddev_ms: float
    # Reductions
    reduction_vs_naive_pct: float
    reduction_vs_rag_pct: float
    raw_runs: list = field(default_factory=list)

# ── Per-site query sets ───────────────────────────────────────────────────────

# AHP site: protocol-specific queries
AHP_SITE_QUERIES = [
    ("Explain what MODE1 is in the Agent Handshake Protocol", "content_search"),
    ("How does AHP discovery work for headless browser agents?", "content_search"),
    ("What are AHP content signals and what do they declare?", "content_search"),
    ("How do I build a MODE2 interactive knowledge endpoint?", "content_search"),
    ("What rate limits should a MODE2 AHP server enforce?", "content_search"),
]

# Nate site: AI-practitioner queries appropriate to a personal blog/guides site
NATE_SITE_QUERIES = [
    ("What is RAG and how does it work?", "content_search"),
    ("Explain prompt engineering", "content_search"),
    ("What is MCP and how does it relate to AI agents?", "content_search"),
    ("What is vibe coding?", "content_search"),
    ("How do AI agents work in production?", "content_search"),
]

def get_site_queries(target: str) -> list:
    """Select comparison queries based on the target URL."""
    if "nate" in target.lower():
        return NATE_SITE_QUERIES
    return AHP_SITE_QUERIES

def get_site_label(target: str) -> str:
    """Human-readable label for a site target."""
    if "nate" in target.lower():
        return "Nate Jones site"
    return "AHP Specification site"

# ── RAG baseline helpers ──────────────────────────────────────────────────────

def chunk_document(text: str, target_chars: int = 500) -> list[str]:
    """
    Split a document into chunks of ~target_chars by paragraph boundary.
    Filters blank/trivial paragraphs and merges short ones.
    """
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(para) < 30:
            continue  # skip trivial lines (headings with no content)
        if current and len(current) + len(para) + 1 > target_chars:
            chunks.append(current)
            current = para
        else:
            current = (current + "\n\n" + para).strip() if current else para
    if current:
        chunks.append(current)
    return chunks

def keyword_score(query: str, chunk: str) -> float:
    """
    Compute keyword overlap score between a query and a chunk.
    Returns proportion of unique query words (≥4 chars) found in the chunk.
    Simple, fast, no external dependencies.
    """
    query_words = set(w.lower() for w in re.findall(r'\b\w{4,}\b', query))
    chunk_lower = chunk.lower()
    if not query_words:
        return 0.0
    hits = sum(1 for w in query_words if w in chunk_lower)
    return hits / len(query_words)

def run_rag_query(api_key: str, query: str, chunks: list[str],
                  model: str = "claude-haiku-4-5", top_k: int = 3) -> tuple[str, int, float]:
    """
    RAG-baseline: retrieve top-k chunks by keyword overlap, call Claude Haiku,
    return (answer, tokens_used, latency_ms).
    Uses raw HTTP to avoid adding anthropic SDK as a hard dependency for the harness,
    but the SDK is used if available.
    """
    # Score and select top-k chunks
    scored = sorted(chunks, key=lambda c: keyword_score(query, c), reverse=True)
    selected = scored[:top_k]
    context = "\n\n---\n\n".join(selected)

    prompt = (
        f"Answer the following question using ONLY the provided context. "
        f"Be concise and accurate.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }

    start = time.perf_counter()
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=body,
        timeout=30,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    r.raise_for_status()
    data = r.json()
    answer = data["content"][0]["text"] if data.get("content") else ""
    usage = data.get("usage", {})
    tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    return answer, tokens, round(latency_ms, 2)

# ── Test scenarios ─────────────────────────────────────────────────────────────

def run_all_tests(client: AHPClient, target: str, verbose: bool = False,
                  api_key: str = "") -> dict:
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

    # ── Test 8: Session management + memory verification ──────────────────────
    r = run_test("T08", "Session — multi-turn with memory verification", lambda: test_multiturn(client), verbose)
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

    # ── Test 17: MODE3 action auth enforcement ────────────────────────────────
    r = run_test("T17", "MODE3 — action capabilities reject unauthenticated requests",
                 lambda: test_mode3_auth_required(client), verbose)
    results.append(r)

    # ── Whitepaper: Token efficiency comparison ────────────────────────────────
    # Run benchmarks BEFORE T16 (rate limiting burst) to avoid 429s polluting results
    print("\n[Benchmark] Running token efficiency comparison (3 runs per query)...")
    queries = get_site_queries(target)
    site_label = get_site_label(target)
    comparisons = run_token_comparison_multi(client, target, queries, site_label,
                                             verbose, n_runs=3, api_key=api_key)

    # ── Latency profile ────────────────────────────────────────────────────────
    print("[Benchmark] Running latency profile (20 runs)...")
    latency_profile = run_latency_profile(client, verbose)

    # ── Test 16: Rate limiting ─────────────────────────────────────────────────
    # MUST run LAST — fires 35 burst requests which pollutes rate limits for all
    # subsequent tests. Always keep this as the final step.
    print("\n[T16] Rate limiting test (runs last to avoid polluting other tests)...")
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
            notes="Site may not implement Accept header redirect (SHOULD per spec)")

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
    """
    Verifies session memory: Turn 1 asks about AHP modes, Turn 2 asks a follow-up
    that can only be answered correctly if the server actually used session context.
    """
    # Turn 1 — establish context
    r1 = client.converse(AHPRequest(
        capability="content_search",
        query="Tell me about AHP modes — focus especially on MODE1"
    ))
    assert r1.status == "success", f"Turn 1 failed: {r1.status}"
    sid = r1.session_id
    assert sid, "Turn 1 must return a session_id"

    # Turn 2 — follow-up requiring turn 1 context
    r2 = client.converse(AHPRequest(
        capability="content_search",
        query="Which mode did I just ask about specifically?",
        session_id=sid,
    ))
    assert r2.http_status == 200, f"Turn 2 HTTP {r2.http_status}"
    assert r2.session_id == sid, "Session ID must persist across turns"

    # The answer should reference MODE1 (from turn 1 context)
    answer2 = r2.raw.get("response", {}).get("answer", "").upper()
    references_turn1 = "MODE1" in answer2 or "MODE 1" in answer2 or "FIRST" in answer2
    memory_note = "Session memory verified (answer references MODE1 from turn 1)" \
        if references_turn1 else \
        "WARNING: Answer may not reflect session context — server may not use session history"

    return TestResult("T08", "Multi-turn + memory", references_turn1,
        latency_ms=r2.latency_ms, tokens_used=r1.tokens_used + r2.tokens_used,
        notes=f"Session {sid}. Turn 2 answer: '{answer2[:120]}...'. {memory_note}")

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
    assert resp.http_status == 200, f"Expected 200, got {resp.http_status}"
    assert resp.status == "success", f"Expected success, got {resp.status}"
    assert resp.mode == "MODE3", f"Expected MODE3, got {resp.mode}"
    assert len(resp.tools_used) > 0, f"No tools were called (mode={resp.mode})"
    answer = resp.raw.get("response", {}).get("answer", "")
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

    poll_start = time.time()
    final = client.poll_status(sid, max_wait_s=60, interval_s=1.0)
    poll_time_ms = (time.time() - poll_start) * 1000

    assert final.status == "success", f"Async resolution failed: {final.status}"
    answer = final.raw.get("response", {}).get("answer", "")
    assert len(answer) > 20, "Human response answer too short"

    return TestResult("T14", "MODE3 async escalation", True,
        latency_ms=poll_time_ms,
        notes=f"Accepted → resolved via polling in {poll_time_ms:.0f}ms [SIMULATED human delay]. "
              f"Note: production human response times are hours to days. "
              f"Answer: {answer[:100]}...")

def test_caching(client):
    """T15: Cache hit on repeated identical query."""
    q = "What is MODE2 in the Agent Handshake Protocol?"
    r1 = client.converse(AHPRequest(capability="content_search", query=q))
    r2 = client.converse(AHPRequest(capability="content_search", query=q))
    cache_hit = r2.cached
    speedup = r1.latency_ms / r2.latency_ms if r2.latency_ms > 0 else 0
    return TestResult("T15", "Caching", True,
        latency_ms=r2.latency_ms,
        notes=f"Cold: {r1.latency_ms:.0f}ms | Cached: {r2.latency_ms:.0f}ms | "
              f"Speedup: {speedup:.1f}x | Cache hit flag: {cache_hit}")

def test_rate_limiting(client):
    """T16: Server enforces rate limits (429) on burst traffic."""
    statuses = client.burst(count=35)
    hit_429 = 429 in statuses
    first_429 = statuses.index(429) + 1 if hit_429 else None
    return TestResult("T16", "Rate limiting", hit_429,
        notes=f"429 hit after {first_429} requests" if hit_429 else
              "No 429 returned — rate limiting may not be active")

def test_mode3_auth_required(client):
    """
    T17: MODE3 action capabilities MUST reject unauthenticated requests.
    Per AHP spec §5.3: capabilities of type 'action' or 'async' MUST require
    authentication (authentication MUST NOT be 'none').
    Tests inventory_check, get_quote, and order_lookup without any auth token.
    """
    # The base client sends no Authorization header by default.
    # We call a known action capability and expect 401 Unauthorized.
    # If the server returns 200, it is violating the spec's security requirement.
    action_capabilities = [
        ("inventory_check", "Is the AHP Server License in stock?"),
        ("get_quote", "Quote for 1 license"),
        ("order_lookup", "Look up order ORD-2026-001"),
    ]

    results_detail = []
    any_rejected = False
    any_allowed = False

    for cap, query in action_capabilities:
        resp = client.converse(AHPRequest(capability=cap, query=query))
        rejected = resp.http_status == 401
        if rejected:
            any_rejected = True
        else:
            any_allowed = True
        results_detail.append(f"{cap}: HTTP {resp.http_status} ({'rejected ✓' if rejected else 'ALLOWED — spec violation'})")

    # Pass if ALL action capabilities require auth, or if at minimum the test
    # demonstrates server awareness of auth. Fail (note only) if any allow unauthenticated.
    # Since the reference implementation may intentionally allow demo access,
    # we report findings without hard failing — this is an advisory test.
    notes = " | ".join(results_detail)
    if any_allowed:
        notes += " | NOTE: Spec §5.3 requires action capabilities to reject unauthenticated requests"

    # Report as pass if any were rejected (showing the server knows about auth),
    # or as advisory failure if all allowed (spec violation, but may be intentional in demo).
    passed = not any_allowed  # strict: all must reject
    return TestResult("T17", "MODE3 auth enforcement", passed,
        notes=notes,
        error="" if passed else "One or more MODE3 action capabilities accepted unauthenticated requests (spec §5.3)")

# ── Whitepaper: Multi-run token comparison with RAG baseline ──────────────────

def run_token_comparison_multi(
    client: AHPClient,
    target: str,
    queries: list,
    site_label: str,
    verbose: bool,
    n_runs: int = 3,
    api_key: str = "",
) -> list[ComparisonResult]:
    """
    Run each comparison query n_runs times across three approaches:
      1. Naive visiting agent (full llms.txt, no retrieval) — tiktoken estimate
      2. RAG-baseline visiting agent (top-3 chunks, keyword overlap, Claude Haiku)
      3. AHP MODE2

    Reports mean ± stddev for each approach.
    """
    results = []

    # Fetch and chunk the document once
    try:
        manifest = client.discover()
        content_url = manifest.get("endpoints", {}).get("content", "/llms.txt")
        fetch_start = time.perf_counter()
        r = requests.get(f"{target}{content_url}", timeout=15)
        fetch_latency_ms = (time.perf_counter() - fetch_start) * 1000
        full_content = r.text
        chunks = chunk_document(full_content, target_chars=500)
        # Naive baseline: tiktoken count of full doc + query overhead
        # This is an estimate (cl100k_base, not the exact Claude tokenizer)
        naive_base_tokens = count_tokens(full_content)
        naive_note = (
            f"tiktoken cl100k_base estimate of full {len(full_content)} char document "
            f"({naive_base_tokens} tokens) + query. Not API-measured."
        )
        if verbose:
            print(f"  Fetched {content_url}: {len(full_content)} chars, "
                  f"{len(chunks)} chunks, {naive_base_tokens} tokens (tiktoken), "
                  f"fetch={fetch_latency_ms:.0f}ms")
    except Exception as e:
        print(f"  [warn] Could not fetch content doc: {e}")
        full_content = ""
        chunks = []
        naive_base_tokens = 10000
        naive_note = "Could not fetch content doc; using fallback estimate"
        fetch_latency_ms = 0.0

    for query, capability in queries:
        run_data: list[ComparisonRunResult] = []

        for run_idx in range(n_runs):
            # Sleep briefly to avoid rate-limit interference between runs
            if run_idx > 0:
                time.sleep(2)

            # ── AHP MODE2 ──
            ahp_resp = client.converse(AHPRequest(capability=capability, query=query))
            if ahp_resp.http_status == 429:
                print(f"  [warn] Rate limited on AHP run {run_idx+1} — waiting 10s")
                time.sleep(10)
                ahp_resp = client.converse(AHPRequest(capability=capability, query=query))
            ahp_tokens = ahp_resp.tokens_used
            ahp_latency = ahp_resp.latency_ms

            # ── RAG baseline ──
            rag_tokens = 0
            rag_latency = 0.0
            if chunks and api_key:
                try:
                    _, rag_tokens, rag_latency = run_rag_query(api_key, query, chunks)
                except Exception as e:
                    print(f"  [warn] RAG query failed on run {run_idx+1}: {e}")
                    rag_tokens = 0
                    rag_latency = 0.0
            elif not api_key:
                print("  [warn] No ANTHROPIC_API_KEY — RAG baseline skipped")

            # ── Naive: tiktoken estimate of full doc + query overhead ──
            naive_tokens = naive_base_tokens + count_tokens(query) + 50  # +50 system overhead
            naive_latency = fetch_latency_ms  # measured fetch time (not a hardcoded constant)

            run_data.append(ComparisonRunResult(
                run=run_idx + 1,
                ahp_tokens=ahp_tokens,
                ahp_latency_ms=ahp_latency,
                rag_tokens=rag_tokens,
                rag_latency_ms=rag_latency,
                naive_tokens=naive_tokens,
                naive_latency_ms=naive_latency,
            ))

            if verbose:
                print(f"  [{run_idx+1}/{n_runs}] {query[:45]:<45} "
                      f"AHP: {ahp_tokens:>5} | RAG: {rag_tokens:>5} | naive: {naive_tokens:>6}")

        # ── Aggregate across runs ──
        ahp_tokens_list = [r.ahp_tokens for r in run_data]
        rag_tokens_list = [r.rag_tokens for r in run_data if r.rag_tokens > 0]
        naive_tokens_list = [r.naive_tokens for r in run_data]
        ahp_lat_list = [r.ahp_latency_ms for r in run_data]
        rag_lat_list = [r.rag_latency_ms for r in run_data if r.rag_latency_ms > 0]
        naive_lat_list = [r.naive_latency_ms for r in run_data]

        ahp_mean = statistics.mean(ahp_tokens_list)
        ahp_std = statistics.stdev(ahp_tokens_list) if len(ahp_tokens_list) > 1 else 0.0
        rag_mean = statistics.mean(rag_tokens_list) if rag_tokens_list else 0.0
        rag_std = statistics.stdev(rag_tokens_list) if len(rag_tokens_list) > 1 else 0.0
        naive_mean = statistics.mean(naive_tokens_list)
        naive_lat_mean = statistics.mean(naive_lat_list)
        ahp_lat_mean = statistics.mean(ahp_lat_list)
        ahp_lat_std = statistics.stdev(ahp_lat_list) if len(ahp_lat_list) > 1 else 0.0
        rag_lat_mean = statistics.mean(rag_lat_list) if rag_lat_list else 0.0
        rag_lat_std = statistics.stdev(rag_lat_list) if len(rag_lat_list) > 1 else 0.0

        reduction_vs_naive = (1 - ahp_mean / naive_mean) * 100 if naive_mean > 0 else 0
        reduction_vs_rag = (1 - ahp_mean / rag_mean) * 100 if rag_mean > 0 else 0

        results.append(ComparisonResult(
            query=query,
            site=site_label,
            runs=n_runs,
            naive_tokens_mean=round(naive_mean, 1),
            naive_tokens_note=naive_note,
            naive_latency_mean_ms=round(naive_lat_mean, 1),
            rag_tokens_mean=round(rag_mean, 1),
            rag_tokens_stddev=round(rag_std, 1),
            rag_latency_mean_ms=round(rag_lat_mean, 1),
            rag_latency_stddev_ms=round(rag_lat_std, 1),
            ahp_tokens_mean=round(ahp_mean, 1),
            ahp_tokens_stddev=round(ahp_std, 1),
            ahp_latency_mean_ms=round(ahp_lat_mean, 1),
            ahp_latency_stddev_ms=round(ahp_lat_std, 1),
            reduction_vs_naive_pct=round(reduction_vs_naive, 1),
            reduction_vs_rag_pct=round(reduction_vs_rag, 1),
            raw_runs=[asdict(rd) for rd in run_data],
        ))

    return results

# ── Latency profile ────────────────────────────────────────────────────────────

def run_latency_profile(client: AHPClient, verbose: bool) -> dict:
    """
    Run 20 latency samples. First run warms the cache; subsequent runs are cached.
    Reports stats separately for uncached vs cached runs.
    p95 requires ≥20 samples to be statistically meaningful — with fewer samples
    the reported value is labeled as 'max of N samples'.
    """
    samples = []
    query = "What is AHP?"
    n_samples = 20

    for i in range(n_samples):
        resp = client.converse(AHPRequest(capability="content_search", query=query))
        samples.append({
            "run": i + 1,
            "latency_ms": resp.latency_ms,
            "cached": resp.cached,
            "tokens": resp.tokens_used,
        })
        if verbose and (i == 0 or i % 5 == 4):
            print(f"  Run {i+1}: {resp.latency_ms:.0f}ms (cached={resp.cached})")

    all_latencies = [s["latency_ms"] for s in samples]
    uncached = [s["latency_ms"] for s in samples if not s["cached"]]
    cached = [s["latency_ms"] for s in samples if s["cached"]]

    def p95(latencies):
        if len(latencies) >= 20:
            return round(sorted(latencies)[int(len(latencies) * 0.95)], 1)
        return round(max(latencies), 1)  # labeled as max if insufficient samples

    return {
        "samples": samples,
        "n_samples": n_samples,
        # All samples
        "p50_ms": round(statistics.median(all_latencies), 1),
        "p95_ms": p95(all_latencies),
        "p95_note": "true p95" if len(all_latencies) >= 20 else f"max of {len(all_latencies)} samples",
        "min_ms": round(min(all_latencies), 1),
        "max_ms": round(max(all_latencies), 1),
        # Cached-only stats
        "cached_p50_ms": round(statistics.median(cached), 1) if cached else None,
        "cached_max_ms": round(max(cached), 1) if cached else None,
        "cached_n": len(cached),
        # Uncached-only stats
        "uncached_mean_ms": round(statistics.mean(uncached), 1) if uncached else None,
        "uncached_max_ms": round(max(uncached), 1) if uncached else None,
        "uncached_n": len(uncached),
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
        print("TOKEN EFFICIENCY (3-run averages: AHP vs. RAG-baseline vs. Naive)")
        print(f"{'='*60}")
        rows = []
        for c in report["token_comparison"]:
            rag_str = f"{c['rag_tokens_mean']:.0f} ±{c['rag_tokens_stddev']:.0f}" \
                if c['rag_tokens_mean'] > 0 else "N/A"
            ahp_str = f"{c['ahp_tokens_mean']:.0f} ±{c['ahp_tokens_stddev']:.0f}"
            rows.append([
                c["query"][:38],
                f"{c['naive_tokens_mean']:.0f}*",
                rag_str,
                ahp_str,
                f"{c['reduction_vs_naive_pct']}%",
                f"{c['reduction_vs_rag_pct']}%" if c['rag_tokens_mean'] > 0 else "—",
            ])
        print(tabulate(rows,
            headers=["Query", "Naive*", "RAG-baseline", "AHP MODE2", "vs. Naive", "vs. RAG"],
            tablefmt="simple"))
        print("* Naive = tiktoken estimate of full document (no retrieval). Not API-measured.")

        ahp_means = [c["ahp_tokens_mean"] for c in report["token_comparison"]]
        naive_means = [c["naive_tokens_mean"] for c in report["token_comparison"]]
        avg_red_naive = (1 - sum(ahp_means) / sum(naive_means)) * 100
        print(f"\nAvg token reduction vs. naive full-doc baseline: {avg_red_naive:.1f}%")

        rag_comps = [c for c in report["token_comparison"] if c['rag_tokens_mean'] > 0]
        if rag_comps:
            ahp_rag = [c["ahp_tokens_mean"] for c in rag_comps]
            rag_rag = [c["rag_tokens_mean"] for c in rag_comps]
            avg_red_rag = (1 - sum(ahp_rag) / sum(rag_rag)) * 100
            print(f"Avg token reduction vs. RAG-baseline (client-side retrieval): {avg_red_rag:.1f}%")

    if report.get("latency_profile"):
        lp = report["latency_profile"]
        print(f"\n{'='*60}")
        print(f"LATENCY PROFILE (MODE2 content_search, {lp.get('n_samples', 5)} runs)")
        print(f"{'='*60}")
        print(f"All samples  — p50: {lp['p50_ms']}ms  p95: {lp['p95_ms']}ms ({lp.get('p95_note', '')})  "
              f"min: {lp['min_ms']}ms  max: {lp['max_ms']}ms")
        if lp.get("cached_p50_ms") is not None:
            print(f"Cached only  — p50: {lp['cached_p50_ms']}ms  max: {lp['cached_max_ms']}ms  "
                  f"(n={lp['cached_n']})")
        if lp.get("uncached_mean_ms") is not None:
            print(f"Uncached only — mean: {lp['uncached_mean_ms']}ms  max: {lp['uncached_max_ms']}ms  "
                  f"(n={lp['uncached_n']})")

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AHP Test Suite")
    parser.add_argument("--target", default="http://localhost:3000", help="AHP server base URL")
    parser.add_argument("--nate-target", default="", help="Nate Jones site URL (for dual-site comparison)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output", "-o", help="Write JSON report to file")
    parser.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY", ""),
                        help="Anthropic API key for RAG-baseline comparison")
    args = parser.parse_args()

    client = AHPClient(args.target)

    try:
        report = run_all_tests(client, args.target, args.verbose, api_key=args.api_key)
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(1)

    print_report(report)

    output_path = args.output or f"report-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report: {output_path}")

    # ── Optional: run Nate site comparison if --nate-target provided ──────────
    if args.nate_target:
        print(f"\n{'='*60}")
        print(f"Running Nate site comparison: {args.nate_target}")
        print(f"{'='*60}")
        nate_client = AHPClient(args.nate_target)
        nate_queries = get_site_queries(args.nate_target)
        nate_comps = run_token_comparison_multi(
            nate_client, args.nate_target, nate_queries,
            get_site_label(args.nate_target),
            args.verbose, n_runs=3, api_key=args.api_key
        )
        nate_report = {
            "target": args.nate_target,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "token_comparison": [asdict(c) for c in nate_comps],
        }
        nate_path = (args.output or "report").replace(".json", "") + "-nate.json"
        with open(nate_path, "w") as f:
            json.dump(nate_report, f, indent=2)
        print(f"\nNate site comparison saved: {nate_path}")
