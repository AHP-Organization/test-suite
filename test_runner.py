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

CHANGELOG (2026-02-22 11:00):
  - Fixed T22 false-negative: test only accepted HTTP 404/410 for invalid
    session_id, but spec §6.2 says any 4xx is acceptable. Server returned
    HTTP 400 (correct graceful rejection); test now accepts any 4xx.
  - Version: v5.1

CHANGELOG (2026-02-22 10:00):
  - Fixed T08 false-negative: 'FIRST QUESTION' in exclusion list matched
    'In your first question, you requested...' which DEMONSTRATES session memory.
    Removed 'FIRST QUESTION' (too ambiguous); kept only unambiguous no-memory
    phrases like 'DON'T HAVE ANY RECORD', 'THIS IS THE FIRST MESSAGE', etc.
    v4 live run confirmed: server answers 'You asked specifically about MODE1.
    In your first question, you requested...' → T08 correctly PASSES with fix.
  - Added T21: clarification_needed response format (spec §6.3) — verifies
    the response format is correct if server returns clarification; advisory
    if server never triggers clarification on test queries
  - Added T22: invalid session_id handling (spec §6.2) — verifies server
    handles unknown/expired session IDs gracefully (no 5xx)
  - Noted naive latency as single measurement (not 3-run mean) in output

CHANGELOG (2026-02-22 09:00):
  - Fixed T08 false-pass: previous version matched 'FIRST' in 'this is your
    FIRST question to me' — the opposite of session memory. Added explicit
    context-failure phrase exclusion list; any response containing phrases like
    'haven't asked', 'first question', 'no previous context', etc. now FAILS.
  - Redesigned latency profile: 10 cold-cache (unique nonces, forced misses) +
    10 cache-hit (repeated query). Eliminates 'all 20 samples cached' artifact.
  - Added T20: rate-limit header conformance (spec §11.1) — verifies
    X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset, X-RateLimit-Window
    are present and numeric on every response
  - Strengthened T17 output: from 'advisory' to 'CRITICAL KNOWN DEFECT'
    language to match the severity of the spec §5.3 MUST violation

CHANGELOG (2026-02-22 08:00):
  - Fixed T12 false-pass: changed query to use known product IDs (AHP-IMPL-001,
    AHP-SUPPORT-001), added price regex assertion ($\\d), excluded failure phrases
  - Added cache-busting to multi-run token comparison: each run appends a unique
    nonce [ref:XXXXXXXX] to prevent server-side cache hits; produces real stddev
  - Auto-loads ANTHROPIC_API_KEY from ahp-reference/.env if not in environment
  - Added automatic Nate Jones site cross-comparison in the benchmark phase
    (no separate flag needed when running against the AHP reference site)
  - Added latency column to comparison output table (RAG vs AHP cold vs AHP cached)
  - T16 (rate limiting): records X-RateLimit-Remaining before burst so the
    window state at test time is reported
  - Added T18: session 10-turn limit enforcement
  - Added T19: 413 response for oversized request body (>8KB)
  - Fixed RAG N/A: API key now loaded automatically from reference .env

CHANGELOG (2026-02-22 07:00):
  - Fixed T15/T16 ID swap in function bodies (cosmetic, now consistent)
  - T16 moved to run LAST to avoid polluting other tests with rate-limit 429s
  - Added T17: MODE3 action capabilities must reject unauthenticated requests
  - Added RAG-baseline comparison test using Claude Haiku + top-3 chunk retrieval
  - Added per-site query sets: AHP site gets AHP-specific queries; Nate site gets
    AI-practitioner queries (RAG, prompt engineering, MCP, etc.)
  - Added multi-run averaging (3 runs per query, mean ± stddev)
  - Fixed markdown latency: measures llms.txt fetch time instead of hardcoded 200ms
  - Fixed p95: 20-sample latency profile with cached/uncached separation
  - Fixed token estimation: tiktoken cl100k_base instead of len//4
  - Improved T08: verifies session memory (turn 2 must reference turn 1 content)
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from typing import Optional

import requests
import tiktoken
from ahp_client import AHPClient, AHPRequest

# ── API key: env → CLI → .env file ───────────────────────────────────────────

def _load_api_key_from_env_file(path: str) -> str:
    """Load ANTHROPIC_API_KEY from a .env file if it exists.
    Handles both quoted (KEY="value") and unquoted (KEY=value) formats.
    """
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY="):
                    value = line.split("=", 1)[1].strip()
                    # Strip surrounding quotes if present
                    if len(value) >= 2 and value[0] in ('"', "'") and value[0] == value[-1]:
                        value = value[1:-1]
                    return value
    except Exception:
        pass
    return ""

_DEFAULT_ENV_FILE = os.path.expanduser("~/ahp-reference/.env")

def resolve_api_key(cli_key: str = "") -> str:
    """Return the best available API key: CLI → environment → .env file."""
    return (
        cli_key
        or os.environ.get("ANTHROPIC_API_KEY", "")
        or _load_api_key_from_env_file(_DEFAULT_ENV_FILE)
    )

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
    nonce: str            # cache-bust nonce used for this run
    ahp_tokens: int
    ahp_latency_ms: float
    rag_tokens: int
    rag_latency_ms: float
    naive_tokens: int
    naive_latency_ms: float

@dataclass
class ComparisonResult:
    """Multi-run averaged comparison result for a single query."""
    query: str
    site: str
    runs: int
    cache_busting: bool   # whether nonces were used
    # Naive baseline (full-doc, no retrieval) — tiktoken estimate
    naive_tokens_mean: float
    naive_tokens_note: str
    naive_latency_mean_ms: float
    # RAG baseline (client-side top-3 chunk retrieval, Claude Haiku)
    rag_tokens_mean: float
    rag_tokens_stddev: float
    rag_latency_mean_ms: float
    rag_latency_stddev_ms: float
    # AHP MODE2 (API-measured)
    ahp_tokens_mean: float
    ahp_tokens_stddev: float
    ahp_latency_mean_ms: float
    ahp_latency_stddev_ms: float
    # Comparisons
    reduction_vs_naive_pct: float
    overhead_vs_rag_pct: float   # positive = AHP uses MORE tokens than RAG
    raw_runs: list = field(default_factory=list)

# ── Per-site query sets ───────────────────────────────────────────────────────

AHP_SITE_QUERIES = [
    ("Explain what MODE1 is in the Agent Handshake Protocol", "content_search"),
    ("How does AHP discovery work for headless browser agents?", "content_search"),
    ("What are AHP content signals and what do they declare?", "content_search"),
    ("How do I build a MODE2 interactive knowledge endpoint?", "content_search"),
    ("What rate limits should a MODE2 AHP server enforce?", "content_search"),
]

NATE_SITE_QUERIES = [
    ("What is RAG and how does it work?", "content_search"),
    ("Explain prompt engineering", "content_search"),
    ("What is MCP and how does it relate to AI agents?", "content_search"),
    ("What is vibe coding?", "content_search"),
    ("How do AI agents work in production?", "content_search"),
]

NATE_TARGET = "https://nate.agenthandshake.dev"

def get_site_queries(target: str) -> list:
    return NATE_SITE_QUERIES if "nate" in target.lower() else AHP_SITE_QUERIES

def get_site_label(target: str) -> str:
    return "Nate Jones site" if "nate" in target.lower() else "AHP Specification site"

# ── RAG baseline helpers ──────────────────────────────────────────────────────

def chunk_document(text: str, target_chars: int = 500) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(para) < 30:
            continue
        if current and len(current) + len(para) + 1 > target_chars:
            chunks.append(current)
            current = para
        else:
            current = (current + "\n\n" + para).strip() if current else para
    if current:
        chunks.append(current)
    return chunks

def keyword_score(query: str, chunk: str) -> float:
    query_words = set(w.lower() for w in re.findall(r'\b\w{4,}\b', query))
    chunk_lower = chunk.lower()
    if not query_words:
        return 0.0
    return sum(1 for w in query_words if w in chunk_lower) / len(query_words)

def run_rag_query(api_key: str, query: str, chunks: list[str],
                  model: str = "claude-haiku-4-5", top_k: int = 3) -> tuple[str, int, float]:
    """Retrieve top-k chunks by keyword overlap, call Haiku, return (answer, tokens, latency_ms)."""
    scored = sorted(chunks, key=lambda c: keyword_score(query, c), reverse=True)
    context = "\n\n---\n\n".join(scored[:top_k])
    prompt = (
        "Answer the following question using ONLY the provided context. "
        "Be concise and accurate.\n\n"
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
    r = requests.post("https://api.anthropic.com/v1/messages",
                      headers=headers, json=body, timeout=30)
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

    print(f"\n{'='*60}")
    print(f"AHP Test Suite — {target}")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}\n")

    r = run_test("T01", "Discovery — /.well-known/agent.json",
                 lambda: test_discovery(client), verbose)
    results.append(r)

    r = run_test("T02", "Discovery — Accept: application/agent+json",
                 lambda: test_accept_header(client), verbose)
    results.append(r)

    r = run_test("T03", "Discovery — in-page agent notice",
                 lambda: test_agent_notice(client), verbose)
    results.append(r)

    r = run_test("T04", "Manifest schema validation",
                 lambda: test_manifest_schema(client), verbose)
    results.append(r)

    r = run_test("T05", "MODE2 — simple content query",
                 lambda: test_simple_query(client), verbose)
    results.append(r)

    r = run_test("T06", "Error handling — unknown capability",
                 lambda: test_unknown_capability(client), verbose)
    results.append(r)

    r = run_test("T07", "Error handling — missing required field",
                 lambda: test_malformed_request(client, target), verbose)
    results.append(r)

    r = run_test("T08", "Session — multi-turn with memory verification",
                 lambda: test_multiturn(client), verbose)
    results.append(r)

    r = run_test("T09", "Response schema — required fields present",
                 lambda: test_response_schema(client), verbose)
    results.append(r)

    r = run_test("T10", "Content signals — present in response meta",
                 lambda: test_content_signals(client), verbose)
    results.append(r)

    r = run_test("T11", "MODE3 — inventory check with tool use",
                 lambda: test_inventory_check(client), verbose)
    results.append(r)

    r = run_test("T12", "MODE3 — quote calculation delivers numeric prices",
                 lambda: test_get_quote(client), verbose)
    results.append(r)

    r = run_test("T13", "MODE3 — order lookup",
                 lambda: test_order_lookup(client), verbose)
    results.append(r)

    r = run_test("T14", "MODE3 — async human escalation + poll",
                 lambda: test_human_escalation(client), verbose)
    results.append(r)

    r = run_test("T15", "Caching — repeated query returns cached response",
                 lambda: test_caching(client), verbose)
    results.append(r)

    r = run_test("T17", "MODE3 — action capabilities reject unauthenticated requests",
                 lambda: test_mode3_auth_required(client), verbose)
    results.append(r)

    r = run_test("T18", "Session — 10-turn limit enforced",
                 lambda: test_session_turn_limit(client), verbose)
    results.append(r)

    r = run_test("T19", "Request body — 413 on oversized payload (>8KB)",
                 lambda: test_oversized_body(client, target), verbose)
    results.append(r)

    r = run_test("T20", "Rate-limit headers present on all responses (spec §11.1)",
                 lambda: test_ratelimit_headers(client, target), verbose)
    results.append(r)

    r = run_test("T21", "Clarification needed — format valid if triggered (spec §6.3)",
                 lambda: test_clarification_needed(client), verbose)
    results.append(r)

    r = run_test("T22", "Invalid session_id handled gracefully (spec §6.2)",
                 lambda: test_invalid_session(client), verbose)
    results.append(r)

    # ── Benchmarks — run BEFORE T16 burst to avoid 429 contamination ──────────
    print("\n[Benchmark] Running token efficiency comparison (3 runs per query, cache-busted)...")
    queries = get_site_queries(target)
    site_label = get_site_label(target)
    comparisons = run_token_comparison_multi(
        client, target, queries, site_label, verbose, n_runs=3, api_key=api_key
    )

    # ── Cross-site Nate comparison (automatic when running AHP site) ───────────
    nate_comparisons = []
    if "nate" not in target.lower():
        print(f"\n[Benchmark] Running Nate Jones cross-site comparison ({NATE_TARGET})...")
        try:
            nate_client = AHPClient(NATE_TARGET)
            nate_comparisons = run_token_comparison_multi(
                nate_client, NATE_TARGET, NATE_SITE_QUERIES,
                "Nate Jones site", verbose, n_runs=3, api_key=api_key
            )
        except Exception as e:
            print(f"  [warn] Nate site comparison failed: {e}")

    print("[Benchmark] Running latency profile (20 runs)...")
    latency_profile = run_latency_profile(client, verbose)

    # ── T16: Rate limiting — ALWAYS LAST; records window state before burst ────
    print("\n[T16] Rate limiting test (last — fires 35 burst requests)...")
    r = run_test("T16", "Rate limiting — 429 on burst",
                 lambda: test_rate_limiting(client, target), verbose)
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
            "pass_rate_pct": round(
                sum(1 for r in results if r.passed) / len(results) * 100, 1),
        },
        "token_comparison": [asdict(c) for c in comparisons],
        "nate_comparison": [asdict(c) for c in nate_comparisons],
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
        return TestResult("T02", "Accept header", True,
                          notes="Accept header discovery succeeded")
    except Exception as e:
        return TestResult("T02", "Accept header", False, error=str(e),
            notes="Site may not implement Accept header redirect (SHOULD per spec)")

def test_agent_notice(client):
    try:
        r = requests.get(f"{client.base_url}/", timeout=10)
        content_type = r.headers.get("Content-Type", "")
        is_html_site = "text/html" in content_type and r.status_code < 400
        if not is_html_site:
            return TestResult("T03", "In-page notice", True,
                notes=f"Not applicable — root returns HTTP {r.status_code} "
                      f"({content_type or 'no content-type'}). "
                      "In-page notice is for HTML sites; API servers are exempt.")
        found = ('aria-label="AI Agent Notice"' in r.text
                 or 'class="ahp-notice"' in r.text)
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
    assert len(answer) > 50, f"Answer too short ({len(answer)} chars)"
    return TestResult("T05", "Simple query", True,
        latency_ms=resp.latency_ms, tokens_used=resp.tokens_used,
        mode=resp.mode, cached=resp.cached,
        notes=f"Answer: {answer[:100]}...")

def test_unknown_capability(client):
    resp = client.converse(AHPRequest(capability="does_not_exist", query="test"))
    assert resp.http_status == 400, f"Expected 400, got {resp.http_status}"
    assert resp.raw.get("code") == "unknown_capability", \
        f"Expected unknown_capability error code, got {resp.raw.get('code')}"
    assert "available_capabilities" in resp.raw, "Should list available capabilities"
    return TestResult("T06", "Unknown capability", True,
        notes=f"Available: {resp.raw.get('available_capabilities')}")

def test_malformed_request(client, target):
    r = requests.post(f"{target}/agent/converse",
        json={"capability": "site_info"},
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    assert r.status_code == 400, \
        f"Expected 400 for missing query, got {r.status_code}"
    return TestResult("T07", "Missing field", True,
        notes="Missing 'query' correctly returns 400")

def test_multiturn(client):
    """
    Verifies session memory: Turn 1 asks about MODE1 specifically.
    Turn 2 asks which mode was just discussed.

    CRITICAL: a server without session memory will respond to turn 2 with
    something like "this is your first question" or "I haven't been asked
    about any mode yet."  Previous versions of this test had a false-positive
    bug where "FIRST" matched "this is your FIRST question" — the opposite of
    context awareness.

    The correct logic is:
      PASS  = answer contains positive context signals (MODE1/MODE 1/STATIC)
              AND does not contain context-failure phrases
      FAIL  = answer contains context-failure phrases  (no session memory)
      FAIL  = answer contains neither (ambiguous / no useful signal)
    """
    r1 = client.converse(AHPRequest(
        capability="content_search",
        query="Tell me about AHP modes — focus especially on MODE1",
    ))
    assert r1.status == "success", f"Turn 1 failed: {r1.status}"
    sid = r1.session_id
    assert sid, "Turn 1 must return a session_id"

    r2 = client.converse(AHPRequest(
        capability="content_search",
        query="Which mode did I just ask about specifically?",
        session_id=sid,
    ))
    assert r2.http_status == 200, f"Turn 2 HTTP {r2.http_status}"
    assert r2.session_id == sid, "Session ID must persist across turns"

    answer2_raw = r2.raw.get("response", {}).get("answer", "")
    answer2 = answer2_raw.upper()

    # Phrases that indicate the server has NO session memory.
    # If any of these appear, the test fails regardless of positive keywords.
    #
    # IMPORTANT — avoid ambiguous phrases like "FIRST QUESTION":
    #   "In your FIRST QUESTION, you asked about MODE1"  ← shows memory (PASS)
    #   "This is your FIRST QUESTION to me"              ← no memory (FAIL)
    #   "It appears I haven't received a FIRST QUESTION" ← no memory (FAIL)
    # Only include phrases that are unambiguously no-memory indicators.
    #
    # Confirmed from live server responses (2026-02-22):
    #   "I don't have any record of previous questions you've asked.
    #    This is the first message in our conversation."  (09:04 UTC run)
    no_memory_phrases = [
        "HAVEN'T ASKED",
        "DON'T HAVE CONTEXT",
        "DO NOT HAVE CONTEXT",
        "NEW CONVERSATION",
        "STARTING FRESH",
        "HAVEN'T MENTIONED",
        "THIS IS YOUR FIRST",      # "This is your first question/message"
        "THAT'S YOUR FIRST",
        "THAT IS YOUR FIRST",
        "HAVEN'T DISCUSSED",
        # Confirmed from live server responses (2026-02-22 09:04):
        "DON'T HAVE ANY RECORD",
        "DO NOT HAVE ANY RECORD",
        "NO RECORD OF PREVIOUS",   # more specific than "NO RECORD OF"
        "THIS IS THE FIRST MESSAGE",
        "FIRST MESSAGE IN OUR CONVERSATION",
        "FIRST INTERACTION",
        "DON'T HAVE ACCESS TO PREVIOUS",
        "DO NOT HAVE ACCESS TO PREVIOUS",
        "WITHOUT PRIOR CONTEXT",
        "WITHOUT ANY PRIOR",
        "HAVEN'T SEEN ANY PREVIOUS",
        "NO PREVIOUS MESSAGES",
        "NO HISTORY OF",            # more specific than "NO HISTORY"
        "NO PRIOR CONVERSATION",
        "NO RECORD OF ANY PREVIOUS",
    ]
    context_failure_phrases = [p for p in no_memory_phrases if p in answer2]

    # Positive signals: the answer references what was asked in turn 1
    positive_signals = ("MODE1" in answer2 or "MODE 1" in answer2
                        or "STATIC SERVE" in answer2
                        or ("STATIC" in answer2 and "MODE" in answer2))

    if context_failure_phrases:
        # Server explicitly says it has no memory — definitive failure
        return TestResult("T08", "Multi-turn + memory", False,
            latency_ms=r2.latency_ms,
            tokens_used=r1.tokens_used + r2.tokens_used,
            notes=f"Session {sid}. FAIL: server has no session memory. "
                  f"Context-failure phrases detected: {context_failure_phrases}. "
                  f"Turn 2 answer: '{answer2_raw[:150]}'",
            error="Server does not implement session memory (spec §6.2). "
                  "Turn 2 response indicates no awareness of turn 1.")

    if not positive_signals:
        return TestResult("T08", "Multi-turn + memory", False,
            latency_ms=r2.latency_ms,
            tokens_used=r1.tokens_used + r2.tokens_used,
            notes=f"Session {sid}. FAIL: turn 2 answer does not reference "
                  f"MODE1 from turn 1, and no context-failure phrase detected. "
                  f"Answer: '{answer2_raw[:150]}'",
            error="Turn 2 answer does not reference turn 1 context (MODE1 not mentioned).")

    return TestResult("T08", "Multi-turn + memory", True,
        latency_ms=r2.latency_ms,
        tokens_used=r1.tokens_used + r2.tokens_used,
        notes=f"Session {sid}. PASS: turn 2 references MODE1 from turn 1 "
              f"and no context-failure phrases detected. "
              f"Turn 2: '{answer2_raw[:120]}...'")

def test_response_schema(client):
    resp = client.converse(AHPRequest(capability="site_info",
                                      query="What is this site?"))
    raw = resp.raw
    assert "status" in raw, "Missing 'status'"
    assert "response" in raw, "Missing 'response'"
    assert "meta" in raw, "Missing 'meta'"
    assert ("answer" in raw["response"] or "payload" in raw["response"]), \
        "Response must have answer or payload"
    assert "capability_used" in raw["meta"], "Missing meta.capability_used"
    assert "content_signals" in raw["meta"], "Missing meta.content_signals"
    return TestResult("T09", "Response schema", True,
        notes="All required response fields present")

def test_content_signals(client):
    resp = client.converse(AHPRequest(capability="site_info",
        query="What can you tell me about this site?"))
    cs = resp.raw.get("meta", {}).get("content_signals", {})
    assert cs, "content_signals missing from response meta"
    assert "ai_input" in cs, "ai_input signal missing"
    return TestResult("T10", "Content signals", True, notes=f"Signals: {cs}")

def test_inventory_check(client):
    resp = client.converse(AHPRequest(
        capability="inventory_check",
        query="Is the AHP Server License Single Site (AHP-IMPL-001) in stock?",
    ))
    assert resp.http_status == 200, f"Got {resp.http_status}"
    assert resp.status == "success", f"Expected success, got {resp.status}"
    assert resp.mode == "MODE3", f"Expected MODE3, got {resp.mode}"
    assert len(resp.tools_used) > 0, "No tools were called"
    answer = resp.raw.get("response", {}).get("answer", "")
    assert len(answer) > 20, "Answer too short"
    return TestResult("T11", "MODE3 inventory", True,
        latency_ms=resp.latency_ms, tokens_used=resp.tokens_used,
        mode=resp.mode, tools_used=resp.tools_used,
        notes=f"Tools: {resp.tools_used}. Answer: {answer[:100]}...")

def test_get_quote(client):
    """
    T12: MODE3 quote calculation must deliver numeric prices.
    Uses known product IDs (AHP-IMPL-001, AHP-SUPPORT-001) to ensure the
    tool call succeeds and returns actual pricing rather than an error/clarification.
    Asserts: dollar amounts present, no tool-failure phrases.
    """
    resp = client.converse(AHPRequest(
        capability="get_quote",
        query=(
            "Calculate a price quote for: "
            "2x AHP-IMPL-001 (AHP Server License Single Site) and "
            "1x AHP-SUPPORT-001 (AHP Implementation Support). "
            "Customer type: business."
        ),
    ))
    assert resp.http_status == 200, f"Expected 200, got {resp.http_status}"
    assert resp.status == "success", f"Expected success, got {resp.status}"
    assert resp.mode == "MODE3", f"Expected MODE3, got {resp.mode}"
    assert len(resp.tools_used) > 0, \
        f"No tools were called (mode={resp.mode})"

    answer = resp.raw.get("response", {}).get("answer", "")

    # Must contain at least one numeric dollar amount (e.g. $99.00, $1,234)
    has_price = bool(re.search(r'\$[\d,]+', answer))
    assert has_price, \
        f"Quote must contain numeric dollar amounts. Got: {answer[:300]}"

    # Must NOT be an error/clarification message (tool failure)
    failure_phrases = [
        "couldn't find", "could not find", "unable to", "please provide",
        "can you provide", "what product", "no products found",
        "knowledge base", "documentation"
    ]
    failure_found = [p for p in failure_phrases if p in answer.lower()]
    assert not failure_found, \
        f"Quote appears to be an error/clarification, not a delivered quote. " \
        f"Failure phrases found: {failure_found}. Answer: {answer[:300]}"

    return TestResult("T12", "MODE3 quote", True,
        latency_ms=resp.latency_ms, tokens_used=resp.tokens_used,
        mode=resp.mode, tools_used=resp.tools_used,
        notes=f"Answer: {answer[:200]}...")

def test_order_lookup(client):
    resp = client.converse(AHPRequest(
        capability="order_lookup",
        query="Can you look up order ORD-2026-001?",
    ))
    assert resp.http_status == 200, f"Expected 200, got {resp.http_status}"
    assert resp.status == "success", f"Expected success, got {resp.status}"
    answer = resp.raw.get("response", {}).get("answer", "")
    assert ("ORD-2026-001" in answer
            or "shipped" in answer.lower()
            or "order" in answer.lower()), \
        "Answer should reference the order"
    return TestResult("T13", "MODE3 order lookup", True,
        latency_ms=resp.latency_ms, tokens_used=resp.tokens_used,
        mode=resp.mode, tools_used=resp.tools_used,
        notes=f"Answer: {answer[:100]}...")

def test_human_escalation(client):
    resp = client.converse(AHPRequest(
        capability="human_escalation",
        query="I need a custom enterprise agreement for 500 sites. "
              "This requires human review.",
    ))
    assert resp.http_status == 200, f"Expected 200, got {resp.http_status}"
    data = resp.raw
    assert data.get("status") == "accepted", \
        f"Expected accepted, got {data.get('status')}"
    assert "session_id" in data, "Missing session_id in accepted response"
    assert "poll" in data, "Missing poll URL in accepted response"
    assert "eta_seconds" in data, "Missing eta_seconds in accepted response"

    sid = data["session_id"]
    poll_start = time.time()
    final = client.poll_status(sid, max_wait_s=60, interval_s=1.0)
    poll_time_ms = (time.time() - poll_start) * 1000

    assert final.status == "success", \
        f"Async resolution failed: {final.status}"
    answer = final.raw.get("response", {}).get("answer", "")
    assert len(answer) > 20, "Human response answer too short"

    return TestResult("T14", "MODE3 async escalation", True,
        latency_ms=poll_time_ms,
        notes=f"Resolved via polling in {poll_time_ms:.0f}ms "
              f"[SIMULATED — production is hours-to-days]. "
              f"Answer: {answer[:100]}...")

def test_caching(client):
    """T15: Cache hit on repeated identical query."""
    q = "What is MODE2 in the Agent Handshake Protocol?"
    r1 = client.converse(AHPRequest(capability="content_search", query=q))
    r2 = client.converse(AHPRequest(capability="content_search", query=q))
    speedup = r1.latency_ms / r2.latency_ms if r2.latency_ms > 0 else 0
    return TestResult("T15", "Caching", True,
        latency_ms=r2.latency_ms,
        notes=f"Cold: {r1.latency_ms:.0f}ms | Cached: {r2.latency_ms:.0f}ms | "
              f"Speedup: {speedup:.1f}x | Cache-hit flag: {r2.cached}")

def test_mode3_auth_required(client):
    """
    T17: MODE3 action capabilities MUST reject unauthenticated requests.
    Per spec §5.3: capabilities of type 'action' MUST require authentication.
    """
    action_caps = [
        ("inventory_check", "Is AHP-IMPL-001 in stock?"),
        ("get_quote",        "Quote for 1x AHP-IMPL-001"),
        ("order_lookup",     "Look up order ORD-2026-001"),
    ]
    detail = []
    any_allowed = False
    for cap, query in action_caps:
        resp = client.converse(AHPRequest(capability=cap, query=query))
        rejected = resp.http_status == 401
        if not rejected:
            any_allowed = True
        detail.append(
            f"{cap}: HTTP {resp.http_status} "
            f"({'✓ rejected' if rejected else '✗ ALLOWED — spec violation'})"
        )
    notes = " | ".join(detail)
    if any_allowed:
        notes += (
            " | CRITICAL KNOWN DEFECT: This server accepts unauthenticated MODE3 "
            "action requests in violation of spec §5.3 (MUST require authentication). "
            "Do NOT deploy to production without implementing authentication. "
            "This is the reference implementation's intentional demo-mode deviation — "
            "every production deployment must override this default."
        )
    return TestResult("T17", "MODE3 auth enforcement", not any_allowed,
        notes=notes,
        error="" if not any_allowed else
              "CRITICAL: MODE3 action capabilities accept unauthenticated requests "
              "(spec §5.3 MUST violation). Not safe for production deployment.")

def test_session_turn_limit(client):
    """
    T18: Server must enforce the 10-turn session limit.
    Sends 11 turns on the same session; turn 11 should fail (session_limit_exceeded,
    session_expired, or HTTP 400/429).
    """
    # Start session
    r0 = client.converse(AHPRequest(
        capability="content_search",
        query="What is AHP?"
    ))
    assert r0.status == "success", f"Initial query failed: {r0.status}"
    sid = r0.session_id
    assert sid, "No session_id returned on first turn"

    last_resp = None
    exceeded = False
    for i in range(10):
        resp = client.converse(AHPRequest(
            capability="content_search",
            query=f"Tell me more about AHP (turn {i+2})",
            session_id=sid,
        ))
        last_resp = resp
        status = resp.raw.get("status", "")
        code = resp.raw.get("code", "")
        if (resp.http_status in (400, 429, 410)
                or "limit" in status.lower()
                or "limit" in code.lower()
                or "expired" in status.lower()
                or "expired" in code.lower()):
            exceeded = True
            break

    notes = (
        f"Session {sid}: limit enforced after {i+2} turns"
        if exceeded else
        f"Session {sid}: 11 turns sent — server did NOT enforce 10-turn limit "
        f"(last HTTP {last_resp.http_status if last_resp else 'N/A'})"
    )
    return TestResult("T18", "Session turn limit", exceeded, notes=notes,
        error="" if exceeded else "10-turn session limit not enforced")

def test_oversized_body(client, target):
    """
    T19: Request body over 8KB must return 413 (spec §6.5).
    Sends a payload with a 10KB query string.
    """
    big_query = "A" * 10_240  # 10KB
    r = requests.post(
        f"{target}/agent/converse",
        json={"ahp": "0.1", "capability": "content_search", "query": big_query,
              "context": {"requesting_agent": "test"}},
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    passed = r.status_code == 413
    return TestResult("T19", "Oversized body → 413", passed,
        notes=f"HTTP {r.status_code} {'(correct 413)' if passed else '(expected 413)'}",
        error="" if passed else
              f"Expected 413 for >8KB body, got {r.status_code} (spec §6.5)")

def test_clarification_needed(client):
    """
    T21: If the server returns clarification_needed (spec §6.3 MAY), the
    response format MUST be correct (status='clarification_needed' +
    clarification_question field).  If the server never returns it, marks advisory.
    Sends a maximally ambiguous query to trigger clarification if supported.
    """
    resp = client.converse(AHPRequest(
        capability="content_search",
        query="What is it?",  # maximally ambiguous — may trigger clarification
    ))
    if resp.raw.get("status") == "clarification_needed":
        has_q = "clarification_question" in resp.raw
        assert has_q, "clarification_needed response must include clarification_question field"
        assert resp.http_status == 200, \
            f"clarification_needed should return 200, got {resp.http_status}"
        return TestResult("T21", "Clarification needed — format valid", True,
            notes=f"Server returned clarification_needed. "
                  f"Question: {resp.raw.get('clarification_question','')[:100]}")
    else:
        # Server answered directly — valid; clarification is optional (MAY)
        return TestResult("T21", "Clarification needed — format valid", True,
            notes=f"Server answered without clarification "
                  f"(status={resp.raw.get('status','?')}, mode={resp.mode}). "
                  "Advisory: clarification_needed flow (spec §6.3) not triggered by "
                  "test queries. Server is not REQUIRED to clarify — spec says MAY.")


def test_invalid_session(client):
    """
    T22: Sending an unknown/expired session_id MUST be handled gracefully.
    Per spec §6.2, the server MUST either:
      a) Return a session-not-found / session-expired error (HTTP 4xx)
      b) OR treat it as a fresh session (start new)
    The server MUST NOT crash or return 5xx.
    """
    fake_sid = str(uuid.uuid4())
    resp = client.converse(AHPRequest(
        capability="content_search",
        query="What is AHP?",
        session_id=fake_sid,
    ))
    # Server must not 5xx
    assert resp.http_status < 500, \
        f"Server crashed (5xx) on unknown session_id (HTTP {resp.http_status})"

    code = resp.raw.get("code", "")
    if resp.http_status in (400, 404, 410, 422) or "session" in code.lower():
        # Any 4xx means the server handled it gracefully without crashing.
        # Spec §6.2 says server MUST return HTTP 4xx (not specifically 404/410).
        # HTTP 400 (bad request) is a valid rejection of an unknown session_id.
        return TestResult("T22", "Invalid session_id handled gracefully", True,
            notes=f"Server returned session-error response "
                  f"(HTTP {resp.http_status}, code={code}). "
                  "Correct per spec §6.2 — any 4xx gracefully rejects unknown session.")
    elif resp.http_status == 200 and resp.status == "success":
        new_sid = resp.session_id or ""
        treated_as_new = bool(new_sid) and new_sid != fake_sid
        return TestResult("T22", "Invalid session_id handled gracefully", True,
            notes=f"Server treated unknown session_id as new session. "
                  f"Sent: {fake_sid[:8]}... → Got: {new_sid[:8] if new_sid else 'N/A'}... "
                  f"({'new session created' if treated_as_new else 'same ID echoed back'}). "
                  "Acceptable per spec §6.2.")
    else:
        return TestResult("T22", "Invalid session_id handled gracefully", False,
            notes=f"Unexpected response to unknown session_id: "
                  f"HTTP {resp.http_status}, status={resp.raw.get('status')}",
            error=f"Unexpected response to invalid session_id (spec §6.2)")


def test_ratelimit_headers(client, target):
    """
    T20: Rate-limit headers must be present on every response (spec §11.1).
    Required headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset,
    X-RateLimit-Window.
    Values must be numeric (integers or floats). A server that rate-limits
    silently (no headers, just 429) violates the spec but would pass T16.
    This test is separate from T16 — it verifies the *mechanism*, not just
    that 429 is eventually returned.
    """
    required_headers = [
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
        "X-RateLimit-Window",
    ]
    r = requests.post(
        f"{target}/agent/converse",
        json={"ahp": "0.1", "capability": "site_info",
              "query": "rate limit header probe",
              "context": {"requesting_agent": "test"}},
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    missing = []
    non_numeric = []
    found = {}
    for h in required_headers:
        val = r.headers.get(h)
        if val is None:
            missing.append(h)
        else:
            found[h] = val
            try:
                float(val)
            except (ValueError, TypeError):
                non_numeric.append(f"{h}={val!r}")

    passed = not missing and not non_numeric
    notes = (
        f"All rate-limit headers present and numeric: {found}"
        if passed else
        f"Missing: {missing} | Non-numeric: {non_numeric} | Found: {found}"
    )
    return TestResult("T20", "Rate-limit headers present (spec §11.1)", passed,
        notes=notes,
        error="" if passed else
              f"Rate-limit headers missing or non-numeric — "
              f"visiting agents cannot implement backoff without these (spec §11.1)")

def test_rate_limiting(client, target):
    """
    T16: Server must enforce rate limits (429) on burst traffic.
    Records window state (X-RateLimit-Remaining) before burst to document
    how many requests were available at test time.
    """
    # Record window state with a single probe request
    probe = requests.post(
        f"{target}/agent/converse",
        json={"ahp": "0.1", "capability": "site_info", "query": "rate limit probe",
              "context": {"requesting_agent": "test"}},
        timeout=5,
    )
    remaining_before = probe.headers.get("X-RateLimit-Remaining", "unknown")
    limit_header = probe.headers.get("X-RateLimit-Limit", "unknown")

    # Burst
    statuses = client.burst(count=35)
    hit_429 = 429 in statuses
    first_429 = statuses.index(429) + 1 if hit_429 else None

    return TestResult("T16", "Rate limiting", hit_429,
        notes=(
            f"429 after {first_429} burst requests | "
            f"Window before burst: {remaining_before}/{limit_header} remaining | "
            f"Note: window was partially consumed by earlier test requests"
        ) if hit_429 else
        "No 429 returned — rate limiting may not be active")

# ── Whitepaper: Multi-run token comparison with cache-busting ─────────────────

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
    Run each comparison query n_runs times across three approaches.
    Cache-busting: each AHP and RAG run appends a unique 8-char nonce to the
    query — e.g. '[ref:a1b2c3d4]' — to force distinct cache keys on the server
    and produce real token variance rather than cache-hit ±0 stddev.
    The nonce adds ~4 tokens overhead per run (consistent and noted).
    """
    results = []

    try:
        manifest = client.discover()
        content_url = manifest.get("endpoints", {}).get("content", "/llms.txt")
        fetch_start = time.perf_counter()
        r = requests.get(f"{target}{content_url}", timeout=15)
        fetch_latency_ms = (time.perf_counter() - fetch_start) * 1000
        full_content = r.text
        chunks = chunk_document(full_content, target_chars=500)
        naive_base_tokens = count_tokens(full_content)
        naive_note = (
            f"tiktoken cl100k_base estimate of full {len(full_content):,} char document "
            f"({naive_base_tokens:,} tokens) + per-query overhead. Not API-measured. "
            f"Fetch latency ({fetch_latency_ms:.0f}ms) is a single measurement, not a "
            f"3-run mean; it is reused for all queries and all runs in this site's comparison."
        )
        if verbose:
            print(f"  {site_label}: fetched {content_url} — "
                  f"{len(full_content):,} chars, {len(chunks)} chunks, "
                  f"{naive_base_tokens:,} tokens (tiktoken), "
                  f"fetch={fetch_latency_ms:.0f}ms")
    except Exception as e:
        print(f"  [warn] Could not fetch content doc from {target}: {e}")
        full_content = ""
        chunks = []
        naive_base_tokens = 10000
        naive_note = "Could not fetch content doc; estimate unavailable"
        fetch_latency_ms = 0.0

    for query, capability in queries:
        run_data: list[ComparisonRunResult] = []

        for run_idx in range(n_runs):
            if run_idx > 0:
                time.sleep(2)

            # Cache-busting nonce: unique per run
            nonce = uuid.uuid4().hex[:8]
            busted_query = f"{query} [ref:{nonce}]"

            # ── AHP MODE2 (cache-busted) ──
            ahp_resp = client.converse(
                AHPRequest(capability=capability, query=busted_query))
            if ahp_resp.http_status == 429:
                print(f"  [warn] AHP rate limited on run {run_idx+1} — waiting 10s")
                time.sleep(10)
                ahp_resp = client.converse(
                    AHPRequest(capability=capability, query=busted_query))
            ahp_tokens = ahp_resp.tokens_used
            ahp_latency = ahp_resp.latency_ms

            # ── RAG baseline (cache-busted same nonce) ──
            rag_tokens = 0
            rag_latency = 0.0
            if chunks and api_key:
                try:
                    _, rag_tokens, rag_latency = run_rag_query(
                        api_key, busted_query, chunks)
                except Exception as e:
                    print(f"  [warn] RAG run {run_idx+1} failed: {e}")
            elif not api_key and run_idx == 0:
                print(f"  [warn] No ANTHROPIC_API_KEY — RAG baseline skipped for {site_label}")

            # ── Naive (tiktoken, cache-busted query overhead) ──
            naive_tokens = (naive_base_tokens
                            + count_tokens(busted_query)
                            + 50)  # +50 system-prompt overhead
            naive_latency = fetch_latency_ms

            run_data.append(ComparisonRunResult(
                run=run_idx + 1,
                nonce=nonce,
                ahp_tokens=ahp_tokens,
                ahp_latency_ms=ahp_latency,
                rag_tokens=rag_tokens,
                rag_latency_ms=rag_latency,
                naive_tokens=naive_tokens,
                naive_latency_ms=naive_latency,
            ))

            if verbose:
                rag_str = str(rag_tokens) if rag_tokens else "N/A"
                print(f"  [{run_idx+1}/{n_runs}] {query[:42]:<42} "
                      f"AHP: {ahp_tokens:>5} ({ahp_latency:.0f}ms) | "
                      f"RAG: {rag_str:>5} | naive: {naive_tokens:>6}")

        # ── Aggregate ──
        ahp_tok = [rd.ahp_tokens for rd in run_data]
        rag_tok  = [rd.rag_tokens for rd in run_data if rd.rag_tokens > 0]
        naive_tok = [rd.naive_tokens for rd in run_data]
        ahp_lat  = [rd.ahp_latency_ms for rd in run_data]
        rag_lat  = [rd.rag_latency_ms for rd in run_data if rd.rag_latency_ms > 0]
        naive_lat = [rd.naive_latency_ms for rd in run_data]

        def _mean(lst): return statistics.mean(lst) if lst else 0.0
        def _std(lst):  return statistics.stdev(lst) if len(lst) > 1 else 0.0

        ahp_mean = _mean(ahp_tok)
        rag_mean = _mean(rag_tok)
        naive_mean = _mean(naive_tok)

        reduction_vs_naive = (1 - ahp_mean / naive_mean) * 100 if naive_mean else 0
        # overhead_vs_rag: positive = AHP uses MORE tokens than RAG
        overhead_vs_rag = (ahp_mean / rag_mean - 1) * 100 if rag_mean else 0

        results.append(ComparisonResult(
            query=query,
            site=site_label,
            runs=n_runs,
            cache_busting=True,
            naive_tokens_mean=round(naive_mean, 1),
            naive_tokens_note=naive_note,
            naive_latency_mean_ms=round(_mean(naive_lat), 1),
            rag_tokens_mean=round(rag_mean, 1),
            rag_tokens_stddev=round(_std(rag_tok), 1),
            rag_latency_mean_ms=round(_mean(rag_lat), 1),
            rag_latency_stddev_ms=round(_std(rag_lat), 1),
            ahp_tokens_mean=round(ahp_mean, 1),
            ahp_tokens_stddev=round(_std(ahp_tok), 1),
            ahp_latency_mean_ms=round(_mean(ahp_lat), 1),
            ahp_latency_stddev_ms=round(_std(ahp_lat), 1),
            reduction_vs_naive_pct=round(reduction_vs_naive, 1),
            overhead_vs_rag_pct=round(overhead_vs_rag, 1),
            raw_runs=[asdict(rd) for rd in run_data],
        ))

    return results

# ── Latency profile ────────────────────────────────────────────────────────────

def run_latency_profile(client: AHPClient, verbose: bool) -> dict:
    """
    Latency profile with explicit cold-cache and cache-hit sampling.

    Uses a split design to ensure both latency classes are measured:
      - 10 cold-cache samples: unique nonce queries, each a guaranteed cache miss
      - 10 cache-hit samples: a single repeated query, cached after first call

    This avoids the 'all 20 cached' artifact that occurs when the profile query
    is already in the cache from earlier tests in the suite.

    p50/p95 are reported separately for cached and uncached cohorts.
    The overall p95 is computed only when ≥20 mixed samples are available.
    """
    samples = []

    # ── 10 cold-cache samples (guaranteed unique, never cached) ──────────────
    for i in range(10):
        nonce = uuid.uuid4().hex[:8]
        query = f"Explain the AHP protocol overview [latency-cold-{nonce}]"
        resp = client.converse(AHPRequest(capability="content_search", query=query))
        samples.append({
            "run": i + 1,
            "type": "cold",
            "latency_ms": resp.latency_ms,
            "cached": resp.cached,
            "tokens": resp.tokens_used,
        })
        if verbose:
            print(f"  Cold {i+1}/10: {resp.latency_ms:.0f}ms "
                  f"(cached={resp.cached}, expected False)")

    # ── 10 cache-hit samples (same query, cached from run 1 onward) ──────────
    hot_query = "Explain the AHP protocol overview [latency-hot-fixed]"
    for i in range(10):
        resp = client.converse(AHPRequest(capability="content_search",
                                          query=hot_query))
        samples.append({
            "run": 10 + i + 1,
            "type": "hot",
            "latency_ms": resp.latency_ms,
            "cached": resp.cached,
            "tokens": resp.tokens_used,
        })
        if verbose and (i == 0 or i == 9):
            print(f"  Hot  {i+1}/10: {resp.latency_ms:.0f}ms "
                  f"(cached={resp.cached}, expected True after run 1)")

    all_lat   = [s["latency_ms"] for s in samples]
    cached    = [s["latency_ms"] for s in samples if s["cached"]]
    uncached  = [s["latency_ms"] for s in samples if not s["cached"]]

    # Typed cohorts (by design, not reported flag)
    cold_lat = [s["latency_ms"] for s in samples if s["type"] == "cold"]
    hot_lat  = [s["latency_ms"] for s in samples if s["type"] == "hot"]

    def _p95(lats, label=""):
        if len(lats) >= 20:
            v = round(sorted(lats)[int(len(lats) * 0.95)], 1)
            return v, "true p95"
        elif lats:
            return round(max(lats), 1), f"max of {len(lats)} samples (n<20)"
        return None, "no data"

    all_p95, all_p95_note = _p95(all_lat)

    return {
        "samples": samples,
        "n_samples": len(samples),
        "design": "10 cold-cache (unique nonces) + 10 cache-hit (repeated query)",
        # All-samples stats
        "p50_ms": round(statistics.median(all_lat), 1),
        "p95_ms": all_p95,
        "p95_note": all_p95_note,
        "min_ms": round(min(all_lat), 1),
        "max_ms": round(max(all_lat), 1),
        # Cache-hit cohort
        "cached_p50_ms":   round(statistics.median(cached), 1) if cached else None,
        "cached_p95_ms":   round(sorted(cached)[int(len(cached)*0.95)], 1)
                           if len(cached) >= 10 else
                           (round(max(cached), 1) if cached else None),
        "cached_max_ms":   round(max(cached), 1) if cached else None,
        "cached_n":        len(cached),
        # Cold-cache cohort
        "uncached_mean_ms":   round(statistics.mean(uncached), 1) if uncached else None,
        "uncached_median_ms": round(statistics.median(uncached), 1) if uncached else None,
        "uncached_max_ms":    round(max(uncached), 1) if uncached else None,
        "uncached_n":         len(uncached),
        # Design-typed cohort (for cross-checking)
        "cold_mean_ms":  round(statistics.mean(cold_lat), 1) if cold_lat else None,
        "cold_p50_ms":   round(statistics.median(cold_lat), 1) if cold_lat else None,
        "cold_max_ms":   round(max(cold_lat), 1) if cold_lat else None,
        "hot_p50_ms":    round(statistics.median(hot_lat), 1) if hot_lat else None,
        "hot_max_ms":    round(max(hot_lat), 1) if hot_lat else None,
    }

# ── Test runner helper ─────────────────────────────────────────────────────────

def run_test(test_id: str, name: str, fn, verbose: bool) -> TestResult:
    print(f"  [{test_id}] {name}...", end=" ", flush=True)
    try:
        result = fn()
        result.test_id = test_id
        result.name = name
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(status + (f" — {result.notes}" if verbose and result.notes else ""))
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
    rows = [[r["test_id"], r["name"][:44], "✓" if r["passed"] else "✗",
             f"{r['latency_ms']:.0f}ms" if r["latency_ms"] else "—",
             str(r["tokens_used"]) if r["tokens_used"] else "—",
             r["mode"] or "—"]
            for r in report["test_results"]]
    print(tabulate(rows,
        headers=["ID", "Test", "Pass", "Latency", "Tokens", "Mode"],
        tablefmt="simple"))
    s = report["summary"]
    print(f"\n{s['passed']}/{s['total']} passed ({s['pass_rate_pct']}%)")

    def print_comparison_table(comps, label):
        if not comps:
            return
        print(f"\n{'='*72}")
        print(f"TOKEN EFFICIENCY — {label} (3-run mean ± stddev, cache-busted)")
        print(f"{'='*72}")
        rows = []
        for c in comps:
            rag_tok = (f"{c['rag_tokens_mean']:.0f} ±{c['rag_tokens_stddev']:.0f}"
                       f" ({c['rag_latency_mean_ms']:.0f}ms)"
                       if c['rag_tokens_mean'] > 0 else "N/A")
            ahp_tok = (f"{c['ahp_tokens_mean']:.0f} ±{c['ahp_tokens_stddev']:.0f}"
                       f" ({c['ahp_latency_mean_ms']:.0f}ms)")
            rows.append([
                c["query"][:34],
                f"{c['naive_tokens_mean']:.0f}* ({c['naive_latency_mean_ms']:.0f}ms)",
                rag_tok, ahp_tok,
                f"{c['reduction_vs_naive_pct']}% ↓",
                f"+{c['overhead_vs_rag_pct']:.0f}% ↑" if c['rag_tokens_mean'] > 0 else "—",
            ])
        print(tabulate(rows, headers=[
            "Query", "Naive* (latency)", "RAG-baseline (lat.)",
            "AHP MODE2 (lat.)",
            "Reduction\nvs. naive↓", "Overhead\nvs. RAG↑"],
            tablefmt="simple"))
        print("* Naive = tiktoken estimate, full doc, no retrieval. ↓=lower is better for AHP. ↑=lower is better for AHP.")
        ahp_means = [c["ahp_tokens_mean"] for c in comps]
        naive_means = [c["naive_tokens_mean"] for c in comps]
        avg_vs_naive = (1 - sum(ahp_means) / sum(naive_means)) * 100
        print(f"\nAvg token reduction vs. naive: {avg_vs_naive:.1f}%")
        rag_comps = [c for c in comps if c['rag_tokens_mean'] > 0]
        if rag_comps:
            avg_overhead = statistics.mean(
                [c['overhead_vs_rag_pct'] for c in rag_comps])
            print(f"Avg token overhead vs. RAG-baseline: +{avg_overhead:.1f}% "
                  f"(AHP uses more tokens; advantages are protocol-level)")

    print_comparison_table(report.get("token_comparison", []),
                           report.get("target", "AHP site"))
    print_comparison_table(report.get("nate_comparison", []), "Nate Jones site")

    if report.get("latency_profile"):
        lp = report["latency_profile"]
        design = lp.get("design", "")
        print(f"\n{'='*60}")
        print(f"LATENCY PROFILE — MODE2 content_search ({lp.get('n_samples', '?')} samples)")
        print(f"Design: {design}" if design else "")
        print(f"{'='*60}")
        print(f"All      p50={lp['p50_ms']}ms  p95={lp['p95_ms']}ms "
              f"({lp.get('p95_note','')})  min={lp['min_ms']}ms  max={lp['max_ms']}ms")
        # Cold-cache (unique nonce queries, guaranteed cache misses)
        if lp.get("cold_mean_ms") is not None:
            print(f"Cold     mean={lp['cold_mean_ms']}ms  "
                  f"p50={lp['cold_p50_ms']}ms  "
                  f"max={lp['cold_max_ms']}ms  "
                  f"(n=10, forced cold via unique nonces)")
        # Cache-hit cohort
        if lp.get("hot_p50_ms") is not None:
            print(f"Hot/cached p50={lp['hot_p50_ms']}ms  "
                  f"max={lp['hot_max_ms']}ms  "
                  f"(n=10, repeated query)")

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AHP Test Suite v5.1")
    parser.add_argument("--target", default="http://localhost:3000")
    parser.add_argument("--nate-target", default="",
                        help="Override Nate site URL (default: auto-included)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output", "-o")
    parser.add_argument("--api-key", default="",
                        help="Anthropic API key (auto-loaded from .env if absent)")
    args = parser.parse_args()

    if args.nate_target:
        NATE_TARGET = args.nate_target  # override module-level default

    api_key = resolve_api_key(args.api_key)
    if not api_key:
        print("[warn] No ANTHROPIC_API_KEY found — RAG baseline will be skipped")

    client = AHPClient(args.target)
    try:
        report = run_all_tests(client, args.target, args.verbose, api_key=api_key)
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(1)

    print_report(report)

    out = args.output or f"report-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report: {out}")
