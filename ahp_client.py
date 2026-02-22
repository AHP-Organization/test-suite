"""
AHP Visiting Agent Client
A minimal AHP client simulating a visiting agent.
Used by the test harness to interact with AHP endpoints.
"""

import requests
import time
import json
from dataclasses import dataclass, field
from typing import Optional, Any

@dataclass
class AHPRequest:
    capability: str
    query: str
    session_id: Optional[str] = None
    clarification: Optional[str] = None
    ahp: str = "0.1"
    context: dict = field(default_factory=dict)

@dataclass
class AHPResponse:
    status: str
    raw: dict
    latency_ms: float
    http_status: int
    tokens_used: int = 0
    mode: str = ""
    cached: bool = False
    tools_used: list = field(default_factory=list)
    session_id: Optional[str] = None
    rate_limit_remaining: Optional[int] = None

class AHPClient:
    def __init__(self, base_url: str, agent_id: str = "ahp-test-suite/0.1"):
        self.base_url = base_url.rstrip("/")
        self.agent_id = agent_id
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": agent_id,
        })

    # ── Discovery ──────────────────────────────────────────────────────────────

    def discover(self) -> dict:
        """Fetch the AHP manifest from /.well-known/agent.json."""
        r = self.session.get(f"{self.base_url}/.well-known/agent.json", timeout=10)
        r.raise_for_status()
        return r.json()

    def discover_via_accept_header(self) -> dict:
        """Discover AHP using Accept: application/agent+json header."""
        r = self.session.get(
            self.base_url,
            headers={"Accept": "application/agent+json"},
            timeout=10,
            allow_redirects=True,
        )
        if r.headers.get("Content-Type", "").startswith("application/json"):
            return r.json()
        # If redirect landed on agent.json
        if "agent.json" in r.url:
            return r.json()
        raise ValueError(f"Accept header discovery failed: got {r.status_code} {r.headers.get('Content-Type')}")

    def check_agent_notice(self, path: str = "/") -> bool:
        """Check if the page contains an AHP in-page agent notice."""
        r = self.session.get(f"{self.base_url}{path}", timeout=10)
        return 'aria-label="AI Agent Notice"' in r.text or 'class="ahp-notice"' in r.text

    def check_link_tag(self, path: str = "/") -> bool:
        """Check if the page contains <link rel='agent-manifest'>."""
        r = self.session.get(f"{self.base_url}{path}", timeout=10)
        return 'rel="agent-manifest"' in r.text or "agent-manifest" in r.text

    # ── Conversational endpoint ────────────────────────────────────────────────

    def converse(self, req: AHPRequest, auth_token: Optional[str] = None) -> AHPResponse:
        """POST to /agent/converse and return a structured response."""
        body = {
            "ahp": req.ahp,
            "capability": req.capability,
            "query": req.query,
            "context": {
                "requesting_agent": self.agent_id,
                **req.context,
            },
        }
        if req.session_id:
            body["session_id"] = req.session_id
        if req.clarification:
            body["clarification"] = req.clarification

        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        start = time.perf_counter()
        r = self.session.post(
            f"{self.base_url}/agent/converse",
            json=body,
            headers=headers,
            timeout=30,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        try:
            data = r.json()
        except Exception:
            data = {"status": "error", "code": "parse_error", "message": r.text}

        meta = data.get("meta", {})
        return AHPResponse(
            status=data.get("status", "unknown"),
            raw=data,
            latency_ms=round(latency_ms, 2),
            http_status=r.status_code,
            tokens_used=meta.get("tokens_used", 0),
            mode=meta.get("mode", ""),
            cached=meta.get("cached", False),
            tools_used=meta.get("tools_used", []),
            session_id=data.get("session_id"),
            rate_limit_remaining=int(r.headers["X-RateLimit-Remaining"])
                if "X-RateLimit-Remaining" in r.headers else None,
        )

    def poll_status(self, session_id: str, max_wait_s: int = 60, interval_s: float = 1.0) -> AHPResponse:
        """Poll /agent/converse/status/:id until resolved or timeout."""
        deadline = time.time() + max_wait_s
        while time.time() < deadline:
            r = self.session.get(
                f"{self.base_url}/agent/converse/status/{session_id}",
                timeout=10,
            )
            data = r.json()
            if data.get("status") in ("success", "failed", "expired"):
                return AHPResponse(
                    status=data["status"],
                    raw=data,
                    latency_ms=0,
                    http_status=r.status_code,
                )
            time.sleep(interval_s)

        return AHPResponse(
            status="timeout",
            raw={"status": "timeout"},
            latency_ms=0,
            http_status=0,
        )

    def burst(self, count: int = 35) -> list[int]:
        """Fire count requests rapidly to test rate limiting. Returns HTTP status codes."""
        statuses = []
        for _ in range(count):
            try:
                r = self.session.post(
                    f"{self.base_url}/agent/converse",
                    json={"capability": "site_info", "query": "test"},
                    timeout=5,
                )
                statuses.append(r.status_code)
            except Exception:
                statuses.append(0)
        return statuses
