# AHP Test Suite

Visiting agent test harness for the [Agent Handshake Protocol](https://agenthandshake.dev).

Runs 16 conformance tests plus whitepaper benchmarks (token efficiency, latency profile) against any live AHP endpoint.

## What it tests

| ID | Test | Category |
|----|------|----------|
| T01 | Discovery via `/.well-known/agent.json` | Protocol |
| T02 | Discovery via `Accept: application/agent+json` | Protocol |
| T03 | In-page agent notice detection | Protocol |
| T04 | Manifest schema validation | Protocol |
| T05 | MODE2 simple content query | Functionality |
| T06 | Unknown capability error handling | Error handling |
| T07 | Missing required field error | Error handling |
| T08 | Multi-turn session exchange | Sessions |
| T09 | Response schema validation | Schema |
| T10 | Content signals in response meta | Content signals |
| T11 | MODE3 inventory check (tool use) | MODE3 |
| T12 | MODE3 custom quote calculation | MODE3 |
| T13 | MODE3 order lookup | MODE3 |
| T14 | MODE3 async human escalation + poll | MODE3 async |
| T15 | Rate limiting (429 on burst) | Infrastructure |
| T16 | Response caching | Infrastructure |

**Whitepaper benchmarks:**
- Token efficiency: AHP MODE2 vs. full markdown doc parse across 5 representative queries
- Latency profile: p50/p95/min/max across repeated runs

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

## Usage

```bash
# Run against the live reference implementation
./venv/bin/python test_runner.py --target https://ref.agenthandshake.dev

# Run against a local dev server
./venv/bin/python test_runner.py --target http://localhost:3000 --verbose

# Save report
./venv/bin/python test_runner.py --target https://ref.agenthandshake.dev --output report.json
```

## Output

The test runner prints a results table and saves a full JSON report:

```
[T01] Discovery — /.well-known/agent.json... ✓ PASS
[T02] Discovery — Accept header...           ✓ PASS
...

16/16 passed (100.0%)

TOKEN EFFICIENCY (Agent vs. Markdown)
Query                                    Markdown tokens  AHP tokens  Reduction
---------------------------------------  ---------------  ----------  ---------
What is MODE1?                                    10050         187     98.1%
How does discovery work in AHP?                   10050         243     97.6%
...

Average token reduction: 97.8%
```

## License

MIT
