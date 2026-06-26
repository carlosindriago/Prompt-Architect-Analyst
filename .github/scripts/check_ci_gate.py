"""Read the needs-context JSON from argv[1] and exit 1 if any job failed."""

import json
import sys

if len(sys.argv) < 2:
    print("Usage: check_ci_gate.py '<needs-json>'")
    sys.exit(1)

try:
    needs = json.loads(sys.argv[1])
except json.JSONDecodeError as exc:
    print(f"Could not parse needs JSON: {exc}")
    sys.exit(1)

failed = [k for k, v in needs.items() if v.get("result") not in ("success", "skipped")]

if failed:
    print(f"FAILED jobs: {failed}")
    sys.exit(1)

print("All CI jobs passed.")
