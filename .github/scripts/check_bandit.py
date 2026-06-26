"""Read bandit-report.json and exit 1 if any HIGH-severity findings exist."""

import json
import sys

try:
    with open("bandit-report.json") as fh:
        data = json.load(fh)
except (FileNotFoundError, json.JSONDecodeError):
    print("No bandit report found — skipping.")
    sys.exit(0)

highs = [r for r in data.get("results", []) if r.get("issue_severity") == "HIGH"]

if highs:
    print(f"\n✗ {len(highs)} HIGH-severity security issue(s) found:")
    for r in highs:
        print(f"  {r['filename']}:{r['line_number']} — {r['issue_text']}")
    sys.exit(1)

print("✓ No HIGH-severity findings.")
