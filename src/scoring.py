"""
Map rule results to overall status: Ready to approve / Needs review / Critical issues.
"""
from __future__ import annotations

from typing import Any


def compute_overall_status(rule_results: list[dict[str, Any]]) -> tuple[str, dict[str, int]]:
    counts = {"pass": 0, "needs_review": 0, "fail": 0}
    for r in rule_results:
        s = (r.get("status") or "pass").lower()
        if s == "fail":
            counts["fail"] += 1
        elif s == "needs_review":
            counts["needs_review"] += 1
        else:
            counts["pass"] += 1

    if counts["fail"] > 0:
        overall = "Critical issues"
    elif counts["needs_review"] > 0:
        overall = "Needs review"
    else:
        overall = "Ready to approve"

    return overall, counts
