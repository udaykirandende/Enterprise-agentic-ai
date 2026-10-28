
import json
import math
import os
import re
import time
from pathlib import Path

import requests

BASE_URL = os.getenv("AGENT_API_URL", "http://127.0.0.1:8000/query")
TESTS_DIR = Path(__file__).parent
CASES_FILE = TESTS_DIR / "eval_cases.json"
RESULTS_FILE = TESTS_DIR / "eval_results.json"

ROUTE_ALIASES = {
    "RETRIEVAL": ("intent: retrieval",),
    "CALCULATOR": ("intent: calculator",),
    "WEB_SEARCH": ("intent: web search",),
    "CONVERSATIONAL": ("intent: conversational",),
    "MCP_TOOL": ("intent: mcp tool",),
}


def normalize_text(value):
    """Normalize Unicode punctuation and whitespace for comparisons."""
    if not isinstance(value, str):
        return ""

    return (
        value.lower()
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .strip()
    )


def check_route(expected_route, thought_process):
    """Check the route reported in the API thought_process."""
    if not expected_route:
        return None

    aliases = ROUTE_ALIASES.get(expected_route)
    if not aliases:
        return False

    trace = normalize_text(" ".join(map(str, thought_process)))
    return any(alias in trace for alias in aliases)


def check_behavior(case, answer, sources):
    """Evaluate answer behavior separately from planner routing."""
    behavior = case["expected_behavior"]
    normalized_answer = normalize_text(answer)
    required_terms = case.get("required_terms", [])

    if behavior == "abstain":
        passed = "i don't know" in normalized_answer
        return passed, "Expected an explicit abstention."

    if behavior in ("answer_from_context", "contains_terms"):
        passed = all(
            normalize_text(term) in normalized_answer
            for term in required_terms
        )
        return passed, "Checked required answer terms."

    if behavior == "web_sources":
        valid_sources = []

        if isinstance(sources, list):
            for source in sources:
                if isinstance(source, str) and re.search(
                    r"https?://\S+", source
                ):
                    valid_sources.append(source)

        passed = bool(answer.strip()) and bool(valid_sources)
        return passed, (
            f"Found {len(valid_sources)} source entries containing URLs."
        )

    if behavior == "nonempty_answer":
        return bool(answer.strip()), "Expected a non-empty answer."

    return False, f"Unknown expected behavior: {behavior}"


def evaluate_case(case):
    payload = {
        "q": case["question"],
        "thread_id": case["thread_id"],
    }

    start = time.perf_counter()

    try:
        response = requests.post(
            BASE_URL,
            json=payload,
            timeout=120,
        )
        latency = round(time.perf_counter() - start, 2)
        response.raise_for_status()
        data = response.json()

        answer = data.get("answer", "")
        sources = data.get("sources", [])
        thought_process = data.get("thought_process", [])

        answer_passed, behavior_detail = check_behavior(
            case,
            answer,
            sources,
        )

        expected_route = case.get("expected_route")
        route_passed = check_route(
            expected_route,
            thought_process,
        )

        checks = {"answer": answer_passed}

        if expected_route is not None:
            checks["route"] = route_passed

        passed = all(checks.values())

        return {
            "id": case["id"],
            "passed": passed,
            "checks": checks,
            "behavior_detail": behavior_detail,
            "expected_route": expected_route,
            "thought_process": thought_process,
            "status": data.get("status", "Unknown"),
            "latency_seconds": latency,
            "answer": answer,
            "source_count": (
                len(sources) if isinstance(sources, list) else 0
            ),
            "sources": sources,
        }

    except Exception as exc:
        return {
            "id": case["id"],
            "passed": False,
            "checks": {"request": False},
            "error": f"{type(exc).__name__}: {exc}",
            "latency_seconds": round(
                time.perf_counter() - start,
                2,
            ),
        }


def main():
    if not CASES_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation cases file not found: {CASES_FILE}"
        )

    cases = json.loads(
        CASES_FILE.read_text(encoding="utf-8-sig")
    )

    if not cases:
        print("No evaluation cases found.")
        return

    print("=== Agentic AI Evaluation ===")
    print(f"Cases: {len(cases)}")
    print(f"Endpoint: {BASE_URL}")

    results = [evaluate_case(case) for case in cases]

    for result in results:
        label = "PASS" if result["passed"] else "FAIL"
        print(
            f"\n[{label}] {result['id']} "
            f"({result['latency_seconds']}s)"
        )

        if "checks" in result:
            print(f"Checks: {json.dumps(result['checks'])}")

        if "behavior_detail" in result:
            print(result["behavior_detail"])

        if "error" in result:
            print(f"Error: {result['error']}")

    passed = sum(result["passed"] for result in results)
    total = len(results)
    latencies = [
        result["latency_seconds"]
        for result in results
        if "latency_seconds" in result
    ]

    sorted_latencies = sorted(latencies)
    p95_latency = (
        sorted_latencies[max(0, math.ceil(0.95 * len(sorted_latencies)) - 1)]
        if sorted_latencies
        else None
    )
    error_count = sum("error" in result for result in results)

    summary = {
        "passed": passed,
        "total": total,
        "pass_rate_percent": round(
            passed / total * 100, 2
        ),
        "p95_latency_seconds": p95_latency,
        "request_error_count": error_count,
        "average_latency_seconds": (
            round(sum(latencies) / len(latencies), 2)
            if latencies
            else None
        ),
        "results": results,
    }

    RESULTS_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== Summary ===")
    print(f"Passed: {passed}/{total}")
    print(f"Pass rate: {summary['pass_rate_percent']}%")
    print(f"P95 latency: {summary['p95_latency_seconds']}s")
    print(f"Request errors: {summary['request_error_count']}")
    print(
        "Average latency: "
        f"{summary['average_latency_seconds']}s"
    )
    print(f"Detailed results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
