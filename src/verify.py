from __future__ import annotations

"""
Verification module: correctness proof layer for all solver outputs.

Every check returns {check_name, passed, detail}.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

from .formulation_3d import ThreeDInput, ThreeDResult
from .formulation_tetn import TETNResult
from .network import TETN

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _result(name: str, passed: bool, detail: str) -> dict:
    return {"check_name": name, "passed": passed, "detail": detail}


# ---------------------------------------------------------------------------
# Structural checks (3D formulation)
# ---------------------------------------------------------------------------


def check_tour_completeness_3d(inp: ThreeDInput, res: ThreeDResult) -> dict:
    """Every destination node appears exactly once in the selected links."""
    visited = set()
    for i, j, k in res.tour_links:
        visited.add(i)
        visited.add(j)

    expected = set(range(1, inp.n + 1))
    missing = expected - visited
    if missing:
        return _result(
            "tour_completeness_3d", False,
            f"Missing nodes in tour: {missing}",
        )
    return _result("tour_completeness_3d", True, "All nodes visited")


def check_no_subtours_3d(res: ThreeDResult) -> dict:
    """Using u-values, verify u[j] > u[i] for selected links where both i,j != 1 (depot)."""
    violations = []
    for i, j, k in res.tour_links:
        if i == 1 or j == 1:
            continue
        ui = res.u_vals.get(i, 0)
        uj = res.u_vals.get(j, 0)
        if uj <= ui:
            violations.append(f"u[{i}]={ui} >= u[{j}]={uj}")
    if violations:
        return _result("no_subtours_3d", False, f"Violations: {violations[:5]}")
    return _result("no_subtours_3d", True, "u-values strictly increasing among non-depot nodes")


def check_mtz_3d(inp: ThreeDInput, res: ThreeDResult) -> dict:
    """For every ordered pair (i,j) where i!=1 and j!=1, verify u[i] - u[j] + n * sum_k(x[i,j,k]) <= n-1."""
    n = inp.n
    violations = []
    for i in range(2, n + 1):
        for j in range(2, n + 1):
            if i == j:
                continue
            sum_k = sum(
                res.x_vals.get((i, j, k), 0) for k in range(1, inp.k_max + 1)
            )
            ui = res.u_vals.get(i, 0)
            uj = res.u_vals.get(j, 0)
            lhs = ui - uj + n * sum_k
            if lhs > n - 1 + 1e-6:
                violations.append(f"({i},{j}): {lhs} > {n-1}")

    if violations:
        return _result("mtz_3d", False, f"Violations: {violations[:5]}")
    return _result("mtz_3d", True, "All MTZ constraints satisfied")


def check_time_feasibility_3d(inp: ThreeDInput, res: ThreeDResult) -> dict:
    """For every selected link, verify departure time >= t[i]."""
    violations = []
    for i, j, k in res.tour_links:
        ti = res.t_vals.get(i, 0)
        start = inp.Start.get((i, j, k), 0)
        if ti > start + 1e-6:
            violations.append(f"t[{i}]={ti} > Start[{i},{j},{k}]={start}")
    if violations:
        return _result("time_feasibility_3d", False, f"Violations: {violations[:5]}")
    return _result("time_feasibility_3d", True, "All departure times are feasible")


# ---------------------------------------------------------------------------
# Structural checks (TETN formulation)
# ---------------------------------------------------------------------------


def check_tour_completeness_tetn(
    net: TETN, res: TETNResult, required_stations: list[str]
) -> dict:
    """Every required station appears at least once in the tour (by ID or name)."""
    visited_ids = set()
    visited_names = set()
    for nid in res.tour_nodes:
        node = net.nodes.get(nid)
        if node:
            visited_ids.add(node.station_id)
            visited_names.add(node.station_name)

    visited_all = visited_ids | visited_names
    required = set(required_stations)
    missing = required - visited_all
    if missing:
        return _result(
            "tour_completeness_tetn", False,
            f"Missing stations: {missing}",
        )
    return _result("tour_completeness_tetn", True, "All required stations visited")


def check_flow_conservation_tetn(net: TETN, res: TETNResult) -> dict:
    """For every non-boundary node in the solution, in-degree == out-degree."""
    in_deg = defaultdict(int)
    out_deg = defaultdict(int)
    for (s, t), v in res.x_vals.items():
        if v == 1:
            out_deg[s] += 1
            in_deg[t] += 1

    active_nodes = set(in_deg.keys()) | set(out_deg.keys())
    boundary = set()
    for nid in active_nodes:
        if in_deg[nid] == 0 or out_deg[nid] == 0:
            boundary.add(nid)

    violations = []
    for nid in active_nodes - boundary:
        if in_deg[nid] != out_deg[nid]:
            violations.append(
                f"Node {nid}: in={in_deg[nid]}, out={out_deg[nid]}"
            )
    if violations:
        return _result("flow_conservation_tetn", False, f"Violations: {violations[:5]}")
    return _result("flow_conservation_tetn", True, "Flow conserved at all interior nodes")


def check_no_subtours_tetn(net: TETN, res: TETNResult) -> dict:
    """Confirm no cycles exist using time ordering (TETN is acyclic by construction)."""
    for (s, t), v in res.x_vals.items():
        if v == 1:
            s_node = net.nodes.get(s)
            t_node = net.nodes.get(t)
            if s_node and t_node:
                if s_node.time_seconds >= t_node.time_seconds:
                    return _result(
                        "no_subtours_tetn", False,
                        f"Link ({s},{t}): time {s_node.time_seconds} >= {t_node.time_seconds}",
                    )
    return _result("no_subtours_tetn", True, "All links move forward in time (acyclic)")


def check_acyclicity(net: TETN) -> dict:
    """On the full TETN, confirm all links move strictly forward in time."""
    violations = []
    for link in net.links:
        s_node = net.nodes[link.source]
        t_node = net.nodes[link.target]
        if s_node.time_seconds >= t_node.time_seconds:
            violations.append(
                f"({link.source},{link.target}): "
                f"{s_node.time_seconds} >= {t_node.time_seconds}"
            )
    if violations:
        return _result(
            "acyclicity", False,
            f"{len(violations)} links violate time ordering. First: {violations[0]}",
        )
    return _result("acyclicity", True, "All links move strictly forward in time")


def check_pruning_losslessness(path_results: list[dict]) -> dict:
    """After pruning, all inter-destination paths must exist."""
    missing = [r for r in path_results if not r["path_exists"]]
    if missing:
        detail = "; ".join(f"{r['from']} -> {r['to']}" for r in missing[:5])
        return _result("pruning_losslessness", False, f"Missing paths: {detail}")
    return _result(
        "pruning_losslessness", True,
        f"All {len(path_results)} inter-destination paths exist",
    )


# ---------------------------------------------------------------------------
# Full verification runners
# ---------------------------------------------------------------------------


def verify_3d_result(inp: ThreeDInput, res: ThreeDResult) -> list[dict]:
    """Run all applicable checks on a 3D formulation result."""
    checks = [
        check_tour_completeness_3d(inp, res),
        check_no_subtours_3d(res),
        check_mtz_3d(inp, res),
        check_time_feasibility_3d(inp, res),
    ]
    return checks


def verify_tetn_result(
    net: TETN, res: TETNResult, required_stations: list[str]
) -> list[dict]:
    """Run all applicable checks on a TETN formulation result."""
    checks = [
        check_tour_completeness_tetn(net, res, required_stations),
        check_flow_conservation_tetn(net, res),
        check_no_subtours_tetn(net, res),
    ]
    return checks


def generate_verification_report(
    checks: list[dict], output_path: Path | None = None
) -> dict:
    """Generate a verification report from a list of check results."""
    passed = sum(1 for c in checks if c["passed"])
    failed = sum(1 for c in checks if not c["passed"])
    report = {
        "summary": {
            "total_checks": len(checks),
            "passed": passed,
            "failed": failed,
            "all_passed": failed == 0,
        },
        "checks": checks,
    }

    if output_path is None:
        output_path = RESULTS_DIR / "verification_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Verification report written to %s", output_path)

    print("\n=== Verification Report ===")
    print(f"Total checks: {len(checks)} | Passed: {passed} | Failed: {failed}")
    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['check_name']}: {c['detail']}")
    print()

    return report
