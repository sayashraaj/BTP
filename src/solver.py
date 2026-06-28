from __future__ import annotations

"""
Unified solver interface for the Journey Planner.

Accepts a query dict, calls the appropriate formulation module,
returns result + verification report with per-stage timing.
"""

import logging
import os
import time
from dataclasses import dataclass

import pandas as pd

from .data_pipeline import run_pipeline
from .formulation_3d import ThreeDInput, ThreeDResult, solve_3d
from .formulation_tetn import TETNResult, solve_mda1, solve_mda2, solve_mda3
from .network import TETN, build_pruned_tetn, check_inter_destination_paths
from .verify import (
    generate_verification_report,
    verify_3d_result,
    verify_tetn_result,
)

logger = logging.getLogger(__name__)


@dataclass
class SolverResult:
    method: str
    result: ThreeDResult | TETNResult
    verification: dict | None
    timing: dict[str, float]
    network_stats: dict | None
    query: dict | None = None


def solve_query(
    origin: str,
    destinations: list[str],
    method: str = "tetn_mda3",
    solver: str | None = None,
    verify: bool = True,
    final_destination: str | None = None,
    schedule: pd.DataFrame | None = None,
) -> SolverResult:
    """
    Unified solver entry point.

    Parameters
    ----------
    origin : str
        Origin station name.
    destinations : list[str]
        Intermediate destination station names.
    method : str
        One of: "3d", "tetn_mda1", "tetn_mda2", "tetn_mda3".
    solver : str or None
        "cbc" or "gurobi". Defaults to env var SOLVER or "cbc".
    verify : bool
        Whether to run verification checks.
    final_destination : str or None
        For non-circular routing. If None, defaults to origin.
    schedule : DataFrame or None
        Pre-loaded schedule. If None, runs the data pipeline.
    """
    if solver is None:
        solver = os.environ.get("SOLVER", "cbc").lower()

    timing = {}

    t0 = time.time()
    if schedule is None:
        schedule = run_pipeline()
    timing["data_pipeline"] = time.time() - t0

    if final_destination is None:
        final_destination = origin

    all_stations = [origin] + destinations + [final_destination]

    if method.startswith("tetn"):
        t0 = time.time()
        net, net_stats = build_pruned_tetn(schedule, all_stations)
        timing["network_construction"] = time.time() - t0

        t0 = time.time()
        if method == "tetn_mda1":
            result = solve_mda1(net, origin, destinations, solver)
        elif method == "tetn_mda2":
            result = solve_mda2(net, origin, destinations, solver)
        else:
            result = solve_mda3(net, origin, destinations, final_destination, solver)
        timing["solve"] = time.time() - t0

        verification = None
        if verify:
            t0 = time.time()
            required = [origin] + destinations + [final_destination]
            checks = verify_tetn_result(net, result, required)

            path_results = check_inter_destination_paths(net, all_stations)
            from .verify import check_pruning_losslessness

            checks.append(check_pruning_losslessness(path_results))
            verification = generate_verification_report(checks)
            timing["verify"] = time.time() - t0

        return SolverResult(
            method=method,
            result=result,
            verification=verification,
            timing=timing,
            network_stats=net_stats,
            query={
                "origin": origin,
                "intermediates": destinations,
                "destination": final_destination,
            },
        )

    elif method == "3d":
        raise NotImplementedError(
            "3D method requires manual input specification. "
            "Use formulation_3d.solve_3d() directly with a ThreeDInput object."
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use '3d', 'tetn_mda1', 'tetn_mda2', or 'tetn_mda3'.")


def format_tour(result: SolverResult) -> str:
    """Pretty-print the optimal tour."""
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  OPTIMAL TOUR — Method: {result.method}")
    lines.append(f"{'='*70}")

    r = result.result
    if isinstance(r, TETNResult):
        lines.append(f"  Status: {r.solver_status}")
        lines.append(f"  Objective (total travel time): {r.objective_value:.0f} seconds")
        lines.append(f"  Solve time: {r.solve_time_seconds:.2f} seconds")
        lines.append(f"\n  {'Node ID':>8}  {'Train':>6}  {'Train Name':<40}  {'Time (s)':>10}  {'Station':<30}  {'Type'}")
        lines.append(f"  {'-'*8}  {'-'*6}  {'-'*40}  {'-'*10}  {'-'*30}  {'-'*4}")
        for d in r.tour_details:
            lines.append(
                f"  {d['node_id']:>8}  {d['train_id']:>6}  {d['train_name']:<40}  "
                f"{d['time_seconds']:>10}  {d['station_name']:<30}  {d['node_type']}"
            )
    elif isinstance(r, ThreeDResult):
        lines.append(f"  Status: {r.solver_status}")
        lines.append(f"  Objective (return time): {r.objective_value:.1f}")
        lines.append(f"  Solve time: {r.solve_time_seconds:.2f} seconds")
        lines.append(f"\n  Tour order: {' -> '.join(str(n) for n in r.tour)}")
        lines.append(f"\n  {'Node':>6}  {'u (rank)':>10}  {'t (exit time)':>14}")
        lines.append(f"  {'-'*6}  {'-'*10}  {'-'*14}")
        for node in r.tour:
            lines.append(
                f"  {node:>6}  {r.u_vals.get(node, '-'):>10}  {r.t_vals.get(node, '-'):>14}"
            )

    lines.append(f"\n  Timing breakdown:")
    for stage, secs in result.timing.items():
        lines.append(f"    {stage}: {secs:.2f}s")
    lines.append(f"{'='*70}\n")

    return "\n".join(lines)
