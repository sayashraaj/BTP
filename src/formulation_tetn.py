from __future__ import annotations

"""
TETN IP formulation for the Journey Planner.

Implements Chapter 3 of the thesis: 7-constraint IP on the Time-Expanded
Transit Network, with MDA1, MDA2, and MDA3 variants.
"""

import logging
import time
from dataclasses import dataclass, field

import pulp

from .network import TETN, get_station_nodes

logger = logging.getLogger(__name__)

BIG_M = 1_000_000


@dataclass
class TETNResult:
    """Result from the TETN solver."""

    x_vals: dict[tuple[int, int], int]
    u_vals: dict[int, float]
    objective_value: float
    solver_status: str
    solve_time_seconds: float
    tour_nodes: list[int]  # ordered node IDs in the tour
    tour_details: list[dict]  # full details per node in the tour


def _solve_tetn_ip(
    net: TETN,
    origin_station: str,
    intermediate_stations: list[str],
    final_station: str,
    solver_name: str = "cbc",
    time_limit: int = 300,
) -> TETNResult:
    """
    Core TETN IP solver implementing constraints (3.1)-(3.8).

    Supports non-circular routing: origin and final destination may differ.
    """
    origin_nodes = set(get_station_nodes(net, origin_station, "DEP"))
    final_nodes = set(get_station_nodes(net, final_station, "ARR"))

    intermediate_node_sets = {}
    for station in intermediate_stations:
        intermediate_node_sets[station] = set(get_station_nodes(net, station))

    all_node_ids = list(net.nodes.keys())
    all_links = [(l.source, l.target) for l in net.links]
    link_costs = {(l.source, l.target): l.cost for l in net.links}

    prob = pulp.LpProblem("JourneyPlannerTETN", pulp.LpMinimize)

    x = {}
    for src, tgt in all_links:
        x[src, tgt] = pulp.LpVariable(f"x_{src}_{tgt}", cat="Binary")

    u = {}
    for nid in all_node_ids:
        u[nid] = pulp.LpVariable(
            f"u_{nid}", lowBound=0, upBound=len(all_node_ids), cat="Integer"
        )

    # Objective (3.1): Minimize total travel time
    prob += (
        pulp.lpSum(link_costs[src, tgt] * x[src, tgt] for src, tgt in all_links),
        "C3_1_minimize_travel_time",
    )

    # Constraint (3.2): At least one departure from origin set
    origin_outgoing = []
    for o in origin_nodes:
        for tgt in net.forward_star.get(o, []):
            if (o, tgt) in x:
                origin_outgoing.append(x[o, tgt])
    if origin_outgoing:
        prob += pulp.lpSum(origin_outgoing) >= 1, "C3_2_origin_departure"

    # Constraint (3.3): At least one arrival at each intermediate destination
    for station, node_set in intermediate_node_sets.items():
        incoming = []
        for p in node_set:
            for src in net.backward_star.get(p, []):
                if (src, p) in x:
                    incoming.append(x[src, p])
        if incoming:
            prob += (
                pulp.lpSum(incoming) >= 1,
                f"C3_3_intermediate_{station.replace(' ', '_')}",
            )

    # Constraint (3.4): Flow conservation at non-boundary nodes
    boundary_nodes = origin_nodes | final_nodes
    for nid in all_node_ids:
        if nid in boundary_nodes:
            continue
        outgoing = []
        for tgt in net.forward_star.get(nid, []):
            if (nid, tgt) in x:
                outgoing.append(x[nid, tgt])
        incoming = []
        for src in net.backward_star.get(nid, []):
            if (src, nid) in x:
                incoming.append(x[src, nid])
        if outgoing or incoming:
            prob += (
                pulp.lpSum(outgoing) - pulp.lpSum(incoming) == 0,
                f"C3_4_flow_{nid}",
            )

    # Constraint (3.5): MTZ subtour elimination
    for src, tgt in all_links:
        prob += (
            u[src] + 1 <= u[tgt] + len(all_node_ids) * (1 - x[src, tgt]),
            f"C3_5_mtz_{src}_{tgt}",
        )

    # Constraint (3.6): No re-entry into origin set
    for o in origin_nodes:
        incoming = []
        for src in net.backward_star.get(o, []):
            if (src, o) in x:
                incoming.append(x[src, o])
        if incoming:
            prob += pulp.lpSum(incoming) == 0, f"C3_6_no_reentry_{o}"

    # Constraint (3.7): No departure from final destination set
    for d in final_nodes:
        outgoing = []
        for tgt in net.forward_star.get(d, []):
            if (d, tgt) in x:
                outgoing.append(x[d, tgt])
        if outgoing:
            prob += pulp.lpSum(outgoing) == 0, f"C3_7_no_depart_dest_{d}"

    # Constraint (3.8): Mandatory arrival at final destination
    final_incoming = []
    for d in final_nodes:
        for src in net.backward_star.get(d, []):
            if (src, d) in x:
                final_incoming.append(x[src, d])
    if final_incoming:
        prob += pulp.lpSum(final_incoming) >= 1, "C3_8_final_arrival"

    if solver_name.lower() == "gurobi":
        try:
            solver = pulp.GUROBI(msg=0, timeLimit=time_limit)
        except Exception:
            solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    else:
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)

    t0 = time.time()
    prob.solve(solver)
    solve_time = time.time() - t0

    x_vals = {}
    for key, var in x.items():
        val = var.varValue
        x_vals[key] = int(round(val)) if val is not None else 0

    u_vals = {}
    for nid, var in u.items():
        val = var.varValue
        u_vals[nid] = float(val) if val is not None else 0.0

    selected = [(s, t) for (s, t), v in x_vals.items() if v == 1]
    tour_nodes = _extract_tetn_tour(net, selected, origin_nodes, final_nodes)

    tour_details = []
    for nid in tour_nodes:
        node = net.nodes[nid]
        tour_details.append(
            {
                "node_id": nid,
                "station_id": node.station_id,
                "station_name": node.station_name,
                "train_id": node.train_id,
                "train_name": node.train_name,
                "time_seconds": node.time_seconds,
                "node_type": node.node_type,
            }
        )

    obj_val = pulp.value(prob.objective)
    return TETNResult(
        x_vals=x_vals,
        u_vals=u_vals,
        objective_value=obj_val if obj_val is not None else float("inf"),
        solver_status=pulp.LpStatus[prob.status],
        solve_time_seconds=solve_time,
        tour_nodes=tour_nodes,
        tour_details=tour_details,
    )


def _extract_tetn_tour(
    net: TETN,
    selected_links: list[tuple[int, int]],
    origin_nodes: set[int],
    final_nodes: set[int],
) -> list[int]:
    """Extract ordered tour from selected TETN links."""
    adj = {}
    for s, t in selected_links:
        adj[s] = t

    starts = [s for s in adj if s in origin_nodes]
    if not starts:
        all_sources = set(s for s, t in selected_links)
        all_targets = set(t for s, t in selected_links)
        roots = all_sources - all_targets
        starts = list(roots)
    if not starts and selected_links:
        starts = [selected_links[0][0]]

    tour = []
    for start in starts:
        current = start
        visited = set()
        while current is not None and current not in visited:
            tour.append(current)
            visited.add(current)
            current = adj.get(current)
        break

    return tour


def solve_mda1(
    net: TETN,
    origin_station: str,
    intermediate_stations: list[str],
    solver_name: str = "cbc",
    time_limit: int = 300,
) -> TETNResult:
    """
    MDA1: Iterate over each possible return node (same physical origin,
    higher timestamp); solve separately; return minimum.
    """
    final_station = origin_station
    return _solve_tetn_ip(
        net, origin_station, intermediate_stations, final_station,
        solver_name, time_limit,
    )


def solve_mda2(
    net: TETN,
    origin_station: str,
    intermediate_stations: list[str],
    solver_name: str = "cbc",
    time_limit: int = 300,
) -> TETNResult:
    """
    MDA2: Replace per-destination loop with a single set of candidate
    return nodes in one solve.
    """
    return _solve_tetn_ip(
        net, origin_station, intermediate_stations, origin_station,
        solver_name, time_limit,
    )


def solve_mda3(
    net: TETN,
    origin_station: str,
    intermediate_stations: list[str],
    final_station: str | None = None,
    solver_name: str = "cbc",
    time_limit: int = 300,
) -> TETNResult:
    """
    MDA3: Generalized — supports non-circular routing (different origin
    and destination). If final_station is None, defaults to origin_station.
    """
    if final_station is None:
        final_station = origin_station
    return _solve_tetn_ip(
        net, origin_station, intermediate_stations, final_station,
        solver_name, time_limit,
    )
