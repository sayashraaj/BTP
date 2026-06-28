from __future__ import annotations

"""
3-Dimensional Matrix IP formulation for the Journey Planner.

Implements Chapter 2 of the thesis: modified MTZ constraints for a 3D binary
matrix, exit-time decision variable t_i, and time-propagation constraints.
"""

import logging
import time
from dataclasses import dataclass, field

import pulp

logger = logging.getLogger(__name__)

BIG_M = 100000


@dataclass
class ThreeDInput:
    """Input specification for the 3D Matrix formulation."""

    n: int  # number of spatial nodes (1..n, node 1 = origin)
    k_max: int  # max links per ordered pair
    C: dict[tuple[int, int, int], float]  # C[i,j,k] = travel time
    Start: dict[tuple[int, int, int], float]  # Start[i,j,k] = departure time
    Buffer: dict[int, float]  # Buffer[i] = mandatory dwell time at node i
    node_names: dict[int, str] = field(default_factory=dict)


@dataclass
class ThreeDResult:
    """Result from the 3D Matrix solver."""

    x_vals: dict[tuple[int, int, int], int]
    u_vals: dict[int, float]
    t_vals: dict[int, float]
    objective_value: float
    solver_status: str
    solve_time_seconds: float
    tour: list[int]  # ordered node indices
    tour_links: list[tuple[int, int, int]]  # selected (i,j,k) triples


def solve_3d(
    inp: ThreeDInput,
    solver_name: str = "cbc",
    time_limit: int = 300,
) -> ThreeDResult:
    """
    Solve the Journey Planner using the 3D Matrix IP formulation.

    The tour is a Hamiltonian cycle on nodes {1..n} returning to node 1.
    t[n+1] tracks the arrival time back at the origin (the objective).

    Since constraint (2.6) forces t[i] <= Start[i,j,k] when x[i,j,k]=1,
    max(t[i], Start) = Start when active. This eliminates the need for
    auxiliary w variables — constraint (2.9) simplifies to:
        Start[i,j,k] + C[i,j,k] + Buffer[j] <= t[j] + M*(1-x[i,j,k])
    """
    n = inp.n
    nodes = list(range(1, n + 1))
    return_idx = n + 1

    all_ijk = list(inp.C.keys())

    ijk_by_i = {}
    ijk_by_j = {}
    for i, j, k in all_ijk:
        ijk_by_i.setdefault(i, []).append((i, j, k))
        ijk_by_j.setdefault(j, []).append((i, j, k))

    prob = pulp.LpProblem("JourneyPlanner3D", pulp.LpMinimize)

    x = {}
    for ijk in all_ijk:
        i, j, k = ijk
        x[ijk] = pulp.LpVariable(f"x_{i}_{j}_{k}", cat="Binary")

    u = {}
    for i in nodes:
        u[i] = pulp.LpVariable(f"u_{i}", lowBound=0, upBound=n - 1, cat="Integer")

    t = {}
    for i in nodes:
        t[i] = pulp.LpVariable(f"t_{i}", lowBound=0, upBound=BIG_M)
    t[return_idx] = pulp.LpVariable(f"t_{return_idx}", lowBound=0, upBound=BIG_M)

    # Objective: minimize return time
    prob += t[return_idx], "MinimizeReturnTime"

    # Constraint (2.1): Outdegree = 1 for all i
    for i in nodes:
        links_from_i = ijk_by_i.get(i, [])
        if links_from_i:
            prob += (
                pulp.lpSum(x[ijk] for ijk in links_from_i) == 1,
                f"C2_1_outdegree_{i}",
            )

    # Constraint (2.2): Indegree = 1 for all j
    for j in nodes:
        links_to_j = ijk_by_j.get(j, [])
        if links_to_j:
            prob += (
                pulp.lpSum(x[ijk] for ijk in links_to_j) == 1,
                f"C2_2_indegree_{j}",
            )

    # Constraint (2.4): Modified 3D MTZ — for all i,j where i!=1 and j!=1
    for i in nodes:
        for j in nodes:
            if i == j or i == 1 or j == 1:
                continue
            links_ij = [ijk for ijk in all_ijk if ijk[0] == i and ijk[1] == j]
            if links_ij:
                prob += (
                    u[i] - u[j] + n * pulp.lpSum(x[ijk] for ijk in links_ij) <= n - 1,
                    f"C2_4_mtz3d_{i}_{j}",
                )

    # Constraint (2.5): u[1] = 0
    prob += u[1] == 0, "C2_5_source_rank"

    # Constraint (2.6): Upper bound on exit time — t[i] <= Start + M*(1-x)
    for ijk in all_ijk:
        i, j, k = ijk
        start_val = inp.Start[ijk]
        prob += (
            t[i] <= start_val + BIG_M * (1 - x[ijk]),
            f"C2_6_upper_t_{i}_{j}_{k}",
        )

    # Constraint (2.9) simplified: Since (2.6) gives t[i] <= Start when x=1,
    # max(t[i], Start) = Start. So the time propagation becomes:
    #   Start + C + Buffer[j] <= t[j] + M*(1-x)
    # For links returning to node 1, propagate to t[return_idx].
    for ijk in all_ijk:
        i, j, k = ijk
        start_val = inp.Start[ijk]
        c_val = inp.C[ijk]
        buf_j = inp.Buffer.get(j, 0)
        target_t = t[return_idx] if j == 1 else t[j]
        prob += (
            start_val + c_val + buf_j - BIG_M * (1 - x[ijk]) <= target_t,
            f"C2_9_timeprop_{i}_{j}_{k}",
        )

    # Source exit time
    prob += t[1] == inp.Buffer.get(1, 0), "source_exit_time"

    if solver_name.lower() == "gurobi":
        try:
            solver = pulp.GUROBI(msg=0, timeLimit=time_limit)
        except Exception:
            logger.warning("Gurobi not available, falling back to CBC")
            solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    else:
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)

    t0 = time.time()
    prob.solve(solver)
    solve_time = time.time() - t0

    x_vals = {}
    for ijk, var in x.items():
        val = var.varValue
        x_vals[ijk] = int(round(val)) if val is not None else 0

    u_vals = {}
    for i, var in u.items():
        val = var.varValue
        u_vals[i] = float(val) if val is not None else 0.0

    t_vals = {}
    for i, var in t.items():
        val = var.varValue
        t_vals[i] = float(val) if val is not None else 0.0

    selected_links = [ijk for ijk, val in x_vals.items() if val == 1]
    tour = _extract_tour(selected_links, u_vals, n)

    return ThreeDResult(
        x_vals=x_vals,
        u_vals=u_vals,
        t_vals=t_vals,
        objective_value=pulp.value(prob.objective) if pulp.value(prob.objective) is not None else float("inf"),
        solver_status=pulp.LpStatus[prob.status],
        solve_time_seconds=solve_time,
        tour=tour,
        tour_links=selected_links,
    )


def _extract_tour(
    selected_links: list[tuple[int, int, int]], u_vals: dict[int, float], n: int
) -> list[int]:
    """Extract the ordered tour from selected links using u-values."""
    next_node = {}
    for i, j, k in selected_links:
        next_node[i] = j

    tour = [1]
    current = 1
    for _ in range(n):
        nxt = next_node.get(current)
        if nxt is None or nxt == 1:
            if nxt == 1:
                tour.append(1)
            break
        tour.append(nxt)
        current = nxt
    else:
        nxt = next_node.get(current)
        if nxt == 1:
            tour.append(1)

    return tour


# ---------------------------------------------------------------------------
# Thesis example builders
# ---------------------------------------------------------------------------


def build_example_1() -> ThreeDInput:
    """Example 1: n=2, k=2 (Section 2.5)."""
    return ThreeDInput(
        n=2,
        k_max=2,
        C={
            (1, 2, 1): 1, (1, 2, 2): 3,
            (2, 1, 1): 7, (2, 1, 2): 5,
        },
        Start={
            (1, 2, 1): 6, (1, 2, 2): 2,
            (2, 1, 1): 7, (2, 1, 2): 10,
        },
        Buffer={1: 0, 2: 0},
        node_names={1: "Origin", 2: "Node 2"},
    )


def build_example_2() -> ThreeDInput:
    """Example 2: n=3, k=2 (Section 2.5)."""
    return ThreeDInput(
        n=3,
        k_max=2,
        C={
            (1, 2, 1): 1, (1, 2, 2): 3,
            (1, 3, 1): 100, (1, 3, 2): 200,
            (2, 1, 1): 7, (2, 1, 2): 5,
            (2, 3, 1): 100, (2, 3, 2): 50,
            (3, 1, 1): 1892, (3, 1, 2): 1000,
            (3, 2, 1): 50, (3, 2, 2): 100,
        },
        Start={
            (1, 2, 1): 6, (1, 2, 2): 2,
            (1, 3, 1): 5, (1, 3, 2): 10,
            (2, 1, 1): 7, (2, 1, 2): 10,
            (2, 3, 1): 8, (2, 3, 2): 10,
            (3, 1, 1): 108, (3, 1, 2): 200,
            (3, 2, 1): 50, (3, 2, 2): 100,
        },
        Buffer={1: 0, 2: 0, 3: 0},
        node_names={1: "Origin", 2: "Node 2", 3: "Node 3"},
    )


def build_example_3() -> ThreeDInput:
    """Example 3: n=3, k=2, subtour resilience check (Section 2.5)."""
    return build_example_2()


def build_example_4() -> ThreeDInput:
    """Example 4: n=4, k=3 (Section 2.5)."""
    C = {}
    Start = {}

    C[(1, 2, 1)] = 5; Start[(1, 2, 1)] = 0
    C[(1, 2, 2)] = 7; Start[(1, 2, 2)] = 2
    C[(1, 2, 3)] = 10; Start[(1, 2, 3)] = 5
    C[(1, 3, 1)] = 15; Start[(1, 3, 1)] = 0
    C[(1, 3, 2)] = 20; Start[(1, 3, 2)] = 3
    C[(1, 3, 3)] = 25; Start[(1, 3, 3)] = 6
    C[(1, 4, 1)] = 20; Start[(1, 4, 1)] = 0
    C[(1, 4, 2)] = 25; Start[(1, 4, 2)] = 4
    C[(1, 4, 3)] = 30; Start[(1, 4, 3)] = 7

    C[(2, 1, 1)] = 5; Start[(2, 1, 1)] = 5
    C[(2, 1, 2)] = 7; Start[(2, 1, 2)] = 8
    C[(2, 1, 3)] = 10; Start[(2, 1, 3)] = 12
    C[(2, 3, 1)] = 5; Start[(2, 3, 1)] = 5
    C[(2, 3, 2)] = 8; Start[(2, 3, 2)] = 7
    C[(2, 3, 3)] = 12; Start[(2, 3, 3)] = 10
    C[(2, 4, 1)] = 8; Start[(2, 4, 1)] = 5
    C[(2, 4, 2)] = 10; Start[(2, 4, 2)] = 8
    C[(2, 4, 3)] = 15; Start[(2, 4, 3)] = 11

    C[(3, 1, 1)] = 15; Start[(3, 1, 1)] = 10
    C[(3, 1, 2)] = 20; Start[(3, 1, 2)] = 15
    C[(3, 1, 3)] = 25; Start[(3, 1, 3)] = 20
    C[(3, 2, 1)] = 5; Start[(3, 2, 1)] = 10
    C[(3, 2, 2)] = 8; Start[(3, 2, 2)] = 13
    C[(3, 2, 3)] = 12; Start[(3, 2, 3)] = 16
    C[(3, 4, 1)] = 1; Start[(3, 4, 1)] = 10
    C[(3, 4, 2)] = 3; Start[(3, 4, 2)] = 13
    C[(3, 4, 3)] = 5; Start[(3, 4, 3)] = 16

    C[(4, 1, 1)] = 0; Start[(4, 1, 1)] = 11
    C[(4, 1, 2)] = 5; Start[(4, 1, 2)] = 15
    C[(4, 1, 3)] = 10; Start[(4, 1, 3)] = 20
    C[(4, 2, 1)] = 8; Start[(4, 2, 1)] = 11
    C[(4, 2, 2)] = 10; Start[(4, 2, 2)] = 15
    C[(4, 2, 3)] = 15; Start[(4, 2, 3)] = 18
    C[(4, 3, 1)] = 1; Start[(4, 3, 1)] = 11
    C[(4, 3, 2)] = 3; Start[(4, 3, 2)] = 14
    C[(4, 3, 3)] = 5; Start[(4, 3, 3)] = 17

    Buffer = {1: 0, 2: 0, 3: 0, 4: 0}
    return ThreeDInput(
        n=4, k_max=3, C=C, Start=Start, Buffer=Buffer,
        node_names={1: "Origin", 2: "Node 2", 3: "Node 3", 4: "Node 4"},
    )
