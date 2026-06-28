"""
Tests for the 3D Matrix IP formulation.

Covers all four thesis examples (Examples 1-4, Chapter 2) plus
subtour resilience and edge cases.
"""

import pytest

from src.formulation_3d import (
    ThreeDInput,
    build_example_1,
    build_example_2,
    build_example_3,
    build_example_4,
    solve_3d,
)
from src.verify import verify_3d_result


class TestExample1:
    """Example 1: n=2, k=2 — base case."""

    @pytest.fixture
    def result(self):
        inp = build_example_1()
        return inp, solve_3d(inp)

    def test_optimal_status(self, result):
        inp, res = result
        assert res.solver_status == "Optimal"

    def test_tour_visits_all_nodes(self, result):
        inp, res = result
        checks = verify_3d_result(inp, res)
        completeness = next(c for c in checks if c["check_name"] == "tour_completeness_3d")
        assert completeness["passed"]

    def test_no_subtours(self, result):
        inp, res = result
        checks = verify_3d_result(inp, res)
        subtour = next(c for c in checks if c["check_name"] == "no_subtours_3d")
        assert subtour["passed"]

    def test_u_values(self, result):
        _, res = result
        assert res.u_vals[1] == 0


class TestExample2:
    """Example 2: n=3, k=2 — three-node tour."""

    @pytest.fixture
    def result(self):
        inp = build_example_2()
        return inp, solve_3d(inp)

    def test_optimal_status(self, result):
        _, res = result
        assert res.solver_status == "Optimal"

    def test_all_checks_pass(self, result):
        inp, res = result
        checks = verify_3d_result(inp, res)
        for c in checks:
            assert c["passed"], f"{c['check_name']}: {c['detail']}"

    def test_tour_order(self, result):
        _, res = result
        assert res.u_vals[1] == 0
        assert res.u_vals[2] < res.u_vals[3] or res.u_vals[3] < res.u_vals[2]


class TestExample3:
    """Example 3: n=3, k=2 — subtour resilience check."""

    @pytest.fixture
    def result(self):
        inp = build_example_3()
        return inp, solve_3d(inp)

    def test_subtour_rejected(self, result):
        """The network admits 1<->2 subtour but MTZ must reject it."""
        inp, res = result
        checks = verify_3d_result(inp, res)
        subtour = next(c for c in checks if c["check_name"] == "no_subtours_3d")
        assert subtour["passed"], "MTZ failed to eliminate subtour"

    def test_all_nodes_visited(self, result):
        inp, res = result
        tour_nodes = set()
        for i, j, k in res.tour_links:
            tour_nodes.add(i)
        assert 1 in tour_nodes
        assert 2 in tour_nodes
        assert 3 in tour_nodes


class TestExample4:
    """Example 4: n=4, k=3 — maximum complexity."""

    @pytest.fixture
    def result(self):
        inp = build_example_4()
        return inp, solve_3d(inp)

    def test_optimal_status(self, result):
        _, res = result
        assert res.solver_status == "Optimal"

    def test_objective_value(self, result):
        _, res = result
        assert res.objective_value <= 11 + 1e-6, f"Objective {res.objective_value} > 11"

    def test_tour_visits_all_4_nodes(self, result):
        inp, res = result
        checks = verify_3d_result(inp, res)
        completeness = next(c for c in checks if c["check_name"] == "tour_completeness_3d")
        assert completeness["passed"]

    def test_all_checks_pass(self, result):
        inp, res = result
        checks = verify_3d_result(inp, res)
        for c in checks:
            assert c["passed"], f"{c['check_name']}: {c['detail']}"

    def test_u_values_increasing(self, result):
        """u-values must be strictly increasing among non-depot tour nodes."""
        _, res = result
        assert res.u_vals[1] == 0
        non_depot = [n for n in res.tour if n != 1 and n in res.u_vals]
        for i in range(len(non_depot) - 1):
            assert res.u_vals[non_depot[i]] < res.u_vals[non_depot[i + 1]]


class TestSubtourElimination:
    """
    The single most important correctness property of the formulation:
    when a subtour is strictly cheaper than the Hamiltonian tour, the
    modified 3D MTZ constraints must force the solver to reject it and
    visit every node.

    These tests construct networks where relaxing subtour elimination
    would produce a lower-cost solution, then confirm the solver picks
    the more expensive Hamiltonian tour instead.
    """

    def test_2d_subtour_strictly_cheaper(self):
        """
        n=3, k=1. Subtour {1,2} costs 2 (return time ~1). Hamiltonian
        through node 3 costs 251 (return time 250). Without MTZ the
        solver would skip node 3 entirely.
        """
        C = {
            (1, 2, 1): 1, (2, 1, 1): 1,
            (1, 3, 1): 100, (3, 1, 1): 100,
            (2, 3, 1): 50, (3, 2, 1): 50,
        }
        Start = {
            (1, 2, 1): 0, (2, 1, 1): 1,
            (1, 3, 1): 0, (3, 1, 1): 150,
            (2, 3, 1): 1, (3, 2, 1): 100,
        }
        inp = ThreeDInput(
            n=3, k_max=1, C=C, Start=Start,
            Buffer={1: 0, 2: 0, 3: 0},
        )
        res = solve_3d(inp)
        assert res.solver_status == "Optimal"

        visited = set()
        for i, j, k in res.tour_links:
            visited.add(i)
        assert 3 in visited, "Node 3 must be visited — subtour {1,2} should be rejected by MTZ"

    def test_3d_subtour_across_link_options(self):
        """
        n=4, k=2. The key 3D-specific scenario: a subtour {1,2} using
        link option k=1 and a disjoint subtour {3,4} using k=2 would
        together satisfy outdegree/indegree constraints (2.1)/(2.2) —
        each node has exactly one outgoing and one incoming link. The
        modified MTZ must detect that these are two disconnected cycles,
        not one Hamiltonian tour, and reject the decomposition.

        Without MTZ, the solver could pick:
          Cycle A: 1 --(k=1)--> 2 --(k=1)--> 1   (return time 2)
          Cycle B: 3 --(k=2)--> 4 --(k=2)--> 3   (return time 2)
        Both cycles satisfy (2.1)/(2.2) independently — each node
        has outdegree 1 and indegree 1. The 3D MTZ must reject this.

        Start times are set so the Hamiltonian tour 1->2->3->4->1 is
        feasible but more expensive (return time 44 vs. 2).
        """
        C = {
            # Cheap subtour links
            (1, 2, 1): 1, (2, 1, 1): 1,  # subtour {1,2} via k=1
            (3, 4, 2): 1, (4, 3, 2): 1,  # subtour {3,4} via k=2
            # Cross-links (needed for a valid Hamiltonian tour)
            (1, 2, 2): 1, (2, 1, 2): 1,
            (1, 3, 1): 10, (1, 3, 2): 10,
            (1, 4, 1): 10, (1, 4, 2): 10,
            (2, 3, 1): 5, (2, 3, 2): 5,
            (2, 4, 1): 5, (2, 4, 2): 5,
            (3, 1, 1): 1, (3, 1, 2): 1,
            (3, 2, 1): 5, (3, 2, 2): 5,
            (3, 4, 1): 1, (4, 3, 1): 1,
            (4, 1, 1): 1, (4, 1, 2): 1,
            (4, 2, 1): 5, (4, 2, 2): 5,
        }
        # Start times ensure the Hamiltonian tour is feasible:
        # t[1]=0, depart at 0, arrive node 2 at 1, depart at 5,
        # arrive node 3 at 10, depart at 20, arrive node 4 at 21,
        # depart at 30, return at 31.
        Start = {}
        for ijk in C:
            i, j, k = ijk
            Start[ijk] = {1: 0, 2: 5, 3: 20, 4: 30}[i]

        inp = ThreeDInput(
            n=4, k_max=2, C=C, Start=Start,
            Buffer={1: 0, 2: 0, 3: 0, 4: 0},
        )
        res = solve_3d(inp)
        assert res.solver_status == "Optimal"

        visited = set()
        for i, j, k in res.tour_links:
            visited.add(i)
            visited.add(j)
        assert {1, 2, 3, 4}.issubset(visited), (
            f"All 4 nodes must be visited — disjoint subtours {{1,2}} and {{3,4}} "
            f"should be rejected by modified MTZ. Visited: {visited}"
        )

        # Verify it's a single connected tour, not two cycles
        next_node = {}
        for i, j, k in res.tour_links:
            next_node[i] = j
        current = 1
        chain = [1]
        for _ in range(4):
            nxt = next_node.get(current)
            if nxt is None or nxt == 1:
                break
            chain.append(nxt)
            current = nxt
        assert len(chain) == 4, (
            f"Tour must visit all 4 nodes in a single chain from node 1. "
            f"Got chain: {chain}"
        )
