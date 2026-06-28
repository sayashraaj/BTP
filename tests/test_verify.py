"""
Tests for the verification module itself.

Ensures that each check correctly identifies passing and failing conditions.
"""

import pytest

from src.formulation_3d import ThreeDInput, ThreeDResult
from src.formulation_tetn import TETNResult
from src.network import TETN, TETNLink, TETNNode
from src.verify import (
    check_acyclicity,
    check_flow_conservation_tetn,
    check_mtz_3d,
    check_no_subtours_3d,
    check_no_subtours_tetn,
    check_pruning_losslessness,
    check_time_feasibility_3d,
    check_tour_completeness_3d,
    check_tour_completeness_tetn,
)


class TestTourCompleteness3D:
    def test_pass(self):
        inp = ThreeDInput(n=3, k_max=1, C={}, Start={}, Buffer={})
        res = ThreeDResult(
            x_vals={(1, 2, 1): 1, (2, 3, 1): 1, (3, 4, 1): 1},
            u_vals={1: 0, 2: 1, 3: 2},
            t_vals={1: 0, 2: 5, 3: 10, 4: 15},
            objective_value=15,
            solver_status="Optimal",
            solve_time_seconds=0.1,
            tour=[1, 2, 3, 4],
            tour_links=[(1, 2, 1), (2, 3, 1), (3, 4, 1)],
        )
        result = check_tour_completeness_3d(inp, res)
        assert result["passed"]

    def test_fail_missing_node(self):
        inp = ThreeDInput(n=3, k_max=1, C={}, Start={}, Buffer={})
        res = ThreeDResult(
            x_vals={(1, 2, 1): 1, (2, 4, 1): 1},  # skips node 3
            u_vals={1: 0, 2: 1},
            t_vals={1: 0, 2: 5, 4: 10},
            objective_value=10,
            solver_status="Optimal",
            solve_time_seconds=0.1,
            tour=[1, 2, 4],
            tour_links=[(1, 2, 1), (2, 4, 1)],
        )
        result = check_tour_completeness_3d(inp, res)
        assert not result["passed"]


class TestNoSubtours3D:
    def test_pass(self):
        res = ThreeDResult(
            x_vals={(1, 2, 1): 1, (2, 3, 1): 1},
            u_vals={1: 0, 2: 1, 3: 2},
            t_vals={}, objective_value=0,
            solver_status="Optimal", solve_time_seconds=0,
            tour=[1, 2, 3], tour_links=[(1, 2, 1), (2, 3, 1)],
        )
        result = check_no_subtours_3d(res)
        assert result["passed"]

    def test_fail_u_not_increasing(self):
        res = ThreeDResult(
            x_vals={(1, 2, 1): 1, (2, 3, 1): 1},
            u_vals={1: 0, 2: 2, 3: 1},  # 3 < 2 but 2->3 selected
            t_vals={}, objective_value=0,
            solver_status="Optimal", solve_time_seconds=0,
            tour=[1, 2, 3], tour_links=[(1, 2, 1), (2, 3, 1)],
        )
        result = check_no_subtours_3d(res)
        assert not result["passed"]


class TestAcyclicity:
    def _make_net(self, links_data):
        net = TETN()
        for nid, t in [(0, 0), (1, 100), (2, 200), (3, 50)]:
            net.nodes[nid] = TETNNode(nid, "S", "S", "T", "T", t, "DEP")
        for s, t in links_data:
            net.links.append(TETNLink(s, t, abs(net.nodes[t].time_seconds - net.nodes[s].time_seconds), "TRAIN"))
        return net

    def test_acyclic(self):
        net = self._make_net([(0, 1), (1, 2)])
        assert check_acyclicity(net)["passed"]

    def test_not_acyclic(self):
        net = self._make_net([(0, 1), (1, 3)])  # node 3 has time 50 < 100
        assert not check_acyclicity(net)["passed"]


class TestPruningLosslessness:
    def test_all_paths_exist(self):
        results = [
            {"from": "A", "to": "B", "path_exists": True},
            {"from": "B", "to": "A", "path_exists": True},
        ]
        assert check_pruning_losslessness(results)["passed"]

    def test_missing_path(self):
        results = [
            {"from": "A", "to": "B", "path_exists": True},
            {"from": "B", "to": "A", "path_exists": False},
        ]
        assert not check_pruning_losslessness(results)["passed"]


class TestFlowConservationTETN:
    def test_pass_simple(self):
        net = TETN()
        res = TETNResult(
            x_vals={(0, 1): 1, (1, 2): 1},
            u_vals={0: 0, 1: 1, 2: 2},
            objective_value=100,
            solver_status="Optimal",
            solve_time_seconds=0.1,
            tour_nodes=[0, 1, 2],
            tour_details=[],
        )
        result = check_flow_conservation_tetn(net, res)
        assert result["passed"]

    def test_fail_imbalanced(self):
        net = TETN()
        res = TETNResult(
            x_vals={(0, 1): 1, (2, 1): 1, (1, 3): 1},  # node 1: in=2, out=1
            u_vals={},
            objective_value=0,
            solver_status="Optimal",
            solve_time_seconds=0,
            tour_nodes=[],
            tour_details=[],
        )
        result = check_flow_conservation_tetn(net, res)
        assert not result["passed"]
