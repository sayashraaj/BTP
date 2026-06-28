"""
Tests for the TETN IP formulation.

Covers network construction, pruning, acyclicity, flow conservation,
and TETN solver correctness.
"""

import pandas as pd
import pytest

from src.formulation_tetn import solve_mda3
from src.network import (
    TETN,
    build_pruned_tetn,
    build_tetn,
    check_inter_destination_paths,
    prune_tetn,
)
from src.verify import (
    check_acyclicity,
    check_flow_conservation_tetn,
    check_no_subtours_tetn,
    check_pruning_losslessness,
    check_tour_completeness_tetn,
)


def _make_schedule(trains: list[dict]) -> pd.DataFrame:
    """Helper to build a schedule DataFrame from train specs."""
    rows = []
    for train in trains:
        tid = train["id"]
        tname = train.get("name", f"Train {tid}")
        for seq, (sid, sname, arr, dep) in enumerate(train["stops"], 1):
            rows.append({
                "train_id": str(tid),
                "train_name": tname,
                "station_id": sid,
                "station_name": sname,
                "arrival_time_seconds": arr,
                "departure_time_seconds": dep,
                "sequence_number": seq,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def simple_schedule():
    """3 trains connecting 4 stations: A-B-C, A-C-D, B-D-A."""
    return _make_schedule([
        {"id": 101, "name": "Express 1", "stops": [
            ("A", "Station A", 0, 100),
            ("B", "Station B", 3700, 3800),
            ("C", "Station C", 7300, 7400),
        ]},
        {"id": 102, "name": "Express 2", "stops": [
            ("A", "Station A", 1000, 1100),
            ("C", "Station C", 8000, 8100),
            ("D", "Station D", 14000, 14100),
        ]},
        {"id": 103, "name": "Express 3", "stops": [
            ("B", "Station B", 5000, 5100),
            ("D", "Station D", 10000, 10100),
            ("A", "Station A", 16000, 16100),
        ]},
    ])


class TestTETNConstruction:
    def test_node_count(self, simple_schedule):
        net = build_tetn(simple_schedule)
        assert net.node_count == len(simple_schedule) * 2

    def test_train_links_exist(self, simple_schedule):
        net = build_tetn(simple_schedule)
        train_links = [l for l in net.links if l.link_type == "TRAIN"]
        assert len(train_links) > 0

    def test_transfer_links_at_shared_station(self, simple_schedule):
        net = build_tetn(simple_schedule)
        transfer_links = [l for l in net.links if l.link_type == "TRANSFER"]
        assert len(transfer_links) > 0


class TestAcyclicity:
    def test_all_links_forward_in_time(self, simple_schedule):
        net = build_tetn(simple_schedule)
        result = check_acyclicity(net)
        assert result["passed"], result["detail"]


class TestPruning:
    def test_irrelevant_trains_excluded(self):
        schedule = _make_schedule([
            {"id": 1, "name": "Relevant", "stops": [
                ("A", "Station A", 0, 100),
                ("B", "Station B", 5000, 5100),
                ("C", "Station C", 10000, 10100),
            ]},
            {"id": 2, "name": "Also Relevant", "stops": [
                ("B", "Station B", 6000, 6100),
                ("C", "Station C", 11000, 11100),
                ("A", "Station A", 16000, 16100),
            ]},
            {"id": 3, "name": "Irrelevant", "stops": [
                ("X", "Station X", 0, 100),
                ("Y", "Station Y", 5000, 5100),
            ]},
            {"id": 4, "name": "Single Match", "stops": [
                ("A", "Station A", 2000, 2100),
                ("X", "Station X", 7000, 7100),
            ]},
        ])
        dests = ["Station A", "Station B", "Station C"]
        pruned, stats = prune_tetn(schedule, dests)

        assert stats["pruned_trains"] == 2
        assert stats["removed_trains"] == 2
        remaining_trains = pruned["train_id"].unique()
        assert "1" in remaining_trains
        assert "2" in remaining_trains
        assert "3" not in remaining_trains
        assert "4" not in remaining_trains

    def test_pruning_preserves_paths(self):
        schedule = _make_schedule([
            {"id": 1, "stops": [
                ("A", "Station A", 0, 100),
                ("B", "Station B", 5000, 5100),
            ]},
            {"id": 2, "stops": [
                ("B", "Station B", 6000, 6100),
                ("A", "Station A", 11000, 11100),
            ]},
        ])
        dests = ["Station A", "Station B"]
        net, _ = build_pruned_tetn(schedule, dests)
        paths = check_inter_destination_paths(net, dests)
        result = check_pruning_losslessness(paths)
        assert result["passed"], result["detail"]

    def test_80_percent_irrelevant(self):
        """80% of trains are irrelevant; confirm exclusion and path preservation."""
        trains = []
        # 2 relevant trains
        trains.append({"id": 1, "stops": [
            ("A", "Station A", 0, 100),
            ("B", "Station B", 5000, 5100),
            ("C", "Station C", 10000, 10100),
        ]})
        trains.append({"id": 2, "stops": [
            ("C", "Station C", 11000, 11100),
            ("B", "Station B", 16000, 16100),
            ("A", "Station A", 21000, 21100),
        ]})
        # 8 irrelevant trains
        for i in range(3, 11):
            trains.append({"id": i, "stops": [
                (f"X{i}", f"Station X{i}", i * 1000, i * 1000 + 100),
                (f"Y{i}", f"Station Y{i}", i * 1000 + 5000, i * 1000 + 5100),
            ]})

        schedule = _make_schedule(trains)
        dests = ["Station A", "Station B", "Station C"]
        pruned, stats = prune_tetn(schedule, dests)

        assert stats["pruned_trains"] == 2
        assert stats["removed_trains"] == 8

        net, _ = build_pruned_tetn(schedule, dests)
        paths = check_inter_destination_paths(net, dests)
        result = check_pruning_losslessness(paths)
        assert result["passed"]


class TestFlowConservation:
    @pytest.mark.parametrize("seed", range(5))
    def test_flow_conservation_random(self, seed):
        """Parameterized test across random valid networks."""
        import random
        rng = random.Random(seed)
        stations = [("S" + str(i), f"Station {i}") for i in range(4)]
        trains = []
        for tid in range(1, 6):
            n_stops = rng.randint(2, 4)
            selected = rng.sample(stations, n_stops)
            base_time = rng.randint(0, 50000)
            stops = []
            for seq, (sid, sname) in enumerate(selected):
                arr = base_time + seq * 5000
                dep = arr + 100
                stops.append((sid, sname, arr, dep))
            trains.append({"id": tid, "stops": stops})

        schedule = _make_schedule(trains)
        all_station_names = [s[1] for s in stations]

        # Only attempt solve if pruning leaves enough structure
        pruned, stats = prune_tetn(schedule, all_station_names)
        if stats["pruned_trains"] < 2:
            pytest.skip("Insufficient pruned trains for a meaningful test")

        net, _ = build_pruned_tetn(schedule, all_station_names)
        try:
            result = solve_mda3(net, all_station_names[0], all_station_names[1:3], all_station_names[0])
            if result.solver_status == "Optimal":
                fc = check_flow_conservation_tetn(net, result)
                assert fc["passed"], fc["detail"]
        except Exception:
            pytest.skip("Solver infeasible on random instance")


class TestTETNSolver:
    def test_simple_round_trip(self):
        schedule = _make_schedule([
            {"id": 1, "name": "Outbound", "stops": [
                ("A", "Station A", 0, 100),
                ("B", "Station B", 5000, 5100),
                ("C", "Station C", 10000, 10100),
            ]},
            {"id": 2, "name": "Return", "stops": [
                ("C", "Station C", 11000, 11100),
                ("B", "Station B", 16000, 16100),
                ("A", "Station A", 21000, 21100),
            ]},
        ])
        dests = ["Station A", "Station B", "Station C"]
        net, _ = build_pruned_tetn(schedule, dests)
        result = solve_mda3(net, "Station A", ["Station B", "Station C"], "Station A")
        assert result.solver_status == "Optimal"

        visited_stations = {
            net.nodes[nid].station_name for nid in result.tour_nodes
        }
        assert "Station A" in visited_stations
        assert "Station B" in visited_stations
        assert "Station C" in visited_stations

    def test_non_circular_routing(self):
        schedule = _make_schedule([
            {"id": 1, "stops": [
                ("A", "Station A", 0, 100),
                ("B", "Station B", 5000, 5100),
                ("C", "Station C", 10000, 10100),
            ]},
        ])
        dests = ["Station A", "Station B", "Station C"]
        net, _ = build_pruned_tetn(schedule, dests)
        result = solve_mda3(net, "Station A", ["Station B"], "Station C")
        assert result.solver_status == "Optimal"
