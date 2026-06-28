from __future__ import annotations

"""
Time-Expanded Transit Network (TETN) construction and pruning.

Implements Chapter 3 of the thesis: node expansion, link generation,
forward/backward star computation, and the pruning algorithm from Section 3.4.1.
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TETNNode:
    node_id: int
    station_id: str
    station_name: str
    train_id: str
    train_name: str
    time_seconds: int
    node_type: str  # "ARR" or "DEP"


@dataclass
class TETNLink:
    source: int
    target: int
    cost: int  # travel time in seconds
    link_type: str  # "TRAIN" or "TRANSFER"


@dataclass
class TETN:
    nodes: dict[int, TETNNode] = field(default_factory=dict)
    links: list[TETNLink] = field(default_factory=list)
    forward_star: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    backward_star: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def link_count(self) -> int:
        return len(self.links)


def build_tetn(schedule: pd.DataFrame) -> TETN:
    """
    Build a Time-Expanded Transit Network from the schedule DataFrame.

    Each row produces two nodes (arrival, departure). Links connect:
    1. Consecutive stops on the same train (DEP at stop i -> ARR at stop i+1)
    2. Dwell links within a stop (ARR -> DEP at same stop, same train)
    3. Transfers at the same station (ARR of one train -> DEP of another, later time)
    """
    net = TETN()
    node_id = 0

    station_dep_nodes: dict[str, list[int]] = defaultdict(list)
    station_arr_nodes: dict[str, list[int]] = defaultdict(list)

    train_nodes: dict[str, list[tuple[int, int, int]]] = defaultdict(list)

    for _, row in schedule.iterrows():
        tid = str(row["train_id"])
        tname = str(row["train_name"])
        sid = str(row["station_id"])
        sname = str(row["station_name"])
        arr_t = int(row["arrival_time_seconds"])
        dep_t = int(row["departure_time_seconds"])
        seq = int(row["sequence_number"])

        arr_node_id = node_id
        net.nodes[arr_node_id] = TETNNode(
            node_id=arr_node_id, station_id=sid, station_name=sname,
            train_id=tid, train_name=tname, time_seconds=arr_t, node_type="ARR",
        )
        station_arr_nodes[sid].append(arr_node_id)
        node_id += 1

        dep_node_id = node_id
        net.nodes[dep_node_id] = TETNNode(
            node_id=dep_node_id, station_id=sid, station_name=sname,
            train_id=tid, train_name=tname, time_seconds=dep_t, node_type="DEP",
        )
        station_dep_nodes[sid].append(dep_node_id)
        node_id += 1

        dwell_cost = dep_t - arr_t
        link = TETNLink(arr_node_id, dep_node_id, max(dwell_cost, 0), "TRAIN")
        net.links.append(link)
        net.forward_star[arr_node_id].append(dep_node_id)
        net.backward_star[dep_node_id].append(arr_node_id)

        train_nodes[tid].append((seq, dep_node_id, arr_node_id))

    for tid, stops in train_nodes.items():
        stops.sort(key=lambda x: x[0])
        for i in range(len(stops) - 1):
            _, dep_id, _ = stops[i]
            _, _, arr_id_next = stops[i + 1]
            dep_node = net.nodes[dep_id]
            arr_node = net.nodes[arr_id_next]
            cost = arr_node.time_seconds - dep_node.time_seconds
            if cost < 0:
                cost += 86400

            link = TETNLink(dep_id, arr_id_next, cost, "TRAIN")
            net.links.append(link)
            net.forward_star[dep_id].append(arr_id_next)
            net.backward_star[arr_id_next].append(dep_id)

    for sid, arr_ids in station_arr_nodes.items():
        dep_ids = station_dep_nodes.get(sid, [])
        for arr_id in arr_ids:
            arr_node = net.nodes[arr_id]
            for dep_id in dep_ids:
                dep_node = net.nodes[dep_id]
                if dep_node.train_id == arr_node.train_id:
                    continue
                if dep_node.time_seconds > arr_node.time_seconds:
                    cost = dep_node.time_seconds - arr_node.time_seconds
                    link = TETNLink(arr_id, dep_id, cost, "TRANSFER")
                    net.links.append(link)
                    net.forward_star[arr_id].append(dep_id)
                    net.backward_star[dep_id].append(arr_id)

    logger.info("Built TETN: %d nodes, %d links", net.node_count, net.link_count)
    return net


def prune_tetn(
    schedule: pd.DataFrame, destination_stations: list[str]
) -> tuple[pd.DataFrame, dict]:
    """
    Pruning algorithm from thesis Section 3.4.1.

    For each train, count how many stations from the destination set it stops at.
    Exclude the train entirely if count < 2.

    Accepts either station IDs (codes) or station names — tries both.
    """
    dest_set = set(destination_stations)
    original_trains = schedule["train_id"].nunique()
    original_rows = len(schedule)

    train_dest_counts = {}
    for tid, grp in schedule.groupby("train_id"):
        ids_in_train = set(grp["station_id"].unique())
        names_in_train = set(grp["station_name"].unique())
        count = len((ids_in_train | names_in_train) & dest_set)
        train_dest_counts[tid] = count

    keep_trains = {tid for tid, count in train_dest_counts.items() if count >= 2}
    pruned = schedule[schedule["train_id"].isin(keep_trains)].copy()

    stats = {
        "original_trains": original_trains,
        "original_rows": original_rows,
        "pruned_trains": len(keep_trains),
        "pruned_rows": len(pruned),
        "removed_trains": original_trains - len(keep_trains),
        "reduction_pct": round(
            (1 - len(pruned) / original_rows) * 100, 2
        ) if original_rows > 0 else 0,
    }

    logger.info(
        "Pruning: %d -> %d trains (%.1f%% rows removed)",
        stats["original_trains"], stats["pruned_trains"], stats["reduction_pct"],
    )
    return pruned, stats


def build_pruned_tetn(
    schedule: pd.DataFrame, destination_stations: list[str]
) -> tuple[TETN, dict]:
    """Prune the schedule then build the TETN on the pruned data."""
    pruned_schedule, stats = prune_tetn(schedule, destination_stations)
    net = build_tetn(pruned_schedule)
    stats["final_nodes"] = net.node_count
    stats["final_links"] = net.link_count
    return net, stats


def get_station_nodes(
    net: TETN, station: str, node_type: str | None = None
) -> list[int]:
    """Get all node IDs for a station (by ID or name), optionally filtered by type."""
    results = []
    for nid, node in net.nodes.items():
        if node.station_id == station or node.station_name == station:
            if node_type is None or node.node_type == node_type:
                results.append(nid)
    return sorted(results, key=lambda nid: net.nodes[nid].time_seconds)


def check_inter_destination_paths(
    net: TETN, destination_stations: list[str]
) -> list[dict]:
    """
    Verify pruning losslessness: for every ordered pair of destinations,
    confirm at least one directed path exists in the TETN.
    """
    results = []
    for src_station in destination_stations:
        for dst_station in destination_stations:
            if src_station == dst_station:
                continue
            src_nodes = get_station_nodes(net, src_station)
            dst_nodes_set = set(get_station_nodes(net, dst_station))

            path_found = False
            for start in src_nodes:
                visited = set()
                queue = deque([start])
                while queue:
                    current = queue.popleft()
                    if current in dst_nodes_set:
                        path_found = True
                        break
                    if current in visited:
                        continue
                    visited.add(current)
                    for neighbor in net.forward_star.get(current, []):
                        if neighbor not in visited:
                            queue.append(neighbor)
                if path_found:
                    break

            results.append({
                "from": src_station, "to": dst_station, "path_exists": path_found,
            })

    return results
