"""
Benchmark runner: reproduces all thesis examples and records solve times.

Writes results to benchmarks/results.json.
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.formulation_3d import (
    build_example_1,
    build_example_2,
    build_example_4,
    solve_3d,
)
from src.verify import verify_3d_result

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


def benchmark_3d_examples():
    """Run all four 3D Matrix examples from Chapter 2."""
    results = []

    examples = [
        ("Example 1 (n=2, k=2)", build_example_1),
        ("Example 2 (n=3, k=2)", build_example_2),
        ("Example 3 (n=3, k=2, subtour check)", build_example_2),
        ("Example 4 (n=4, k=3)", build_example_4),
    ]

    for name, builder in examples:
        inp = builder()
        t0 = time.time()
        res = solve_3d(inp)
        wall_time = time.time() - t0

        checks = verify_3d_result(inp, res)
        all_passed = all(c["passed"] for c in checks)

        result = {
            "name": name,
            "method": "3D Matrix",
            "status": res.solver_status,
            "objective": res.objective_value,
            "tour": res.tour,
            "u_vals": {str(k): v for k, v in res.u_vals.items()},
            "t_vals": {str(k): v for k, v in res.t_vals.items()},
            "solve_time_seconds": round(res.solve_time_seconds, 4),
            "wall_time_seconds": round(wall_time, 4),
            "verification_passed": all_passed,
            "checks": checks,
        }
        results.append(result)

        status = "PASS" if all_passed else "FAIL"
        logger.info(
            "[%s] %s: objective=%.1f, tour=%s, solve=%.3fs",
            status, name, res.objective_value, res.tour, res.solve_time_seconds,
        )

    return results


def benchmark_tetn_real_data():
    """Run TETN benchmarks on real Indian Railways data if available."""
    from src.data_pipeline import REAL_DATA_PATH, run_pipeline
    from src.network import build_pruned_tetn, check_inter_destination_paths
    from src.formulation_tetn import solve_mda3
    from src.verify import verify_tetn_result, check_pruning_losslessness

    results = []

    if not REAL_DATA_PATH.exists():
        logger.warning("Real data not found at %s; skipping TETN benchmarks", REAL_DATA_PATH)
        return results

    t0 = time.time()
    schedule = run_pipeline()
    data_time = time.time() - t0
    logger.info("Data pipeline: %.1fs, %d rows, %d trains",
                data_time, len(schedule), schedule["train_id"].nunique())

    test_cases = [
        {
            "name": "MDA3: Chennai->Vijayawada->Warangal->Chennai (thesis MDA1 route)",
            "origin": "MAS",
            "intermediates": ["BZA", "WL"],
            "final": "MAS",
        },
        {
            "name": "MDA3(c): New Delhi->Bhopal->Jhansi->Warangal->Vijayawada->Tiruchchirappalli",
            "origin": "NDLS",
            "intermediates": ["BPL", "JHS", "WL", "BZA"],
            "final": "TPJ",
        },
        {
            "name": "MDA3(d): Kanpur->Bhopal->Jhansi->Warangal->Vijayawada->Tiruchchirappalli",
            "origin": "CNB",
            "intermediates": ["BPL", "JHS", "WL", "BZA"],
            "final": "TPJ",
        },
    ]

    for case in test_cases:
        logger.info("Running: %s", case["name"])
        all_stations = [case["origin"]] + case["intermediates"] + [case["final"]]

        t0 = time.time()
        net, net_stats = build_pruned_tetn(schedule, all_stations)
        network_time = time.time() - t0

        t0 = time.time()
        res = solve_mda3(
            net, case["origin"], case["intermediates"], case["final"],
            solver_name="cbc", time_limit=600,
        )
        solve_time = time.time() - t0

        checks = []
        if res.solver_status == "Optimal":
            checks = verify_tetn_result(net, res, all_stations)
            path_results = check_inter_destination_paths(net, all_stations)
            checks.append(check_pruning_losslessness(path_results))

        all_passed = all(c["passed"] for c in checks) if checks else False

        tour_summary = []
        for d in res.tour_details:
            tour_summary.append({
                "node_id": d["node_id"],
                "train_id": d["train_id"],
                "train_name": d["train_name"],
                "time_seconds": d["time_seconds"],
                "station": d["station_name"],
            })

        result = {
            "name": case["name"],
            "method": "TETN MDA3",
            "status": res.solver_status,
            "objective": res.objective_value,
            "network_stats": net_stats,
            "network_build_seconds": round(network_time, 4),
            "solve_time_seconds": round(res.solve_time_seconds, 4),
            "total_time_seconds": round(network_time + solve_time, 4),
            "verification_passed": all_passed,
            "tour_node_count": len(res.tour_nodes),
            "tour_summary": tour_summary[:30],
            "checks": checks,
        }
        results.append(result)

        status = "PASS" if all_passed else "FAIL"
        logger.info(
            "[%s] %s: status=%s, objective=%.0f, nodes=%d, total=%.1fs",
            status, case["name"], res.solver_status,
            res.objective_value, len(res.tour_nodes),
            network_time + solve_time,
        )

    return results


def main():
    all_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "3d_matrix": benchmark_3d_examples(),
        "tetn": benchmark_tetn_real_data(),
    }

    RESULTS_PATH.write_text(json.dumps(all_results, indent=2, default=str))
    logger.info("Results written to %s", RESULTS_PATH)

    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)

    print("\n  3D Matrix Examples:")
    for r in all_results["3d_matrix"]:
        v = "PASS" if r["verification_passed"] else "FAIL"
        print(f"    [{v}] {r['name']}: obj={r['objective']}, "
              f"tour={r['tour']}, solve={r['solve_time_seconds']}s")

    if all_results["tetn"]:
        print("\n  TETN (Real Indian Railways Data):")
        for r in all_results["tetn"]:
            v = "PASS" if r["verification_passed"] else "FAIL"
            stats = r.get("network_stats", {})
            print(f"    [{v}] {r['name']}:")
            print(f"         status={r['status']}, objective={r['objective']}")
            print(f"         trains: {stats.get('original_trains', '?')} -> {stats.get('pruned_trains', '?')} "
                  f"({stats.get('reduction_pct', '?')}% removed)")
            print(f"         nodes={stats.get('final_nodes', '?')}, links={stats.get('final_links', '?')}")
            print(f"         solve={r['solve_time_seconds']}s, total={r['total_time_seconds']}s")

    print("=" * 70)


if __name__ == "__main__":
    main()
