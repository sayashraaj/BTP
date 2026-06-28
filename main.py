"""
Journey Planner — CLI entry point.

Usage:
    python main.py --origin "Chennai Central" \
                   --destinations "Vijayawada Junction" "Warangal" \
                   --method tetn_mda3 \
                   --solver cbc \
                   --verify --visualize
"""

import argparse
import logging
import sys

from src.solver import format_tour, solve_query


def main():
    parser = argparse.ArgumentParser(
        description="Journey Planner: optimal multi-city train routing via Integer Programming"
    )
    parser.add_argument(
        "--origin", required=True, help="Origin station name or code"
    )
    parser.add_argument(
        "--destinations", nargs="+", required=True,
        help="Intermediate destination station names or codes",
    )
    parser.add_argument(
        "--final-destination", default=None,
        help="Final destination (defaults to origin for circular tours)",
    )
    parser.add_argument(
        "--method",
        choices=["3d", "tetn_mda1", "tetn_mda2", "tetn_mda3"],
        default="tetn_mda3",
        help="Solver method (default: tetn_mda3)",
    )
    parser.add_argument(
        "--solver", choices=["cbc", "gurobi"], default="cbc",
        help="IP solver backend (default: cbc)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run verification checks on the solution",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate an HTML visualization of the solution",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output path for the HTML visualization (default: results/solution.html)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        result = solve_query(
            origin=args.origin,
            destinations=args.destinations,
            method=args.method,
            solver=args.solver,
            verify=args.verify,
            final_destination=args.final_destination,
        )
        print(format_tour(result))

        if args.visualize:
            from src.visualize import save_html

            path = save_html(result, output_path=args.output)
            print(f"Visualization saved to {path}")

    except Exception as e:
        logging.error("Solver failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
