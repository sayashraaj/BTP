# Journey Planner — Integer Programming for Optimal Train Routing

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Exact, provably optimal multi-city train journey planning on the Indian Railways network (237,449 schedule entries, 12,873 trains, 8,601 stations), formulated as a constrained time-dependent variant of the Travelling Salesman Problem and solved via Integer Programming. Based on a B.Tech thesis at IIT Madras (2023).

## Mathematical contributions

### Modified MTZ constraints for 3D binary matrices

The Miller-Tucker-Zemlin (1960) subtour-elimination constraints are among the most widely used tools in IP for the TSP. They were formulated for a standard 2D assignment matrix where each ordered node-pair has one arc. This work independently derives a modification for a **3-dimensional** cost structure where each ordered node-pair *(i, j)* admits *k* time-dependent link options, each with its own departure time and travel cost:

```
u_i - u_j + n * sum_k(x_ijk) <= n - 1    for all i != j
```

Since the outdegree and indegree constraints force `sum_k(x_ijk) in {0, 1}`, the subtour logic is preserved exactly: when any link *k* is selected between *i* and *j*, the visit-rank of *j* must exceed that of *i*. This is a direct extension of the original MTZ to 3D binary tensors — to the author's knowledge not present in prior TSP literature in this form.

### Exit-time variable and time-propagation IP

Standard TSP minimises a fixed arc cost. The journey planning problem requires tracking *when* the tour is ready to depart each node, accounting for arrival time, mandatory dwell periods, and discrete train departure schedules. This thesis introduces a continuous decision variable *t_i* (exit time) and constraint families that propagate timing through the network:

- **Upper bound (2.6):** `t_i <= Start_ijk + M * (1 - x_ijk)` — the tour cannot board a train that has already departed
- **Time propagation (2.9):** `Start_ijk + C_ijk + Buffer_j <= t_j + M * (1 - x_ijk)` — arrival at *j* sets a lower bound on *t_j*

The objective minimises `t_destination` (earliest return time to origin) rather than sum of arc costs — reframing the classical TSP as a time-minimisation problem.

### Time-Expanded Transit Network (TETN)

The TETN method encodes time directly into node identities: each physical station becomes a set of time-stamped nodes (arrival and departure). The resulting directed acyclic graph eliminates cycles by construction and removes the need for the *t_i* variable entirely. A 7-constraint IP formulation with asymmetric boundary conditions on origin, intermediate, and destination node sets handles both circular and non-circular routing. A pruning algorithm reduces the full Indian Railways schedule to a tractable subgraph by excluding trains that stop at fewer than 2 query destinations.

For a detailed walkthrough of why the 3D MTZ modification is valid — including why `Σ_k x_ijk` can substitute for `x_ij` and how disjoint subtours across different link options are caught — see [docs/mathematical_notes.md](docs/mathematical_notes.md).

## Design decisions

### Eliminating the auxiliary `max()` linearization

The thesis formulation (2.9) contains `max(t_i, Start_ijk)` — the departure time is whichever is later: when the tour is ready, or when the train departs. Standard Big-M linearization would introduce auxiliary variables *w_ijk* with four additional constraints per link. However, constraint (2.6) already forces `t_i <= Start_ijk` when `x_ijk = 1`, which means `max(t_i, Start) = Start` whenever a link is active. This collapses the time propagation to:

```
Start_ijk + C_ijk + Buffer_j <= t_j + M * (1 - x_ijk)
```

This eliminates the entire *w* variable family and its 4 constraints per link, producing a tighter formulation that solves faster and avoids numerical issues from contradictory upper/lower bounds on the auxiliary variables.

## Empirical results

| Test Case | Method | Network | Pruning | CBC | Gurobi (thesis) |
|-----------|--------|---------|---------|-----|-----------------|
| Examples 1-4 | 3D Matrix | 2-4 nodes, synthetic | N/A | < 0.1s each | < 1s each |
| Chennai-Vijayawada-Warangal-Chennai | TETN MDA3 | 12,516 nodes, 136K links | 12,873 -> 198 trains (97.4%) | 598s | 8s |
| NDLS-BPL-JHS-WL-BZA-TPJ | TETN MDA3 | 16,952 nodes, 209K links | 12,873 -> 301 trains (96.4%) | 69s | 12s |
| CNB-BPL-JHS-WL-BZA-TPJ | TETN MDA3 | 18,528 nodes, 228K links | 12,873 -> 325 trains (96.1%) | 44s | 13s |

All solutions are **exact** (provably optimal). CBC is the default open-source solver; Gurobi (thesis column) requires an academic license but solves 3-50x faster. Set `SOLVER=gurobi` to use Gurobi if installed. Run `python benchmarks/run_benchmarks.py` to reproduce.

## Installation

```bash
git clone <repo-url>
cd journey-planner-ip
pip install -r requirements.txt
```

The real Indian Railways schedule (`data/raw/TrainScheduleDB.csv`, 237,449 rows) is included. The pipeline also supports downloading from data.gov.in or generating a synthetic fallback.

## Usage

### CLI

```bash
# Circular tour (station codes)
python main.py --origin MAS --destinations BZA WL --method tetn_mda3 --verify

# Non-circular tour
python main.py --origin NDLS \
               --destinations BPL JHS WL BZA \
               --final-destination TPJ \
               --method tetn_mda3 --verify

# Full station names also work
python main.py --origin "CHENNAI CENTRAL" \
               --destinations "VIJAYAWADA JUNCTION" "WARANGAL" \
               --method tetn_mda3 --verify
```

### As a library

```python
from src.solver import solve_query, format_tour

result = solve_query(
    origin="MAS",
    destinations=["BZA", "WL"],
    method="tetn_mda3",
    verify=True,
)
print(format_tour(result))
```

### 3D Matrix examples (Chapter 2)

```python
from src.formulation_3d import build_example_4, solve_3d
from src.verify import verify_3d_result

inp = build_example_4()
res = solve_3d(inp)
# objective=11.0, tour=[1, 2, 3, 4, 1] — matches thesis exactly
```

### Benchmarks

```bash
python benchmarks/run_benchmarks.py
# Writes detailed results to benchmarks/results.json
```

## Verification methodology

Every solution is verified programmatically against the thesis constraints:

- **Tour completeness** — all destinations visited
- **Flow conservation** — in-degree equals out-degree at interior nodes
- **Subtour elimination** — u-values strictly increase among non-depot nodes (3D); all links move forward in time (TETN)
- **Time feasibility** — no train boarded after departure
- **MTZ constraint satisfaction** — `u[i] - u[j] + n * sum_k(x[i,j,k]) <= n-1` for all non-depot pairs
- **Pruning losslessness** — all inter-destination paths preserved after pruning
- **TETN acyclicity** — all links move strictly forward in time

## Data source

**Indian Railways Train Time Table** — 237,449 schedule entries covering 12,873 trains and 8,601 stations. Originally sourced from [data.gov.in](https://data.gov.in/catalog/indian-railways-train-time-table). The dataset includes multi-day journey encoding (Day field, 1-5) and handles Source/Destination markers for terminus stations.

## Project structure

```
journey-planner-ip/
├── data/
│   ├── raw/               # TrainScheduleDB.csv (237K rows)
│   └── processed/         # cleaned schedule.csv
├── src/
│   ├── data_pipeline.py   # fetch, parse, validate
│   ├── network.py         # TETN construction + pruning
│   ├── formulation_3d.py  # 3D Matrix IP (Chapter 2)
│   ├── formulation_tetn.py# TETN IP MDA1/2/3 (Chapter 3)
│   ├── solver.py          # unified solver interface
│   └── verify.py          # correctness verification
├── tests/                 # pytest test suite (39 tests)
├── benchmarks/            # benchmark runner + results
├── notebooks/demo.ipynb   # end-to-end walkthrough
├── results/               # verification reports
├── main.py                # CLI entry point
└── requirements.txt
```

## References

1. Miller, Tucker & Zemlin (1960). Integer Programming Formulation of TSP. *J. ACM*, 7, 326-329.
2. Dantzig, Fulkerson & Johnson (1954). Solution of a Large-Scale TSP. *JORSA*, 2(4), 393-410.
3. Silver & de Weck (2007). Time-Expanded Decision Networks. *AIAA-2006-6964*.
