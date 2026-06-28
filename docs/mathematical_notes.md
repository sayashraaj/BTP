# Modified MTZ Constraints for a 3D Binary Matrix

## The problem

The standard Travelling Salesman Problem has one link per ordered node-pair: you decide whether to go from city *i* to city *j*, and there's a single cost *C_ij* associated with that arc. The MTZ subtour-elimination constraints (Miller, Tucker & Zemlin, 1960) work on this 2D binary matrix *x_ij*:

```
u_i - u_j + n * x_ij  <=  n - 1      for all i, j != depot
```

Where *u_i* is an integer variable representing the visit-rank of node *i* in the tour. The logic is simple: if *x_ij = 1* (we travel from *i* to *j*), then *u_j >= u_i + 1* — *j* must come strictly after *i* in the tour ordering. If *x_ij = 0*, the constraint reduces to *u_i - u_j <= n - 1*, which is always satisfied. This prevents any subset of nodes from forming a disconnected cycle.

## The 3D extension

In the journey planning problem, each ordered node-pair *(i, j)* has *k* time-dependent link options — different trains running between the same two stations at different times, each with its own departure time *Start_ijk* and travel cost *C_ijk*. The decision variable becomes a 3D binary tensor *x_ijk*: "do we take the *k*-th train from *i* to *j*?"

The modified MTZ constraint replaces the scalar *x_ij* with the aggregate link-selection indicator across all *k* options:

```
u_i - u_j + n * Σ_k x_ijk  <=  n - 1      for all i, j != depot
```

## Why this works

The substitution *x_ij → Σ_k x_ijk* is valid **only because** constraints (2.1) and (2.2) guarantee that `Σ_k x_ijk ∈ {0, 1}`:

- **(2.1) Outdegree = 1:** `Σ_{j,k} x_ijk = 1` for all *i*. Each node has exactly one outgoing link selected (to some *j*, via some *k*).
- **(2.2) Indegree = 1:** `Σ_{i,k} x_ijk = 1` for all *j*. Each node has exactly one incoming link.

Together, these mean that for any specific ordered pair *(i, j)*, at most one link option *k* is selected. So `Σ_k x_ijk` is either 0 (no direct link between *i* and *j* in the tour) or 1 (exactly one of the *k* options is used). This is the same binary behavior as the original *x_ij*, so the subtour logic carries through unchanged.

Without (2.1) and (2.2) holding first, `Σ_k x_ijk` could be 2 or more (multiple trains from *i* to *j*), and the constraint would become `u_j >= u_i + 2`, which is a stronger-than-necessary restriction that could make feasible tours infeasible.

## Why disjoint subtours using different *k* are caught

Consider 4 nodes and 2 link options. Without MTZ, the solver could form two disjoint cycles:

- Cycle A: `1 → 2 → 1` using link option *k=1*
- Cycle B: `3 → 4 → 3` using link option *k=2*

Both cycles independently satisfy (2.1) and (2.2) — every node has outdegree 1 and indegree 1. Each cycle is internally consistent.

But the MTZ constraint for the pair *(2, 3)* says: if no link connects 2 to 3 (i.e., `Σ_k x_{2,3,k} = 0`), then `u_2 - u_3 <= n - 1`. This alone doesn't prevent the two cycles. However, combining all the MTZ constraints for the pairs within and between the two cycles forces the *u*-values into a single total order. Since *u_2 > u_1* (from cycle A) and *u_4 > u_3* (from cycle B), and the cross-constraints link the *u*-values of both cycles together, the system cannot assign consistent ranks to two disconnected cycles that both include the depot.

Specifically: cycle A requires `u_2 >= u_1 + 1 = 1`, and cycle B exists entirely among non-depot nodes where MTZ is active. The MTZ constraints for pairs like *(3, 2)* and *(4, 2)* force the *u*-values of cycle B to be ordered relative to cycle A's nodes. Since a valid tour must start and end at the depot (node 1, *u_1 = 0*), and there's no link from cycle B back to the depot, the rank assignments become contradictory.

The test `test_3d_subtour_across_link_options` in `tests/test_3d.py` constructs exactly this scenario: two cheap disjoint subtours where the Hamiltonian tour is strictly more expensive, and confirms the solver picks the expensive Hamiltonian tour.
