# Process-Constrained Laser Drilling (PC-LDPP)

Code, instance data, and archived computational results for **Process-Constrained
Laser Drilling (PC-LDPP)**. This repository provides the final fused CBIS
heuristic, GLNS- and GLKH-driven block methods, and independent MTZ and DFJ
formulations.

PC-LDPP requires a drilling route to cover all required holes under process
constraints. The implementations use horizontal and vertical motion and
evaluate the trade-off between travel distance and changes in travel direction.
CBIS organizes holes into scan lines, constructs and joins cycles, and improves
the resulting route through a fused variable neighborhood search.

## Repository structure

```text
.
|-- README.md
|-- code/
|   |-- CBIS/              Scan generation, cycles, refinement, and fused search
|   |   `-- search/        Shake operators and descent neighborhoods
|   |-- GLNS/              GLNS entry point
|   |-- GLKH/              GLKH entry point
|   |-- MTZ&DFJ/           Exact formulations and solver worker
|   |-- common/            Input, scoring, geometry, block models, native binaries
|   |-- tests/             Small correctness checks
|   |-- config.json        Default heuristic settings
|   |-- run.py             Shared single-instance command-line interface
|   `-- doctor.py          Dependency, input, and license checks
|-- data/
|   |-- points/            Twenty canonical XY hole files
|   |-- gtsp/              Supplementary GTSP matrices and cluster partitions
`-- result/
```

## Objective and methods

All PC-LDPP routes are closed and consist of horizontal or vertical segments.
The distance coefficient is fixed at 1:

```text
W = D + lambda * T
```

- **D** is the complete Manhattan path length, in the coordinate units of the
  input. For scan-based routes, it includes both scan traversal and connections.
- **T** counts changes in signed travel direction, including the closing
  connection. A 90-degree turn and a 180-degree reversal each count as one change.
- **lambda** is the turn coefficient; its default value is 1.

Results also report 90-degree and 180-degree counts separately.

| Method | Public implementation |
| --- | --- |
| CBIS | Hole-to-scan preprocessing, matching, cycle patching, singleton refinement, and fused search |
| GLNS | Point-derived blocks optimized through the bundled native GLNS solver |
| GLKH | Point-derived blocks optimized through the bundled native GLKH/LKH solvers |
| MTZ | Full-hole MILP with lifted Miller–Tucker–Zemlin ordering constraints |
| DFJ | The same full-hole objective with Dantzig–Fulkerson–Johnson subtour constraints added through a lazy-cut callback |

GLNS and GLKH reconstruct a route through every hole and check its physical
score against the integer solver objective within the recorded rounding
tolerance. They are block-based heuristic methods.

MTZ and DFJ start independently without an external route, objective, variable
hint, or MIP start. Both exact implementations currently use lambda=1. Equal
proven optima are expected for equivalent formulations; resource-limited
incumbents need not be equal.

Scan-based, block-based, and full-hole formulations have different feasible
solution spaces. A common scoring function alone does not make their modeling
restrictions identical.

## Data

The twenty canonical PC-LDPP instances are in [data/points](data/points/).
Each row is one unique hole center:

```text
x y
```

Files contain numeric coordinates without a header. Row positions define hole
IDs and must be preserved. The registry records measured counts and SHA-256
hashes for the distributed files.

The supplementary GTSP data use their original edge-weight matrices and cluster
partitions. Their objective is the closed GTSP matrix cost, without an added
PC-LDPP distance or turn penalty. The archived GTSP observations are included;
the GTSP-specific experiment runner is not distributed.

The exact archive contains 40 recorded outcomes, three feasible incumbents,
and no proven optima. Outcomes without an incumbent or bound retain missing
values. Resource prechecks, interrupted attempts, feasible solutions, and
proven optima must be distinguished.

## Licensing and citation
Third-party software retains its own terms. The bundled GLNS license notice is
preserved in [code/common/bin/licenses](code/common/bin/licenses/).
The GLNS implementation originates from
[glns-cpp](https://github.com/stephenlsmith/glns-cpp); GLKH and LKH are described
on the [author's software page](https://keldhelsgaun.github.io/).

