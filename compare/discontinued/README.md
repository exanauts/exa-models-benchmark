# DISCONTINUED — do not use, do not reinstate

Nothing in this directory is on any code path. It is kept only so that a
formulation we measured and rejected does not get proposed again as if it were
a new idea.

**Do not import these files, do not wire them into `compare_ad.py`, and do not
"restore" them.** If a measurement here looks worth revisiting, re-measure it
first — every file below records the numbers that retired it and the machine
they were taken on.

| file | retired | why |
|---|---|---|
| `opf_casadi_mtimes.py` | 2026-08-05 (`960dfc7`) | Hessian 4.11× slower than index gathers |

See `benchmark/compare/compare_ad.py::_casadi_opf` for the formulation actually
in use.
