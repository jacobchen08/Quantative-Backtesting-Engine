"""Quantitative Backtesting Engine: the backend (API and engine)."""

import os

# OpenBLAS runs every matrix operation on a pool of threads. The engine's matrices are small (the
# two Engle–Granger tests fit 53 least-squares regressions of about 2,600 rows and up to 28
# columns), and at that size the threads cost more than they save: one thread nearly halves the
# tests' time, with bit-identical results (measured). OpenBLAS reads this once, when NumPy loads,
# so it is set here, before any app module imports NumPy; a value already in the environment wins.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

__version__ = "0.1.0"
