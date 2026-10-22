"""Domain services. A request flows through them top to bottom:

    data_loader     prices and the risk-free rate from the cache, downloading what is missing
    calendar        the run calendar: today in New York, alignment, rf from ^IRX
    pipeline        run_strategy(): score, signal, weights, sizing, then the backtest (§5.0)
      strategies      the scores, entry and exit rules and unit weights of each strategy (§5)
      sizing          the volatility-targeting overlay (§5.6, §5.8)
      backtester      execution lag, daily returns, costs, ruin and trades (§6, §7)
    report          build_report(): metrics, benchmark, diagnostics and chart series (§8)
      metrics         Sharpe and the other §8 statistics, the cost sweep, warnings
      stats           Engle–Granger, half-life, Kalman filter, deflated Sharpe
    sensitivity     run_sensitivity(): the §9 parameter grid
    walk_forward    run_walk_forward(): re-optimization over that grid (§9.2)
    monte_carlo     GBM and bootstrap projections of the daily returns (§10)

Everything except data_loader and calendar is pure (§0, rule 8): arrays and Series in,
arrays, Series and dataclasses out, with no I/O.

The § numbers in docstrings and comments are sections of CLAUDE.md, the specification. A
plain-language tour, with the short names used here (t0, e, k, W, ...), is docs/ARCHITECTURE.md.
"""
