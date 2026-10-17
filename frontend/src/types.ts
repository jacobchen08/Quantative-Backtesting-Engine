// API types. The response types copy CLAUDE.md §11.3 exactly; backend/app/schemas.py mirrors the
// same contract. Returns, rates and weights are decimals (0.01 = 1%); dates are "YYYY-MM-DD".

export type Strategy = 'mean_reversion' | 'pairs' | 'trend'
export type SignalFrequency = 'monthly' | 'daily'
export type HedgeMode = 'fixed_at_entry' | 'rolling'
export type VolRefresh = 'daily' | 'weekly' | 'monthly'

// ---------------------------------------------------------------- requests (§11.2)
// Optional fields fall back to the server defaults. The dashboard omits every field that
// doesn't apply to the chosen strategy (§13: hidden controls are not sent).

export type BacktestRequest = {
  strategy: Strategy
  tickers: string[]
  start: string
  end: string
  L?: number | null
  z_entry?: number
  z_exit?: number
  signal_frequency?: SignalFrequency
  cost_bps?: number
  execution_lag?: number
  hedge_mode?: HedgeMode
  borrow_bps?: number
  allow_short?: boolean
  vol_target?: number | null
  vol_halflife?: number
  vol_refresh?: VolRefresh
  max_leverage?: number
  split_date?: string | null
  risk_free?: 'tbill' | number
  initial_capital?: number
}

export type SensitivityRequest = BacktestRequest & {
  L_values?: number[] | null
  z_entry_values?: number[] | null
}

export type MonteCarloRequest = {
  source: 'strategy' | 'ticker'
  backtest?: BacktestRequest | null
  window?: 'full' | 'is'
  ticker?: string | null
  start?: string | null
  end?: string | null
  n_paths?: number
  horizon?: number
  block_len?: number
  seed?: number | null
  zero_drift?: boolean
}

// "Resolved" = the request with every default filled in, `end` clamped, and fields that don't
// apply echoed as null (§11.2).
export type ResolvedBacktestRequest = {
  strategy: Strategy
  tickers: string[]
  start: string
  end: string
  L: number
  z_entry: number | null
  z_exit: number | null
  signal_frequency: SignalFrequency | null
  cost_bps: number
  execution_lag: number
  hedge_mode: HedgeMode | null
  borrow_bps: number
  allow_short: boolean
  vol_target: number | null
  vol_halflife: number | null
  vol_refresh: VolRefresh | null
  max_leverage: number | null
  split_date: string | null
  risk_free: 'tbill' | number
  initial_capital: number
}

export type ResolvedMonteCarloRequest = {
  source: 'strategy' | 'ticker'
  backtest: ResolvedBacktestRequest | null
  window: 'full' | 'is' | null
  ticker: string | null
  start: string | null
  end: string | null
  n_paths: number
  horizon: number
  block_len: number
  seed: number | null
  zero_drift: boolean
}

// ---------------------------------------------------------------- responses (§11.3, verbatim)

export type Metrics = {
  n_days: number; total_return: number; cagr: number | null; ann_vol: number | null;   // ann_vol null when n_days < 2
  sharpe: number | null; sharpe_ci95: [number, number] | null; sortino: number | null;
  max_drawdown: number; beta_spy: number | null; corr_spy: number | null;
  pct_days_in_market: number; n_closed_trades: number | null; n_open_trades: number | null;
  win_rate: number | null; avg_holding_days: number | null;
  ann_turnover: number | null; ann_cost_drag: number | null;   // trade/cost fields are null for benchmarks
  ann_borrow_drag: number | null;                               // Phase 5; null for benchmarks
  avg_gross_exposure: number;                                   // 1 for benchmarks
};
export type Trade = {
  id: number; side: "long" | "short"; segment: "is" | "oos"; open: boolean;
  signal_date: string; entry_date: string; exit_signal_date: string | null; exit_date: string | null;
  holding_days: number; pnl: number; beta_at_entry: number | null;
};
type BySegment<T> = { is: T; oos: T | null; full: T };          // oos is null only if ruin happened in IS
export type BacktestResponse = {
  run_id: number; engine_version: string;
  resolved: ResolvedBacktestRequest & { split_date: string; first_evaluated_date: string; rf_source: string };
  series: {                        // aligned arrays from s0 = min(t0, e - 1) to the end (or the ruin day)
    dates: string[];
    equity: (number | null)[]; benchmark_equity: (number | null)[]; drawdown: (number | null)[];
                                              // null before the baseline day e - 1; E0 / E0 / 0 at the baseline
    daily_return: (number | null)[];          // daily TOTAL return (matches equity); null up to and including the baseline
    position: number[];                       // executed position pos
    gross_exposure: number[];                 // executed sum of |w| at the end of each day (shows leverage over time)
    score: (number | null)[];                 // z (mean_reversion, pairs) or sampled trailing excess return m (trend)
    beta: (number | null)[] | null; beta_used: (number | null)[] | null;   // pairs only
    vol_forecast: (number | null)[] | null;   // annualized sigma (§5.6); null when the overlay is off
    is_oos: boolean[];
    rolling_sharpe: (number | null)[]; rolling_beta: (number | null)[];   // Phase 5, §8.2
  };
  sizing: { vol_target: number; max_leverage: number;
            pct_days_at_cap: number | null } | null;   // whole object null when the overlay is off; pct null if never invested
  trades: Trade[];
  metrics: { is: Metrics; oos: Metrics | null; full: Metrics };            // oos is null only if ruin happened in IS
  benchmark_metrics: { is: Metrics; oos: Metrics | null; full: Metrics };  // same window as the strategy
  diagnostics: { coint_p_is: number | null; coint_p_oos: number | null; half_life_is: number | null;
                 static_beta_is: number | null; static_alpha_is: number | null } | null;    // pairs only (null for mean_reversion and trend)
  cost_sensitivity: { cost_bps: number[];                       // Phase 5, §8.1; null if ruined
                      sharpe: BySegment<(number | null)[]>;     // one value per cost_bps entry
                      break_even_bps: BySegment<number | null> } | null;
  calendar: { monthly: { year: number; month: number; strategy: number; benchmark: number; partial: boolean }[];
              annual: { year: number; strategy: number; benchmark: number; partial: boolean }[] };   // Phase 5, §8.2
  ruined: boolean; ruin_date: string | null; warnings: string[];
};
export type SensitivityResponse = {
  L_values: number[]; z_entry_values: number[] | null;          // null for trend
  sharpe: (number | null)[][];                                  // rows = L; one column for trend
  n_configs_tested: number; window: { start: string; end: string };
  selected: { L: number; z_entry: number | null } | null;       // the request's own cell, if it is in the grid
  deflated: { n_trials: number; sr0_star: number | null;        // Phase 5, §9.1; sr0_star annualized for display
              selected_dsr: number | null;
              best: { L: number; z_entry: number | null; sharpe: number; dsr: number | null } | null };
};
export type ModelResult = {
  bands: { p5: number[]; p25: number[]; p50: number[]; p75: number[]; p95: number[] };   // length T+1
  sample_paths: number[][];        // 20 x (T+1) equity multiples
  hist_counts: number[];           // 50
  var95: number; var99: number; cvar95: number; cvar99: number; elapsed_ms: number;
};
export type MonteCarloResponse = {
  resolved: ResolvedMonteCarloRequest & { seed: number; n_hist: number; mu_daily: number; sigma_daily: number };
  gbm: ModelResult; bootstrap: ModelResult;
  closed_form_gbm: { var95: number; var99: number; cvar95: number; cvar99: number };
  hist_edges: number[];            // 51, shared by both models
  elapsed_ms_total: number;
};

// ---------------------------------------------------------------- aliases of the types above

export type Series = BacktestResponse['series']
export type SegmentMetrics = BacktestResponse['metrics']
export type SegmentKey = keyof SegmentMetrics
export type ResolvedBacktestRun = BacktestResponse['resolved']
export type CostSensitivity = NonNullable<BacktestResponse['cost_sensitivity']>
export type CalendarReturns = BacktestResponse['calendar']
export type Deflated = SensitivityResponse['deflated']
