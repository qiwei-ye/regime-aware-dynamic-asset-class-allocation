"""Authoritative research settings, calendar, paths, grids, and seed identities.

Importing this module creates no directories, reads no credentials, and starts
no fits. Calendar dates are scheduled dates; trading sessions, actual closes,
release timestamps, and realized targets must be resolved from source data.
"""

from datetime import date, timedelta
from itertools import product
from math import prod
import os
from operator import index
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd


# Declare destinations here; create them only when writing their contents.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SELECTION_DIR = OUTPUT_DIR / "selection"
LOCKED_TEST_DIR = OUTPUT_DIR / "locked_test"
FIGURES_DIR = OUTPUT_DIR / "figures"
SEED_MAPPING_PATH = OUTPUT_DIR / "seed_mapping.jsonl"

FRED_API_KEY_ENV = "FRED_API_KEY"
MARKET_DATA_SOURCE = "Yahoo Finance"
MACRO_DATA_SOURCE = "ALFRED decision-time vintages"
ASSETS = ("VT", "IEF", "LQD", "VNQ", "DBC", "BIL")
CASH_ASSET = "BIL"
FORECAST_ASSETS = tuple(asset for asset in ASSETS if asset != CASH_ASSET)
MACRO_SERIES = ("VIXCLS", "T10Y2Y", "PPIACO")
MACRO_CHANGE_METHODS = MappingProxyType(
    {"VIXCLS": "difference", "T10Y2Y": "difference", "PPIACO": "pct_change"}
)
PRICE_AUTO_ADJUST = True

RESEARCH_START = date(2014, 7, 28)
WEEKS_PER_YEAR = 52
RESEARCH_WEEKS = 624
DAYS_PER_WEEK = 7
FIRST_RETURN_PRICE_BOUNDARY = RESEARCH_START - timedelta(days=3)
DECISION_WEEKDAY = 4  # Friday; Monday is zero.
EXECUTION_DELAY_DAYS = 3  # Following Monday, before holiday adjustment.
MARKET_TIMEZONE = "America/New_York"
DECISION_CUTOFF_RULE = "latest available trading close by Friday"
EXECUTION_RULE = "Monday close, or next trading close on a market holiday"
TARGET_ENDPOINT_RULE = "execution close to trading close four weeks later"
SPLIT_WEEKS = MappingProxyType(
    {
        "warm_up": (1, 52),
        "initial_training": (53, 312),
        "validation": (313, 520),
        "locked_test": (521, 624),
    }
)
TRAINING_WINDOW_WEEKS = 260
REFIT_EVERY_WEEKS = 1
FORECAST_HORIZON_WEEKS = 4
SUPERVISED_TARGET_RULE = "fully realized by the forecast origin cutoff"
SELECTION_TARGET_RULE = "realized strictly before the first locked-test decision"
LOCKED_TEST_PASSES = 1
MACRO_VINTAGE_RULE = "available at cutoff; carry forward only after release"

RETURN_LAG_WEEKS = 1
TRAILING_WINDOWS_WEEKS = (4, 13, 52)
EXCESS_RETURN_COMPOUNDING = "product(1 + weekly asset return - weekly BIL return) - 1"
VOLATILITY_DDOF = 1
MACRO_CHANGE_WEEKS = 4
CASH_EXCESS_FORECAST = 0.0

HMM_EMISSIONS = MappingProxyType(
    {
        "hmm_equity_stress": ("VIXCLS", "level"),
        "hmm_yield_curve_stress": ("T10Y2Y", "level"),
        "hmm_producer_price_pressure": ("PPIACO", "pct_change_4w"),
    }
)
HMM_SETTINGS = MappingProxyType(
    {
        "n_components": 2,
        "covariance_type": "diag",
        "min_covar": 0.001,
        "n_iter": 200,
        "tol": 0.001,
    }
)
HMM_STANDARDIZATION = "training-window mean and population standard deviation"
HMM_PROBABILITIES = "causal filtering; states ordered by ascending emission mean"
HMM_INITIAL_HISTORY = "shorter available history before a full rolling window"

RETURN_MODEL_SETTINGS = MappingProxyType(
    {
        "loss": "squared_error",
        "early_stopping": False,
        "categorical_features": ("asset",),
    }
)
# This insertion order and the value orders define permanent candidate IDs.
RETURN_MODEL_GRID = MappingProxyType(
    {
        "learning_rate": tuple(i / 100 for i in range(1, 11)),
        "max_iter": tuple(range(100, 501, 100)),
        "max_leaf_nodes": tuple(range(5, 16)),
        "min_samples_leaf": tuple(range(5, 26, 5)),
        "l2_regularization": tuple(10.0**i for i in (-1, -2, -3, -4, -5)),
    }
)
RETURN_MODEL_SELECTION = "lowest pooled walk-forward mean squared error"
ALLOCATION_GRID = MappingProxyType(
    {
        "risk_aversion": tuple(range(1, 6)),
        "turnover_penalty": tuple(i / 1000 for i in range(1, 6)),
    }
)
ALLOCATION_SELECTION = "highest annualized validation excess-return Sharpe ratio"
SELECTION_TIE_BREAK = "lowest candidate ID in the declared Cartesian grid order"

RETURN_MODEL_CANDIDATES = prod(len(values) for values in RETURN_MODEL_GRID.values())
ALLOCATION_CANDIDATES = prod(len(values) for values in ALLOCATION_GRID.values())
VALIDATION_ORIGINS = range(
    SPLIT_WEEKS["validation"][0], SPLIT_WEEKS["validation"][1] + 1
)
LOCKED_TEST_ORIGINS = range(
    SPLIT_WEEKS["locked_test"][0], SPLIT_WEEKS["locked_test"][1] + 1
)
# Every candidate is fit at every validation origin, even if its target cannot
# be scored before the locked-test boundary. Only scoring excludes those labels.
VALIDATION_RETURN_MODEL_FITS = RETURN_MODEL_CANDIDATES * len(VALIDATION_ORIGINS)
LOCKED_TEST_RETURN_MODEL_FITS = len(LOCKED_TEST_ORIGINS)
HMM_FITS_PER_ORIGIN = len(HMM_EMISSIONS)

COVARIANCE_ESTIMATOR = "LedoitWolf"
COVARIANCE_WINDOW_WEEKS = 260
COVARIANCE_HORIZON_MULTIPLIER = FORECAST_HORIZON_WEEKS
MIN_ASSET_WEIGHT = 0.05
MAX_ASSET_WEIGHT = 0.50
WEIGHT_SUM = 1.0
WEIGHT_TOLERANCE = 1e-7
INITIAL_WEIGHTS = tuple(WEIGHT_SUM / len(ASSETS) for _ in ASSETS)
ALLOCATION_OBJECTIVE = "mu @ w - risk_aversion / 2 * w @ covariance_4w @ w - turnover_penalty * norm1(w - w_previous)"
ALLOCATION_REFERENCE = "holdings drifted to the Friday cutoff"
TURNOVER_RULE = "0.5 * sum(abs(target weights - pre-trade execution-time weights))"
BENCHMARKS = ("equal_weight", "inverse_volatility")
INVERSE_VOLATILITY_WINDOW_WEEKS = 52
INVERSE_VOLATILITY_PROJECTION = "Euclidean projection onto the strategy weight constraints"
BENCHMARK_EXECUTION_RULE = EXECUTION_RULE
TRANSACTION_COST_RATE = 0.0
SLIPPAGE_RATE = 0.0
TAX_RATE = 0.0
EVALUATION_METRICS = (
    "pooled_forecast_mse",
    "annualized_return",
    "annualized_volatility",
    "excess_return_sharpe",
    "maximum_drawdown",
    "mean_weekly_turnover",
)

MASTER_SEED = 42
# Explicit IDs are permanent: adding a component must not renumber old IDs.
SEED_COMPONENT_IDS = MappingProxyType(
    {
        "hmm_equity_stress": 0,
        "hmm_yield_curve_stress": 1,
        "hmm_producer_price_pressure": 2,
        "return_model": 3,
    }
)
SEED_KEY_FIELDS = (
    "component_id", "forecast_origin_week", "candidate_id", "repeated_fit_id"
)


def fred_api_key() -> str:
    """Read the credential only from the current process environment."""
    value = os.environ.get(FRED_API_KEY_ENV, "").strip()
    if not value:
        raise RuntimeError(f"Set {FRED_API_KEY_ENV} in the process environment.")
    return value


def research_calendar() -> pd.DataFrame:
    """Return all research weeks; endpoints beyond week 624 are scoring only.

    Scheduled dates are not session timestamps. Holiday-adjusted execution and
    target endpoints must be resolved before any label-availability check.
    """
    starts = pd.date_range(
        RESEARCH_START, periods=RESEARCH_WEEKS, freq=pd.Timedelta(days=DAYS_PER_WEEK)
    )
    decisions = starts + pd.Timedelta(days=DECISION_WEEKDAY)
    executions = decisions + pd.Timedelta(days=EXECUTION_DELAY_DAYS)
    calendar = pd.DataFrame(
        {
            "week_start": starts,
            "decision_date": decisions,
            "scheduled_execution_date": executions,
            "scheduled_target_end_date": executions
            + pd.Timedelta(weeks=FORECAST_HORIZON_WEEKS),
            "split": "",
        },
        index=pd.RangeIndex(1, RESEARCH_WEEKS + 1, name="research_week"),
    )
    for split, (first, last) in SPLIT_WEEKS.items():
        calendar.loc[first:last, "split"] = split
    return calendar


def first_locked_test_decision() -> date:
    """Scheduled Friday boundary; resolve the actual cutoff from market data."""
    return RESEARCH_START + timedelta(
        weeks=SPLIT_WEEKS["locked_test"][0] - 1, days=DECISION_WEEKDAY
    )


def grid_candidates(grid):
    """Yield (stable candidate ID, parameters) for the complete Cartesian grid."""
    for candidate_id, values in enumerate(product(*grid.values())):
        yield candidate_id, dict(zip(grid, values))


def seed_sequence(
    component: str,
    forecast_origin_week: int,
    *,
    candidate_id: int = 0,
    repeated_fit_id: int = 0,
) -> np.random.SeedSequence:
    """Address a named child directly, independent of call or worker order.

    Forecast origins are one-based research weeks. Candidate IDs come from
    grid_candidates; repeated_fit_id distinguishes repeated fits of the same
    candidate. Reconstructing a child never consumes a shared spawn counter.
    """
    if component not in SEED_COMPONENT_IDS:
        raise ValueError(f"Unknown seed component: {component!r}")
    identifiers = (forecast_origin_week, candidate_id, repeated_fit_id)
    if any(isinstance(value, (bool, np.bool_)) for value in identifiers):
        raise TypeError("Seed identifiers must be integers, not booleans.")
    origin, candidate, repeated_fit = map(index, identifiers)
    if not 1 <= origin <= RESEARCH_WEEKS:
        raise ValueError(f"Forecast origin must be in 1..{RESEARCH_WEEKS}.")
    if candidate < 0 or repeated_fit < 0:
        raise ValueError("Candidate and repeated-fit IDs must be nonnegative.")
    return np.random.SeedSequence(
        MASTER_SEED,
        spawn_key=(SEED_COMPONENT_IDS[component], origin, candidate, repeated_fit),
    )


def seed_record(
    component: str,
    forecast_origin_week: int,
    *,
    candidate_id: int = 0,
    repeated_fit_id: int = 0,
) -> dict:
    """Return a JSON-ready mapping for a fit, without writing an output file.

    Use random_state for sklearn/hmmlearn and the full SeedSequence for NumPy.
    The 32-bit adapter is not a unique fit ID; never deduplicate fits by it.
    """
    sequence = seed_sequence(
        component,
        forecast_origin_week,
        candidate_id=candidate_id,
        repeated_fit_id=repeated_fit_id,
    )
    return {
        "component": component,
        **dict(zip(SEED_KEY_FIELDS, sequence.spawn_key)),
        "master_seed": MASTER_SEED,
        "spawn_key": list(sequence.spawn_key),
        "pool_size": sequence.pool_size,
        "random_state": int(sequence.generate_state(1, dtype=np.uint32)[0]),
    }
