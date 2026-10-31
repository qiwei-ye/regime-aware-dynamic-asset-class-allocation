"""Constrained targets and self-financing execution-time holdings accounting."""

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from . import config as c


class AllocationError(RuntimeError):
    """An allocation, solver result, or holdings identity is unusable."""


def check_weights(weights):
    values = np.asarray(weights, dtype=float)
    if (values.shape != (len(c.ASSETS),) or not np.isfinite(values).all()
            or abs(values.sum() - c.WEIGHT_SUM) > c.WEIGHT_TOLERANCE
            or values.min() < c.MIN_ASSET_WEIGHT - c.WEIGHT_TOLERANCE
            or values.max() > c.MAX_ASSET_WEIGHT + c.WEIGHT_TOLERANCE):
        raise AllocationError("Weights violate the fully invested bounded allocation constraints.")
    return values


def project_weights(values):
    """Euclidean projection onto the bounded fully invested simplex."""
    values = np.asarray(values, dtype=float)
    if values.shape != (len(c.ASSETS),) or not np.isfinite(values).all():
        raise AllocationError("Projection requires six finite weights.")
    low = np.min(values - c.MAX_ASSET_WEIGHT)
    high = np.max(values - c.MIN_ASSET_WEIGHT)
    for _ in range(100):
        midpoint = (low + high) / 2
        projected = np.clip(values - midpoint, c.MIN_ASSET_WEIGHT, c.MAX_ASSET_WEIGHT)
        if projected.sum() > c.WEIGHT_SUM:
            low = midpoint
        else:
            high = midpoint
    return check_weights(np.clip(values - (low + high) / 2, c.MIN_ASSET_WEIGHT, c.MAX_ASSET_WEIGHT))


def drift_weights(weights, total_return_factors):
    grown = np.asarray(weights, dtype=float) * np.asarray(total_return_factors, dtype=float)
    if not np.isfinite(grown).all() or grown.sum() <= 0:
        raise AllocationError("Holdings drift has nonpositive or nonfinite wealth.")
    return grown / grown.sum()


def one_way_turnover(pretrade_weights, executed_weights):
    return float(np.abs(np.asarray(executed_weights) - np.asarray(pretrade_weights)).sum() / 2)


def covariance_at(weekly_returns, origin):
    first = origin - c.COVARIANCE_WINDOW_WEEKS + 1
    history = weekly_returns.loc[first:origin, list(c.ASSETS)]
    if len(history) != c.COVARIANCE_WINDOW_WEEKS or not np.isfinite(history.to_numpy()).all():
        raise AllocationError(f"Week {origin}: covariance needs 260 available weekly total returns.")
    return LedoitWolf().fit(history.to_numpy()).covariance_ * c.COVARIANCE_HORIZON_MULTIPLIER


def benchmark_target(weekly_returns, origin, name):
    if name == "equal_weight":
        return np.asarray(c.INITIAL_WEIGHTS)
    if name != "inverse_volatility":
        raise ValueError(f"Unknown benchmark: {name}")
    first = origin - c.INVERSE_VOLATILITY_WINDOW_WEEKS + 1
    history = weekly_returns.loc[first:origin, list(c.ASSETS)]
    if len(history) != c.INVERSE_VOLATILITY_WINDOW_WEEKS:
        raise AllocationError("Inverse-volatility benchmark lacks 52 weeks of history.")
    volatility = history.std(ddof=c.VOLATILITY_DDOF).to_numpy()
    if not np.isfinite(volatility).all() or (volatility <= c.ZERO_VOLATILITY_TOLERANCE).any():
        raise AllocationError("Inverse-volatility weights require strictly positive sample volatilities.")
    inverse = 1 / volatility
    return project_weights(inverse / inverse.sum())


class ConstrainedAllocator:
    """Reuse a DPP problem with a supplied square root of horizon covariance."""

    def __init__(self, risk_aversion, turnover_penalty):
        count = len(c.ASSETS)
        self.mean = cp.Parameter(count)
        self.factor = cp.Parameter((count, count))
        self.reference = cp.Parameter(count)
        self.weights = cp.Variable(count)
        self.problem = cp.Problem(cp.Maximize(
            self.mean @ self.weights
            - risk_aversion / 2 * cp.sum_squares(self.factor @ self.weights)
            - turnover_penalty * cp.norm1(self.weights - self.reference)
        ), [cp.sum(self.weights) == c.WEIGHT_SUM,
            self.weights >= c.MIN_ASSET_WEIGHT, self.weights <= c.MAX_ASSET_WEIGHT])

    def solve(self, mean, covariance, reference):
        mean, covariance, reference = map(np.asarray, (mean, covariance, reference))
        if not all(np.isfinite(value).all() for value in (mean, covariance, reference)):
            raise AllocationError("Allocation inputs contain unavailable values.")
        if not np.allclose(covariance, covariance.T, atol=c.COVARIANCE_TOLERANCE, rtol=0):
            raise AllocationError("Covariance is not symmetric.")
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if eigenvalues.min() < -c.COVARIANCE_TOLERANCE:
            raise AllocationError("Covariance is not positive semidefinite.")
        self.mean.value = mean
        self.factor.value = np.sqrt(np.maximum(eigenvalues, 0))[:, None] * eigenvectors.T
        self.reference.value = reference
        try:
            self.problem.solve(solver=c.ALLOCATION_SOLVER, **c.SOLVER_SETTINGS)
        except cp.error.SolverError as error:
            raise AllocationError(f"{c.ALLOCATION_SOLVER} failed: {error}") from None
        if self.problem.status != cp.OPTIMAL:
            raise AllocationError(f"Solver status is {self.problem.status}; no fallback allocation was used.")
        weights = check_weights(self.weights.value)
        return weights, {"solver_status": self.problem.status,
                         "solver_iterations": self.problem.solver_stats.num_iters,
                         "objective": float(self.problem.value)}


def simulate(prices, weekly_returns, calendar, forecasts, origins, *, name,
             parameters=None, cutoff=None, initial_state=None):
    """Decide on Friday, trade on Monday, and value the same holdings until rebalance.

    Adjusted-price units represent distribution-reinvested holdings. A validation
    interval ending at/after the locked cutoff is retained as an unscored row.
    """
    origins = list(origins)
    first = calendar.loc[origins[0]]
    if initial_state is None:
        initial_prices = prices.loc[first["predictor_close"], list(c.ASSETS)].to_numpy()
        units = np.asarray(c.INITIAL_WEIGHTS) / initial_prices
    else:
        units = np.asarray(initial_state["units"], dtype=float)
    if units.shape != (len(c.ASSETS),) or not np.isfinite(units).all() or (units < 0).any():
        raise AllocationError("Initial holdings must contain six finite nonnegative units.")
    allocator = ConstrainedAllocator(**parameters) if name == "strategy" else None
    allocations, results = [], []
    for origin in origins:
        row = calendar.loc[origin]
        close = row["execution_close"]
        if cutoff is not None and close >= cutoff:
            raise AllocationError("Execution crosses the selection cutoff.")
        friday_prices = prices.loc[row["predictor_close"], list(c.ASSETS)].to_numpy()
        friday_values = units * friday_prices
        reference = friday_values / friday_values.sum()
        if allocator is not None:
            mean = forecasts.loc[origin].reindex(c.ASSETS).to_numpy()
            target, diagnostic = allocator.solve(mean, covariance_at(weekly_returns, origin), reference)
        else:
            target = benchmark_target(weekly_returns, origin, name)
            diagnostic = {"solver_status": "deterministic_benchmark", "solver_iterations": 0,
                          "objective": np.nan}
        execution_prices = prices.loc[close, list(c.ASSETS)].to_numpy()
        before = units * execution_prices
        wealth = float(before.sum())
        pretrade = before / wealth
        units = wealth * target / execution_prices
        after = units * execution_prices
        executed = after / after.sum()
        check_weights(executed)
        if not np.isclose(after.sum(), wealth, atol=c.ACCOUNTING_TOLERANCE, rtol=0):
            raise AllocationError("Rebalancing does not conserve wealth.")
        turnover = one_way_turnover(pretrade, executed)
        end = row["next_execution_close"]
        scored = cutoff is None or end < cutoff
        if scored:
            end_wealth = float(units @ prices.loc[end, list(c.ASSETS)].to_numpy())
            realized = end_wealth / wealth - 1
            cash_return = float(prices.at[end, c.CASH_ASSET] / prices.at[close, c.CASH_ASSET] - 1)
        else:
            end_wealth = realized = cash_return = np.nan
        results.append({
            "evaluation_period": row["split"], "portfolio": name, "research_week": origin,
            "decision_cutoff": row["decision_cutoff"], "execution_close": close,
            "realization_close": end, "weekly_total_return": realized,
            "cash_return": cash_return, "weekly_excess_return": realized - cash_return,
            "one_way_turnover": turnover, "wealth_start": wealth, "wealth_end": end_wealth,
            "scored": scored, "exclusion_reason": "" if scored else "holding_period_crosses_locked_cutoff",
        })
        for position, asset in enumerate(c.ASSETS):
            allocations.append({
                "evaluation_period": row["split"], "portfolio": name, "research_week": origin,
                "asset": asset, "decision_cutoff": row["decision_cutoff"],
                "reference_close": row["predictor_close"], "execution_close": close,
                "reference_weight": reference[position], "pretrade_weight": pretrade[position],
                "target_weight": target[position], "executed_weight": executed[position],
                "holding_units": units[position], "wealth_before_trade": wealth,
                "wealth_after_trade": float(after.sum()), "one_way_turnover": turnover,
                "risk_aversion": parameters["risk_aversion"] if parameters else np.nan,
                "turnover_penalty": parameters["turnover_penalty"] if parameters else np.nan,
                **diagnostic,
            })
    state = {"units": units.tolist(), "last_execution": close.isoformat(), "wealth_at_execution": wealth}
    return pd.DataFrame(allocations), pd.DataFrame(results), state
