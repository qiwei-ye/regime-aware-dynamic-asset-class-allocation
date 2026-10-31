"""Causal Friday predictors and execution-to-realization supervised labels."""

import numpy as np
import pandas as pd

from . import config as c
from .data import DataUnavailableError


def weekly_returns(prices, calendar):
    """Adjusted total returns between consecutive Friday price observations."""
    current = prices.reindex(calendar["predictor_close"]).set_axis(calendar.index)
    previous = prices.reindex(calendar["previous_predictor_close"]).set_axis(calendar.index)
    returns = current / previous - 1
    if not np.isfinite(returns.to_numpy()).all():
        raise DataUnavailableError("Weekly returns lack a required adjusted-price boundary.")
    return returns


def _asset_rows(frames):
    """Keep the configured asset order and a categorical identifier in the index."""
    blocks = []
    for asset, frame in frames.items():
        block = frame.copy()
        block["asset"] = pd.Categorical([asset] * len(block), categories=c.ASSETS)
        blocks.append(block.reset_index().set_index(["research_week", "asset"]))
    return pd.concat(blocks).sort_index()


def return_predictors(returns):
    """Lag one-week excess returns; trailing windows include the current week."""
    excess = returns.sub(returns[c.CASH_ASSET], axis=0)
    frames = {}
    for asset in c.FORECAST_ASSETS:
        frame = pd.DataFrame({"excess_lag_1w": excess[asset].shift(c.RETURN_LAG_WEEKS)})
        for window in c.TRAILING_WINDOWS_WEEKS:
            rolling = excess[asset].rolling(window, min_periods=window)
            frame[f"excess_compounded_{window}w"] = (1 + excess[asset]).rolling(
                window, min_periods=window
            ).apply(np.prod, raw=True) - 1
            frame[f"excess_volatility_{window}w"] = rolling.std(ddof=c.VOLATILITY_DDOF)
        frames[asset] = frame
    return _asset_rows(frames)


def macro_predictors(availability, research_weeks):
    """Changes compare levels available at each of the two historical cutoffs.

    Later revisions never overwrite a previously constructed weekly level.
    PPI percentage changes are fractions, so 0.02 means a 2% increase.
    """
    levels = availability["value"].unstack("series").reindex(
        range(research_weeks.min() - c.MACRO_CHANGE_WEEKS, research_weeks.max() + 1)
    )
    predictors = pd.DataFrame(index=levels.index)
    for series in c.MACRO_SERIES:
        current = levels[series]
        previous = current.shift(c.MACRO_CHANGE_WEEKS)
        predictors[f"{series}_level"] = current
        if c.MACRO_CHANGE_METHODS[series] == "pct_change":
            if previous.eq(0).any():
                raise DataUnavailableError(f"{series}: zero denominator in a four-week percentage change.")
            change = current / previous - 1
        else:
            change = current - previous
        predictors[f"{series}_change_4w"] = change
    predictors = predictors.reindex(research_weeks)
    if not np.isfinite(predictors.to_numpy()).all():
        raise DataUnavailableError("Required macro predictors or their four-week history are unavailable.")
    return predictors


def execution_targets(prices, calendar):
    """Four-calendar-week total returns less BIL, using actual endpoint closes."""
    start = prices.reindex(calendar["execution_close"]).set_axis(calendar.index)
    end = prices.reindex(calendar["realization_close"]).set_axis(calendar.index)
    total = end / start - 1
    excess = total.sub(total[c.CASH_ASSET], axis=0)
    if not np.isfinite(excess.to_numpy()).all():
        raise DataUnavailableError("Targets lack a required execution or realization price.")
    locked_cutoff = calendar.at[c.SPLIT_WEEKS["locked_test"][0], "decision_cutoff"]
    frames = {}
    for asset in c.ASSETS:
        frame = calendar[["split", "decision_cutoff", "execution_close", "realization_close"]].copy()
        frame["target_4w_excess_return"] = excess[asset]
        frame["selection_eligible"] = (
            frame["split"].eq("validation") & frame["realization_close"].lt(locked_cutoff)
        )
        frames[asset] = frame
    return _asset_rows(frames)


def training_mask(targets, forecast_origin_week, calendar):
    """A 260-calendar-week lookback containing only fully observed labels.

    This mask also applies during locked testing: earlier test labels enter
    scheduled refits only after realization. It is not a selection mask.
    """
    cutoff = calendar.at[forecast_origin_week, "decision_cutoff"]
    weeks = targets.index.get_level_values("research_week")
    first = max(c.SPLIT_WEEKS["initial_training"][0], forecast_origin_week - c.TRAINING_WINDOW_WEEKS)
    return (
        (weeks >= first) & (weeks < forecast_origin_week)
        & targets["realization_close"].le(cutoff)
    )


def build_dataset(prices, calendar, availability):
    returns = weekly_returns(prices, calendar)
    macro = macro_predictors(availability, calendar.index)
    predictors = return_predictors(returns).join(macro, on="research_week")
    weeks = predictors.index.get_level_values("research_week")
    required = predictors.loc[weeks >= c.SPLIT_WEEKS["initial_training"][0]]
    if not np.isfinite(required.to_numpy()).all():
        raise DataUnavailableError("Return predictors remain unavailable after the 52-week warm-up.")
    targets = execution_targets(prices, calendar)
    return {
        "calendar": calendar, "adjusted_prices": prices, "weekly_returns": returns,
        "macro_predictors": macro, "macro_availability": availability,
        "features": predictors, "targets": targets,
    }
