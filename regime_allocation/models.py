"""Origin-time HMM filtering and complete pooled walk-forward model selection."""

from functools import lru_cache
import hashlib
import json
import math
import time

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from joblib import Parallel, delayed, parallel_config
from scipy.special import logsumexp
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from . import config as c
from .features import training_mask


class ModelFitError(RuntimeError):
    """A fit cannot be used for reporting or freezing."""


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def frame_hash(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def _save_cache(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary, compress=0)
    temporary.replace(path)


def forward_filter(log_emissions, start_probability, transition):
    """P(state_t | observations_1:t), with no backward/smoothing pass."""
    emissions = np.asarray(log_emissions, dtype=float)
    start = np.asarray(start_probability, dtype=float)
    transition = np.asarray(transition, dtype=float)
    if (emissions.ndim != 2 or len(start) != emissions.shape[1]
            or transition.shape != (len(start), len(start))):
        raise ValueError("Incompatible filtering dimensions.")
    if (start < 0).any() or (transition < 0).any() or not np.isclose(start.sum(), 1):
        raise ValueError("Invalid starting probabilities.")
    if not np.allclose(transition.sum(axis=1), 1):
        raise ValueError("Transition rows must sum to one.")
    with np.errstate(divide="ignore"):
        log_start, log_transition = np.log(start), np.log(transition)
    posterior = np.empty_like(emissions)
    for position, emission in enumerate(emissions):
        prediction = log_start if position == 0 else logsumexp(
            log_posterior[:, None] + log_transition, axis=0
        )
        joint = prediction + emission
        normalization = logsumexp(joint)
        if not np.isfinite(normalization):
            raise ModelFitError("Filtering assigned zero probability to an observation.")
        log_posterior = joint - normalization
        posterior[position] = np.exp(log_posterior)
    return posterior


def gaussian_filtered_probabilities(observations, model):
    values = np.asarray(observations, dtype=float).reshape(-1, 1)
    means = model.means_.reshape(-1)
    variances = model._covars_.reshape(-1)
    emissions = -0.5 * (np.log(2 * np.pi * variances)
                        + (values - means) ** 2 / variances)
    probabilities = forward_filter(emissions, model.startprob_, model.transmat_)
    return probabilities[:, np.argsort(means, kind="stable")]


class _FlooredGaussianHMM(GaussianHMM):
    """Apply the specified covariance floor after every diagonal M step.

    hmmlearn 0.3.3 applies min_covar at initialization only. Constrained maximum
    likelihood with a zero covariance prior preserves a genuine variance floor.
    """

    def _do_mstep(self, statistics):
        super()._do_mstep(statistics)
        self._covars_ = np.maximum(self._covars_, self.min_covar)


def fit_regime(history, component, origin):
    values = np.asarray(history, dtype=float)
    if len(values) < c.HMM_MIN_HISTORY_WEEKS or not np.isfinite(values).all():
        raise ModelFitError(f"{component}, week {origin}: insufficient finite HMM history.")
    center = float(values.mean())
    scale = float(values.std(ddof=0))
    scale = scale if scale > 0 else 1.0
    standardized = ((values - center) / scale).reshape(-1, 1)
    diagnostics = []
    for attempt in range(c.HMM_MAX_FIT_ATTEMPTS):
        seed = c.seed_record(component, origin, repeated_fit_id=attempt)
        model = _FlooredGaussianHMM(**c.HMM_SETTINGS, random_state=seed["random_state"])
        reason = ""
        try:
            model.fit(standardized)
            trace = list(model.monitor_.history)
            improvement = float(trace[-1] - trace[-2]) if len(trace) >= 2 else math.inf
            converged = (-c.HMM_MONOTONICITY_TOLERANCE <= improvement < c.HMM_SETTINGS["tol"])
            finite = np.isfinite(model.means_).all() and np.isfinite(model._covars_).all()
            if not converged or not finite:
                reason = "nonconvergence" if not converged else "nonfinite_parameters"
            probability = gaussian_filtered_probabilities(standardized, model)[-1] if not reason else None
        except (ValueError, FloatingPointError) as error:
            improvement, probability = math.nan, None
            reason = f"{type(error).__name__}: {error}"
        diagnostics.append({
            **seed, "fit_observations": len(values), "standardization_mean": center,
            "standardization_scale": scale, "iterations": model.monitor_.iter,
            "last_improvement": improvement, "accepted": not reason, "failure": reason,
        })
        if not reason:
            order = np.argsort(model.means_.reshape(-1), kind="stable")
            return probability, diagnostics, model.means_.reshape(-1)[order] * scale + center
    raise ModelFitError(
        f"{component}, week {origin}: all {len(diagnostics)} seeded fits failed: "
        f"{diagnostics[-1]['failure']}"
    )


def causal_regimes(tables, previous=None):
    """Historical rows are fitted once with their own origin's available window."""
    macro, calendar = tables["macro_predictors"], tables["calendar"]
    signature = fingerprint({
        "macro": frame_hash(macro), "settings": dict(c.HMM_SETTINGS),
        "minimum_history": c.HMM_MIN_HISTORY_WEEKS, "master_seed": c.MASTER_SEED,
        "components": dict(c.SEED_COMPONENT_IDS), "method": "origin_window_floor_forward_v1",
    })
    if previous is None and c.REGIME_CACHE_PATH.exists():
        cached = joblib.load(c.REGIME_CACHE_PATH)
        if cached["signature"] == signature:
            return cached
    wide, probabilities, fits = [], [], []
    completed = set()
    if previous is not None:
        wide = previous["features"].reset_index().to_dict("records")
        probabilities = previous["probabilities"].to_dict("records")
        fits = list(previous["fits"])
        completed = set(previous["features"].index)
    with threadpool_limits(limits=c.MODEL_THREADS):
        for origin in calendar.index:
            if origin < c.HMM_MIN_HISTORY_WEEKS or origin in completed:
                continue
            first = max(1, origin - c.TRAINING_WINDOW_WEEKS + 1)
            row = {"research_week": origin}
            for component, (series, transformation) in c.HMM_EMISSIONS.items():
                column = f"{series}_{'level' if transformation == 'level' else 'change_4w'}"
                history = macro.loc[first:origin, column]
                probability, attempts, means = fit_regime(history, component, origin)
                fits.extend({**attempt, "first_training_week": first,
                             "last_training_week": origin,
                             "available_through": calendar.at[origin, "decision_cutoff"]}
                            for attempt in attempts)
                for state in range(c.HMM_SETTINGS["n_components"]):
                    row[f"{component}_state_{state}"] = probability[state]
                probabilities.append({
                    "evaluation_period": calendar.at[origin, "split"], "research_week": origin,
                    "decision_cutoff": calendar.at[origin, "decision_cutoff"], "component": component,
                    "probability_low": probability[0], "probability_high": probability[1],
                    "emission_mean_low": means[0], "emission_mean_high": means[1],
                    "first_training_week": first, "last_training_week": origin,
                    "fit_attempts": len(attempts), "converged": True,
                    "iterations": attempts[-1]["iterations"],
                    "last_improvement": attempts[-1]["last_improvement"],
                    "random_state": attempts[-1]["random_state"],
                })
            wide.append(row)
    result = {"signature": signature, "features": pd.DataFrame(wide).set_index("research_week"),
              "probabilities": pd.DataFrame(probabilities), "fits": fits}
    _save_cache(result, c.REGIME_CACHE_PATH)
    return result


def origin_dataset(tables, regime_features, origin):
    frame = tables["features"].join(regime_features, on="research_week")
    eligible = tables["targets"].loc[training_mask(tables["targets"], origin, tables["calendar"])]
    eligible = eligible.loc[eligible.index.get_level_values("asset").isin(c.FORECAST_ASSETS)]
    training = frame.join(eligible[["target_4w_excess_return"]], how="inner").reset_index()
    training["asset"] = pd.Categorical(training["asset"], categories=c.FORECAST_ASSETS)
    training = training.sort_values(["research_week", "asset"])
    x = training.drop(columns=["research_week", "target_4w_excess_return"])
    y = training["target_4w_excess_return"].to_numpy()
    query = frame.xs(origin, level="research_week").reindex(c.FORECAST_ASSETS).reset_index()
    query["asset"] = pd.Categorical(query["asset"], categories=c.FORECAST_ASSETS)
    if not len(x) or not np.isfinite(y).all() or x.isna().any().any() or query.isna().any().any():
        raise ModelFitError(f"Week {origin}: unavailable training labels or predictors.")
    if eligible["realization_close"].max() > tables["calendar"].at[origin, "decision_cutoff"]:
        raise ModelFitError("A training label is not mature at its forecast origin.")
    return {"x": x, "y": y, "query": query,
            "training_observations": len(y),
            "latest_training_realization": eligible["realization_close"].max()}


def candidate_groups():
    groups = {}
    for candidate, parameters in c.grid_candidates(c.RETURN_MODEL_GRID):
        key = tuple((name, value) for name, value in parameters.items() if name != "max_iter")
        groups.setdefault(key, []).append((candidate, parameters))
    return list(groups.values())


@lru_cache(maxsize=1)
def _origin_payloads(path):
    return joblib.load(path, mmap_mode="r")


def _search_origin(origin, signature):
    """Every grid candidate receives a fit call; max_iter checkpoints reuse trees."""
    dataset = _origin_payloads(str(c.ORIGIN_DATASETS_PATH))["datasets"][origin]
    if len(dataset["y"]) >= c.HGB_BINNING_SUBSAMPLE:
        raise ModelFitError("Checkpoint equivalence requires full-row binning without subsampling.")
    path = c.RETURN_SEARCH_DIR / f"origin_{origin}.joblib"
    groups = candidate_groups()
    completed = 0
    predictions = np.full((c.RETURN_MODEL_CANDIDATES, len(c.FORECAST_ASSETS)), np.nan)
    if path.exists():
        checkpoint = joblib.load(path)
        if checkpoint["signature"] == signature:
            completed, predictions = checkpoint["completed_groups"], checkpoint["predictions"]
    started = time.perf_counter()
    with threadpool_limits(limits=c.MODEL_THREADS):
        for position in range(completed, len(groups)):
            model = HistGradientBoostingRegressor(**c.RETURN_MODEL_SETTINGS, warm_start=True)
            if model.max_features != 1.0 or model.early_stopping:
                raise ModelFitError("Checkpoint equivalence requires all features and no early stopping.")
            for candidate, parameters in groups[position]:
                seed = c.seed_record("return_model", origin, candidate_id=candidate)
                model.set_params(**parameters, random_state=seed["random_state"])
                model.fit(dataset["x"], dataset["y"])
                predictions[candidate] = model.predict(dataset["query"])
            if ((position + 1) % c.SELECTION_CHECKPOINT_GROUPS == 0
                    or position + 1 == len(groups)):
                checkpoint = {
                    "signature": signature, "origin": origin,
                    "completed_groups": position + 1, "predictions": predictions,
                    "fit_calls": (position + 1) * len(c.RETURN_MODEL_GRID["max_iter"]),
                    "elapsed_seconds_this_run": time.perf_counter() - started,
                }
                _save_cache(checkpoint, path)
                progress = {name: checkpoint[name] for name in (
                    "origin", "completed_groups", "fit_calls", "elapsed_seconds_this_run"
                )}
                path.with_suffix(".progress.json").write_text(json.dumps(progress) + "\n", encoding="utf-8")
    return origin, path


def fit_forecast(tables, regime_features, origin, candidate, parameters):
    dataset = origin_dataset(tables, regime_features, origin)
    seed = c.seed_record("return_model", origin, candidate_id=candidate)
    with threadpool_limits(limits=c.MODEL_THREADS):
        model = HistGradientBoostingRegressor(
            **c.RETURN_MODEL_SETTINGS, **parameters, random_state=seed["random_state"]
        ).fit(dataset["x"], dataset["y"])
        prediction = model.predict(dataset["query"])
    return prediction, dataset


def select_return_model(tables, regimes):
    origins = list(c.VALIDATION_ORIGINS)
    datasets = {origin: origin_dataset(tables, regimes["features"], origin) for origin in origins}
    signature = fingerprint({
        "features": frame_hash(tables["features"]), "regimes": regimes["signature"],
        "targets": frame_hash(tables["targets"]), "grid": dict(c.RETURN_MODEL_GRID),
        "settings": dict(c.RETURN_MODEL_SETTINGS), "seed": c.MASTER_SEED,
        "method": "all_candidates_origin_checkpoint_v1",
    })
    if not c.ORIGIN_DATASETS_PATH.exists() or joblib.load(c.ORIGIN_DATASETS_PATH)["signature"] != signature:
        _save_cache({"signature": signature, "datasets": datasets}, c.ORIGIN_DATASETS_PATH)
    c.RETURN_SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    with parallel_config(backend="loky", n_jobs=c.SELECTION_WORKERS, inner_max_num_threads=c.MODEL_THREADS):
        results = Parallel(return_as="generator_unordered", temp_folder=str(c.JOBLIB_TEMP_DIR))(
            delayed(_search_origin)(origin, signature) for origin in origins
        )
        for origin, _ in results:
            print(f"Return grid: week {origin} complete ({len(origins)} origins required).", flush=True)
    predictions = np.stack([
        joblib.load(c.RETURN_SEARCH_DIR / f"origin_{origin}.joblib")["predictions"]
        for origin in origins
    ])
    if not np.isfinite(predictions).all():
        raise ModelFitError("The full return grid is incomplete.")
    actual = np.stack([
        tables["targets"].xs(origin, level="research_week")
        .reindex(c.FORECAST_ASSETS)["target_4w_excess_return"].to_numpy()
        for origin in origins
    ])
    scored = np.isfinite(actual).all(axis=1)
    mse = np.mean((predictions[scored] - actual[scored, None, :]) ** 2, axis=(0, 2))
    best = int(np.lexsort((np.arange(len(mse)), mse))[0])
    rows = [{
        "evaluation_period": "validation", "model": "return_model", "candidate_id": candidate,
        **parameters, "pooled_mse": mse[candidate], "fit_origins": len(origins),
        "scored_origins": int(scored.sum()), "scored_observations": int(scored.sum()) * len(c.FORECAST_ASSETS),
        "first_scored_origin": origins[np.flatnonzero(scored)[0]],
        "last_scored_origin": origins[np.flatnonzero(scored)[-1]], "selected": candidate == best,
    } for candidate, parameters in c.grid_candidates(c.RETURN_MODEL_GRID)]
    return {"scores": pd.DataFrame(rows), "candidate_id": best,
            "parameters": list(c.grid_candidates(c.RETURN_MODEL_GRID))[best][1],
            "forecasts": pd.DataFrame(predictions[:, best], index=origins, columns=c.FORECAST_ASSETS),
            "signature": signature, "fit_calls": len(origins) * c.RETURN_MODEL_CANDIDATES,
            "origin_metadata": {origin: {name: datasets[origin][name] for name in (
                "training_observations", "latest_training_realization"
            )} for origin in origins}}
