"""Validation selection, frozen research choices, and one locked-test assessment."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json

import joblib
import numpy as np
import pandas as pd

from . import config as c, data, models, portfolio


class EvaluationError(RuntimeError):
    """The requested evaluation would violate the frozen experiment."""


def performance_measures(returns, cash_returns, turnover):
    values = np.asarray(returns, dtype=float)
    cash = np.asarray(cash_returns, dtype=float)
    turnover = np.asarray(turnover, dtype=float)
    if (len(values) < 2 or cash.shape != values.shape or turnover.shape != values.shape
            or not all(np.isfinite(array).all() for array in (values, cash, turnover))
            or (values < -1).any()):
        raise ValueError("Evaluation requires at least two finite matched weekly observations.")
    excess = values - cash
    excess_volatility = excess.std(ddof=c.VOLATILITY_DDOF)
    wealth = np.r_[1.0, np.cumprod(1 + values)]
    return {
        "annualized_return": float(np.prod(1 + values) ** (c.WEEKS_PER_YEAR / len(values)) - 1),
        "annualized_volatility": float(values.std(ddof=c.VOLATILITY_DDOF) * np.sqrt(c.WEEKS_PER_YEAR)),
        "excess_return_sharpe": float(excess.mean() / excess_volatility * np.sqrt(c.WEEKS_PER_YEAR))
        if excess_volatility > c.ZERO_VOLATILITY_TOLERANCE else np.nan,
        "maximum_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1)),
        "mean_weekly_turnover": float(turnover.mean()),
    }


def forecast_score(forecasts):
    selected = forecasts.loc[forecasts["scored"] & forecasts["asset"].isin(c.FORECAST_ASSETS)]
    if selected.empty or not np.isfinite(selected[["forecast_4w_excess_return", "realized_target"]]).all().all():
        raise EvaluationError("Scored forecasts have unavailable outcomes.")
    sizes = selected.groupby("research_week").size()
    if not sizes.eq(len(c.FORECAST_ASSETS)).all():
        raise EvaluationError("Pooled scoring requires all five non-cash assets at each origin.")
    return float(np.mean((selected["forecast_4w_excess_return"] - selected["realized_target"]) ** 2))


def forecast_table(predictions, tables, metadata, period):
    rows = []
    for origin, prediction in predictions.iterrows():
        timing = tables["calendar"].loc[origin]
        for asset in c.ASSETS:
            target = tables["targets"].loc[(origin, asset)]
            scored = (period == "locked_test" or bool(target["selection_eligible"]))
            realized = target["target_4w_excess_return"] if scored else np.nan
            if scored and not np.isfinite(realized):
                raise EvaluationError("A scored target is unavailable.")
            rows.append({
                "evaluation_period": period, "research_week": origin, "asset": asset,
                "decision_cutoff": timing["decision_cutoff"], "execution_close": timing["execution_close"],
                "realization_close": timing["realization_close"],
                "forecast_4w_excess_return": float(prediction[asset]), "realized_target": realized,
                "scored": scored, "exclusion_reason": "" if scored else "target_crosses_locked_cutoff",
                **metadata[origin],
            })
    return pd.DataFrame(rows)


def _cash_rows(return_frame):
    cash = return_frame.copy()
    cash["portfolio"] = "cash"
    cash["weekly_total_return"] = cash["cash_return"]
    cash["weekly_excess_return"] = 0.0
    cash.loc[~cash["scored"], "weekly_excess_return"] = np.nan
    cash["one_way_turnover"] = 0.0
    wealth = 1.0
    for position in cash.index:
        cash.at[position, "wealth_start"] = wealth
        if cash.at[position, "scored"]:
            wealth *= 1 + cash.at[position, "cash_return"]
            cash.at[position, "wealth_end"] = wealth
        else:
            cash.at[position, "wealth_end"] = np.nan
    return cash


def performance_table(returns, forecasts):
    rows = []
    for (period, name), frame in returns.groupby(["evaluation_period", "portfolio"], sort=False):
        selected = frame.loc[frame["scored"]]
        period_forecasts = forecasts.loc[forecasts["evaluation_period"].eq(period)]
        rows.append({
            "evaluation_period": period, "portfolio": name,
            "pooled_forecast_mse": forecast_score(period_forecasts) if name == "strategy" else np.nan,
            "forecast_scored_origins": int(period_forecasts.loc[period_forecasts["scored"], "research_week"].nunique())
            if name == "strategy" else 0,
            "return_observations": len(selected), "first_return_origin": int(selected["research_week"].min()),
            "last_return_origin": int(selected["research_week"].max()),
            "period_start": selected["execution_close"].min(), "period_end": selected["realization_close"].max(),
            **performance_measures(selected["weekly_total_return"], selected["cash_return"], selected["one_way_turnover"]),
        })
    return pd.DataFrame(rows)


def _input_provenance(tables):
    manifest = tables["retrieval_manifest"]
    return {
        "source_snapshots": {entry["local_snapshot"]: entry["sha256"] for entry in manifest["retrievals"]},
        "prepared_tables": {entry["local_path"]: entry["sha256"] for entry in manifest["local_artifacts"]},
        "availability_policy": c.MACRO_AVAILABILITY_RULE,
    }


def _code_hashes():
    names = [".gitattributes", "regime_allocation/config.py", "regime_allocation/data.py", "regime_allocation/features.py",
             "regime_allocation/models.py", "regime_allocation/portfolio.py", "regime_allocation/evaluation.py",
             "run_pipeline.py", "requirements.txt"]
    return {name: data._code_hash(c.PROJECT_ROOT / name) for name in names}


def _dependency_versions():
    return {name: version(name) for name in (
        "numpy", "pandas", "scipy", "scikit-learn", "hmmlearn", "cvxpy", "clarabel", "joblib",
    )}


def _write_json(path, value, *, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=str)
        stream.write("\n")


def verify_frozen():
    if not c.FROZEN_CONFIG_PATH.exists():
        raise EvaluationError("Validation selection must finish and freeze settings before locked testing.")
    frozen = json.loads(c.FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))
    expected_seal = models.fingerprint({name: value for name, value in frozen.items() if name != "seal"})
    if frozen.get("seal") != expected_seal:
        raise EvaluationError("Frozen configuration integrity check failed.")
    if frozen["code_hashes"] != _code_hashes():
        raise EvaluationError("Research code differs from the frozen experiment.")
    if frozen["dependency_versions"] != _dependency_versions():
        raise EvaluationError("Installed numerical dependencies differ from the frozen experiment.")
    for paths in frozen["input_provenance"].values():
        if not isinstance(paths, dict):
            continue
        for name, digest in paths.items():
            path = c.PROJECT_ROOT / name
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise EvaluationError(f"Frozen input differs: {name}")
    return frozen


def _export(frames):
    hashes = {}
    for name, frame in frames.items():
        path = c.RESULT_PATHS[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False, float_format="%.17g", lineterminator="\n")
        temporary.replace(path)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _read_result(name):
    """Round-trip 17-digit numbers without changing frozen historical rows."""
    path = c.RESULT_PATHS[name]
    columns = pd.read_csv(path, nrows=0).columns
    strings = {
        "evaluation_period", "portfolio", "asset", "component", "model", "solver_status",
        "decision_cutoff", "reference_close", "execution_close", "realization_close",
        "latest_training_realization", "period_start", "period_end", "exclusion_reason",
        "selected", "scored", "converged",
    }
    converters = {column: (lambda value: float(value) if value else np.nan)
                  for column in columns if column not in strings}
    frame = pd.read_csv(path, converters=converters, keep_default_na=False)
    if "research_week" in frame:
        frame["research_week"] = frame["research_week"].astype(int)
    return frame


def _recorded_validation_regimes(frozen):
    """Restore origin-time historical features from the verified public table."""
    probabilities = _read_result("regime_probabilities")
    probabilities = probabilities.loc[probabilities["research_week"] <= c.SPLIT_WEEKS["validation"][1]].copy()
    encoded = probabilities.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode()
    if hashlib.sha256(encoded).hexdigest() != frozen["validation_output_hashes"]["regime_probabilities"]:
        raise EvaluationError("Frozen historical regime probabilities differ.")
    probabilities["decision_cutoff"] = pd.to_datetime(probabilities["decision_cutoff"], utc=True)
    indexed = probabilities.set_index(["research_week", "component"])
    blocks = []
    for component in c.HMM_EMISSIONS:
        block = indexed.xs(component, level="component")[["probability_low", "probability_high"]]
        blocks.append(block.rename(columns={"probability_low": f"{component}_state_0",
                                            "probability_high": f"{component}_state_1"}))
    features = pd.concat(blocks, axis=1).sort_index()
    expected = list(range(c.HMM_MIN_HISTORY_WEEKS, c.SPLIT_WEEKS["validation"][1] + 1))
    if features.index.tolist() != expected or not np.isfinite(features.to_numpy()).all():
        raise EvaluationError("Frozen historical regime coverage is incomplete.")
    return {"features": features, "probabilities": probabilities,
            "fits": frozen["seed_mapping"]["hmm_fit_attempts"], "signature": frozen["seal"]}


def run_selection():
    if c.FROZEN_CONFIG_PATH.exists():
        frozen = verify_frozen()
        if c.TEST_RECEIPT_PATH.exists():
            receipt = json.loads(c.TEST_RECEIPT_PATH.read_text(encoding="utf-8"))
            if receipt["status"] == "complete":
                _verify_test_receipt(frozen)
            elif receipt["frozen_sha256"] != frozen["seal"]:
                raise EvaluationError("An incomplete test belongs to another frozen experiment.")
        else:
            for name, digest in frozen["validation_output_hashes"].items():
                if hashlib.sha256(c.RESULT_PATHS[name].read_bytes()).hexdigest() != digest:
                    raise EvaluationError(f"Frozen validation output differs: {name}")
        print("Using the complete frozen validation selection.", flush=True)
        return frozen
    tables = data.load_prepared("validation")
    regimes = models.causal_regimes(tables)
    search = models.select_return_model(tables, regimes)
    models._save_cache(search, c.RETURN_SELECTION_CACHE_PATH)
    predictions = search["forecasts"].assign(**{c.CASH_ASSET: c.CASH_EXCESS_FORECAST}).reindex(columns=c.ASSETS)
    forecast_rows = forecast_table(predictions, tables, search["origin_metadata"], "validation")
    forecast_rows["return_candidate_id"] = search["candidate_id"]
    cutoff = tables["locked_test_cutoff"]
    best_key, best_result = None, None
    allocation_scores = []
    for candidate, parameters in c.grid_candidates(c.ALLOCATION_GRID):
        allocations, returns, state = portfolio.simulate(
            tables["adjusted_prices"], tables["weekly_returns"], tables["calendar"], predictions,
            c.VALIDATION_ORIGINS, name="strategy", parameters=parameters, cutoff=cutoff,
        )
        allocations["allocation_candidate_id"] = candidate
        scored = returns.loc[returns["scored"]]
        measures = performance_measures(scored["weekly_total_return"], scored["cash_return"], scored["one_way_turnover"])
        sharpe = measures["excess_return_sharpe"]
        if not np.isfinite(sharpe):
            raise EvaluationError(f"Allocation candidate {candidate} has undefined validation Sharpe.")
        allocation_scores.append({
            "evaluation_period": "validation", "model": "allocation", "candidate_id": candidate,
            **parameters, "validation_sharpe": sharpe, "allocation_decisions": len(returns),
            "scored_origins": len(scored), "scored_observations": len(scored),
            "first_scored_origin": int(scored["research_week"].min()),
            "last_scored_origin": int(scored["research_week"].max()), "selected": False,
        })
        key = (-sharpe, candidate)
        if best_key is None or key < best_key:
            best_key, best_result = key, (allocations, returns, state, parameters)
    allocation_scores[best_key[1]]["selected"] = True
    allocation_frames, return_frames, states = [best_result[0]], [best_result[1]], {"strategy": best_result[2]}
    for name in c.BENCHMARKS:
        allocations, returns, state = portfolio.simulate(
            tables["adjusted_prices"], tables["weekly_returns"], tables["calendar"], predictions,
            c.VALIDATION_ORIGINS, name=name, cutoff=cutoff,
        )
        allocation_frames.append(allocations)
        return_frames.append(returns)
        states[name] = state
    return_frames.append(_cash_rows(best_result[1]))
    returns = pd.concat(return_frames, ignore_index=True)
    frames = {
        "selection": pd.concat([search["scores"], pd.DataFrame(allocation_scores)], ignore_index=True),
        "weekly_forecasts": forecast_rows, "regime_probabilities": regimes["probabilities"],
        "weekly_allocations": pd.concat(allocation_frames, ignore_index=True),
        "weekly_returns": returns, "performance": performance_table(returns, forecast_rows),
    }
    output_hashes = _export(frames)
    seed_attempts = [{name: row[name] for name in (
        "component", "component_id", "forecast_origin_week", "candidate_id", "repeated_fit_id",
        "random_state", "accepted", "iterations", "first_training_week", "last_training_week", "failure",
    )} for row in regimes["fits"]]
    frozen = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "first_locked_test_origin": c.SPLIT_WEEKS["locked_test"][0], "locked_test_cutoff": cutoff.isoformat(),
        "locked_test_executed_at_freeze": False,
        "return_candidate_id": search["candidate_id"], "return_model_parameters": search["parameters"],
        "allocation_candidate_id": best_key[1], "allocation_parameters": best_result[3],
        "hmm_settings": dict(c.HMM_SETTINGS), "return_model_settings": dict(c.RETURN_MODEL_SETTINGS),
        "regime_rules": {
            "minimum_history_weeks": c.HMM_MIN_HISTORY_WEEKS,
            "maximum_seeded_attempts": c.HMM_MAX_FIT_ATTEMPTS,
            "state_order": "ascending emission mean",
            "probabilities": "forward filtering under each row's origin-time fitted model",
            "covariance_floor": "enforced after every diagonal maximum-likelihood M step",
            "historical_rows": "retained unchanged across later refits",
            "reconstruction": "restore frozen validation probabilities; append later origin-time fits",
        },
        "feature_rules": {
            "return_lag_weeks": c.RETURN_LAG_WEEKS, "trailing_windows_weeks": c.TRAILING_WINDOWS_WEEKS,
            "excess_compounding": c.EXCESS_RETURN_COMPOUNDING,
            "volatility_ddof": c.VOLATILITY_DDOF, "macro_change_weeks": c.MACRO_CHANGE_WEEKS,
            "macro_change_methods": dict(c.MACRO_CHANGE_METHODS),
        },
        "return_model_grid": dict(c.RETURN_MODEL_GRID), "allocation_grid": dict(c.ALLOCATION_GRID),
        "grid_key_order": {"return_model": list(c.RETURN_MODEL_GRID), "allocation": list(c.ALLOCATION_GRID)},
        "tie_break": c.SELECTION_TIE_BREAK,
        "selection_metrics": {"return_model": c.RETURN_MODEL_SELECTION, "allocation": c.ALLOCATION_SELECTION},
        "feature_columns": list(tables["features"].columns) + list(regimes["features"].columns),
        "calendar_splits": dict(c.SPLIT_WEEKS), "training_window_weeks": c.TRAINING_WINDOW_WEEKS,
        "weight_bounds": [c.MIN_ASSET_WEIGHT, c.MAX_ASSET_WEIGHT], "weight_tolerance": c.WEIGHT_TOLERANCE,
        "solver": c.ALLOCATION_SOLVER, "solver_settings": dict(c.SOLVER_SETTINGS),
        "covariance": {"estimator": c.COVARIANCE_ESTIMATOR, "window_weeks": c.COVARIANCE_WINDOW_WEEKS,
                       "horizon_multiplier": c.COVARIANCE_HORIZON_MULTIPLIER},
        "allocation_objective": c.ALLOCATION_OBJECTIVE, "allocation_reference": c.ALLOCATION_REFERENCE,
        "benchmarks": {"names": c.BENCHMARKS, "inverse_volatility_window_weeks": c.INVERSE_VOLATILITY_WINDOW_WEEKS,
                       "projection": c.INVERSE_VOLATILITY_PROJECTION},
        "accounting": {"turnover": c.TURNOVER_RULE, "execution": c.EXECUTION_RULE,
                       "transaction_cost_rate": c.TRANSACTION_COST_RATE, "slippage_rate": c.SLIPPAGE_RATE,
                       "tax_rate": c.TAX_RATE, "annualization_weeks": c.WEEKS_PER_YEAR},
        "numerical_tolerances": {"weights": c.WEIGHT_TOLERANCE,
                                 "covariance": c.COVARIANCE_TOLERANCE,
                                 "accounting": c.ACCOUNTING_TOLERANCE},
        "input_provenance": _input_provenance(tables), "code_hashes": _code_hashes(),
        "dependency_versions": _dependency_versions(),
        "validation_output_hashes": output_hashes, "validation_end_holdings": states,
        "seed_mapping": {
            "master_seed": c.MASTER_SEED, "component_ids": dict(c.SEED_COMPONENT_IDS),
            "spawn_key_fields": c.SEED_KEY_FIELDS,
            "random_state_rule": "SeedSequence(master_seed, spawn_key=key).generate_state(1, uint32)[0]",
            "return_model_domain": {"forecast_origins_inclusive": list(c.SPLIT_WEEKS["validation"]),
                                    "candidate_ids_inclusive": [0, c.RETURN_MODEL_CANDIDATES - 1],
                                    "repeated_fit_id": 0},
            "locked_test_return_model_domain": {
                "forecast_origins_inclusive": list(c.SPLIT_WEEKS["locked_test"]), "candidate_id": search["candidate_id"],
                "repeated_fit_id": 0,
            },
            "hmm_fit_attempts": seed_attempts,
        },
        "fit_counts": {"return_model": search["fit_calls"], "hmm": len(regimes["fits"]),
                       "hmm_rejected_attempts": sum(not row["accepted"] for row in regimes["fits"]),
                       "allocation_candidates": len(allocation_scores),
                       "planned_locked_test_return_fits": len(c.LOCKED_TEST_ORIGINS)},
        "fit_strategy": "five within-origin max_iter checkpoint fit calls per parameter group; every grid candidate evaluated",
        "selection_coverage": {"forecast_origins": int(search["scores"].iloc[0]["scored_origins"]),
                               "weekly_portfolio_periods": int(best_result[1]["scored"].sum())},
        "selection_excluded_origins": {
            "forecasts": sorted(forecast_rows.loc[~forecast_rows["scored"], "research_week"].unique().tolist()),
            "weekly_returns": best_result[1].loc[~best_result[1]["scored"], "research_week"].tolist(),
        },
    }
    frozen["seal_format"] = "sha256 of canonical JSON excluding the seal field"
    frozen["seal"] = models.fingerprint(frozen)
    _write_json(c.FROZEN_CONFIG_PATH, frozen, exclusive=True)
    print("Validation complete and choices frozen. Locked testing has not run.", flush=True)
    return frozen


def _verify_test_receipt(frozen):
    receipt = json.loads(c.TEST_RECEIPT_PATH.read_text(encoding="utf-8"))
    if receipt["status"] != "complete":
        raise EvaluationError("A locked-test run is incomplete; resume --stage test using the frozen inputs.")
    if receipt["frozen_sha256"] != frozen["seal"]:
        raise EvaluationError("Locked-test receipt refers to a different frozen experiment.")
    for name, digest in receipt["output_hashes"].items():
        if hashlib.sha256(c.RESULT_PATHS[name].read_bytes()).hexdigest() != digest:
            raise EvaluationError(f"Completed locked-test output differs: {name}")
    return receipt


def run_locked_test():
    """An explicit test stage evaluates the frozen choice once; reruns verify outputs."""
    frozen = verify_frozen()
    frozen_hash = frozen["seal"]
    resuming = c.TEST_RECEIPT_PATH.exists()
    if c.TEST_RECEIPT_PATH.exists():
        receipt = json.loads(c.TEST_RECEIPT_PATH.read_text(encoding="utf-8"))
        if receipt["status"] == "complete":
            _verify_test_receipt(frozen)
            print("Reusing the completed locked-test assessment.", flush=True)
            return receipt
        if receipt["frozen_sha256"] != frozen_hash:
            raise EvaluationError("An incomplete test belongs to a different frozen experiment.")
    if not resuming:
        for name, digest in frozen["validation_output_hashes"].items():
            path = c.RESULT_PATHS[name]
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise EvaluationError(f"Frozen validation output differs before testing: {name}")
    previous = _recorded_validation_regimes(frozen)
    _write_json(c.TEST_RECEIPT_PATH, {"status": "running", "frozen_sha256": frozen_hash})
    tables = data.load_prepared("locked_test")
    if resuming and c.REGIME_CACHE_PATH.exists():
        cached_regimes = joblib.load(c.REGIME_CACHE_PATH)
        historic = cached_regimes["features"].reindex(previous["features"].index)
        if not np.array_equal(historic.to_numpy(), previous["features"].to_numpy()):
            raise EvaluationError("Test cache changes frozen historical regime features.")
        regimes = models.causal_regimes(tables)
    else:
        regimes = models.causal_regimes(tables, previous=previous)
    cached = joblib.load(c.TEST_FORECAST_CACHE_PATH) if c.TEST_FORECAST_CACHE_PATH.exists() else {
        "frozen_sha256": frozen_hash, "predictions": {}, "metadata": {},
    }
    if cached["frozen_sha256"] != frozen_hash:
        raise EvaluationError("Test forecast checkpoints belong to another frozen experiment.")
    for origin in c.LOCKED_TEST_ORIGINS:
        if origin in cached["predictions"]:
            continue
        prediction, metadata = models.fit_forecast(
            tables, regimes["features"], origin, frozen["return_candidate_id"], frozen["return_model_parameters"]
        )
        cached["predictions"][origin] = prediction
        cached["metadata"][origin] = {name: metadata[name] for name in (
            "training_observations", "latest_training_realization"
        )}
        models._save_cache(cached, c.TEST_FORECAST_CACHE_PATH)
    predictions = pd.DataFrame.from_dict(cached["predictions"], orient="index", columns=c.FORECAST_ASSETS)
    predictions[c.CASH_ASSET] = c.CASH_EXCESS_FORECAST
    predictions = predictions.reindex(columns=c.ASSETS).sort_index()
    forecasts = forecast_table(predictions, tables, cached["metadata"], "locked_test")
    forecasts["return_candidate_id"] = frozen["return_candidate_id"]
    allocations, returns = [], []
    for name in ("strategy", *c.BENCHMARKS):
        allocated, returned, _ = portfolio.simulate(
            tables["adjusted_prices"], tables["weekly_returns"], tables["calendar"], predictions,
            c.LOCKED_TEST_ORIGINS, name=name, parameters=frozen["allocation_parameters"] if name == "strategy" else None,
            initial_state=frozen["validation_end_holdings"][name],
        )
        allocated["allocation_candidate_id"] = frozen["allocation_candidate_id"] if name == "strategy" else np.nan
        allocations.append(allocated)
        returns.append(returned)
    returns.append(_cash_rows(returns[0]))
    test_returns = pd.concat(returns, ignore_index=True)
    frames = {
        "selection": _read_result("selection"),
        "regime_probabilities": regimes["probabilities"],
    }
    for name, new in {
        "weekly_forecasts": forecasts, "weekly_allocations": pd.concat(allocations, ignore_index=True),
        "weekly_returns": test_returns, "performance": performance_table(test_returns, forecasts),
    }.items():
        old = _read_result(name)
        frames[name] = pd.concat([old.loc[old["evaluation_period"].eq("validation")], new], ignore_index=True)
    output_hashes = _export(frames)
    receipt = {"status": "complete", "frozen_sha256": frozen_hash,
               "completed_at_utc": datetime.now(timezone.utc).isoformat(), "output_hashes": output_hashes,
               "return_model_fits": len(c.LOCKED_TEST_ORIGINS)}
    _write_json(c.TEST_RECEIPT_PATH, receipt)
    return receipt
