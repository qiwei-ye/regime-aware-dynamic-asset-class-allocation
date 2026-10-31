"""Cold verification of frozen choices, without repeating model selection."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as c, data, evaluation, models, portfolio


def compare_table(expected, reproduced):
    """Require identical structure/metadata and absolute numeric error <= 1e-10."""
    first, second = pd.read_csv(expected), pd.read_csv(reproduced)
    if first.shape != second.shape or first.columns.tolist() != second.columns.tolist():
        raise ValueError(f"Reproduction changes the table structure: {expected.name}")
    maximum = 0.0
    for column in first:
        left, right = first[column], second[column]
        identifier = column.endswith("_id") or column in ("research_week", "random_state")
        if (not identifier and pd.api.types.is_float_dtype(left.dtype)
                and pd.api.types.is_numeric_dtype(right.dtype)):
            if not left.isna().equals(right.isna()):
                raise ValueError(f"Reproduction changes unavailable values: {column}")
            values = np.abs(left.dropna().to_numpy() - right.loc[left.notna()].to_numpy())
            difference = float(values.max(initial=0))
            if not np.isfinite(difference) or difference > c.ACCOUNTING_TOLERANCE:
                raise ValueError(f"Reproduction exceeds numeric tolerance: {column}")
            maximum = max(maximum, difference)
        elif not left.equals(right):
            raise ValueError(f"Reproduction changes identifiers or metadata: {column}")
    digest = hashlib.sha256(reproduced.read_bytes()).hexdigest()
    return {"rows": len(first), "columns": len(first.columns),
            "byte_identical": expected.read_bytes() == reproduced.read_bytes(),
            "maximum_absolute_numeric_difference": maximum, "sha256": digest}


def _replay_validation():
    """Rebuild origin-time HMMs and the selected validation model, never the grid."""
    frozen = evaluation.verify_frozen()
    tables = data.load_prepared("validation")
    print("Rebuilding validation HMMs from recorded inputs.", flush=True)
    regimes = models.causal_regimes(tables)
    predictions, metadata = {}, {}
    for origin in c.VALIDATION_ORIGINS:
        prediction, dataset = models.fit_forecast(
            tables, regimes["features"], origin,
            frozen["return_candidate_id"], frozen["return_model_parameters"],
        )
        predictions[origin] = prediction
        metadata[origin] = {name: dataset[name] for name in (
            "training_observations", "latest_training_realization"
        )}
        if (origin-c.VALIDATION_ORIGINS.start+1) % 52 == 0:
            print(f"Frozen validation refits: {origin-c.VALIDATION_ORIGINS.start+1}/208.", flush=True)
    forecasts = pd.DataFrame.from_dict(predictions, orient="index", columns=c.FORECAST_ASSETS)
    forecasts[c.CASH_ASSET] = c.CASH_EXCESS_FORECAST
    forecasts = forecasts.reindex(columns=c.ASSETS)
    forecast_rows = evaluation.forecast_table(forecasts, tables, metadata, "validation")
    forecast_rows["return_candidate_id"] = frozen["return_candidate_id"]
    allocated, returned = [], []
    for name in ("strategy", *c.BENCHMARKS):
        allocations, returns, _ = portfolio.simulate(
            tables["adjusted_prices"], tables["weekly_returns"], tables["calendar"], forecasts,
            c.VALIDATION_ORIGINS, name=name, cutoff=tables["locked_test_cutoff"],
            parameters=frozen["allocation_parameters"] if name == "strategy" else None,
        )
        if name == "strategy":
            allocations["allocation_candidate_id"] = frozen["allocation_candidate_id"]
        allocated.append(allocations)
        returned.append(returns)
    returned.append(evaluation._cash_rows(returned[0]))
    returns = pd.concat(returned, ignore_index=True)
    evaluation._export({
        "weekly_forecasts": forecast_rows, "regime_probabilities": regimes["probabilities"],
        "weekly_allocations": pd.concat(allocated, ignore_index=True),
        "weekly_returns": returns, "performance": evaluation.performance_table(returns, forecast_rows),
    })
    for name, digest in frozen["validation_output_hashes"].items():
        if hashlib.sha256(c.RESULT_PATHS[name].read_bytes()).hexdigest() != digest:
            raise ValueError(f"Cold validation differs from the frozen handoff: {name}")


def verify_reproduction():
    """Preserve completed outputs, run a fresh isolated replay, and record evidence."""
    frozen = evaluation.verify_frozen()
    receipt = evaluation._verify_test_receipt(frozen)
    dependencies = {}
    for line in (c.PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            name, expected = line.split("==")
            dependencies[name] = version(name)
            if dependencies[name] != expected:
                raise ValueError(f"Installed dependency differs from the pinned environment: {name}")
    figures = [c.FIGURES_DIR / name for name in ("performance.png", "allocations.png")]
    protected = [*c.RESULT_PATHS.values(), *figures, c.FROZEN_CONFIG_PATH,
                 c.TEST_RECEIPT_PATH, c.RETRIEVAL_MANIFEST_PATH]
    original = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    with tempfile.TemporaryDirectory(prefix="reproduction-", dir=c.DATA_DIR) as directory:
        temporary = Path(directory).resolve()
        if not temporary.is_relative_to(c.DATA_DIR.resolve()):
            raise ValueError("Temporary reproduction directory is outside local data storage.")
        preserved, fresh = temporary / "preserved", temporary / "fresh"
        shutil.copytree(c.OUTPUT_DIR, preserved / "results")
        shutil.copytree(c.ARTIFACTS_DIR, preserved / "artifacts")
        shutil.copytree(c.PROJECT_ROOT / "regime_allocation", fresh / "regime_allocation",
                        ignore=shutil.ignore_patterns("__pycache__"))
        for name in (".gitattributes", "requirements.txt", "run_pipeline.py"):
            shutil.copy2(c.PROJECT_ROOT / name, fresh / name)
        shutil.copytree(c.RAW_DATA_DIR, fresh / "data" / "raw")
        (fresh / "artifacts").mkdir()
        shutil.copy2(c.FROZEN_CONFIG_PATH, fresh / "artifacts" / c.FROZEN_CONFIG_PATH.name)
        (fresh / "results").mkdir()
        shutil.copy2(c.RESULT_PATHS["selection"], fresh / "results" / "selection.csv")
        assert not (fresh / "data" / "processed").exists()
        assert not (fresh / "data" / "model_cache").exists()
        assert not (fresh / "artifacts" / c.TEST_RECEIPT_PATH.name).exists()
        print("Completed outputs preserved; fresh workspace contains raw snapshots and frozen choices only.", flush=True)
        subprocess.run([sys.executable, "run_pipeline.py", "--stage", "data"], cwd=fresh,
                       env=environment, check=True)
        subprocess.run([sys.executable, "-c",
                        "from regime_allocation.reproduction import _replay_validation; _replay_validation()"],
                       cwd=fresh, env=environment, check=True)
        assert not (fresh / "data" / "model_cache" / c.TEST_FORECAST_CACHE_PATH.name).exists()
        print("Running python run_pipeline.py without a locked-test cache or receipt.", flush=True)
        subprocess.run([sys.executable, "run_pipeline.py"], cwd=fresh, env=environment, check=True)
        subprocess.run([sys.executable, "-c",
                        "from regime_allocation.reporting import export_figures; export_figures()"],
                       cwd=fresh, env=environment, check=True)
        tables = {name: compare_table(preserved / "results" / path.name, fresh / "results" / path.name)
                  for name, path in c.RESULT_PATHS.items()}
        figure_checks = {path.name: {
            "byte_identical": (fresh / "results" / "figures" / path.name).read_bytes() == path.read_bytes(),
            "sha256": hashlib.sha256((fresh / "results" / "figures" / path.name).read_bytes()).hexdigest(),
        } for path in figures}
        if not all(value["byte_identical"] for value in figure_checks.values()):
            raise ValueError("Reproduced figures differ from the completed outputs.")
        manifest = json.loads((fresh / "results" / c.RETRIEVAL_MANIFEST_PATH.name).read_text(encoding="utf-8"))
        provenance = {
            "source_snapshots": {row["local_snapshot"]: row["sha256"] for row in manifest["retrievals"]},
            "prepared_tables": {row["local_path"]: row["sha256"] for row in manifest["local_artifacts"]},
        }
        for name, hashes in provenance.items():
            if hashes != frozen["input_provenance"][name]:
                raise ValueError(f"Reproduced input provenance differs: {name}")
        assert all(row["reused_local_snapshot"] for row in manifest["retrievals"])
        reproduced_receipt = json.loads((fresh / "artifacts" / c.TEST_RECEIPT_PATH.name).read_text(encoding="utf-8"))
        assert reproduced_receipt["status"] == "complete"
        assert reproduced_receipt["frozen_sha256"] == frozen["seal"]
        assert (fresh / "artifacts" / c.FROZEN_CONFIG_PATH.name).read_bytes() == c.FROZEN_CONFIG_PATH.read_bytes()
    if original != {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}:
        raise ValueError("Verification changed the preserved project outputs.")
    report = {
        "status": "passed", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_seal": frozen["seal"], "original_assessment_completed_at_utc": receipt["completed_at_utc"],
        "verification_assessment_completed_at_utc": reproduced_receipt["completed_at_utc"],
        "python": platform.python_version(), "dependency_versions": dependencies,
        "verifier_code_sha256": data._code_hash(Path(__file__)),
        "research_code_hashes": frozen["code_hashes"], "input_provenance": provenance,
        "selection_retuned": False, "grid_scores": "preserved frozen validation scores; grid search not repeated",
        "replayed_return_fits": {"validation": len(c.VALIDATION_ORIGINS), "locked_test": len(c.LOCKED_TEST_ORIGINS)},
        "cold_start": "no prepared tables, fitted-model caches, or completed receipt copied into verification workspace",
        "pipeline_command": "python run_pipeline.py", "absolute_numeric_tolerance": c.ACCOUNTING_TOLERANCE,
        "tables": tables, "figures": figure_checks, "original_outputs_unchanged": True,
    }
    evaluation._write_json(c.ARTIFACTS_DIR / "reproduction_verification.json", report)
    print("Clean frozen-choice reproduction passed; original assessment and outputs preserved.", flush=True)
    return report
