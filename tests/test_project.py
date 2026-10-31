"""Independent numerical, timing, availability, and reproducibility checks."""

from datetime import date
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests

from regime_allocation import config, data, features, models, portfolio, evaluation, reporting, reproduction
from sklearn.ensemble import HistGradientBoostingRegressor
import run_pipeline


class AdjustedReturnTests(unittest.TestCase):
    def test_split_and_distribution_adjusted_returns(self):
        # A 2:1 split, then a $1 distribution, then a 10% price gain.
        raw = pd.DataFrame({
            "Open": [100, 50, 49, 53.9], "High": [102, 51, 50, 55],
            "Low": [98, 49, 48, 53], "Close": [100, 50, 49, 53.9],
            "Adj Close": [49, 49, 49, 53.9],
        }, index=pd.date_range("2024-01-02", periods=4, tz="UTC"))
        adjusted = data.adjusted_prices(raw)
        np.testing.assert_allclose(adjusted["Close"].pct_change().iloc[1:], [0, 0, 0.1])
        self.assertAlmostEqual(adjusted.iloc[0]["High"], 49.98)
        self.assertAlmostEqual(adjusted.iloc[0]["Low"], 48.02)
        self.assertEqual(raw.iloc[0]["Close"], 100)
        with self.assertRaises(data.DataUnavailableError):
            data.adjusted_prices(raw.drop(columns="Adj Close"))
        with self.assertRaises(data.DataUnavailableError):
            data.adjusted_prices(raw.assign(Close=0))

    def test_excess_compounding_sample_volatility_and_lag(self):
        returns = pd.DataFrame(0.01, index=pd.RangeIndex(1, 55, name="research_week"), columns=config.ASSETS)
        returns["VT"] += [0.01, 0.02, -0.01, 0.03] + [0.005] * 50
        predictors = features.return_predictors(returns)
        row = predictors.loc[(4, "VT")]
        self.assertAlmostEqual(row["excess_lag_1w"], -0.01)
        self.assertAlmostEqual(row["excess_compounded_4w"], 0.05049494)
        self.assertAlmostEqual(row["excess_volatility_4w"], np.sqrt(0.000875 / 3))
        self.assertTrue(np.isnan(predictors.loc[(51, "VT"), "excess_compounded_52w"]))
        self.assertAlmostEqual(
            predictors.loc[(52, "VT"), "excess_compounded_52w"], 1.05049494 * 1.005**48 - 1
        )
        changed = returns.copy()
        changed.loc[21:, "VT"] = 0.99
        pd.testing.assert_frame_equal(
            predictors.loc[1:20], features.return_predictors(changed).loc[1:20]
        )


class SessionTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sessions = data.session_schedule()
        cls.calendar = data.align_calendar(cls.sessions)
        cls.prices = pd.DataFrame(
            100.0, index=pd.DatetimeIndex(cls.sessions["market_close"].array, name="market_close"),
            columns=config.ASSETS,
        )

    def week(self, friday):
        matches = self.calendar["decision_date"].eq(pd.Timestamp(friday))
        return self.calendar.index[matches][0]

    def test_friday_holiday_retains_cutoff_and_uses_prior_close(self):
        row = self.calendar.loc[self.week("2024-03-29")]
        self.assertEqual(row["decision_cutoff"], pd.Timestamp("2024-03-29T20:00:00Z"))
        self.assertEqual(row["predictor_close"], pd.Timestamp("2024-03-28T20:00:00Z"))
        self.assertEqual(row["execution_close"], pd.Timestamp("2024-04-01T20:00:00Z"))

    def test_monday_holiday_moves_execution_and_four_week_origin(self):
        row = self.calendar.loc[self.week("2024-08-30")]
        self.assertEqual(row["execution_close"], pd.Timestamp("2024-09-03T20:00:00Z"))
        self.assertEqual(row["realization_close"], pd.Timestamp("2024-10-01T20:00:00Z"))

    def test_early_closes_dst_and_extraordinary_closures(self):
        row = self.calendar.loc[self.week("2024-11-29")]
        self.assertEqual(row["decision_cutoff"], pd.Timestamp("2024-11-29T18:00:00Z"))
        row = self.calendar.loc[self.week("2024-03-08")]
        self.assertEqual(row["decision_cutoff"], pd.Timestamp("2024-03-08T21:00:00Z"))
        self.assertEqual(row["execution_close"], pd.Timestamp("2024-03-11T20:00:00Z"))
        self.assertNotIn(pd.Timestamp("2025-01-09"), self.sessions.index)

    def test_weekly_returns_use_resolved_closes_and_initial_boundary(self):
        prices = self.prices.copy()
        prices.loc[pd.Timestamp("2014-07-25T20:00:00Z"), "VT"] = 80
        prices.loc[pd.Timestamp("2014-08-01T20:00:00Z"), "VT"] = 100
        prices.loc[pd.Timestamp("2014-07-25T20:00:00Z"), "BIL"] = 99
        returns = features.weekly_returns(prices, self.calendar)
        self.assertAlmostEqual(returns.at[1, "VT"], 0.25)
        self.assertAlmostEqual(returns.at[1, "BIL"], 1 / 99)

    def test_missing_session_stops_instead_of_forward_filling(self):
        absent = pd.Timestamp("2024-05-15T20:00:00Z")
        with self.assertRaisesRegex(data.DataUnavailableError, "Required Yahoo trading sessions"):
            data.validate_price_coverage(self.prices.drop(index=absent), self.calendar, self.sessions)

    def test_targets_start_at_execution_and_shift_holiday_endpoint(self):
        prices = self.prices.copy()
        prices.loc[pd.Timestamp("2014-08-01T20:00:00Z"), "VT"] = 500
        prices.loc[pd.Timestamp("2014-09-02T20:00:00Z"), "VT"] = 110
        prices.loc[pd.Timestamp("2014-09-02T20:00:00Z"), "BIL"] = 102
        targets = features.execution_targets(prices, self.calendar)
        self.assertEqual(targets.loc[(1, "VT"), "execution_close"], pd.Timestamp("2014-08-04T20:00:00Z"))
        self.assertEqual(targets.loc[(1, "VT"), "realization_close"], pd.Timestamp("2014-09-02T20:00:00Z"))
        self.assertAlmostEqual(targets.loc[(1, "VT"), "target_4w_excess_return"], 0.08)
        np.testing.assert_array_equal(targets.xs("BIL", level="asset")["target_4w_excess_return"], 0)
        self.assertEqual(targets.loc[(624, "VT"), "realization_close"], pd.Timestamp("2026-08-10T20:00:00Z"))

    def test_selection_excludes_boundary_crossing_labels(self):
        targets = features.execution_targets(self.prices, self.calendar)
        self.assertEqual(targets.loc[(516, "VT"), "realization_close"], pd.Timestamp("2024-07-15T20:00:00Z"))
        self.assertEqual(targets.loc[(517, "VT"), "realization_close"], pd.Timestamp("2024-07-22T20:00:00Z"))
        self.assertTrue(targets.loc[(516, "VT"), "selection_eligible"])
        self.assertFalse(targets.loc[(517, "VT"), "selection_eligible"])
        self.assertFalse(targets.loc[(521, "VT"), "selection_eligible"])
        self.assertEqual(targets.xs("VT", level="asset")["selection_eligible"].sum(), 204)

    def test_rolling_training_requires_realization_including_earlier_test_labels(self):
        targets = features.execution_targets(self.prices, self.calendar)
        mask = features.training_mask(targets, 313, self.calendar)
        weeks = targets.loc[mask].index.get_level_values("research_week").unique().tolist()
        self.assertEqual(weeks, list(range(53, 309)))
        self.assertFalse(features.training_mask(targets, 521, self.calendar).loc[(517, "VT")])
        self.assertFalse(features.training_mask(targets, 525, self.calendar).loc[(521, "VT")])
        self.assertTrue(features.training_mask(targets, 526, self.calendar).loc[(521, "VT")])


class MacroAvailabilityTests(unittest.TestCase):
    def vintage_fixture(self):
        rows = []
        for series in config.MACRO_SERIES:
            for observation, release, value in (
                ("2024-06-01", "2024-07-01", "100"),
                ("2024-06-01", "2024-07-05", "999"),
                ("2024-07-01", "2024-08-01", "200"),
            ):
                rows.append({"series": series, "date": observation, "realtime_start": release,
                             "realtime_end": "2024-07-04" if release == "2024-07-01" else "2024-12-31", "value": value})
        return data.normalize_vintages(pd.DataFrame(rows))

    def test_date_only_release_is_not_available_at_midday_or_before_release(self):
        cutoffs = pd.Series(pd.to_datetime([
            "2024-07-05T20:00:00Z", "2024-07-06T03:59:59Z", "2024-07-06T04:00:00Z"
        ], utc=True), index=[1, 2, 3])
        known = data.available_macro(self.vintage_fixture(), cutoffs)
        np.testing.assert_array_equal(known.xs("PPIACO", level="series")["value"], [100, 100, 999])
        self.assertTrue((known["available_at"] <= known["decision_cutoff"]).all())
        self.assertTrue(known["observation_date"].eq(pd.Timestamp("2024-06-01")).all())
        with self.assertRaisesRegex(data.DataUnavailableError, "VIXCLS"):
            data.available_macro(self.vintage_fixture(), pd.Series([pd.Timestamp("2024-06-30T20:00:00Z")], index=[1]))

    def test_next_midnight_uses_local_dates_across_dst(self):
        rows = pd.DataFrame([
            {"series": "VIXCLS", "date": observation, "realtime_start": release,
             "realtime_end": "2024-12-31", "value": "10"}
            for observation, release in (("2024-03-08", "2024-03-09"), ("2024-03-09", "2024-03-10"))
        ])
        normalized = data.normalize_vintages(rows)
        self.assertEqual(normalized.iloc[0]["available_at"], pd.Timestamp("2024-03-10T05:00:00Z"))
        self.assertEqual(normalized.iloc[1]["available_at"], pd.Timestamp("2024-03-11T04:00:00Z"))

    def test_missing_release_is_not_filled_with_a_later_revision(self):
        rows = pd.DataFrame([
            {"series": series, "date": observation, "realtime_start": start,
             "realtime_end": end, "value": value}
            for series in config.MACRO_SERIES
            for observation, start, end, value in (
                ("2024-06-01", "2024-07-01", "2024-07-31", "100"),
                ("2024-07-01", "2024-07-05", "2024-07-11", "."),
                ("2024-07-01", "2024-07-12", "9999-12-31", "200"),
            )
        ])
        history = data.normalize_vintages(rows)
        cutoffs = pd.Series(pd.to_datetime([
            "2024-07-05T20:00:00Z", "2024-07-12T20:00:00Z", "2024-07-19T20:00:00Z"
        ], utc=True), index=[1, 2, 3])
        known = data.available_macro(history, cutoffs)
        np.testing.assert_array_equal(known.xs("PPIACO", level="series")["value"], [100, 100, 200])
        with self.assertRaises(data.DataUnavailableError):
            data.available_macro(history.loc[history["date"].eq("2024-07-01")], cutoffs.iloc[:2])

    def test_withdrawn_observation_does_not_live_past_its_vintage_interval(self):
        rows = pd.DataFrame([
            {"series": series, "date": "2024-06-01", "realtime_start": "2024-07-01",
             "realtime_end": "2024-07-04", "value": "100"}
            for series in config.MACRO_SERIES
        ])
        history = data.normalize_vintages(rows)
        before = data.available_macro(history, pd.Series([pd.Timestamp("2024-07-05T20:00:00Z")], index=[1]))
        self.assertTrue(before["value"].eq(100).all())
        with self.assertRaises(data.DataUnavailableError):
            data.available_macro(history, pd.Series([pd.Timestamp("2024-07-06T04:00:00Z")], index=[1]))

    def test_macro_changes_use_historical_levels_and_fractional_ppi_change(self):
        sequences = {
            "VIXCLS": [10, 11, 12, 13, 16, 18],
            "T10Y2Y": [1, 2, 3, 4, 6, 9],
            "PPIACO": [100, 101, 102, 103, 110, 121],
        }
        rows = [{"research_week": week, "series": series, "value": value}
                for series, values in sequences.items() for week, value in zip(range(-3, 3), values)]
        availability = pd.DataFrame(rows).set_index(["research_week", "series"]).sort_index()
        predictors = features.macro_predictors(availability, pd.Index([1, 2], name="research_week"))
        self.assertEqual(predictors.at[1, "VIXCLS_change_4w"], 6)
        self.assertEqual(predictors.at[1, "T10Y2Y_change_4w"], 5)
        self.assertAlmostEqual(predictors.at[1, "PPIACO_change_4w"], 0.1)
        self.assertAlmostEqual(predictors.at[2, "PPIACO_change_4w"], 121 / 101 - 1)
        with self.assertRaises(data.DataUnavailableError):
            features.macro_predictors(availability.loc[0:], pd.Index([1, 2], name="research_week"))
        with self.assertRaises(data.DataUnavailableError):
            features.macro_predictors(availability.drop(index=-2, level="research_week"), pd.Index([1, 2], name="research_week"))

    def test_query_boundaries_do_not_create_new_releases(self):
        rows = pd.DataFrame([
            {"series": "PPIACO", "date": "2024-01-01", "realtime_start": start,
             "realtime_end": end, "value": value}
            for start, end, value in (
                ("2024-01-15", "2024-01-31", "100"),
                ("2024-02-01", "2024-02-29", "100"),
                ("2024-03-01", "2024-03-31", "110"),
                ("2024-04-01", "2024-04-30", "100"),
            )
        ])
        normalized = data.normalize_vintages(rows)
        self.assertEqual(len(normalized), 3)
        self.assertEqual(normalized.iloc[0]["realtime_start"], "2024-01-15")
        self.assertEqual(normalized.iloc[0]["realtime_end"], "2024-02-29")
        self.assertEqual(normalized.iloc[0]["available_at"], pd.Timestamp("2024-01-16T05:00:00Z"))
        self.assertEqual(normalized.iloc[-1]["realtime_start"], "2024-04-01")


class SnapshotTests(unittest.TestCase):
    def test_source_hash_is_independent_of_git_newline_conversion(self):
        with tempfile.TemporaryDirectory(dir=config.PROJECT_ROOT / ".venv") as directory:
            root = Path(directory).resolve()
            self.assertTrue(root.is_relative_to((config.PROJECT_ROOT / ".venv").resolve()))
            path = root / "source.py"
            expected = hashlib.sha256(b'print("hello")\n').hexdigest()
            for contents in (b'print("hello")\n', b'print("hello")\r\n'):
                path.write_bytes(contents)
                self.assertEqual(data._code_hash(path), expected)

    def test_authenticated_success_keeps_key_out_of_snapshots_and_metadata(self):
        with tempfile.TemporaryDirectory(dir=config.PROJECT_ROOT / ".venv") as directory:
            root = Path(directory).resolve()
            self.assertTrue(root.is_relative_to((config.PROJECT_ROOT / ".venv").resolve()))
            response = Mock(status_code=200, content=b'{"count":0}')
            manifest = {"retrievals": []}
            with (
                patch.object(config, "PROJECT_ROOT", root),
                patch.object(config, "RAW_DATA_DIR", root / "raw"),
                patch.dict(os.environ, {config.FRED_API_KEY_ENV: "secret-for-success-test"}),
                patch.object(data.requests, "get", return_value=response) as fetch,
            ):
                data._retrieve("fixture", "https://example.test/source", {"series_id": "PPIACO"}, manifest, fred=True)
                self.assertEqual(fetch.call_args.kwargs["params"]["api_key"], "secret-for-success-test")
                self.assertNotIn("secret-for-success-test", json.dumps(manifest))
                for path in (root / "raw").iterdir():
                    self.assertNotIn(b"secret-for-success-test", path.read_bytes())

    def test_exact_bytes_cache_replay_and_corruption_detection(self):
        with tempfile.TemporaryDirectory(dir=config.PROJECT_ROOT / ".venv") as directory:
            root = Path(directory).resolve()
            self.assertTrue(root.is_relative_to((config.PROJECT_ROOT / ".venv").resolve()))
            manifest = {"retrievals": []}
            response = Mock(status_code=200, content=b'{"ok":true}\n')
            with (
                patch.object(config, "PROJECT_ROOT", root),
                patch.object(config, "RAW_DATA_DIR", root / "raw"),
                patch.object(data.requests, "get", return_value=response) as fetch,
            ):
                first = data._retrieve("fixture", "https://example.test/source", {"interval": "1d"}, manifest)
                second = data._retrieve("fixture", "https://example.test/source", {"interval": "1d"}, manifest)
                self.assertEqual(first, second)
                fetch.assert_called_once()
                record = manifest["retrievals"][0]
                snapshot = root / record["local_snapshot"]
                self.assertEqual(snapshot.read_bytes(), b'{"ok":true}\n')
                self.assertEqual(record["sha256"], hashlib.sha256(snapshot.read_bytes()).hexdigest())
                self.assertEqual(record["retrieved_at_utc"], manifest["retrievals"][1]["retrieved_at_utc"])
                snapshot.write_bytes(b'{"ok":false}\n')
                with self.assertRaisesRegex(data.DataUnavailableError, "hash mismatch"):
                    data._retrieve("fixture", "https://example.test/source", {"interval": "1d"}, manifest)

    def test_authenticated_failure_does_not_expose_or_write_key(self):
        with tempfile.TemporaryDirectory(dir=config.PROJECT_ROOT / ".venv") as directory:
            root = Path(directory).resolve()
            self.assertTrue(root.is_relative_to((config.PROJECT_ROOT / ".venv").resolve()))
            with (
                patch.object(config, "PROJECT_ROOT", root),
                patch.object(config, "RAW_DATA_DIR", root / "raw"),
                patch.dict(os.environ, {config.FRED_API_KEY_ENV: "secret-for-error-test"}),
                patch.object(data.requests, "get", side_effect=requests.Timeout("api_key=secret-for-error-test")),
            ):
                with self.assertRaises(data.DataUnavailableError) as failure:
                    data._retrieve("fixture", "https://example.test/source", {}, {"retrievals": []}, fred=True)
                self.assertNotIn("secret-for-error-test", str(failure.exception))
                self.assertFalse((root / "raw").exists())


class CalendarTests(unittest.TestCase):
    def test_calendar_boundaries_and_scoring_endpoints(self):
        calendar = config.research_calendar()
        expected = {
            1: ("2014-07-28", "2014-08-01", "warm_up"),
            53: ("2015-07-27", "2015-07-31", "initial_training"),
            313: ("2020-07-20", "2020-07-24", "validation"),
            521: ("2024-07-15", "2024-07-19", "locked_test"),
            624: ("2026-07-06", "2026-07-10", "locked_test"),
        }
        self.assertEqual(len(calendar), 624)
        self.assertTrue(calendar.index.is_unique)
        for week, (monday, friday, split) in expected.items():
            with self.subTest(week=week):
                self.assertEqual(str(calendar.at[week, "week_start"].date()), monday)
                self.assertEqual(str(calendar.at[week, "decision_date"].date()), friday)
                self.assertEqual(calendar.at[week, "split"], split)
        self.assertEqual(
            str(calendar.at[624, "scheduled_target_end_date"].date()), "2026-08-10"
        )
        self.assertEqual(config.first_locked_test_decision(), date(2024, 7, 19))
        self.assertTrue(calendar["week_start"].diff().dropna().dt.days.eq(7).all())

    def test_splits_partition_all_weeks_without_gaps(self):
        counts = config.research_calendar()["split"].value_counts().to_dict()
        self.assertEqual(
            counts,
            {"warm_up": 52, "initial_training": 260, "validation": 208, "locked_test": 104},
        )
        self.assertEqual(list(config.VALIDATION_ORIGINS), list(range(313, 521)))
        self.assertEqual(list(config.LOCKED_TEST_ORIGINS), list(range(521, 625)))


class GridTests(unittest.TestCase):
    def test_complete_return_grid_and_stable_candidate_ids(self):
        candidates = list(config.grid_candidates(config.RETURN_MODEL_GRID))
        self.assertEqual(len(candidates), 13_750)
        self.assertEqual([candidate for candidate, _ in candidates], list(range(13_750)))
        self.assertEqual(len({tuple(p.values()) for _, p in candidates}), 13_750)
        expected_values = {
            "learning_rate": {i / 100 for i in range(1, 11)},
            "max_iter": {100, 200, 300, 400, 500},
            "max_leaf_nodes": set(range(5, 16)),
            "min_samples_leaf": {5, 10, 15, 20, 25},
            "l2_regularization": {0.1, 0.01, 0.001, 0.0001, 0.00001},
        }
        for parameter, values in expected_values.items():
            self.assertEqual({p[parameter] for _, p in candidates}, values)
        self.assertEqual(config.VALIDATION_RETURN_MODEL_FITS, 2_860_000)
        self.assertEqual(config.LOCKED_TEST_RETURN_MODEL_FITS, 104)
        self.assertEqual(config.HMM_FITS_PER_ORIGIN, 3)

    def test_allocation_grid_and_feasible_initial_holdings(self):
        candidates = list(config.grid_candidates(config.ALLOCATION_GRID))
        self.assertEqual(
            {(p["risk_aversion"], p["turnover_penalty"]) for _, p in candidates},
            {(risk, penalty / 1000) for risk in range(1, 6) for penalty in range(1, 6)},
        )
        self.assertAlmostEqual(sum(config.INITIAL_WEIGHTS), 1)
        self.assertEqual(len(config.INITIAL_WEIGHTS), len(config.ASSETS))
        self.assertTrue(
            all(config.MIN_ASSET_WEIGHT <= w <= config.MAX_ASSET_WEIGHT
                for w in config.INITIAL_WEIGHTS)
        )


class SeedTests(unittest.TestCase):
    @staticmethod
    def state(component, origin, candidate=0, repeat=0):
        return tuple(
            config.seed_sequence(
                component, origin, candidate_id=candidate, repeated_fit_id=repeat
            ).generate_state(8)
        )

    def test_components_origins_candidates_and_repeats_have_distinct_streams(self):
        identities = [
            (component, origin, candidate, repeat)
            for component in config.SEED_COMPONENT_IDS
            for origin in (53, 313, 521, 624)
            for candidate in (0, 13_749)
            for repeat in (0, 1, 2)
        ]
        forward = {identity: self.state(*identity) for identity in identities}
        reverse = {identity: self.state(*identity) for identity in reversed(identities)}
        self.assertEqual(forward, reverse)
        self.assertEqual(len(set(forward.values())), len(identities))

    def test_repeated_calls_do_not_consume_a_shared_spawn_counter(self):
        expected = self.state("return_model", 313, 42, 2)
        sequence = config.seed_sequence("return_model", 313, candidate_id=42, repeated_fit_id=2)
        sequence.spawn(10)
        config.seed_sequence("hmm_equity_stress", 624).generate_state(100)
        self.assertEqual(self.state("return_model", 313, 42, 2), expected)

    def test_mapping_round_trip_reconstructs_child_and_random_state(self):
        record = json.loads(json.dumps(config.seed_record(
            "return_model", 520, candidate_id=13_749, repeated_fit_id=7
        )))
        replay = np.random.SeedSequence(
            record["master_seed"], spawn_key=record["spawn_key"], pool_size=record["pool_size"]
        )
        self.assertEqual(record["master_seed"], 42)
        self.assertEqual(record["spawn_key"], [3, 520, 13_749, 7])
        self.assertEqual(record["random_state"], int(replay.generate_state(1)[0]))
        self.assertEqual(record["forecast_origin_week"], 520)
        self.assertEqual(record["candidate_id"], 13_749)
        self.assertEqual(record["repeated_fit_id"], 7)
        self.assertEqual(tuple(replay.generate_state(8)), self.state("return_model", 520, 13_749, 7))

    def test_invalid_seed_identities_fail_clearly(self):
        for parameters in (
            {"forecast_origin_week": 0},
            {"forecast_origin_week": 625},
            {"forecast_origin_week": 313, "candidate_id": -1},
            {"forecast_origin_week": 313, "repeated_fit_id": -1},
        ):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                config.seed_sequence("return_model", **parameters)
        for origin in (True, np.bool_(False), 313.0, "313"):
            with self.subTest(origin=origin), self.assertRaises(TypeError):
                config.seed_sequence("return_model", origin)
        with self.assertRaises(ValueError):
            config.seed_sequence("unknown", 313)


class CredentialTests(unittest.TestCase):
    def test_credentials_are_read_from_process_environment_on_demand(self):
        with patch.dict(os.environ, {config.FRED_API_KEY_ENV: "first-test-key"}, clear=True):
            self.assertEqual(config.fred_api_key(), "first-test-key")
            os.environ[config.FRED_API_KEY_ENV] = "second-test-key"
            self.assertEqual(config.fred_api_key(), "second-test-key")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "FRED_API_KEY"):
                config.fred_api_key()



class FilteringTests(unittest.TestCase):
    def test_forward_recursion_matches_hand_calculation_and_has_no_smoothing(self):
        emissions = np.log([[0.4, 0.1], [0.2, 0.5], [0.01, 0.9]])
        transition = np.array([[0.8, 0.2], [0.3, 0.7]])
        filtered = models.forward_filter(emissions, [0.6, 0.4], transition)
        np.testing.assert_allclose(filtered[0], [6 / 7, 1 / 7])
        np.testing.assert_allclose(filtered[1], [102 / 197, 95 / 197])
        prefix = models.forward_filter(emissions[:2], [0.6, 0.4], transition)
        np.testing.assert_array_equal(filtered[:2], prefix)

    def test_gaussian_state_columns_follow_emission_means(self):
        high_first = SimpleNamespace(
            means_=np.array([[1.0], [-1.0]]), _covars_=np.ones((2, 1)),
            startprob_=np.array([0.5, 0.5]), transmat_=np.array([[0.8, 0.2], [0.3, 0.7]]),
        )
        low_first = SimpleNamespace(
            means_=high_first.means_[::-1], _covars_=high_first._covars_[::-1],
            startprob_=high_first.startprob_[::-1], transmat_=high_first.transmat_[::-1, ::-1],
        )
        actual = models.gaussian_filtered_probabilities([1, 0, -1], high_first)
        np.testing.assert_allclose(actual[0], [1 / (1 + np.exp(2)), 1 / (1 + np.exp(-2))])
        np.testing.assert_allclose(actual, models.gaussian_filtered_probabilities([1, 0, -1], low_first))

    def test_covariance_floor_is_applied_after_estimation(self):
        values = np.r_[np.full(20, -1.0), np.full(20, 1.0)].reshape(-1, 1)
        model = models._FlooredGaussianHMM(**config.HMM_SETTINGS, random_state=42).fit(values)
        self.assertGreaterEqual(model._covars_.min(), 0.001)

    def test_failed_fits_are_exposed_after_seeded_attempts(self):
        with patch.object(config, "HMM_MAX_FIT_ATTEMPTS", 2), patch.object(
            models._FlooredGaussianHMM, "fit", side_effect=ValueError("invalid transition")
        ) as fit:
            with self.assertRaisesRegex(models.ModelFitError, "all 2 seeded fits failed"):
                models.fit_regime(np.arange(20.0), "hmm_equity_stress", 53)
            self.assertEqual(fit.call_count, 2)

    def test_later_macro_values_do_not_rewrite_historical_regime_features(self):
        weeks = pd.RangeIndex(1, 13, name="research_week")
        macro = pd.DataFrame({
            "VIXCLS_level": np.sin(np.arange(12)) + np.arange(12) / 10,
            "T10Y2Y_level": np.cos(np.arange(12)) + np.arange(12) / 20,
            "PPIACO_change_4w": np.sin(np.arange(12) * 1.7) / 10,
        }, index=weeks)
        calendar = data.align_calendar(data.session_schedule()).loc[weeks]
        with tempfile.TemporaryDirectory(dir=config.PROJECT_ROOT / ".venv") as directory:
            root = Path(directory).resolve()
            self.assertTrue(root.is_relative_to((config.PROJECT_ROOT / ".venv").resolve()))
            with patch.object(config, "REGIME_CACHE_PATH", root / "regimes.joblib"):
                original = models.causal_regimes({"macro_predictors": macro, "calendar": calendar})
                changed = macro.copy()
                changed.loc[12] += 0.25
                later = models.causal_regimes({"macro_predictors": changed, "calendar": calendar})
            pd.testing.assert_frame_equal(original["features"].loc[:11], later["features"].loc[:11])
            self.assertTrue(original["probabilities"]["last_training_week"].eq(
                original["probabilities"]["research_week"]
            ).all())


class RollingModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calendar = data.align_calendar(data.session_schedule())
        index = pd.MultiIndex.from_product(
            [range(53, 316), config.FORECAST_ASSETS], names=["research_week", "asset"]
        )
        signal = np.sin(index.get_level_values("research_week").to_numpy() / 7)
        cls.features = pd.DataFrame({"signal": signal}, index=index)
        cls.targets = pd.DataFrame({
            "target_4w_excess_return": signal / 100,
            "realization_close": [cls.calendar.at[week, "realization_close"]
                                  for week in index.get_level_values("research_week")],
        }, index=index)
        cls.regimes = pd.DataFrame({"regime": np.linspace(0.1, 0.9, 315)},
                                  index=pd.RangeIndex(1, 316, name="research_week"))

    def inputs(self, targets=None):
        return {"features": self.features, "targets": self.targets if targets is None else targets,
                "calendar": self.calendar}

    def test_origin_training_contains_only_mature_labels_in_calendar_window(self):
        origin = models.origin_dataset(self.inputs(), self.regimes, 313)
        self.assertEqual(origin["training_observations"], 256 * 5)
        self.assertEqual(origin["latest_training_realization"], pd.Timestamp("2020-07-20T20:00:00Z"))
        changed = self.targets.copy()
        changed.loc[changed.index.get_level_values("research_week") >= 309, "target_4w_excess_return"] = 10000
        first_id, parameters = next(config.grid_candidates(config.RETURN_MODEL_GRID))
        actual, _ = models.fit_forecast(self.inputs(), self.regimes, 313, first_id, parameters)
        altered, _ = models.fit_forecast(self.inputs(changed), self.regimes, 313, first_id, parameters)
        np.testing.assert_array_equal(actual, altered)

    def test_iteration_checkpoints_equal_independent_seeded_fits(self):
        dataset = models.origin_dataset(self.inputs(), self.regimes, 313)
        warm = HistGradientBoostingRegressor(**config.RETURN_MODEL_SETTINGS, warm_start=True)
        for candidate, parameters in models.candidate_groups()[0][:2]:
            seed = config.seed_record("return_model", 313, candidate_id=candidate)["random_state"]
            warm.set_params(**parameters, random_state=seed)
            warm.fit(dataset["x"], dataset["y"])
            cold = HistGradientBoostingRegressor(
                **config.RETURN_MODEL_SETTINGS, **parameters, random_state=seed
            ).fit(dataset["x"], dataset["y"])
            np.testing.assert_array_equal(warm.predict(dataset["query"]), cold.predict(dataset["query"]))

    def test_checkpoint_groups_preserve_every_candidate_and_fit_count(self):
        groups = models.candidate_groups()
        identifiers = [candidate for group in groups for candidate, _ in group]
        self.assertEqual(len(groups), 2750)
        self.assertEqual(sorted(identifiers), list(range(13750)))
        self.assertTrue(all([p["max_iter"] for _, p in group] == [100, 200, 300, 400, 500]
                            for group in groups))
        self.assertEqual(len(identifiers) * 208, 2_860_000)


class PortfolioTests(unittest.TestCase):
    def test_projection_drift_and_turnover_match_hand_calculations(self):
        np.testing.assert_allclose(portfolio.project_weights([1, 0, 0, 0, 0, 0]), [0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
        drifted = portfolio.drift_weights([0.5, 0.1, 0.1, 0.1, 0.1, 0.1], [2, 1, 1, 1, 1, 1])
        np.testing.assert_allclose(drifted, [2 / 3, 1 / 15, 1 / 15, 1 / 15, 1 / 15, 1 / 15])
        self.assertAlmostEqual(portfolio.one_way_turnover(drifted, np.full(6, 1 / 6)), 0.5)
        with self.assertRaises(portfolio.AllocationError):
            portfolio.check_weights(drifted)

    def test_covariance_uses_available_window_and_four_week_horizon(self):
        time = np.arange(260)
        values = np.column_stack([np.cos(2 * np.pi * frequency * time / 260) for frequency in range(1, 7)])
        returns = pd.DataFrame(values, index=range(1, 261), columns=config.ASSETS)
        returns.loc[261] = 10000
        np.testing.assert_allclose(portfolio.covariance_at(returns, 260), np.eye(6) * 2, atol=1e-12)
        with self.assertRaises(portfolio.AllocationError):
            portfolio.covariance_at(returns.loc[2:], 260)

    def test_inverse_volatility_uses_sample_volatility_and_same_constraints(self):
        pattern = np.array([1, -1] * 26)
        values = pattern[:, None] * np.arange(1, 7)[None, :]
        returns = pd.DataFrame(values, index=range(1, 53), columns=config.ASSETS)
        returns.loc[53] = 10000
        inverse = 1 / np.arange(1, 7, dtype=float)
        np.testing.assert_allclose(portfolio.benchmark_target(returns, 52, "inverse_volatility"), inverse / inverse.sum())
        np.testing.assert_allclose(portfolio.benchmark_target(returns, 52, "equal_weight"), np.full(6, 1 / 6))
        with self.assertRaises(portfolio.AllocationError):
            portfolio.benchmark_target(returns.assign(BIL=0), 52, "inverse_volatility")

    def test_solver_symmetric_and_linear_optima_are_feasible(self):
        reference = np.full(6, 1 / 6)
        solver = portfolio.ConstrainedAllocator(3, 0.003)
        weights, diagnostic = solver.solve(np.zeros(6), np.eye(6), reference)
        np.testing.assert_allclose(weights, reference, atol=1e-8)
        self.assertEqual(diagnostic["solver_status"], "optimal")
        weights, _ = solver.solve([1, 0, 0, 0, 0, 0], np.zeros((6, 6)), reference)
        self.assertAlmostEqual(weights[0], 0.5, places=7)
        portfolio.check_weights(weights)
        with self.assertRaises(portfolio.AllocationError):
            solver.solve(np.zeros(6), -np.eye(6), reference)

    def test_solver_failure_is_exposed_without_substituting_weights(self):
        solver = portfolio.ConstrainedAllocator(1, 0.001)
        with patch.object(solver.problem, "solve", side_effect=portfolio.cp.error.SolverError("failed")):
            with self.assertRaisesRegex(portfolio.AllocationError, "failed"):
                solver.solve(np.zeros(6), np.eye(6), np.full(6, 1 / 6))

    def holdings_fixture(self):
        calendar = data.align_calendar(data.session_schedule()).loc[[313, 314]]
        dates = [calendar.at[313, "predictor_close"], calendar.at[313, "execution_close"],
                 calendar.at[314, "predictor_close"], calendar.at[314, "execution_close"],
                 calendar.at[314, "next_execution_close"]]
        prices = pd.DataFrame(100.0, index=dates, columns=config.ASSETS)
        prices["VT"] = [100, 200, 300, 400, 400]
        return calendar, prices

    def test_execution_holdings_and_turnover_reconcile(self):
        calendar, prices = self.holdings_fixture()
        allocations, returns, state = portfolio.simulate(
            prices, pd.DataFrame(), calendar, None, [313, 314], name="equal_weight"
        )
        first = allocations.loc[allocations["research_week"].eq(313)].set_index("asset")
        self.assertAlmostEqual(first.at["VT", "pretrade_weight"], 2 / 7)
        self.assertAlmostEqual(first.at["VT", "reference_weight"], 1 / 6)
        self.assertAlmostEqual(returns.iloc[0]["one_way_turnover"], 5 / 42)
        self.assertAlmostEqual(returns.iloc[0]["weekly_total_return"], 1 / 6)
        self.assertAlmostEqual(returns.iloc[1]["weekly_total_return"], 0)
        self.assertAlmostEqual(state["wealth_at_execution"], 49 / 36)
        np.testing.assert_allclose(allocations.groupby("research_week")["executed_weight"].sum(), 1)
        np.testing.assert_allclose(allocations["wealth_before_trade"], allocations["wealth_after_trade"])

    def test_optimizer_reference_is_friday_holdings_before_monday_drift(self):
        calendar, prices = self.holdings_fixture()
        forecasts = pd.DataFrame(0.0, index=[313, 314], columns=config.ASSETS)
        references = []
        def solve(_, mean, covariance, reference):
            references.append(reference.copy())
            return np.full(6, 1 / 6), {"solver_status": "fixture", "solver_iterations": 0, "objective": 0}
        with patch.object(portfolio.ConstrainedAllocator, "solve", solve), patch.object(portfolio, "covariance_at", return_value=np.eye(6)):
            allocations, _, _ = portfolio.simulate(
                prices, pd.DataFrame(), calendar, forecasts, [313, 314], name="strategy",
                parameters={"risk_aversion": 1, "turnover_penalty": 0.001},
            )
        self.assertAlmostEqual(references[1][0], 3 / 13)
        self.assertAlmostEqual(allocations.loc[allocations["research_week"].eq(314)].iloc[0]["pretrade_weight"], 2 / 7)

    def test_selection_does_not_read_prices_beyond_its_cutoff(self):
        calendar, prices = self.holdings_fixture()
        available = prices.loc[prices.index < calendar.at[313, "next_execution_close"]]
        _, returns, _ = portfolio.simulate(
            available, pd.DataFrame(), calendar, None, [313], name="equal_weight",
            cutoff=calendar.at[313, "next_execution_close"],
        )
        self.assertFalse(returns.iloc[0]["scored"])
        self.assertTrue(np.isnan(returns.iloc[0]["weekly_total_return"]))


class EvaluationTests(unittest.TestCase):
    def test_result_round_trip_preserves_frozen_float_values_and_hashes(self):
        with tempfile.TemporaryDirectory(dir=config.PROJECT_ROOT / ".venv") as directory:
            root = Path(directory).resolve()
            self.assertTrue(root.is_relative_to((config.PROJECT_ROOT / ".venv").resolve()))
            frame = pd.DataFrame({
                "research_week": [313, 314], "evaluation_period": ["validation"] * 2,
                "weekly_total_return": [np.nextafter(1.0, 2.0), 1.2345678901234567e-15],
                "scored": [True, False], "exclusion_reason": ["", "boundary"],
            })
            with patch.object(config, "RESULT_PATHS", {"weekly_returns": root / "returns.csv"}):
                original = evaluation._export({"weekly_returns": frame})
                replayed = evaluation._read_result("weekly_returns")
                self.assertEqual(evaluation._export({"weekly_returns": replayed}), original)
                np.testing.assert_array_equal(replayed["weekly_total_return"], frame["weekly_total_return"])

    def test_metrics_match_independently_derived_formulas(self):
        returns, cash, turnover = [0.1, -0.2, 0.05], [0.01] * 3, [0.1, 0.2, 0.3]
        measures = evaluation.performance_measures(returns, cash, turnover)
        self.assertAlmostEqual(measures["annualized_return"], 0.924 ** (52 / 3) - 1)
        mean = -0.05 / 3
        variance = sum((value - mean) ** 2 for value in returns) / 2
        self.assertAlmostEqual(measures["annualized_volatility"], np.sqrt(variance * 52))
        self.assertAlmostEqual(measures["excess_return_sharpe"], (mean - 0.01) / np.sqrt(variance) * np.sqrt(52))
        self.assertAlmostEqual(measures["maximum_drawdown"], -0.2)
        self.assertAlmostEqual(measures["mean_weekly_turnover"], 0.2)
        self.assertAlmostEqual(evaluation.performance_measures([-0.1, 0.2], [0, 0], [0, 0])["maximum_drawdown"], -0.1)

    def test_pooled_error_excludes_cash_and_boundary_crossing_outcomes(self):
        forecasts = pd.DataFrame([
            {"research_week": 313, "asset": asset, "forecast_4w_excess_return": position / 100,
             "realized_target": 0, "scored": True}
            for position, asset in enumerate(config.FORECAST_ASSETS)
        ] + [
            {"research_week": 313, "asset": "BIL", "forecast_4w_excess_return": 100,
             "realized_target": 0, "scored": True},
            {"research_week": 520, "asset": "VT", "forecast_4w_excess_return": 100,
             "realized_target": np.nan, "scored": False},
        ])
        self.assertAlmostEqual(evaluation.forecast_score(forecasts), 0.0006)

    def test_locked_stage_requires_a_frozen_experiment(self):
        with tempfile.TemporaryDirectory(dir=config.PROJECT_ROOT / ".venv") as directory:
            root = Path(directory).resolve()
            self.assertTrue(root.is_relative_to((config.PROJECT_ROOT / ".venv").resolve()))
            with patch.object(config, "FROZEN_CONFIG_PATH", root / "absent.json"), patch.object(config, "TEST_RECEIPT_PATH", root / "receipt.json"):
                with self.assertRaisesRegex(evaluation.EvaluationError, "freeze settings"):
                    evaluation.run_locked_test()
                self.assertFalse((root / "receipt.json").exists())

    def test_default_cli_dispatches_all_stages_and_select_does_not_test(self):
        calls = []
        manifest = {"coverage": {"research_weeks": 624, "required_daily_sessions_per_asset": 3028}}
        with patch.object(run_pipeline, "prepare_data", side_effect=lambda: calls.append("data") or manifest), patch.object(evaluation, "run_selection", side_effect=lambda: calls.append("select")), patch.object(evaluation, "run_locked_test", side_effect=lambda: calls.append("test")):
            self.assertEqual(run_pipeline.main([]), 0)
            self.assertEqual(calls, ["data", "select", "test"])
            calls.clear()
            self.assertEqual(run_pipeline.main(["--stage", "select"]), 0)
            self.assertEqual(calls, ["select"])

    def test_frozen_reproduction_preserves_choices_and_detects_changed_inputs(self):
        with tempfile.TemporaryDirectory(dir=config.PROJECT_ROOT / ".venv") as directory:
            root = Path(directory).resolve()
            self.assertTrue(root.is_relative_to((config.PROJECT_ROOT / ".venv").resolve()))
            input_path = root / "input.csv"
            input_path.write_bytes(b"recorded input\n")
            frozen = {"code_hashes": {"model.py": "fixture"}, "dependency_versions": {"numpy": "fixture"},
                      "input_provenance": {"prepared_tables": {"input.csv": hashlib.sha256(input_path.read_bytes()).hexdigest()}},
                      "validation_output_hashes": {}, "return_candidate_id": 42}
            frozen["seal"] = models.fingerprint(frozen)
            frozen_path = root / "frozen.json"
            frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
            with (
                patch.object(config, "PROJECT_ROOT", root),
                patch.object(config, "FROZEN_CONFIG_PATH", frozen_path),
                patch.object(config, "TEST_RECEIPT_PATH", root / "absent.json"),
                patch.object(evaluation, "_code_hashes", return_value={"model.py": "fixture"}),
                patch.object(evaluation, "_dependency_versions", return_value={"numpy": "fixture"}),
                patch.object(models, "select_return_model") as select,
            ):
                self.assertEqual(evaluation.run_selection()["return_candidate_id"], 42)
                select.assert_not_called()
                input_path.write_bytes(b"changed input\n")
                with self.assertRaisesRegex(evaluation.EvaluationError, "Frozen input differs"):
                    evaluation.verify_frozen()
                frozen["return_candidate_id"] = 43
                frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
                with self.assertRaisesRegex(evaluation.EvaluationError, "integrity"):
                    evaluation.verify_frozen()


class ReportingTests(unittest.TestCase):
    def test_cumulative_paths_reset_periods_and_exclude_unscored_outcomes(self):
        returns = pd.DataFrame({
            "evaluation_period": ["validation"] * 3 + ["locked_test"],
            "portfolio": ["strategy"] * 4,
            "execution_close": ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"],
            "realization_close": ["2024-01-08", "2024-01-15", "2024-01-22", "2024-01-29"],
            "weekly_total_return": [0.1, -0.2, np.nan, 0.25],
            "scored": [True, True, False, True],
        })
        paths = reporting.cumulative_performance(returns)
        validation = paths.loc[paths["evaluation_period"].eq("validation")]
        locked = paths.loc[paths["evaluation_period"].eq("locked_test")]
        np.testing.assert_allclose(validation["wealth"], [1, 1.1, 0.88])
        np.testing.assert_allclose(locked["wealth"], [1, 1.25])
        self.assertEqual(validation["market_close"].max(), pd.Timestamp("2024-01-15", tz="UTC"))
        self.assertEqual(locked["market_close"].min(), pd.Timestamp("2024-01-22", tz="UTC"))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            reporting.cumulative_performance(returns.assign(
                execution_close=["2024-01-01", "2024-01-09", "2024-01-15", "2024-01-22"]
            ))

    def test_allocation_steps_begin_at_execution_and_extend_last_holding(self):
        targets = [[0.5, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.5, 0.1, 0.1, 0.1, 0.1]]
        allocations = pd.DataFrame([
            {"portfolio": "strategy", "execution_close": date, "asset": asset, "executed_weight": weight}
            for date, values in zip(["2024-01-08", "2024-01-15"], targets)
            for asset, weight in zip(config.ASSETS, values)
        ])
        dates, values = reporting.allocation_steps(allocations, pd.Timestamp("2024-01-22", tz="UTC"))
        self.assertEqual(dates.tolist(), list(pd.to_datetime(["2024-01-08", "2024-01-15", "2024-01-22"], utc=True)))
        np.testing.assert_allclose(values, [targets[0], targets[1], targets[1]])


class ReproductionTests(unittest.TestCase):
    def test_numeric_reproduction_tolerance_accepts_roundoff_and_rejects_material_change(self):
        with tempfile.TemporaryDirectory(dir=config.PROJECT_ROOT / ".venv") as directory:
            root = Path(directory).resolve()
            self.assertTrue(root.is_relative_to((config.PROJECT_ROOT / ".venv").resolve()))
            expected, replayed = root / "expected.csv", root / "replayed.csv"
            pd.DataFrame({"return": [1.0, np.nan]}).to_csv(expected, index=False)
            pd.DataFrame({"return": [1.0 + 5e-11, np.nan]}).to_csv(replayed, index=False)
            result = reproduction.compare_table(expected, replayed)
            self.assertFalse(result["byte_identical"])
            self.assertGreater(result["maximum_absolute_numeric_difference"], 0)
            self.assertLess(result["maximum_absolute_numeric_difference"], 1e-10)
            pd.DataFrame({"return": [1.0 + 5e-9, np.nan]}).to_csv(replayed, index=False)
            with self.assertRaisesRegex(ValueError, "numeric tolerance"):
                reproduction.compare_table(expected, replayed)

    def test_reproduction_does_not_tolerate_changed_identifiers_or_unavailable_values(self):
        with tempfile.TemporaryDirectory(dir=config.PROJECT_ROOT / ".venv") as directory:
            root = Path(directory).resolve()
            self.assertTrue(root.is_relative_to((config.PROJECT_ROOT / ".venv").resolve()))
            expected, replayed = root / "expected.csv", root / "replayed.csv"
            frame = pd.DataFrame({"candidate_id": [24, 24], "return": [0.1, np.nan]})
            frame.to_csv(expected, index=False)
            frame.assign(candidate_id=[25, 24]).to_csv(replayed, index=False)
            with self.assertRaisesRegex(ValueError, "identifiers"):
                reproduction.compare_table(expected, replayed)
            frame.assign(**{"return": [0.1, 0.0]}).to_csv(replayed, index=False)
            with self.assertRaisesRegex(ValueError, "unavailable"):
                reproduction.compare_table(expected, replayed)
            nullable = pd.DataFrame({"allocation_candidate_id": [14.0, np.nan]})
            nullable.to_csv(expected, index=False)
            nullable.assign(allocation_candidate_id=[14.0 + 5e-11, np.nan]).to_csv(replayed, index=False)
            with self.assertRaisesRegex(ValueError, "identifiers"):
                reproduction.compare_table(expected, replayed)


if __name__ == "__main__":
    unittest.main()
