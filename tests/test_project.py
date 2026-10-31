"""Independent numerical, timing, availability, and reproducibility checks."""

from datetime import date
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests

from regime_allocation import config, data, features


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



if __name__ == "__main__":
    unittest.main()
