"""Check consequential calendar, grid, credential, and seed behavior."""

from datetime import date
import json
import os
import unittest
from unittest.mock import patch

import numpy as np

from regime_allocation import config


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
