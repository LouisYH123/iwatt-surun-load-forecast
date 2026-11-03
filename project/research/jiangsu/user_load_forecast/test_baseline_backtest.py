"""Unit tests for the Surun baseline time contract and metrics."""

from __future__ import annotations

import unittest
from datetime import date, time, timedelta

import numpy as np

from project.research.jiangsu.user_load_forecast.baseline_backtest import (
    daily_metrics,
    run_backtest,
    summary_metrics,
)
from project.research.jiangsu.user_load_forecast.contracts import (
    build_availability_ledger,
    validate_visible_source_dates,
)
from project.research.jiangsu.user_load_forecast.load_data import (
    AggregatedLoadDataset,
    interval_end_label,
)


def synthetic_dataset() -> AggregatedLoadDataset:
    start = date(2026, 4, 1)
    curves = {}
    for offset in range(60):
        trade_date = start + timedelta(days=offset)
        curves[trade_date] = np.full(96, 100.0 + trade_date.weekday())
    counts = {item: 10 for item in curves}
    return AggregatedLoadDataset(
        curves=curves,
        daily_account_count=counts,
        daily_user_count=counts,
        row_count=600,
        distinct_account_count=10,
        distinct_user_count=10,
        distinct_retailer_count=1,
        negative_value_count=0,
        value_units=("source_unspecified",),
        source_sha256="synthetic",
        source_path="synthetic",
    )


class AvailabilityContractTest(unittest.TestCase):
    def test_normal_t_plus_2_contract(self) -> None:
        observed = {
            date(2026, 4, 1) + timedelta(days=offset) for offset in range(40)
        }
        ledger = build_availability_ledger(
            [date(2026, 5, 1)],
            observed,
            forecast_as_of_time=time(8, 45),
        )
        row = ledger.iloc[0]
        self.assertEqual(row["run_date"], date(2026, 4, 30))
        self.assertEqual(
            row["latest_available_label_date"],
            date(2026, 4, 28),
        )
        self.assertEqual(row["label_age_days"], 3)
        self.assertEqual(row["latency_level"], "L0_NORMAL")

    def test_extra_delay_moves_latest_label_back(self) -> None:
        observed = {
            date(2026, 4, 1) + timedelta(days=offset) for offset in range(40)
        }
        ledger = build_availability_ledger(
            [date(2026, 5, 1)],
            observed,
            extra_delay_days=1,
        )
        self.assertEqual(
            ledger.iloc[0]["latest_available_label_date"],
            date(2026, 4, 27),
        )
        self.assertEqual(ledger.iloc[0]["latency_level"], "L1_LIGHT_DELAY")

    def test_invisible_actual_load_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_visible_source_dates(
                target_date=date(2026, 5, 1),
                latest_available_label_date=date(2026, 4, 28),
                source_dates=[date(2026, 4, 29)],
            )


class BaselineBacktestTest(unittest.TestCase):
    def test_same_weekday_baselines_are_exact_on_weekly_pattern(self) -> None:
        predictions, ledger = run_backtest(
            synthetic_dataset(),
            target_start=date(2026, 5, 1),
            target_end=date(2026, 5, 3),
        )
        self.assertEqual(len(ledger), 3)
        self.assertEqual(len(predictions), 3 * 2 * 96)
        self.assertTrue(np.allclose(predictions["absolute_error"], 0.0))
        self.assertTrue(np.allclose(daily_metrics(predictions)["wmape_pct"], 0.0))
        self.assertTrue(
            np.allclose(summary_metrics(predictions)["wmape_pct"], 0.0)
        )

    def test_interval_end_mapping(self) -> None:
        self.assertEqual(interval_end_label(1), "00:15")
        self.assertEqual(interval_end_label(4), "01:00")
        self.assertEqual(interval_end_label(96), "24:00")


if __name__ == "__main__":
    unittest.main()
