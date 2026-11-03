"""Dynamic Top-N and long-tail user-structure load forecast experiment."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd

from project.research.jiangsu.user_load_forecast.calendar_features import day_type
from project.research.jiangsu.user_load_forecast.load_data import (
    AggregatedLoadDataset,
    read_aggregated_load_snapshot,
)


USER_MODEL_VERSION = "surun_topn_longtail_v1.0"


@dataclass
class UserPanel:
    curves: dict[str, dict[date, np.ndarray]]
    daily_users: dict[date, set[str]]
    daily_accounts: dict[date, set[str]]
    negative_users: set[str]


def _open_text(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _anonymous_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def read_user_panel(path: Path) -> UserPanel:
    """Aggregate accounts to anonymous retail users without exporting identities."""
    curves: dict[str, dict[date, np.ndarray]] = defaultdict(
        lambda: defaultdict(lambda: np.zeros(96, dtype=float))
    )
    daily_users: dict[date, set[str]] = defaultdict(set)
    daily_accounts: dict[date, set[str]] = defaultdict(set)
    negative_users: set[str] = set()
    with _open_text(path) as handle:
        for row in csv.DictReader(handle):
            trade_date = date.fromisoformat(row["trade_date"])
            user_id = _anonymous_id(row["retail_user_name"].strip())
            account_id = _anonymous_id(row["account_id"].strip())
            raw = row["actual_values"].strip()
            values = np.asarray(
                [float(item) for item in raw[1:-1].split(",")],
                dtype=float,
            )
            if len(values) != 96 or not np.isfinite(values).all():
                raise ValueError("Invalid user actual_values array")
            curves[user_id][trade_date] += values
            daily_users[trade_date].add(user_id)
            daily_accounts[trade_date].add(account_id)
            if np.any(values < 0):
                negative_users.add(user_id)
    return UserPanel(
        curves={key: dict(value) for key, value in curves.items()},
        daily_users=dict(daily_users),
        daily_accounts=dict(daily_accounts),
        negative_users=negative_users,
    )


def membership_diagnostics(panel: UserPanel) -> pd.DataFrame:
    """Post-label diagnostics; these fields are never used as target-day features."""
    dates = sorted(panel.daily_users)
    rows = []
    for index, trade_date in enumerate(dates):
        previous = dates[index - 1] if index else None
        next_date = dates[index + 1] if index + 1 < len(dates) else None
        users = panel.daily_users[trade_date]
        accounts = panel.daily_accounts[trade_date]
        previous_users = panel.daily_users.get(previous, set())
        previous_accounts = panel.daily_accounts.get(previous, set())
        disappeared_accounts = previous_accounts - accounts
        returned_next_day = (
            len(disappeared_accounts & panel.daily_accounts.get(next_date, set()))
            if next_date
            else 0
        )
        rows.append(
            {
                "trade_date": trade_date,
                "diagnostic_visibility": "POST_LABEL_ONLY",
                "user_count": len(users),
                "account_count": len(accounts),
                "new_users_vs_previous": len(users - previous_users),
                "missing_users_vs_previous": len(previous_users - users),
                "new_accounts_vs_previous": len(accounts - previous_accounts),
                "missing_accounts_vs_previous": len(disappeared_accounts),
                "missing_accounts_returned_next_day": returned_next_day,
            }
        )
    return pd.DataFrame(rows)


def _available_curve_mean(
    user_curves: dict[date, np.ndarray],
    dates: list[date],
) -> tuple[np.ndarray | None, list[date]]:
    available = [item for item in dates if item in user_curves]
    if not available:
        return None, []
    return (
        np.mean(np.vstack([user_curves[item] for item in available]), axis=0),
        available,
    )


def _trend_factor(
    user_curves: dict[date, np.ndarray],
    *,
    visible_dates: list[date],
    reference_dates: list[date],
    lower: float = 0.75,
    upper: float = 1.25,
) -> float:
    recent = [item for item in visible_dates[-7:] if item in user_curves]
    reference = [item for item in reference_dates if item in user_curves]
    if not recent or not reference:
        return 1.0
    recent_mean = np.mean([user_curves[item].sum() for item in recent])
    reference_mean = np.mean([user_curves[item].sum() for item in reference])
    if not np.isfinite(reference_mean) or abs(reference_mean) < 1e-9:
        return 1.0
    return float(np.clip(recent_mean / reference_mean, lower, upper))


def dynamic_topn_prediction(
    panel: UserPanel,
    aggregate: AggregatedLoadDataset,
    *,
    target_date: date,
    coverage_target: float = 0.75,
) -> tuple[np.ndarray, dict]:
    """Forecast Top-N users and the remaining long tail using visible history only."""
    cutoff = target_date - timedelta(days=3)
    visible_dates = [item for item in aggregate.dates if item <= cutoff]
    if not visible_dates:
        raise ValueError(f"No visible history for {target_date}")

    user_energy = {}
    for user_id, curves in panel.curves.items():
        values = [curves[item].sum() for item in visible_dates if item in curves]
        if values:
            user_energy[user_id] = float(np.sum(values))
    ranked = sorted(
        user_energy,
        key=lambda item: max(user_energy[item], 0.0),
        reverse=True,
    )
    positive_total = sum(max(value, 0.0) for value in user_energy.values())
    selected = []
    selected_positive = 0.0
    for user_id in ranked:
        if positive_total and selected_positive / positive_total >= coverage_target:
            break
        selected.append(user_id)
        selected_positive += max(user_energy[user_id], 0.0)
    selected_set = set(selected)

    reference_dates = [
        target_date - timedelta(days=lag) for lag in (7, 14, 21, 28)
    ]
    prediction = np.zeros(96, dtype=float)
    cold_start_users = 0
    for user_id in selected:
        curves = panel.curves[user_id]
        base, available_reference = _available_curve_mean(curves, reference_dates)
        if base is None:
            recent = [item for item in visible_dates[-7:] if item in curves]
            base, available_reference = _available_curve_mean(curves, recent[-3:])
            cold_start_users += 1
        if base is None:
            continue
        factor = _trend_factor(
            curves,
            visible_dates=visible_dates,
            reference_dates=available_reference,
        )
        prediction += base * factor

    def long_tail_curve(trade_date: date) -> np.ndarray:
        value = np.zeros(96, dtype=float)
        for user_id, curves in panel.curves.items():
            if user_id in selected_set:
                continue
            if trade_date in curves:
                value += curves[trade_date]
        return value

    reference_long_tail_dates = [
        item for item in reference_dates if item in aggregate.curves
    ]
    if reference_long_tail_dates:
        reference_long_tail = np.vstack(
            [long_tail_curve(item) for item in reference_long_tail_dates]
        )
        long_tail_base = reference_long_tail.mean(axis=0)
        recent_long_tail_dates = visible_dates[-7:]
        recent_long_tail_total = np.mean(
            [long_tail_curve(item).sum() for item in recent_long_tail_dates]
        )
        reference_long_tail_total = np.mean(reference_long_tail.sum(axis=1))
        long_tail_factor = (
            float(
                np.clip(
                    recent_long_tail_total / reference_long_tail_total,
                    0.75,
                    1.25,
                )
            )
            if abs(reference_long_tail_total) > 1e-9
            else 1.0
        )
        prediction += long_tail_base * long_tail_factor
    else:
        long_tail_factor = 1.0

    metadata = {
        "target_date": target_date,
        "latest_visible_label_date": cutoff,
        "selected_user_count": len(selected),
        "selected_history_coverage": (
            selected_positive / positive_total if positive_total else 0.0
        ),
        "long_tail_user_count": len(user_energy) - len(selected),
        "cold_start_selected_users": cold_start_users,
        "long_tail_trend_factor": long_tail_factor,
        "negative_user_count_in_history": len(panel.negative_users),
    }
    return np.clip(prediction, 0.0, None), metadata


def run_user_backtest(
    panel: UserPanel,
    aggregate: AggregatedLoadDataset,
    *,
    target_start: date,
    target_end: date,
    coverage_target: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    daily_rows = []
    target_date = target_start
    while target_date <= target_end:
        prediction, metadata = dynamic_topn_prediction(
            panel,
            aggregate,
            target_date=target_date,
            coverage_target=coverage_target,
        )
        actual = aggregate.curves[target_date]
        for interval in range(1, 97):
            predicted = float(prediction[interval - 1])
            actual_value = float(actual[interval - 1])
            rows.append(
                {
                    "model": "topn75_longtail_trend",
                    "target_date": target_date,
                    "trade_interval": interval,
                    "train_label_cutoff": target_date - timedelta(days=3),
                    "actual_load": actual_value,
                    "predicted_load": predicted,
                    "error": predicted - actual_value,
                    "absolute_error": abs(predicted - actual_value),
                }
            )
        denominator = np.abs(actual).sum()
        daily_rows.append(
            {
                **metadata,
                "wmape_pct": np.abs(prediction - actual).sum() / denominator * 100,
                "actual_total": actual.sum(),
                "predicted_total": prediction.sum(),
                "total_bias_pct": (
                    (prediction.sum() - actual.sum()) / actual.sum() * 100
                ),
                "day_type": day_type(target_date),
            }
        )
        target_date += timedelta(days=1)
    return pd.DataFrame(rows), pd.DataFrame(daily_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-start", type=date.fromisoformat, default=date(2026, 5, 1))
    parser.add_argument("--target-end", type=date.fromisoformat, default=date(2026, 7, 25))
    parser.add_argument("--coverage-target", type=float, default=0.75)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = read_aggregated_load_snapshot(args.input)
    panel = read_user_panel(args.input)
    predictions, daily = run_user_backtest(
        panel,
        aggregate,
        target_start=args.target_start,
        target_end=args.target_end,
        coverage_target=args.coverage_target,
    )
    diagnostics = membership_diagnostics(panel)
    overall_wmape = (
        predictions["absolute_error"].sum()
        / predictions["actual_load"].abs().sum()
        * 100
    )
    predictions.to_csv(
        args.output_dir / "predictions_96.csv.gz",
        index=False,
        compression="gzip",
    )
    daily.to_csv(
        args.output_dir / "daily_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    diagnostics.to_csv(
        args.output_dir / "membership_diagnostics_post_label.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "version": USER_MODEL_VERSION,
        "source_sha256": aggregate.source_sha256,
        "coverage_target": args.coverage_target,
        "target_start": args.target_start.isoformat(),
        "target_end": args.target_end.isoformat(),
        "overall_wmape_pct": overall_wmape,
        "user_status_policy": "HISTORY_ONLY_NO_CONTRACT_EVENT_FEED",
        "membership_diagnostics_visibility": "POST_LABEL_ONLY_NOT_A_FEATURE",
        "negative_values_policy": "preserve_in_history; clip final company forecast at zero",
        "output_contains_user_identifiers": False,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
