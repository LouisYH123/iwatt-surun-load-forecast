"""Calendar-aware similar-day baseline with strict D-3 visibility."""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from project.research.jiangsu.user_load_forecast.calendar_features import (
    CALENDAR_VERSION,
    calendar_features,
    day_type,
)
from project.research.jiangsu.user_load_forecast.load_data import (
    read_aggregated_load_snapshot,
)


VERSION = "surun_calendar_similar_day_v1.0"


def select_similar_dates(
    target_date: date,
    available_dates: list[date],
    *,
    max_dates: int = 4,
) -> list[date]:
    cutoff = target_date - timedelta(days=3)
    candidates = [item for item in available_dates if item <= cutoff]
    target_type = day_type(target_date)
    same_type = [item for item in candidates if day_type(item) == target_type]
    if target_type == "holiday":
        target_index = calendar_features(target_date)["holiday_day_index"]
        same_stage = [
            item
            for item in same_type
            if calendar_features(item)["holiday_day_index"] == target_index
        ]
        selected = same_stage or same_type
    else:
        same_weekday = [
            item for item in same_type if item.weekday() == target_date.weekday()
        ]
        selected = same_weekday or same_type
    if not selected:
        selected = candidates
    if not selected:
        raise ValueError(f"No visible similar date for {target_date}")
    return selected[-max_dates:]


def run_calendar_baseline(
    load_input: Path,
    *,
    target_start: date,
    target_end: date,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dataset = read_aggregated_load_snapshot(load_input)
    rows = []
    target_date = target_start
    while target_date <= target_end:
        selected = select_similar_dates(target_date, dataset.dates)
        cutoff = target_date - timedelta(days=3)
        if max(selected) > cutoff:
            raise AssertionError("Calendar baseline visibility leakage")
        prediction = np.mean(
            np.vstack([dataset.curves[item] for item in selected]),
            axis=0,
        )
        actual = dataset.curves[target_date]
        for interval in range(1, 97):
            predicted = float(prediction[interval - 1])
            actual_value = float(actual[interval - 1])
            rows.append(
                {
                    "model": "calendar_similar_day_mean4",
                    "target_date": target_date,
                    "trade_interval": interval,
                    "latest_available_label_date": cutoff,
                    "source_dates": "|".join(item.isoformat() for item in selected),
                    "source_max_date": max(selected),
                    "day_type": day_type(target_date),
                    "actual_load": actual_value,
                    "predicted_load": predicted,
                    "error": predicted - actual_value,
                    "absolute_error": abs(predicted - actual_value),
                }
            )
        target_date += timedelta(days=1)
    predictions = pd.DataFrame(rows)
    daily = (
        predictions.groupby(["target_date", "day_type"], as_index=False)
        .agg(
            absolute_error_sum=("absolute_error", "sum"),
            actual_abs_sum=("actual_load", lambda value: value.abs().sum()),
            actual_total=("actual_load", "sum"),
            predicted_total=("predicted_load", "sum"),
        )
    )
    daily["wmape_pct"] = (
        daily["absolute_error_sum"] / daily["actual_abs_sum"] * 100
    )
    daily["total_bias_pct"] = (
        (daily["predicted_total"] - daily["actual_total"])
        / daily["actual_total"]
        * 100
    )
    metadata = {
        "version": VERSION,
        "calendar_version": CALENDAR_VERSION,
        "source_sha256": dataset.source_sha256,
        "target_start": target_start.isoformat(),
        "target_end": target_end.isoformat(),
        "latency_policy": "SIMULATED_T_PLUS_2",
        "leakage_check": "PASS",
        "overall_wmape_pct": (
            predictions["absolute_error"].sum()
            / predictions["actual_load"].abs().sum()
            * 100
        ),
        "daily_wmape_p90_pct": daily["wmape_pct"].quantile(0.9),
        "worst_date": str(
            daily.loc[daily["wmape_pct"].idxmax(), "target_date"]
        ),
    }
    return predictions, daily, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-start", type=date.fromisoformat, default=date(2026, 5, 1))
    parser.add_argument("--target-end", type=date.fromisoformat, default=date(2026, 7, 25))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions, daily, metadata = run_calendar_baseline(
        args.input,
        target_start=args.target_start,
        target_end=args.target_end,
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
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
