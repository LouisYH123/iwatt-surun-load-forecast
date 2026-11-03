"""Operational walk-forward check for Jiangsu expanded market features.

For every target day D, the model is refit using portfolio labels no later
than D-3.  This is a research proxy because the historical market export does
not preserve issue_at/available_at.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from project.research.jiangsu.user_load_forecast.market_d2_curve_backtest import (
    build_features,
    curve_metrics,
    fit_candidate,
)
from project.research.jiangsu.user_load_forecast.market_level_shape_backtest import (
    MARKET_POLICY,
    add_expanded_market_features,
    read_benchmark,
    read_market_matrix,
    sha256_file,
)
from project.research.jiangsu.user_load_forecast.model_backtest import (
    prepare_output_dir,
)


VERSION = "surun_market_feature_walkforward_v1"


def run(args: argparse.Namespace) -> None:
    prepare_output_dir(args.output_dir)
    base = build_features(args.calendar_prediction_file, args.market_input)
    frame, feature_groups = add_expanded_market_features(
        base,
        read_market_matrix(args.market_input),
    )
    requested_groups = [
        item.strip() for item in args.feature_groups.split(",") if item.strip()
    ]
    unknown = sorted(set(requested_groups) - set(feature_groups))
    if unknown:
        raise ValueError(f"Unknown feature groups: {unknown}")
    target_dates = sorted(
        value
        for value in frame["target_date"].unique()
        if args.target_start <= value <= args.target_end
    )
    if not target_dates:
        raise ValueError("No target dates in requested walk-forward window")

    prediction_frames = []
    training_rows = []
    for group_name in requested_groups:
        daily_predictions = []
        for index, target_date in enumerate(target_dates, start=1):
            train_end = target_date - timedelta(days=3)
            candidate = fit_candidate(
                frame,
                model_name=f"walkforward_gbdt__{group_name}",
                feature_columns=feature_groups[group_name],
                train_end=train_end,
                holdout_start=target_date,
            )
            candidate = candidate.loc[
                candidate["target_date"].eq(target_date)
            ].copy()
            if len(candidate) != 96:
                raise ValueError(
                    f"{target_date}: expected 96 predictions, got {len(candidate)}"
                )
            daily_predictions.append(candidate)
            training_rows.append(
                {
                    "model": f"walkforward_gbdt__{group_name}",
                    "target_date": target_date,
                    "train_end": train_end,
                    "training_days": int(
                        frame.loc[
                            frame["target_date"] <= train_end,
                            "target_date",
                        ].nunique()
                    ),
                }
            )
            if index % 10 == 0 or index == len(target_dates):
                print(
                    f"[walk-forward] {group_name} "
                    f"{index}/{len(target_dates)}",
                    flush=True,
                )
        prediction_frames.append(pd.concat(daily_predictions, ignore_index=True))

    benchmark = read_benchmark(args.benchmark_prediction_file)
    benchmark = benchmark.loc[
        benchmark["target_date"].between(
            args.target_start,
            args.target_end,
        )
    ].copy()
    prediction_frames.append(benchmark)
    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)

    summary_rows = []
    daily_frames = []
    for model_name, group in predictions.groupby("model", sort=False):
        daily, summary = curve_metrics(group)
        daily.insert(0, "model", model_name)
        daily_frames.append(daily)
        summary_rows.append({"model": model_name, **summary})
    summary = pd.DataFrame(summary_rows).sort_values("wmape_pct")
    daily_metrics = pd.concat(daily_frames, ignore_index=True)

    predictions.to_csv(
        args.output_dir / "predictions_96.csv.gz",
        index=False,
        compression="gzip",
    )
    summary.to_csv(
        args.output_dir / "summary_curve_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    daily_metrics.to_csv(
        args.output_dir / "daily_curve_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(training_rows).to_csv(
        args.output_dir / "training_ledger.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "version": VERSION,
        "province_display": "江苏",
        "province_slug": "jiangsu",
        "target_start": args.target_start.isoformat(),
        "target_end": args.target_end.isoformat(),
        "feature_groups": requested_groups,
        "training_policy": "for each D, labels through D-3",
        "market_policy": MARKET_POLICY,
        "champion_eligible": False,
        "eligibility_reason": (
            "market export lacks historical issue_at/available_at evidence"
        ),
        "calendar_prediction_sha256": sha256_file(
            args.calendar_prediction_file
        ),
        "market_input_sha256": sha256_file(args.market_input),
        "benchmark_prediction_sha256": sha256_file(
            args.benchmark_prediction_file
        ),
        "pii_output_check": "aggregate-only outputs",
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar-prediction-file", type=Path, required=True)
    parser.add_argument("--market-input", type=Path, required=True)
    parser.add_argument("--benchmark-prediction-file", type=Path, required=True)
    parser.add_argument("--feature-groups", default="core,physical")
    parser.add_argument(
        "--target-start",
        type=date.fromisoformat,
        default=date(2026, 6, 15),
    )
    parser.add_argument(
        "--target-end",
        type=date.fromisoformat,
        default=date(2026, 7, 25),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
