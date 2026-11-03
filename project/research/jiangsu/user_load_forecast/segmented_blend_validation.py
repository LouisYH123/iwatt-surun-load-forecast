"""Calibrate a fixed base/segmented-trend blend and validate on later dates."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from project.research.jiangsu.user_load_forecast.model_backtest import (
    prepare_output_dir,
)


VERSION = "surun_segmented_blend_validation_v1.1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(frame: pd.DataFrame) -> dict:
    daily = (
        frame.groupby("target_date", as_index=False)
        .agg(
            absolute_error_sum=("absolute_error", "sum"),
            actual_abs_sum=("actual_load", lambda value: value.abs().sum()),
        )
    )
    daily["wmape_pct"] = daily["absolute_error_sum"] / daily["actual_abs_sum"] * 100
    return {
        "target_days": len(daily),
        "wmape_pct": (
            frame["absolute_error"].sum()
            / frame["actual_load"].abs().sum()
            * 100
        ),
        "daily_wmape_p90_pct": daily["wmape_pct"].quantile(0.9),
        "daily_wmape_max_pct": daily["wmape_pct"].max(),
    }


def run(args: argparse.Namespace) -> None:
    prepare_output_dir(args.output_dir)
    base_path = args.base_prediction_file or args.prediction_file
    trend_path = args.trend_prediction_file or args.prediction_file
    if base_path is None or trend_path is None:
        raise ValueError(
            "Provide --prediction-file, or both --base-prediction-file "
            "and --trend-prediction-file"
        )
    base_frame = pd.read_csv(base_path)
    trend_frame = pd.read_csv(trend_path)
    base_frame["target_date"] = pd.to_datetime(
        base_frame["target_date"]
    ).dt.date
    trend_frame["target_date"] = pd.to_datetime(
        trend_frame["target_date"]
    ).dt.date
    base = base_frame.loc[
        base_frame["model"].eq(args.base_model),
        ["target_date", "trade_interval", "actual_load", "predicted_load"],
    ].rename(columns={"predicted_load": "base_prediction"})
    trend = trend_frame.loc[
        trend_frame["model"].eq(args.trend_model),
        ["target_date", "trade_interval", "actual_load", "predicted_load"],
    ].rename(
        columns={
            "actual_load": "trend_actual_load",
            "predicted_load": "segmented_trend_prediction",
        }
    )
    if base.empty:
        raise ValueError(f"Base model not found: {args.base_model}")
    if trend.empty:
        raise ValueError(f"Trend model not found: {args.trend_model}")
    merged = base.merge(
        trend,
        on=["target_date", "trade_interval"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(base) or len(merged) != len(trend):
        raise ValueError("Base and trend predictions have different key coverage")
    if not np.allclose(
        merged["actual_load"],
        merged["trend_actual_load"],
        equal_nan=False,
    ):
        raise ValueError("Base and trend predictions disagree on actual load")
    intervals_by_day = merged.groupby("target_date")["trade_interval"].agg(
        lambda value: set(value.astype(int))
    )
    expected_intervals = set(range(1, 97))
    if not intervals_by_day.map(lambda value: value == expected_intervals).all():
        raise ValueError("Predictions must contain exactly intervals 1..96 per day")
    merged = merged.drop(columns="trend_actual_load")
    holdout_start = getattr(args, "holdout_start", None) or (
        args.calibration_end + timedelta(days=3)
    )
    if holdout_start < args.calibration_end + timedelta(days=3):
        raise ValueError(
            "holdout_start must be at least calibration_end + 3 days "
            "under the D-3 label-maturity policy"
        )
    calibration_mask = merged["target_date"] <= args.calibration_end
    holdout_mask = merged["target_date"] >= holdout_start
    legal_mask = calibration_mask | holdout_mask
    base_calibration = merged.loc[calibration_mask].copy()
    base_calibration["predicted_load"] = base_calibration["base_prediction"]
    base_calibration["absolute_error"] = (
        base_calibration["predicted_load"] - base_calibration["actual_load"]
    ).abs()
    base_calibration_metrics = _metrics(base_calibration)

    grid_rows = []
    alpha_values = np.arange(
        args.alpha_min,
        args.alpha_max + args.alpha_step / 2,
        args.alpha_step,
    )
    for alpha in alpha_values:
        candidate = merged.copy()
        candidate["predicted_load"] = (
            (1.0 - alpha) * candidate["base_prediction"]
            + alpha * candidate["segmented_trend_prediction"]
        )
        candidate["absolute_error"] = (
            candidate["predicted_load"] - candidate["actual_load"]
        ).abs()
        for window, mask in (
            ("calibration", calibration_mask),
            ("holdout", holdout_mask),
            ("overall", legal_mask),
        ):
            grid_rows.append(
                {
                    "alpha": float(alpha),
                    "window": window,
                    **_metrics(candidate.loc[mask]),
                }
            )
    grid = pd.DataFrame(grid_rows)
    calibration = grid.loc[grid["window"].eq("calibration")].copy()
    eligible = calibration.loc[
        calibration["daily_wmape_p90_pct"]
        <= base_calibration_metrics["daily_wmape_p90_pct"]
        + args.calibration_p90_tolerance_pp
    ]
    if eligible.empty:
        raise ValueError("No alpha satisfies the calibration P90 guard")
    selected_alpha = float(
        eligible.loc[eligible["wmape_pct"].idxmin(), "alpha"]
    )

    predictions = merged.copy()
    predictions.insert(0, "model", args.output_model)
    predictions["alpha"] = selected_alpha
    predictions["window"] = np.select(
        [calibration_mask, holdout_mask],
        ["calibration", "holdout"],
        default="label_maturity_embargo",
    )
    predictions["predicted_load"] = (
        (1.0 - selected_alpha) * predictions["base_prediction"]
        + selected_alpha * predictions["segmented_trend_prediction"]
    )
    predictions["error"] = predictions["predicted_load"] - predictions["actual_load"]
    predictions["absolute_error"] = predictions["error"].abs()
    predictions = predictions.loc[legal_mask].copy()
    daily = (
        predictions.groupby(["window", "target_date"], as_index=False)
        .agg(
            absolute_error_sum=("absolute_error", "sum"),
            actual_abs_sum=("actual_load", lambda value: value.abs().sum()),
        )
    )
    daily["wmape_pct"] = daily["absolute_error_sum"] / daily["actual_abs_sum"] * 100
    selected_summary = grid.loc[
        np.isclose(grid["alpha"], selected_alpha)
    ].copy()

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
    grid.to_csv(
        args.output_dir / "alpha_grid_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected_summary.to_csv(
        args.output_dir / "selected_alpha_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "version": VERSION,
        "target_start": min(merged["target_date"]).isoformat(),
        "target_end": max(merged["target_date"]).isoformat(),
        "calibration_end": args.calibration_end.isoformat(),
        "holdout_start": holdout_start.isoformat(),
        "label_maturity_embargo_days": (
            holdout_start - args.calibration_end
        ).days - 1,
        "selection_rule": (
            "minimum calibration WMAPE subject to calibration P90 no more "
            f"than baseline + {args.calibration_p90_tolerance_pp:.3f}pp"
        ),
        "selected_alpha": selected_alpha,
        "alpha_applies_to": args.trend_model,
        "base_model": args.base_model,
        "base_weight": 1.0 - selected_alpha,
        "base_prediction_file": str(base_path.resolve()),
        "trend_prediction_file": str(trend_path.resolve()),
        "base_prediction_sha256": _sha256(base_path),
        "trend_prediction_sha256": _sha256(trend_path),
        "selected_metrics": {
            row["window"]: {
                key: (
                    int(row[key])
                    if key == "target_days"
                    else float(row[key])
                )
                for key in (
                    "target_days",
                    "wmape_pct",
                    "daily_wmape_p90_pct",
                    "daily_wmape_max_pct",
                )
            }
            for _, row in selected_summary.iterrows()
        },
        "load_visibility": "segment definitions frozen at first target D-3",
        "market_features_used": False,
        "weather_features_used": False,
        "champion_eligible": True,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(selected_summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-file",
        type=Path,
        help="Legacy convenience path containing both base and trend models.",
    )
    parser.add_argument("--base-prediction-file", type=Path)
    parser.add_argument("--trend-prediction-file", type=Path)
    parser.add_argument(
        "--base-model",
        default="calendar_similar_day_mean4",
    )
    parser.add_argument(
        "--trend-model",
        default="industry_behavior__recent_trend",
    )
    parser.add_argument(
        "--output-model",
        default="calendar_segmented_trend_fixed_blend",
    )
    parser.add_argument(
        "--calibration-end",
        type=date.fromisoformat,
        default=date(2026, 6, 14),
    )
    parser.add_argument("--holdout-start", type=date.fromisoformat)
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--alpha-step", type=float, default=0.1)
    parser.add_argument("--calibration-p90-tolerance-pp", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
