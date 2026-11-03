"""Calibrate a fixed non-negative fusion including the public DA signal."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from project.research.jiangsu.user_load_forecast.fusion import (
    nonnegative_weights,
)
from project.research.jiangsu.user_load_forecast.model_backtest import (
    prepare_output_dir,
)


VERSION = "surun_market_model_fixed_fusion_v1"
CANDIDATES = {
    "calendar": ("calendar_prediction_file", "calendar_similar_day_mean4"),
    "strict_fusion": (
        "strict_prediction_file",
        "dynamic_nonnegative_fusion",
    ),
    "industry_behavior": (
        "industry_prediction_file",
        "industry_behavior__recent_trend",
    ),
    "xgboost": ("xgboost_prediction_file", "xgboost"),
    "market_da": (
        "market_prediction_file",
        "market_da_elasticity__calendar__target_da_vs_source_rt",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_candidate(
    path: Path,
    *,
    model: str,
    candidate: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "model",
        "target_date",
        "trade_interval",
        "actual_load",
        "predicted_load",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame = frame.loc[frame["model"].eq(model)].copy()
    if frame.empty:
        raise ValueError(f"{path}: model not found: {model}")
    frame["target_date"] = pd.to_datetime(frame["target_date"]).dt.date
    frame = frame[
        ["target_date", "trade_interval", "actual_load", "predicted_load"]
    ].rename(
        columns={
            "actual_load": f"actual__{candidate}",
            "predicted_load": candidate,
        }
    )
    if frame.duplicated(["target_date", "trade_interval"]).any():
        raise ValueError(f"{path}: duplicate keys for {model}")
    return frame


def metrics(frame: pd.DataFrame) -> dict:
    daily = (
        frame.groupby("target_date", as_index=False)
        .agg(
            absolute_error_sum=("absolute_error", "sum"),
            actual_abs_sum=("actual_load", lambda value: value.abs().sum()),
        )
    )
    daily["wmape_pct"] = (
        daily["absolute_error_sum"] / daily["actual_abs_sum"] * 100
    )
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
    frames = []
    input_metadata = {}
    for candidate, (path_attribute, model) in CANDIDATES.items():
        path = getattr(args, path_attribute)
        frames.append(
            read_candidate(path, model=model, candidate=candidate)
        )
        input_metadata[candidate] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "source_model": model,
        }
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(
            frame,
            on=["target_date", "trade_interval"],
            validate="one_to_one",
        )
    actual_columns = [column for column in merged if column.startswith("actual__")]
    actual = merged[actual_columns[0]].to_numpy(dtype=float)
    for column in actual_columns[1:]:
        if not np.allclose(actual, merged[column].to_numpy(dtype=float)):
            raise ValueError(f"Candidate actual values disagree: {column}")
    merged["actual_load"] = actual
    merged = merged.drop(columns=actual_columns)
    interval_sets = merged.groupby("target_date")["trade_interval"].agg(
        lambda value: set(value.astype(int))
    )
    if not interval_sets.map(lambda value: value == set(range(1, 97))).all():
        raise ValueError("Every date must contain intervals 1..96")

    candidates = list(CANDIDATES)
    holdout_start = args.holdout_start or (
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
    weights, optimizer_success = nonnegative_weights(
        merged.loc[calibration_mask, candidates].to_numpy(dtype=float),
        merged.loc[calibration_mask, "actual_load"].to_numpy(dtype=float),
    )
    predictions = merged.loc[
        legal_mask,
        ["target_date", "trade_interval", "actual_load"]
    ].copy()
    predictions.insert(0, "model", "market_model_fixed_nonnegative_fusion")
    predictions["window"] = np.where(
        calibration_mask.loc[legal_mask],
        "calibration",
        "holdout",
    )
    predictions["predicted_load"] = (
        merged.loc[legal_mask, candidates].to_numpy(dtype=float) @ weights
    )
    predictions["error"] = (
        predictions["predicted_load"] - predictions["actual_load"]
    )
    predictions["absolute_error"] = predictions["error"].abs()

    metric_rows = []
    for window, mask in (
        ("calibration", calibration_mask),
        ("holdout", holdout_mask),
        ("overall", legal_mask),
    ):
        prediction_mask = mask.loc[legal_mask]
        metric_rows.append(
            {"window": window, **metrics(predictions.loc[prediction_mask])}
        )
    summary = pd.DataFrame(metric_rows)
    daily = (
        predictions.groupby(["window", "target_date"], as_index=False)
        .agg(
            absolute_error_sum=("absolute_error", "sum"),
            actual_abs_sum=("actual_load", lambda value: value.abs().sum()),
        )
    )
    daily["wmape_pct"] = (
        daily["absolute_error_sum"] / daily["actual_abs_sum"] * 100
    )
    weight_frame = pd.DataFrame(
        {"candidate": candidates, "weight": weights}
    )
    correlation_rows = []
    for window, mask in (
        ("calibration", calibration_mask),
        ("holdout", holdout_mask),
    ):
        errors = merged.loc[mask, candidates].sub(
            merged.loc[mask, "actual_load"],
            axis=0,
        )
        correlation = errors.corr()
        for left in candidates:
            for right in candidates:
                correlation_rows.append(
                    {
                        "window": window,
                        "candidate_left": left,
                        "candidate_right": right,
                        "error_correlation": correlation.loc[left, right],
                    }
                )
    predictions.to_csv(
        args.output_dir / "predictions_96.csv.gz",
        index=False,
        compression="gzip",
    )
    summary.to_csv(
        args.output_dir / "summary_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    daily.to_csv(
        args.output_dir / "daily_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    weight_frame.to_csv(
        args.output_dir / "fusion_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(correlation_rows).to_csv(
        args.output_dir / "error_correlations.csv",
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
        "weight_training_window": "calibration only",
        "optimizer_success": bool(optimizer_success),
        "weights": {
            candidate: float(weight)
            for candidate, weight in zip(candidates, weights, strict=True)
        },
        "input_candidates": input_metadata,
        "market_policy": "RESEARCH_PROXY_UNVERIFIED_PUBLICATION_TIME",
        "champion_eligible": False,
        "eligibility_reason": (
            "market DA source lacks historical issue_at/available_at evidence"
        ),
        "selected_metrics": {
            row["window"]: {
                "target_days": int(row["target_days"]),
                "wmape_pct": float(row["wmape_pct"]),
                "daily_wmape_p90_pct": float(
                    row["daily_wmape_p90_pct"]
                ),
                "daily_wmape_max_pct": float(
                    row["daily_wmape_max_pct"]
                ),
            }
            for _, row in summary.iterrows()
        },
        "pii_output_check": "aggregate-only outputs",
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(weight_frame.to_string(index=False))
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar-prediction-file", type=Path, required=True)
    parser.add_argument("--strict-prediction-file", type=Path, required=True)
    parser.add_argument("--industry-prediction-file", type=Path, required=True)
    parser.add_argument("--xgboost-prediction-file", type=Path, required=True)
    parser.add_argument("--market-prediction-file", type=Path, required=True)
    parser.add_argument(
        "--calibration-end",
        type=date.fromisoformat,
        default=date(2026, 6, 14),
    )
    parser.add_argument("--holdout-start", type=date.fromisoformat)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
