"""Mature-label-only non-negative fusion of Surun OOS predictions."""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from project.research.jiangsu.user_load_forecast.calendar_features import day_type


FUSION_VERSION = "surun_dynamic_nonnegative_fusion_v1.0"


def read_prediction_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "model",
        "target_date",
        "trade_interval",
        "actual_load",
        "predicted_load",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing prediction columns {sorted(missing)}")
    frame["target_date"] = pd.to_datetime(frame["target_date"]).dt.date
    candidate = frame["model"].astype(str)
    if "feature_variant" in frame.columns:
        candidate = candidate + "__" + frame["feature_variant"].astype(str)
    if "target_mode" in frame.columns:
        candidate = candidate + "__" + frame["target_mode"].astype(str)
    frame["candidate"] = candidate
    return frame[
        [
            "candidate",
            "target_date",
            "trade_interval",
            "actual_load",
            "predicted_load",
        ]
    ].copy()


def combine_predictions(
    paths: list[Path],
    *,
    include_candidates: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    frames = [read_prediction_file(path) for path in paths]
    long = pd.concat(frames, ignore_index=True)
    if include_candidates is not None:
        long = long.loc[long["candidate"].isin(include_candidates)].copy()
    duplicate = long.duplicated(
        ["candidate", "target_date", "trade_interval"],
        keep=False,
    )
    if duplicate.any():
        sample = long.loc[
            duplicate,
            ["candidate", "target_date", "trade_interval"],
        ].head()
        raise ValueError(f"Duplicate candidate predictions:\n{sample}")
    actual_counts = long.groupby(["target_date", "trade_interval"])[
        "actual_load"
    ].nunique()
    if (actual_counts != 1).any():
        raise ValueError("Candidate files disagree on actual load")
    actual = long.groupby(["target_date", "trade_interval"])[
        "actual_load"
    ].first()
    matrix = long.pivot(
        index=["target_date", "trade_interval"],
        columns="candidate",
        values="predicted_load",
    ).sort_index()
    if matrix.isna().any().any():
        missing = matrix.columns[matrix.isna().any()].tolist()
        raise ValueError(f"Missing candidate predictions: {missing}")
    actual = actual.reindex(matrix.index)
    return matrix, actual


def nonnegative_weights(
    predictions: np.ndarray,
    actual: np.ndarray,
) -> tuple[np.ndarray, bool]:
    candidate_count = predictions.shape[1]
    initial = np.full(candidate_count, 1.0 / candidate_count)
    scale = max(float(np.mean(np.abs(actual))), 1e-9)

    def objective(weights: np.ndarray) -> float:
        error = predictions @ weights - actual
        return float(np.mean(np.abs(error)) / scale + 0.005 * np.sum(weights**2))

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * candidate_count,
        constraints={"type": "eq", "fun": lambda value: value.sum() - 1.0},
        options={"maxiter": 300, "ftol": 1e-10},
    )
    if result.success and np.isfinite(result.x).all():
        weights = np.clip(result.x, 0.0, 1.0)
        weights /= weights.sum()
        return weights, True

    errors = np.mean(np.abs(predictions - actual[:, None]), axis=0)
    inverse = 1.0 / np.maximum(errors, 1e-9)
    return inverse / inverse.sum(), False


def dynamic_fusion(
    matrix: pd.DataFrame,
    actual: pd.Series,
    *,
    fallback_candidate: str,
    min_history_days: int = 14,
    history_window_days: int = 28,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fallback_candidate not in matrix.columns:
        raise ValueError(f"Fallback candidate not found: {fallback_candidate}")
    target_dates = sorted({index[0] for index in matrix.index})
    prediction_rows = []
    weight_rows = []
    for target_date in target_dates:
        matured_cutoff = target_date - timedelta(days=3)
        matured_dates = [
            item for item in target_dates
            if item <= matured_cutoff and item < target_date
        ][-history_window_days:]
        if len(matured_dates) >= min_history_days:
            history_mask = matrix.index.get_level_values("target_date").isin(
                matured_dates
            )
            weights, optimizer_success = nonnegative_weights(
                matrix.loc[history_mask].to_numpy(dtype=float),
                actual.loc[history_mask].to_numpy(dtype=float),
            )
            mode = "optimized"
        else:
            weights = np.zeros(len(matrix.columns), dtype=float)
            weights[matrix.columns.get_loc(fallback_candidate)] = 1.0
            optimizer_success = True
            mode = "fallback"

        target_mask = matrix.index.get_level_values("target_date") == target_date
        target_matrix = matrix.loc[target_mask]
        target_actual = actual.loc[target_mask]
        fused = target_matrix.to_numpy(dtype=float) @ weights
        for (index, actual_value, predicted_value) in zip(
            target_matrix.index,
            target_actual.to_numpy(dtype=float),
            fused,
            strict=True,
        ):
            prediction_rows.append(
                {
                    "model": "dynamic_nonnegative_fusion",
                    "target_date": index[0],
                    "trade_interval": index[1],
                    "matured_label_cutoff": matured_cutoff,
                    "history_start_date": (
                        matured_dates[0] if matured_dates else None
                    ),
                    "history_days": len(matured_dates),
                    "weight_mode": mode,
                    "actual_load": actual_value,
                    "predicted_load": predicted_value,
                    "error": predicted_value - actual_value,
                    "absolute_error": abs(predicted_value - actual_value),
                }
            )
        for candidate, weight in zip(matrix.columns, weights, strict=True):
            weight_rows.append(
                {
                    "target_date": target_date,
                    "matured_label_cutoff": matured_cutoff,
                    "history_start_date": (
                        matured_dates[0] if matured_dates else None
                    ),
                    "history_days": len(matured_dates),
                    "weight_mode": mode,
                    "optimizer_success": optimizer_success,
                    "candidate": candidate,
                    "weight": weight,
                }
            )
    return pd.DataFrame(prediction_rows), pd.DataFrame(weight_rows)


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    daily = (
        predictions.groupby("target_date", as_index=False)
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
    daily["day_type"] = daily["target_date"].map(day_type)
    overall = {
        "wmape_pct": (
            predictions["absolute_error"].sum()
            / predictions["actual_load"].abs().sum()
            * 100
        ),
        "daily_wmape_mean_pct": daily["wmape_pct"].mean(),
        "daily_wmape_p90_pct": daily["wmape_pct"].quantile(0.9),
        "daily_wmape_max_pct": daily["wmape_pct"].max(),
        "worst_date": str(
            daily.loc[daily["wmape_pct"].idxmax(), "target_date"]
        ),
        "mean_abs_total_bias_pct": daily["total_bias_pct"].abs().mean(),
    }
    return daily, overall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-file", type=Path, action="append", required=True)
    parser.add_argument("--include-candidate", action="append")
    parser.add_argument("--fallback-candidate", default="same_weekday_mean_4")
    parser.add_argument("--min-history-days", type=int, default=14)
    parser.add_argument("--history-window-days", type=int, default=28)
    parser.add_argument("--weather-policy", default="NONE")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    include = set(args.include_candidate) if args.include_candidate else None
    matrix, actual = combine_predictions(
        args.prediction_file,
        include_candidates=include,
    )
    predictions, weights = dynamic_fusion(
        matrix,
        actual,
        fallback_candidate=args.fallback_candidate,
        min_history_days=args.min_history_days,
        history_window_days=args.history_window_days,
    )
    daily, overall = summarize(predictions)
    predictions.to_csv(
        args.output_dir / "predictions_96.csv.gz",
        index=False,
        compression="gzip",
    )
    weights.to_csv(
        args.output_dir / "daily_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )
    daily.to_csv(
        args.output_dir / "daily_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "version": FUSION_VERSION,
        "candidate_count": len(matrix.columns),
        "candidates": matrix.columns.tolist(),
        "fallback_candidate": args.fallback_candidate,
        "min_history_days": args.min_history_days,
        "history_window_days": args.history_window_days,
        "label_maturity_policy": "only OOS labels with target_date <= D-3",
        "weather_policy": args.weather_policy,
        "champion_eligible": args.weather_policy in {"NONE", "STRICT_FORECAST_SNAPSHOT"},
        **overall,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
