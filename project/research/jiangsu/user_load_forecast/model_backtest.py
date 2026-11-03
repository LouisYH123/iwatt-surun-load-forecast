"""Walk-forward global-point model comparison for the Surun load target."""

from __future__ import annotations

import argparse
import json
import time as wall_time
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from project.research.jiangsu.user_load_forecast.calendar_features import (
    CALENDAR_VERSION,
    day_type,
)
from project.research.jiangsu.user_load_forecast.features import (
    FEATURE_VERSION,
    build_feature_frame,
    model_feature_columns,
)
from project.research.jiangsu.user_load_forecast.load_data import (
    read_aggregated_load_snapshot,
)
from project.research.jiangsu.user_load_forecast.weather_features import (
    WEATHER_POLICY_PROXY,
    WeatherFeatureStore,
    read_weather_snapshot,
)


MODEL_BACKTEST_VERSION = "surun_model_pool_v1.0"
DEFAULT_MODELS = (
    "ridge",
    "huber",
    "random_forest",
    "gbdt",
    "hist_gbdt",
    "extra_trees",
    "xgboost",
)


@dataclass(frozen=True)
class ModelAvailability:
    name: str
    available: bool
    reason: str | None = None


def _linear_pipeline(model: RegressorMixin) -> TransformedTargetRegressor:
    regressor = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )
    return TransformedTargetRegressor(
        regressor=regressor,
        transformer=StandardScaler(),
    )


def _tree_pipeline(model: RegressorMixin) -> TransformedTargetRegressor:
    regressor = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("model", model),
        ]
    )
    return TransformedTargetRegressor(
        regressor=regressor,
        transformer=StandardScaler(),
    )


def build_model(name: str) -> tuple[RegressorMixin | None, ModelAvailability]:
    if name == "ridge":
        return (
            _linear_pipeline(Ridge(alpha=20.0)),
            ModelAvailability(name=name, available=True),
        )
    if name == "huber":
        return (
            _linear_pipeline(
                HuberRegressor(
                    epsilon=1.35,
                    alpha=0.0005,
                    max_iter=160,
                    tol=1e-4,
                )
            ),
            ModelAvailability(name=name, available=True),
        )
    if name == "random_forest":
        return (
            _tree_pipeline(
                RandomForestRegressor(
                    n_estimators=80,
                    max_depth=12,
                    min_samples_leaf=4,
                    max_features=0.75,
                    random_state=42,
                    n_jobs=-1,
                )
            ),
            ModelAvailability(name=name, available=True),
        )
    if name == "gbdt":
        return (
            _tree_pipeline(
                GradientBoostingRegressor(
                    loss="huber",
                    n_estimators=90,
                    learning_rate=0.04,
                    max_depth=2,
                    min_samples_leaf=12,
                    random_state=42,
                )
            ),
            ModelAvailability(name=name, available=True),
        )
    if name == "hist_gbdt":
        return (
            _tree_pipeline(
                HistGradientBoostingRegressor(
                    loss="absolute_error",
                    max_iter=120,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    random_state=42,
                )
            ),
            ModelAvailability(name=name, available=True),
        )
    if name == "extra_trees":
        return (
            _tree_pipeline(
                ExtraTreesRegressor(
                    n_estimators=100,
                    max_depth=14,
                    min_samples_leaf=3,
                    max_features=0.8,
                    random_state=42,
                    n_jobs=-1,
                )
            ),
            ModelAvailability(name=name, available=True),
        )
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError:
            return (
                None,
                ModelAvailability(
                    name=name,
                    available=False,
                    reason=(
                        "Python package xgboost is not installed; network package "
                        "installation is unavailable in the current environment"
                    ),
                ),
            )
        return (
            _tree_pipeline(
                XGBRegressor(
                    n_estimators=180,
                    max_depth=4,
                    learning_rate=0.03,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=2.0,
                    objective="reg:absoluteerror",
                    random_state=42,
                    n_jobs=-1,
                )
            ),
            ModelAvailability(name=name, available=True),
        )
    raise ValueError(f"Unknown model: {name}")


def _date_list(start: date, end: date) -> list[date]:
    return [
        start + timedelta(days=offset)
        for offset in range((end - start).days + 1)
    ]


def walk_forward_model(
    feature_frame: pd.DataFrame,
    *,
    model_name: str,
    target_start: date,
    target_end: date,
    include_weather: bool,
    target_mode: str = "residual",
    min_train_days: int = 14,
    anchor_column: str = "baseline_same_weekday_mean_4",
) -> tuple[pd.DataFrame, ModelAvailability, list[dict]]:
    """Refit one fixed model for each target date using labels through D-3."""
    model_template, availability = build_model(model_name)
    if not availability.available:
        return pd.DataFrame(), availability, []
    if target_mode not in {"direct", "residual"}:
        raise ValueError("target_mode must be direct or residual")

    features = model_feature_columns(
        feature_frame,
        include_weather=include_weather,
    )
    rows = []
    failures = []
    for target_date in _date_list(target_start, target_end):
        label_cutoff = target_date - timedelta(days=3)
        train = feature_frame.loc[
            pd.to_datetime(feature_frame["trade_date"]).dt.date <= label_cutoff
        ].copy()
        target = feature_frame.loc[
            pd.to_datetime(feature_frame["trade_date"]).dt.date == target_date
        ].copy()
        if len(target) != 96:
            raise ValueError(f"Expected 96 target rows for {target_date}, got {len(target)}")
        train_dates = sorted(pd.to_datetime(train["trade_date"]).dt.date.unique())
        if train_dates and max(train_dates) > label_cutoff:
            raise AssertionError("Training label cutoff leakage")
        if len(train_dates) < min_train_days:
            raise ValueError(
                f"Only {len(train_dates)} training dates for {target_date}; "
                f"minimum={min_train_days}"
            )

        if anchor_column not in train or anchor_column not in target:
            raise ValueError(f"Missing anchor column: {anchor_column}")
        anchor_train = train[anchor_column]
        anchor_target = target[anchor_column]
        if anchor_target.isna().any():
            raise ValueError(f"Missing same-weekday anchor for {target_date}")
        if target_mode == "residual":
            usable = anchor_train.notna()
            train = train.loc[usable].copy()
            y_train = (
                train["actual_load"] - train[anchor_column]
            )
        else:
            y_train = train["actual_load"]

        model, _ = build_model(model_name)
        assert model is not None
        started = wall_time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(train[features], y_train)
            raw = np.asarray(model.predict(target[features]), dtype=float)
            prediction = (
                raw + anchor_target.to_numpy(dtype=float)
                if target_mode == "residual"
                else raw
            )
            prediction = np.clip(prediction, 0.0, None)
            elapsed = wall_time.perf_counter() - started
        except Exception as exc:
            failures.append(
                {
                    "model": model_name,
                    "target_date": target_date.isoformat(),
                    "error": repr(exc),
                }
            )
            prediction = anchor_target.to_numpy(dtype=float)
            raw = np.full(96, np.nan)
            elapsed = wall_time.perf_counter() - started

        for idx, target_row in target.reset_index(drop=True).iterrows():
            actual = float(target_row["actual_load"])
            predicted = float(prediction[idx])
            rows.append(
                {
                    "model": model_name,
                    "feature_variant": (
                        "calendar_load_weather_proxy"
                        if include_weather
                        else "calendar_load"
                    ),
                    "target_mode": target_mode,
                    "anchor_column": anchor_column,
                    "target_date": target_date,
                    "trade_interval": int(target_row["trade_interval"]),
                    "train_label_cutoff": label_cutoff,
                    "train_start_date": min(train_dates),
                    "train_days": len(train_dates),
                    "feature_count": len(features),
                    "fit_seconds": elapsed,
                    "used_fallback": bool(
                        failures and failures[-1]["target_date"] == target_date.isoformat()
                    ),
                    "actual_load": actual,
                    "predicted_load": predicted,
                    "error": predicted - actual,
                    "absolute_error": abs(predicted - actual),
                }
            )
    return pd.DataFrame(rows), availability, failures


def summarize_predictions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = (
        predictions.groupby(
            ["model", "feature_variant", "target_mode", "target_date"],
            as_index=False,
        )
        .agg(
            absolute_error_sum=("absolute_error", "sum"),
            actual_abs_sum=("actual_load", lambda value: value.abs().sum()),
            mae=("absolute_error", "mean"),
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
    daily["day_type"] = pd.to_datetime(daily["target_date"]).dt.date.map(day_type)

    rows = []
    keys = ["model", "feature_variant", "target_mode"]
    for key, group in daily.groupby(keys, sort=True):
        point_group = predictions
        for column, value in zip(keys, key, strict=True):
            point_group = point_group.loc[point_group[column].eq(value)]
        overall_denominator = point_group["actual_load"].abs().sum()
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "segment_type": "overall",
                "segment": "all",
                "target_days": len(group),
                "wmape_pct": (
                    point_group["absolute_error"].sum()
                    / overall_denominator
                    * 100
                ),
                "daily_wmape_mean_pct": group["wmape_pct"].mean(),
                "daily_wmape_p90_pct": group["wmape_pct"].quantile(0.9),
                "daily_wmape_max_pct": group["wmape_pct"].max(),
                "worst_date": group.loc[
                    group["wmape_pct"].idxmax(),
                    "target_date",
                ],
                "mean_abs_total_bias_pct": group["total_bias_pct"].abs().mean(),
            }
        )
        for month, month_daily in group.groupby(
            pd.to_datetime(group["target_date"]).dt.month
        ):
            month_dates = set(month_daily["target_date"])
            month_points = point_group.loc[
                point_group["target_date"].isin(month_dates)
            ]
            rows.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "segment_type": "month",
                    "segment": str(month),
                    "target_days": len(month_daily),
                    "wmape_pct": (
                        month_points["absolute_error"].sum()
                        / month_points["actual_load"].abs().sum()
                        * 100
                    ),
                    "daily_wmape_mean_pct": month_daily["wmape_pct"].mean(),
                    "daily_wmape_p90_pct": month_daily["wmape_pct"].quantile(0.9),
                    "daily_wmape_max_pct": month_daily["wmape_pct"].max(),
                    "worst_date": month_daily.loc[
                        month_daily["wmape_pct"].idxmax(),
                        "target_date",
                    ],
                    "mean_abs_total_bias_pct": (
                        month_daily["total_bias_pct"].abs().mean()
                    ),
                }
            )
    return daily, pd.DataFrame(rows)


def run_model_pool(
    feature_frame: pd.DataFrame,
    *,
    models: tuple[str, ...],
    target_start: date,
    target_end: date,
    include_weather: bool,
    target_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    frames = []
    statuses = []
    failures = []
    for model_name in models:
        started = wall_time.perf_counter()
        print(
            f"[model-pool] start model={model_name} "
            f"weather={include_weather} target_mode={target_mode}",
            flush=True,
        )
        predictions, status, model_failures = walk_forward_model(
            feature_frame,
            model_name=model_name,
            target_start=target_start,
            target_end=target_end,
            include_weather=include_weather,
            target_mode=target_mode,
        )
        statuses.append(
            {
                "model": status.name,
                "available": status.available,
                "reason": status.reason,
            }
        )
        failures.extend(model_failures)
        if not predictions.empty:
            frames.append(predictions)
        print(
            f"[model-pool] done model={model_name} available={status.available} "
            f"rows={len(predictions)} seconds={wall_time.perf_counter() - started:.1f}",
            flush=True,
        )
    all_predictions = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )
    if all_predictions.empty:
        return (
            all_predictions,
            pd.DataFrame(),
            pd.DataFrame(statuses),
            failures,
        )
    daily, summary = summarize_predictions(all_predictions)
    return all_predictions, summary, pd.DataFrame(statuses), failures


def prepare_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-input", type=Path, required=True)
    parser.add_argument("--weather-input", type=Path)
    parser.add_argument("--weather-source", default="Ts_weather_1")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
    )
    parser.add_argument("--target-start", type=date.fromisoformat, default=date(2026, 5, 1))
    parser.add_argument("--target-end", type=date.fromisoformat, default=date(2026, 7, 25))
    parser.add_argument("--target-mode", choices=["direct", "residual"], default="residual")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    prepare_output_dir(output_dir)
    dataset = read_aggregated_load_snapshot(args.load_input)
    weather: WeatherFeatureStore | None = None
    if args.weather_input:
        weather = read_weather_snapshot(
            args.weather_input,
            source=args.weather_source,
            policy=WEATHER_POLICY_PROXY,
        )
    feature_frame = build_feature_frame(dataset, weather=weather)
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    predictions, summary, statuses, failures = run_model_pool(
        feature_frame,
        models=models,
        target_start=args.target_start,
        target_end=args.target_end,
        include_weather=weather is not None,
        target_mode=args.target_mode,
    )
    daily, _ = summarize_predictions(predictions)
    feature_frame.to_csv(
        output_dir / "feature_snapshot.csv.gz",
        index=False,
        compression="gzip",
    )
    predictions.to_csv(output_dir / "predictions_96.csv.gz", index=False, compression="gzip")
    daily.to_csv(output_dir / "daily_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    statuses.to_csv(output_dir / "model_status.csv", index=False, encoding="utf-8-sig")
    (output_dir / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "version": MODEL_BACKTEST_VERSION,
        "feature_version": FEATURE_VERSION,
        "calendar_version": CALENDAR_VERSION,
        "load_source_sha256": dataset.source_sha256,
        "weather_policy": weather.policy if weather else "NONE",
        "weather_champion_eligible": (
            weather.is_champion_eligible if weather else True
        ),
        "weather_source_sha256": weather.source_sha256 if weather else None,
        "target_start": args.target_start.isoformat(),
        "target_end": args.target_end.isoformat(),
        "target_mode": args.target_mode,
        "models_requested": list(models),
        "prediction_rows": len(predictions),
        "failures": len(failures),
        "training_policy": "daily expanding window; labels through D-3; full dates only",
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        summary.loc[summary["segment_type"].eq("overall"), [
            "model", "feature_variant", "wmape_pct", "daily_wmape_p90_pct"
        ]].sort_values("wmape_pct").to_string(index=False)
    )


if __name__ == "__main__":
    main()
