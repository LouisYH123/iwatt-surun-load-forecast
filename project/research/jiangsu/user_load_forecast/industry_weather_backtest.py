"""Industry-segment daily weather correction over the calendar baseline.

The experiment compares:

1. the calendar-aware similar-day baseline;
2. one aggregate Ridge weather correction using load-weighted city weather;
3. separate Ridge corrections for major name-derived industry segments.

All load labels used for a target date end at D-3.  City weights are frozen at
the first target date's D-3 cutoff.  Weather is still an unversioned diagnostic
proxy, so no result from this module is Champion-eligible.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from project.research.jiangsu.user_load_forecast.calendar_baseline import (
    select_similar_dates,
)
from project.research.jiangsu.user_load_forecast.load_data import (
    read_aggregated_load_snapshot,
)
from project.research.jiangsu.user_load_forecast.model_backtest import (
    prepare_output_dir,
)
from project.research.jiangsu.user_load_forecast.regional_weather import (
    CityWeatherPanel,
    read_city_weather_panel,
)
from project.research.jiangsu.user_load_forecast.user_segmentation import (
    SEGMENTATION_VERSION,
    SegmentedUserPanel,
    city_load_weights,
    read_segmented_user_panel,
    segment_profile,
)


VERSION = "surun_industry_weather_correction_v1"
FEATURE_COLUMNS = (
    "delta_temp_mean",
    "delta_cooling_degree",
    "delta_humidity_mean",
    "delta_rain_total",
    "temp_mean",
    "cooling_degree",
)


def _city_daily_weather(panel: CityWeatherPanel) -> pd.DataFrame:
    frame = panel.frame.copy()
    frame["trade_date"] = frame["timestamp"].dt.date
    return (
        frame.groupby(["city", "trade_date"], as_index=False)
        .agg(
            temp_mean=("temp_2m", "mean"),
            humidity_mean=("humidity_2m", "mean"),
            rain_total=("rain", "sum"),
        )
    )


def _weighted_daily_weather(
    city_daily: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    frame = city_daily.copy()
    frame["_weight"] = frame["city"].map(weights).astype(float)
    for column in ("temp_mean", "humidity_mean", "rain_total"):
        frame[column] *= frame["_weight"]
    daily = frame.groupby("trade_date", as_index=True)[
        ["temp_mean", "humidity_mean", "rain_total"]
    ].sum()
    daily["cooling_degree"] = (daily["temp_mean"] - 26.0).clip(lower=0.0)
    return daily


def _segment_groups(
    panel: SegmentedUserPanel,
    *,
    cutoff: date,
    minimum_share: float,
) -> dict[str, set[str]]:
    profile = segment_profile(panel, dimension="industry", cutoff=cutoff)
    major = set(
        profile.loc[
            profile["positive_energy_share"] >= minimum_share,
            "segment",
        ]
    )
    groups: dict[str, set[str]] = {segment: set() for segment in sorted(major)}
    groups["其他行业"] = set()
    for user_id, label in panel.industry.items():
        target = label.value if label.value in major else "其他行业"
        groups[target].add(user_id)
    return {key: value for key, value in groups.items() if value}


def _group_curves(
    panel: SegmentedUserPanel,
    users: set[str],
) -> dict[date, np.ndarray]:
    curves: dict[date, np.ndarray] = defaultdict(lambda: np.zeros(96))
    for user_id in users:
        for trade_date, curve in panel.curves[user_id].items():
            curves[trade_date] += curve
    return dict(curves)


def _weather_features(
    daily_weather: pd.DataFrame,
    target_date: date,
    source_dates: list[date],
) -> dict[str, float] | None:
    if target_date not in daily_weather.index:
        return None
    available = [item for item in source_dates if item in daily_weather.index]
    if not available:
        return None
    target = daily_weather.loc[target_date]
    reference = daily_weather.loc[available].mean()
    return {
        "delta_temp_mean": float(target["temp_mean"] - reference["temp_mean"]),
        "delta_cooling_degree": float(
            target["cooling_degree"] - reference["cooling_degree"]
        ),
        "delta_humidity_mean": float(
            target["humidity_mean"] - reference["humidity_mean"]
        ),
        "delta_rain_total": float(
            target["rain_total"] - reference["rain_total"]
        ),
        "temp_mean": float(target["temp_mean"]),
        "cooling_degree": float(target["cooling_degree"]),
    }


def _training_frame(
    curves: dict[date, np.ndarray],
    daily_weather: pd.DataFrame,
    *,
    label_cutoff: date,
) -> pd.DataFrame:
    dates = sorted(curves)
    rows = []
    for train_date in dates:
        if train_date > label_cutoff:
            break
        visible_before_train = [item for item in dates if item < train_date]
        try:
            source_dates = select_similar_dates(
                train_date,
                visible_before_train,
            )
        except ValueError:
            continue
        base = np.mean(np.vstack([curves[item] for item in source_dates]), axis=0)
        base_total = float(base.sum())
        if abs(base_total) < 1e-9:
            continue
        weather_features = _weather_features(
            daily_weather,
            train_date,
            source_dates,
        )
        if weather_features is None:
            continue
        rows.append(
            {
                "trade_date": train_date,
                **weather_features,
                "residual_ratio": (
                    (float(curves[train_date].sum()) - base_total)
                    / abs(base_total)
                ),
            }
        )
    return pd.DataFrame(rows)


def _predict_correction(
    curves: dict[date, np.ndarray],
    daily_weather: pd.DataFrame,
    *,
    target_date: date,
    source_dates: list[date],
    min_train_days: int,
    clip: float,
) -> tuple[float, dict]:
    label_cutoff = target_date - timedelta(days=3)
    train = _training_frame(
        curves,
        daily_weather,
        label_cutoff=label_cutoff,
    )
    target_features = _weather_features(
        daily_weather,
        target_date,
        source_dates,
    )
    if len(train) < min_train_days or target_features is None:
        return 0.0, {
            "train_days": len(train),
            "used_fallback": True,
            "raw_correction": 0.0,
        }
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=20.0)),
        ]
    )
    model.fit(train[list(FEATURE_COLUMNS)], train["residual_ratio"])
    raw = float(
        model.predict(pd.DataFrame([target_features])[list(FEATURE_COLUMNS)])[0]
    )
    return float(np.clip(raw, -clip, clip)), {
        "train_days": len(train),
        "used_fallback": False,
        "raw_correction": raw,
    }


def _summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = (
        predictions.groupby(["model", "target_date"], as_index=False)
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
    rows = []
    for model, group in daily.groupby("model"):
        points = predictions.loc[predictions["model"].eq(model)]
        rows.append(
            {
                "model": model,
                "target_days": len(group),
                "wmape_pct": (
                    points["absolute_error"].sum()
                    / points["actual_load"].abs().sum()
                    * 100
                ),
                "daily_wmape_mean_pct": group["wmape_pct"].mean(),
                "daily_wmape_p90_pct": group["wmape_pct"].quantile(0.9),
                "daily_wmape_max_pct": group["wmape_pct"].max(),
                "mean_abs_total_bias_pct": group["total_bias_pct"].abs().mean(),
                "worst_date": group.loc[group["wmape_pct"].idxmax(), "target_date"],
            }
        )
    return daily, pd.DataFrame(rows).sort_values("wmape_pct")


def run_backtest(
    load_input: Path,
    weather_input: Path,
    *,
    weather_source: str,
    target_start: date,
    target_end: date,
    minimum_industry_share: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    aggregate = read_aggregated_load_snapshot(load_input)
    panel = read_segmented_user_panel(load_input)
    weather = read_city_weather_panel(weather_input, source=weather_source)
    weight_cutoff = target_start - timedelta(days=3)
    city_daily = _city_daily_weather(weather)

    aggregate_weights, aggregate_weight_meta = city_load_weights(
        panel,
        cutoff=weight_cutoff,
    )
    aggregate_weather = _weighted_daily_weather(city_daily, aggregate_weights)
    groups = _segment_groups(
        panel,
        cutoff=weight_cutoff,
        minimum_share=minimum_industry_share,
    )
    segment_curves = {
        segment: _group_curves(panel, users)
        for segment, users in groups.items()
    }
    segment_weather = {}
    segment_weight_meta = {}
    for segment, users in groups.items():
        weights, metadata = city_load_weights(
            panel,
            cutoff=weight_cutoff,
            user_subset=users,
        )
        segment_weather[segment] = _weighted_daily_weather(city_daily, weights)
        segment_weight_meta[segment] = metadata

    prediction_rows = []
    correction_rows = []
    target_date = target_start
    while target_date <= target_end:
        label_cutoff = target_date - timedelta(days=3)
        source_dates = select_similar_dates(target_date, aggregate.dates)
        calendar_prediction = np.mean(
            np.vstack([aggregate.curves[item] for item in source_dates]),
            axis=0,
        )
        aggregate_correction, aggregate_meta = _predict_correction(
            aggregate.curves,
            aggregate_weather,
            target_date=target_date,
            source_dates=source_dates,
            min_train_days=14,
            clip=0.20,
        )
        aggregate_prediction = calendar_prediction * (1.0 + aggregate_correction)

        industry_prediction = np.zeros(96)
        for segment, curves in segment_curves.items():
            available_sources = [item for item in source_dates if item in curves]
            if not available_sources:
                continue
            segment_base = np.mean(
                np.vstack([curves[item] for item in available_sources]),
                axis=0,
            )
            correction, correction_meta = _predict_correction(
                curves,
                segment_weather[segment],
                target_date=target_date,
                source_dates=available_sources,
                min_train_days=14,
                clip=0.20,
            )
            industry_prediction += segment_base * (1.0 + correction)
            correction_rows.append(
                {
                    "model": "industry_segment_weather_ridge",
                    "target_date": target_date,
                    "segment": segment,
                    "train_label_cutoff": label_cutoff,
                    "source_dates": "|".join(
                        item.isoformat() for item in available_sources
                    ),
                    "correction_ratio": correction,
                    **correction_meta,
                }
            )
        correction_rows.append(
            {
                "model": "aggregate_weather_ridge",
                "target_date": target_date,
                "segment": "公司总量",
                "train_label_cutoff": label_cutoff,
                "source_dates": "|".join(item.isoformat() for item in source_dates),
                "correction_ratio": aggregate_correction,
                **aggregate_meta,
            }
        )

        actual = aggregate.curves[target_date]
        for model, prediction in (
            ("calendar_similar_day_mean4", calendar_prediction),
            ("aggregate_weather_ridge", aggregate_prediction),
            ("industry_segment_weather_ridge", industry_prediction),
        ):
            prediction = np.clip(prediction, 0.0, None)
            for interval in range(1, 97):
                predicted = float(prediction[interval - 1])
                actual_value = float(actual[interval - 1])
                prediction_rows.append(
                    {
                        "model": model,
                        "target_date": target_date,
                        "trade_interval": interval,
                        "train_label_cutoff": label_cutoff,
                        "source_dates": "|".join(
                            item.isoformat() for item in source_dates
                        ),
                        "actual_load": actual_value,
                        "predicted_load": predicted,
                        "error": predicted - actual_value,
                        "absolute_error": abs(predicted - actual_value),
                    }
                )
        target_date += timedelta(days=1)

    predictions = pd.DataFrame(prediction_rows)
    corrections = pd.DataFrame(correction_rows)
    daily, summary = _summarize(predictions)
    metadata = {
        "version": VERSION,
        "segmentation_version": SEGMENTATION_VERSION,
        "province_display": "江苏",
        "province_slug": "jiangsu",
        "raw_schema": "jiangsu",
        "target_start": target_start.isoformat(),
        "target_end": target_end.isoformat(),
        "weight_history_cutoff": weight_cutoff.isoformat(),
        "aggregate_weight_policy": aggregate_weight_meta,
        "industry_segments": {
            segment: {
                "user_count": len(groups[segment]),
                "weight_policy": segment_weight_meta[segment],
            }
            for segment in groups
        },
        "minimum_industry_share": minimum_industry_share,
        "load_as_of_policy": "labels through D-3; full target dates",
        "weather_policy": "RESEARCH_PROXY_UNVERIFIED_ISSUE_TIME",
        "champion_eligible": False,
        "pii_output_check": "aggregate segment labels only",
    }
    return predictions, daily, summary, corrections, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-input", type=Path, required=True)
    parser.add_argument("--weather-input", type=Path, required=True)
    parser.add_argument("--weather-source", default="Ts_weather_1")
    parser.add_argument(
        "--target-start",
        type=date.fromisoformat,
        default=date(2026, 5, 1),
    )
    parser.add_argument(
        "--target-end",
        type=date.fromisoformat,
        default=date(2026, 7, 25),
    )
    parser.add_argument("--minimum-industry-share", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir)
    predictions, daily, summary, corrections, metadata = run_backtest(
        args.load_input,
        args.weather_input,
        weather_source=args.weather_source,
        target_start=args.target_start,
        target_end=args.target_end,
        minimum_industry_share=args.minimum_industry_share,
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
    summary.to_csv(
        args.output_dir / "summary_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    corrections.to_csv(
        args.output_dir / "segment_corrections.csv.gz",
        index=False,
        compression="gzip",
    )
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
