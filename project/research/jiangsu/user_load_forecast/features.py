"""Leakage-safe calendar, load-history, and optional weather features."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from project.research.jiangsu.user_load_forecast.calendar_features import (
    CALENDAR_VERSION,
    calendar_features,
)
from project.research.jiangsu.user_load_forecast.load_data import (
    AggregatedLoadDataset,
)
from project.research.jiangsu.user_load_forecast.weather_features import (
    WeatherFeatureStore,
)


FEATURE_VERSION = "surun_global_point_v1"
LOAD_LAGS = (3, 4, 7, 10, 14, 21, 28)
ROLL_WINDOWS = (3, 7, 14, 28)


def _curve_or_nan(
    curves: dict[date, np.ndarray],
    source_date: date,
) -> np.ndarray:
    value = curves.get(source_date)
    if value is None:
        return np.full(96, np.nan)
    return value


def _history_dates(
    dataset: AggregatedLoadDataset,
    cutoff: date,
    count: int,
) -> list[date]:
    return [item for item in dataset.dates if item <= cutoff][-count:]


def build_feature_frame(
    dataset: AggregatedLoadDataset,
    *,
    weather: WeatherFeatureStore | None = None,
) -> pd.DataFrame:
    """Build one global-model row for every date and 15-minute interval.

    Each sample date uses actual-load history only through ``sample_date-3``.
    The target actual value is retained solely as the training/evaluation
    label and is never copied into a feature column.
    """
    rows = []
    for sample_date in dataset.dates:
        latest_visible = sample_date - timedelta(days=3)
        source_dates: set[date] = set()
        lag_curves = {}
        for lag in LOAD_LAGS:
            source_date = sample_date - timedelta(days=lag)
            lag_curves[lag] = _curve_or_nan(dataset.curves, source_date)
            if source_date in dataset.curves:
                source_dates.add(source_date)

        roll_curves = {}
        roll_counts = {}
        for window in ROLL_WINDOWS:
            dates = _history_dates(dataset, latest_visible, window)
            roll_counts[window] = len(dates)
            if dates:
                source_dates.update(dates)
                stack = np.vstack([dataset.curves[item] for item in dates])
                roll_curves[(window, "mean")] = np.mean(stack, axis=0)
                roll_curves[(window, "std")] = np.std(stack, axis=0)
            else:
                roll_curves[(window, "mean")] = np.full(96, np.nan)
                roll_curves[(window, "std")] = np.full(96, np.nan)

        same_weekday_dates = [
            sample_date - timedelta(days=lag) for lag in (7, 14, 21, 28)
        ]
        same_weekday_dates = [
            item for item in same_weekday_dates if item in dataset.curves
        ]
        source_dates.update(same_weekday_dates)
        if same_weekday_dates:
            same_weekday_curve = np.mean(
                np.vstack([dataset.curves[item] for item in same_weekday_dates]),
                axis=0,
            )
        else:
            same_weekday_curve = np.full(96, np.nan)

        recent_dates = _history_dates(dataset, latest_visible, 7)
        recent_totals = np.asarray(
            [dataset.curves[item].sum() for item in recent_dates],
            dtype=float,
        )
        weekday_totals = np.asarray(
            [dataset.curves[item].sum() for item in same_weekday_dates],
            dtype=float,
        )
        recent_total_mean = (
            float(np.mean(recent_totals)) if len(recent_totals) else np.nan
        )
        same_weekday_total_mean = (
            float(np.mean(weekday_totals)) if len(weekday_totals) else np.nan
        )
        total_trend_ratio = (
            recent_total_mean / same_weekday_total_mean
            if same_weekday_total_mean and np.isfinite(same_weekday_total_mean)
            else np.nan
        )

        max_source_date = max(source_dates) if source_dates else None
        if max_source_date and max_source_date > latest_visible:
            raise AssertionError(
                f"Load feature leakage for {sample_date}: "
                f"{max_source_date} > {latest_visible}"
            )

        day_features = calendar_features(sample_date)
        weather_frame = (
            weather.features_for_day(sample_date)
            if weather is not None
            else pd.DataFrame({"trade_interval": range(1, 97)})
        )
        weather_by_interval = weather_frame.set_index("trade_interval")
        actual_curve = dataset.curves[sample_date]

        for interval in range(1, 97):
            idx = interval - 1
            interval_angle = 2 * np.pi * idx / 96
            interval_angle2 = 4 * np.pi * idx / 96
            row = {
                "trade_date": sample_date,
                "trade_interval": interval,
                "latest_visible_label_date": latest_visible,
                "max_load_feature_source_date": max_source_date,
                "actual_load": float(actual_curve[idx]),
                "interval_sin": float(np.sin(interval_angle)),
                "interval_cos": float(np.cos(interval_angle)),
                "interval_sin2": float(np.sin(interval_angle2)),
                "interval_cos2": float(np.cos(interval_angle2)),
                "same_weekday_available_count": float(len(same_weekday_dates)),
                "baseline_same_weekday_mean_4": float(same_weekday_curve[idx]),
                "recent_total_mean_7": recent_total_mean,
                "same_weekday_total_mean_4": same_weekday_total_mean,
                "total_trend_ratio": total_trend_ratio,
                **day_features,
            }
            for lag, curve in lag_curves.items():
                row[f"load_lag_{lag}d"] = float(curve[idx])
            for window in ROLL_WINDOWS:
                row[f"load_roll_{window}d_mean"] = float(
                    roll_curves[(window, "mean")][idx]
                )
                row[f"load_roll_{window}d_std"] = float(
                    roll_curves[(window, "std")][idx]
                )
                row[f"load_roll_{window}d_count"] = float(roll_counts[window])
            if weather is not None:
                weather_values = weather_by_interval.loc[interval].to_dict()
                row.update(
                    {
                        key: float(value) if pd.notna(value) else np.nan
                        for key, value in weather_values.items()
                    }
                )
                row["weather_temp_interval_sin"] = (
                    row["weather_temp_2m"] * row["interval_sin"]
                )
                row["weather_temp_interval_cos"] = (
                    row["weather_temp_2m"] * row["interval_cos"]
                )
                row["weather_cooling_degree"] = max(
                    row["weather_temp_2m"] - 26.0,
                    0.0,
                )
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.attrs["feature_version"] = FEATURE_VERSION
    frame.attrs["calendar_version"] = CALENDAR_VERSION
    frame.attrs["weather_policy"] = weather.policy if weather else "NONE"
    return frame


def model_feature_columns(frame: pd.DataFrame, *, include_weather: bool) -> list[str]:
    excluded = {
        "trade_date",
        "latest_visible_label_date",
        "max_load_feature_source_date",
        "actual_load",
    }
    columns = []
    for column in frame.columns:
        if column in excluded:
            continue
        if not include_weather and column.startswith("weather_"):
            continue
        columns.append(column)
    return columns
