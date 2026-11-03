"""City-resolved weather loading and leakage-safe load-weighted aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from project.research.jiangsu.user_load_forecast.load_data import sha256_file
from project.research.jiangsu.user_load_forecast.user_segmentation import (
    JIANGSU_CITIES,
)
from project.research.jiangsu.user_load_forecast.weather_features import (
    WEATHER_COLUMNS,
    WEATHER_POLICY_PROXY,
    WeatherFeatureStore,
)


REGIONAL_WEATHER_VERSION = "jiangsu_city_weighted_weather_v1"


@dataclass(frozen=True)
class CityWeatherPanel:
    frame: pd.DataFrame
    source: str
    source_path: str
    source_sha256: str

    @property
    def cities(self) -> tuple[str, ...]:
        return tuple(sorted(self.frame["city"].unique()))


def read_city_weather_panel(
    path: Path,
    *,
    source: str = "Ts_weather_1",
) -> CityWeatherPanel:
    path = path.resolve()
    usecols = ["city", "source", "timestamp", *WEATHER_COLUMNS]
    frame = pd.read_csv(path, usecols=usecols)
    frame = frame.loc[frame["source"].eq(source)].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    for column in WEATHER_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    duplicates = frame.duplicated(["city", "timestamp"])
    if duplicates.any():
        raise ValueError(
            f"Duplicate city/timestamp weather rows: {int(duplicates.sum())}"
        )
    missing_cities = sorted(set(JIANGSU_CITIES) - set(frame["city"].unique()))
    if missing_cities:
        raise ValueError(f"Weather input is missing cities: {missing_cities}")
    return CityWeatherPanel(
        frame=frame,
        source=source,
        source_path=str(path),
        source_sha256=sha256_file(path),
    )


def weighted_weather_store(
    panel: CityWeatherPanel,
    *,
    city_weights: dict[str, float],
    variant: str,
) -> WeatherFeatureStore:
    """Aggregate city weather with fixed, externally supplied city weights."""
    missing = sorted(set(JIANGSU_CITIES) - set(city_weights))
    extra = sorted(set(city_weights) - set(JIANGSU_CITIES))
    if missing or extra:
        raise ValueError(f"Invalid city weights: missing={missing}, extra={extra}")
    weight_sum = float(sum(city_weights.values()))
    if not np.isclose(weight_sum, 1.0, atol=1e-9):
        raise ValueError(f"City weights must sum to 1, got {weight_sum}")
    if any(value < 0 for value in city_weights.values()):
        raise ValueError("City weights must be nonnegative")

    frame = panel.frame.copy()
    frame["_weight"] = frame["city"].map(city_weights).astype(float)
    rows = []
    for timestamp, group in frame.groupby("timestamp", sort=True):
        row = {"timestamp": timestamp}
        for column in WEATHER_COLUMNS:
            values = group[column].to_numpy(dtype=float)
            weights = group["_weight"].to_numpy(dtype=float)
            usable = np.isfinite(values) & np.isfinite(weights)
            available_weight = weights[usable].sum()
            row[column] = (
                float(np.dot(values[usable], weights[usable]) / available_weight)
                if available_weight > 0
                else np.nan
            )
        rows.append(row)
    hourly = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return WeatherFeatureStore(
        hourly=hourly,
        source=f"{panel.source}:{variant}",
        policy=WEATHER_POLICY_PROXY,
        source_path=panel.source_path,
        source_sha256=panel.source_sha256,
        snapshots=None,
    )


def daily_weighted_weather(
    panel: CityWeatherPanel,
    *,
    city_weights: dict[str, float],
) -> pd.DataFrame:
    """Return daily weighted weather for aggregate sensitivity diagnostics."""
    store = weighted_weather_store(
        panel,
        city_weights=city_weights,
        variant="diagnostic",
    )
    hourly = store.hourly.reset_index()
    hourly["trade_date"] = hourly["timestamp"].dt.date
    daily = (
        hourly.groupby("trade_date", as_index=False)
        .agg(
            temp_mean=("temp_2m", "mean"),
            temp_max=("temp_2m", "max"),
            humidity_mean=("humidity_2m", "mean"),
            rain_total=("rain", "sum"),
            radiation_mean=("radiation", "mean"),
            wind_mean=("wind_10m", "mean"),
        )
    )
    daily["cooling_degree"] = (daily["temp_mean"] - 26.0).clip(lower=0.0)
    return daily
