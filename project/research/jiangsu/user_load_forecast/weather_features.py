"""Weather feature loading with explicit historical visibility status."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


WEATHER_POLICY_STRICT = "STRICT_FORECAST_SNAPSHOT"
WEATHER_POLICY_PROXY = "RESEARCH_PROXY_UNVERIFIED_ISSUE_TIME"
WEATHER_FEATURE_VERSION = "jiangsu_13city_hourly_v1"
WEATHER_COLUMNS = (
    "temp_2m",
    "humidity_2m",
    "rain",
    "radiation",
    "wind_10m",
)


@dataclass(frozen=True)
class WeatherFeatureStore:
    """Province-mean hourly weather indexed by the hour-ending timestamp."""

    hourly: pd.DataFrame
    source: str
    policy: str
    source_path: str
    source_sha256: str
    snapshots: pd.DataFrame | None = None

    @property
    def is_champion_eligible(self) -> bool:
        return self.policy == WEATHER_POLICY_STRICT

    def _hourly_for_day(self, target_day: pd.Timestamp) -> pd.DataFrame:
        if self.policy != WEATHER_POLICY_STRICT:
            return self.hourly
        if self.snapshots is None:
            raise ValueError("Strict weather store has no versioned snapshots")
        run_day = target_day.date() - timedelta(days=1)
        forecast_as_of = pd.Timestamp(
            datetime.combine(
                run_day,
                time(8, 45),
                tzinfo=ZoneInfo("Asia/Shanghai"),
            )
        ).tz_convert("UTC")
        snapshots = self.snapshots.loc[
            self.snapshots["available_at"] <= forecast_as_of
        ].copy()
        if snapshots.empty:
            return pd.DataFrame(columns=WEATHER_COLUMNS)
        target_start = target_day.tz_localize("Asia/Shanghai").tz_convert("UTC")
        target_end = target_start + pd.Timedelta(days=1)
        snapshots = snapshots.loc[
            (snapshots["forecast_valid_at"] > target_start)
            & (snapshots["forecast_valid_at"] <= target_end)
        ].copy()
        snapshots = (
            snapshots.sort_values(
                ["city", "forecast_valid_at", "forecast_issue_at", "available_at"]
            )
            .groupby(["city", "forecast_valid_at"], as_index=False)
            .tail(1)
        )
        if snapshots.empty:
            return pd.DataFrame(columns=WEATHER_COLUMNS)
        valid_local = snapshots["forecast_valid_at"].dt.tz_convert(
            "Asia/Shanghai"
        ).dt.tz_localize(None)
        snapshots["timestamp"] = valid_local
        return (
            snapshots.groupby("timestamp", as_index=True)[list(WEATHER_COLUMNS)]
            .mean()
            .sort_index()
        )

    def features_for_day(self, day: pd.Timestamp | str) -> pd.DataFrame:
        """Return 96 point-level and repeated daily weather features."""
        target_day = pd.Timestamp(day).normalize()
        hourly = self._hourly_for_day(target_day)
        rows = []
        for interval in range(1, 97):
            hour_end = (interval + 3) // 4
            timestamp = target_day + pd.Timedelta(hours=hour_end)
            if timestamp in hourly.index:
                values = hourly.loc[timestamp]
                if isinstance(values, pd.DataFrame):
                    values = values.iloc[-1]
            else:
                values = pd.Series({column: np.nan for column in WEATHER_COLUMNS})
            rows.append(
                {
                    "trade_interval": interval,
                    **{
                        f"weather_{column}": float(values.get(column, np.nan))
                        for column in WEATHER_COLUMNS
                    },
                }
            )
        frame = pd.DataFrame(rows)
        daily = {
            "weather_temp_mean": frame["weather_temp_2m"].mean(),
            "weather_temp_max": frame["weather_temp_2m"].max(),
            "weather_temp_min": frame["weather_temp_2m"].min(),
            "weather_temp_range": (
                frame["weather_temp_2m"].max() - frame["weather_temp_2m"].min()
            ),
            "weather_humidity_mean": frame["weather_humidity_2m"].mean(),
            "weather_humidity_max": frame["weather_humidity_2m"].max(),
            # Hourly values repeat four times in the 96-point frame.
            "weather_rain_daily": frame["weather_rain"].sum(min_count=1) / 4,
            "weather_radiation_mean": frame["weather_radiation"].mean(),
            "weather_radiation_max": frame["weather_radiation"].max(),
            "weather_wind_mean": frame["weather_wind_10m"].mean(),
            "weather_wind_max": frame["weather_wind_10m"].max(),
        }
        for key, value in daily.items():
            frame[key] = value
        frame["weather_high_temp"] = (
            frame["weather_temp_max"] >= 35.0
        ).astype(float)
        frame["weather_muggy"] = (
            (frame["weather_temp_2m"] >= 30.0)
            & (frame["weather_humidity_2m"] >= 70.0)
        ).astype(float)
        frame["weather_heavy_rain"] = (
            frame["weather_rain_daily"] >= 25.0
        ).astype(float)
        return frame


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_weather_snapshot(
    path: Path,
    *,
    source: str = "Ts_weather_1",
    policy: str = WEATHER_POLICY_PROXY,
    chunksize: int = 100_000,
) -> WeatherFeatureStore:
    """Read a Jiangsu weather snapshot without inventing missing issue times.

    The current server-sync table has ``created_at`` and ``updated_at`` but no
    historical forecast issue/availability timestamp. Strict mode therefore
    rejects it. Proxy mode is permitted only for research diagnostics and is
    explicitly ineligible for Champion promotion.
    """
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    strict_columns = {"forecast_issue_at", "available_at", "forecast_valid_at"}
    if policy == WEATHER_POLICY_STRICT and not strict_columns.issubset(header):
        missing = sorted(strict_columns - set(header))
        raise ValueError(
            "Strict weather backtest requires historical forecast snapshots; "
            f"missing columns={missing}"
        )
    if policy not in {WEATHER_POLICY_STRICT, WEATHER_POLICY_PROXY}:
        raise ValueError(f"Unsupported weather policy: {policy}")

    if policy == WEATHER_POLICY_STRICT:
        strict_usecols = [
            "city",
            "source",
            "forecast_valid_at",
            "forecast_issue_at",
            "available_at",
            *WEATHER_COLUMNS,
        ]
        snapshots = pd.read_csv(path, usecols=strict_usecols)
        snapshots = snapshots.loc[snapshots["source"].eq(source)].copy()
        for column in ("forecast_valid_at", "forecast_issue_at", "available_at"):
            snapshots[column] = pd.to_datetime(
                snapshots[column],
                utc=True,
                errors="raise",
            )
        for column in WEATHER_COLUMNS:
            snapshots[column] = pd.to_numeric(
                snapshots[column],
                errors="coerce",
            )
        return WeatherFeatureStore(
            hourly=pd.DataFrame(columns=WEATHER_COLUMNS),
            source=source,
            policy=policy,
            source_path=str(path),
            source_sha256=_sha256_file(path),
            snapshots=snapshots,
        )

    usecols = ["city", "source", "timestamp", *WEATHER_COLUMNS]
    frames = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk = chunk.loc[chunk["source"].eq(source)].copy()
        if chunk.empty:
            continue
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])
        for column in WEATHER_COLUMNS:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        # Preserve city rows until all file chunks are concatenated.  Averaging
        # within each chunk and then averaging those chunk means would weight
        # chunks equally rather than cities equally when a timestamp is split
        # across chunks or when the file is ordered by city.
        frames.append(chunk[["timestamp", *WEATHER_COLUMNS]])
    if not frames:
        raise ValueError(f"No weather rows found for source={source!r}")
    hourly = (
        pd.concat(frames, ignore_index=True)
        .groupby("timestamp", as_index=True)[list(WEATHER_COLUMNS)]
        .mean()
        .sort_index()
    )
    return WeatherFeatureStore(
        hourly=hourly,
        source=source,
        policy=policy,
        source_path=str(path),
        source_sha256=_sha256_file(path),
        snapshots=None,
    )
