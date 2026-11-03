"""Read and aggregate Jiangsu retail-user actual-load snapshots."""

from __future__ import annotations

import csv
import gzip
import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "retailer_name",
    "retail_user_name",
    "account_id",
    "trade_date",
    "actual_values",
    "nonnull_count",
    "value_unit",
}
INTERVALS_PER_DAY = 96


@dataclass(frozen=True)
class AggregatedLoadDataset:
    """An anonymized company-level view plus source quality metadata."""

    curves: dict[date, np.ndarray]
    daily_account_count: dict[date, int]
    daily_user_count: dict[date, int]
    row_count: int
    distinct_account_count: int
    distinct_user_count: int
    distinct_retailer_count: int
    negative_value_count: int
    value_units: tuple[str, ...]
    source_sha256: str
    source_path: str

    @property
    def dates(self) -> list[date]:
        return sorted(self.curves)

    def validate_continuous_dates(self) -> None:
        dates = self.dates
        if not dates:
            raise ValueError("No load dates were parsed")
        expected = {
            dates[0] + timedelta(days=offset)
            for offset in range((dates[-1] - dates[0]).days + 1)
        }
        missing = sorted(expected - set(dates))
        if missing:
            raise ValueError(f"Missing calendar dates in load data: {missing}")

    def to_long_frame(self) -> pd.DataFrame:
        rows = []
        for trade_date in self.dates:
            for interval, value in enumerate(self.curves[trade_date], start=1):
                rows.append(
                    {
                        "trade_date": trade_date,
                        "trade_interval": interval,
                        "interval_end": interval_end_label(interval),
                        "actual_load": float(value),
                        "account_count": self.daily_account_count[trade_date],
                        "user_count": self.daily_user_count[trade_date],
                    }
                )
        return pd.DataFrame(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_text(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _parse_actual_values(raw: str, row_number: int) -> np.ndarray:
    value = raw.strip()
    if not (value.startswith("{") and value.endswith("}")):
        raise ValueError(f"Row {row_number}: actual_values is not a PostgreSQL array")
    parts = value[1:-1].split(",")
    if len(parts) != INTERVALS_PER_DAY:
        raise ValueError(
            f"Row {row_number}: expected 96 actual values, got {len(parts)}"
        )
    values = np.asarray([float(item) for item in parts], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Row {row_number}: actual_values contains NaN or Inf")
    return values


def read_aggregated_load_snapshot(path: Path) -> AggregatedLoadDataset:
    """Stream a server-sync CSV/CSV.GZ and aggregate to one 96-point curve/day."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    curves: dict[date, np.ndarray] = defaultdict(
        lambda: np.zeros(INTERVALS_PER_DAY, dtype=float)
    )
    daily_account_count: dict[date, int] = defaultdict(int)
    daily_users: dict[date, set[str]] = defaultdict(set)
    accounts: set[str] = set()
    users: set[str] = set()
    retailers: set[str] = set()
    value_units: set[str] = set()
    seen_account_days: set[tuple[date, str]] = set()
    negative_value_count = 0
    row_count = 0

    with _open_text(path) as handle:
        reader = csv.DictReader(handle)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing required source columns: {sorted(missing_columns)}")
        for row_number, row in enumerate(reader, start=2):
            trade_date = date.fromisoformat(row["trade_date"])
            account_id = row["account_id"].strip()
            retail_user_name = row["retail_user_name"].strip()
            if not account_id or not retail_user_name:
                raise ValueError(f"Row {row_number}: empty account or retail user")
            account_day = (trade_date, account_id)
            if account_day in seen_account_days:
                raise ValueError(f"Duplicate account/date source row: {account_day}")
            seen_account_days.add(account_day)

            values = _parse_actual_values(row["actual_values"], row_number)
            declared_nonnull = int(row["nonnull_count"])
            if declared_nonnull != INTERVALS_PER_DAY:
                raise ValueError(
                    f"Row {row_number}: nonnull_count={declared_nonnull}, expected 96"
                )
            curves[trade_date] += values
            daily_account_count[trade_date] += 1
            daily_users[trade_date].add(retail_user_name)
            accounts.add(account_id)
            users.add(retail_user_name)
            retailers.add(row["retailer_name"].strip())
            value_units.add(row["value_unit"].strip())
            negative_value_count += int(np.count_nonzero(values < 0))
            row_count += 1

    dataset = AggregatedLoadDataset(
        curves=dict(curves),
        daily_account_count=dict(daily_account_count),
        daily_user_count={key: len(value) for key, value in daily_users.items()},
        row_count=row_count,
        distinct_account_count=len(accounts),
        distinct_user_count=len(users),
        distinct_retailer_count=len(retailers),
        negative_value_count=negative_value_count,
        value_units=tuple(sorted(value_units)),
        source_sha256=sha256_file(path),
        source_path=str(path),
    )
    dataset.validate_continuous_dates()
    if any(not math.isfinite(float(curve.sum())) for curve in dataset.curves.values()):
        raise ValueError("Aggregated company curve contains a non-finite value")
    return dataset


def interval_end_label(interval: int) -> str:
    if not 1 <= interval <= INTERVALS_PER_DAY:
        raise ValueError(f"trade_interval must be 1..96, got {interval}")
    total_minutes = interval * 15
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"
