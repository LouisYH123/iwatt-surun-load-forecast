"""Time and visibility contracts for Jiangsu D+1 portfolio load forecasts."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd


TIMEZONE = ZoneInfo("Asia/Shanghai")
LATENCY_POLICY = "SIMULATED_T_PLUS_2"
NORMAL_LABEL_AGE_DAYS = 3


def latency_level(label_age_days: int) -> str:
    """Return the operating level associated with a target-to-label age."""
    if label_age_days <= 3:
        return "L0_NORMAL"
    if label_age_days == 4:
        return "L1_LIGHT_DELAY"
    if label_age_days <= 6:
        return "L2_MATERIAL_DELAY"
    return "L3_SEVERE_DELAY"


def build_availability_ledger(
    target_dates: list[date],
    observed_dates: set[date],
    *,
    forecast_as_of_time: time = time(8, 45),
    extra_delay_days: int = 0,
) -> pd.DataFrame:
    """Build a simulated T+2 availability ledger for historical replay.

    For a target date ``D``, the run date is ``R=D-1``. Under normal T+2
    publication, the latest legal actual-load date is ``D-3``. Additional
    delay scenarios move that cap further back without pretending that an
    unavailable label exists.
    """
    if extra_delay_days < 0:
        raise ValueError("extra_delay_days must be non-negative")
    if not observed_dates:
        raise ValueError("observed_dates must not be empty")

    sorted_observed = sorted(observed_dates)
    rows: list[dict] = []
    for target_date in sorted(target_dates):
        run_date = target_date - timedelta(days=1)
        theoretical_latest = target_date - timedelta(days=NORMAL_LABEL_AGE_DAYS)
        scenario_cap = theoretical_latest - timedelta(days=extra_delay_days)
        eligible = [item for item in sorted_observed if item <= scenario_cap]
        if not eligible:
            raise ValueError(
                f"No visible load label for target_date={target_date} "
                f"under cap={scenario_cap}"
            )
        latest_available = eligible[-1]
        label_age_days = (target_date - latest_available).days
        forecast_as_of = datetime.combine(
            run_date,
            forecast_as_of_time,
            tzinfo=TIMEZONE,
        )
        rows.append(
            {
                "run_date": run_date,
                "target_date": target_date,
                "forecast_as_of": forecast_as_of.isoformat(),
                "theoretical_latest_label_date": theoretical_latest,
                "scenario_label_date_cap": scenario_cap,
                "latest_available_label_date": latest_available,
                "label_age_days": label_age_days,
                "latency_level": latency_level(label_age_days),
                "latency_policy": LATENCY_POLICY,
                "extra_delay_days": extra_delay_days,
            }
        )
    return pd.DataFrame(rows)


def validate_visible_source_dates(
    *,
    target_date: date,
    latest_available_label_date: date,
    source_dates: list[date],
) -> None:
    """Reject a forecast if any actual-load feature exceeds its visible cutoff."""
    if not source_dates:
        raise ValueError("At least one source date is required")
    future_or_target = [item for item in source_dates if item >= target_date]
    if future_or_target:
        raise ValueError(
            f"Target/future actual load used for {target_date}: {future_or_target}"
        )
    invisible = [
        item for item in source_dates if item > latest_available_label_date
    ]
    if invisible:
        raise ValueError(
            "Actual-load leakage: "
            f"target={target_date}, latest_available={latest_available_label_date}, "
            f"invisible_sources={invisible}"
        )
