"""Versioned Chinese calendar features used by the Surun backtest."""

from __future__ import annotations

from datetime import date, timedelta


CALENDAR_VERSION = "cn_official_2026_v1"

# The current load history spans April-July 2026. Full-year entries are kept so
# future rolling runs do not silently treat later public holidays as weekdays.
HOLIDAY_PERIODS = (
    (date(2026, 1, 1), date(2026, 1, 3), "new_year"),
    (date(2026, 2, 15), date(2026, 2, 23), "spring_festival"),
    (date(2026, 4, 4), date(2026, 4, 6), "qingming"),
    (date(2026, 5, 1), date(2026, 5, 5), "labour_day"),
    (date(2026, 6, 19), date(2026, 6, 21), "dragon_boat"),
    (date(2026, 9, 25), date(2026, 9, 27), "mid_autumn"),
    (date(2026, 10, 1), date(2026, 10, 7), "national_day"),
)

ADJUSTED_WORKDAYS = {
    date(2026, 2, 14),
    date(2026, 3, 1),
    date(2026, 4, 26),
    date(2026, 5, 9),
    date(2026, 9, 20),
    date(2026, 10, 10),
}

HOLIDAY_DATES = {
    start + timedelta(days=offset)
    for start, end, _ in HOLIDAY_PERIODS
    for offset in range((end - start).days + 1)
}


def _holiday_period(day: date) -> tuple[date, date, str] | None:
    for start, end, name in HOLIDAY_PERIODS:
        if start <= day <= end:
            return start, end, name
    return None


def day_type(day: date) -> str:
    if day in ADJUSTED_WORKDAYS:
        return "adjusted_workday"
    if day in HOLIDAY_DATES:
        return "holiday"
    if day.weekday() >= 5:
        return "weekend"
    return "workday"


def calendar_features(day: date) -> dict[str, float]:
    """Return numeric calendar features known before the forecast is issued."""
    import numpy as np

    period = _holiday_period(day)
    kind = day_type(day)
    nearest_distance = min(
        (abs((holiday - day).days) for holiday in HOLIDAY_DATES),
        default=366,
    )
    if period:
        start, end, _ = period
        holiday_index = (day - start).days + 1
        holiday_length = (end - start).days + 1
    else:
        holiday_index = 0
        holiday_length = 0

    weekday_angle = 2 * np.pi * day.weekday() / 7
    year_angle = 2 * np.pi * (day.timetuple().tm_yday - 1) / 366
    month_angle = 2 * np.pi * (day.month - 1) / 12
    return {
        "weekday": float(day.weekday()),
        "weekday_sin": float(np.sin(weekday_angle)),
        "weekday_cos": float(np.cos(weekday_angle)),
        "month": float(day.month),
        "month_sin": float(np.sin(month_angle)),
        "month_cos": float(np.cos(month_angle)),
        "day_of_year_sin": float(np.sin(year_angle)),
        "day_of_year_cos": float(np.cos(year_angle)),
        "is_workday": float(kind in {"workday", "adjusted_workday"}),
        "is_weekend": float(kind == "weekend"),
        "is_holiday": float(kind == "holiday"),
        "is_adjusted_workday": float(kind == "adjusted_workday"),
        "is_pre_holiday": float(day + timedelta(days=1) in HOLIDAY_DATES),
        "is_post_holiday": float(day - timedelta(days=1) in HOLIDAY_DATES),
        "holiday_day_index": float(holiday_index),
        "holiday_length": float(holiday_length),
        "distance_to_holiday": float(min(nearest_distance, 30)),
    }
