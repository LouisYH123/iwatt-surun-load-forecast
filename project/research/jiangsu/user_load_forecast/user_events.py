"""Known-ahead user event adjustments for portfolio load forecasts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


EVENT_CONTRACT_VERSION = "surun_user_event_v1"
ALLOWED_EVENTS = {"ADD", "UNBIND", "PAUSE", "RESUME"}
POSITIVE_EVENTS = {"ADD", "RESUME"}
NEGATIVE_EVENTS = {"UNBIND", "PAUSE"}


@dataclass(frozen=True)
class UserEvent:
    event_id: str
    anonymous_user_id: str
    event_type: str
    effective_date: date
    known_at: datetime
    estimated_curve: np.ndarray
    confidence: float


def _parse_curve(raw: str) -> np.ndarray:
    value = raw.strip()
    if not (value.startswith("{") and value.endswith("}")):
        raise ValueError("estimated_values must be a PostgreSQL-style 96-point array")
    curve = np.asarray([float(item) for item in value[1:-1].split(",")])
    if len(curve) != 96 or not np.isfinite(curve).all():
        raise ValueError("estimated_values must contain 96 finite values")
    return curve


def read_user_events(path: Path) -> list[UserEvent]:
    events = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            event_type = row["event_type"].strip().upper()
            if event_type not in ALLOWED_EVENTS:
                raise ValueError(f"Unsupported user event type: {event_type}")
            known_at = datetime.fromisoformat(row["known_at"])
            if known_at.tzinfo is None:
                raise ValueError("known_at must include a timezone offset")
            events.append(
                UserEvent(
                    event_id=row["event_id"].strip(),
                    anonymous_user_id=row["anonymous_user_id"].strip(),
                    event_type=event_type,
                    effective_date=date.fromisoformat(row["effective_date"]),
                    known_at=known_at,
                    estimated_curve=_parse_curve(row["estimated_values"]),
                    confidence=float(row.get("confidence") or 1.0),
                )
            )
    return events


def visible_events_for_target(
    events: list[UserEvent],
    target_date: date,
    *,
    forecast_as_of_time: time = time(8, 45),
) -> list[UserEvent]:
    forecast_as_of = datetime.combine(
        target_date - timedelta(days=1),
        forecast_as_of_time,
        tzinfo=ZoneInfo("Asia/Shanghai"),
    )
    return [
        event
        for event in events
        if event.known_at <= forecast_as_of
        and event.effective_date <= target_date
    ]


def apply_user_event_adjustments(
    base_prediction: np.ndarray,
    events: list[UserEvent],
    target_date: date,
) -> tuple[np.ndarray, dict]:
    """Apply only events that were known before the target-day declaration."""
    visible = visible_events_for_target(events, target_date)
    adjustment = np.zeros(96, dtype=float)
    for event in visible:
        signed = event.estimated_curve * event.confidence
        if event.event_type in POSITIVE_EVENTS:
            adjustment += signed
        elif event.event_type in NEGATIVE_EVENTS:
            adjustment -= signed
    final = np.clip(np.asarray(base_prediction, dtype=float) + adjustment, 0.0, None)
    return final, {
        "visible_event_count": len(visible),
        "positive_event_count": sum(
            event.event_type in POSITIVE_EVENTS for event in visible
        ),
        "negative_event_count": sum(
            event.event_type in NEGATIVE_EVENTS for event in visible
        ),
        "adjustment_total": float(adjustment.sum()),
    }
