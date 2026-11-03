from datetime import date

import numpy as np
import pandas as pd

from project.research.jiangsu.user_load_forecast.pv_sensitive_segment_backtest import (
    apply_guard,
    build_strict_features,
    calendar_frame,
    component_curves,
    members_for_scope,
)


def test_component_curves_sum_to_portfolio() -> None:
    target_date = date(2026, 6, 15)
    portfolio = {target_date: np.full(96, 10.0)}
    users = {
        ("candidate", target_date): np.full(96, 2.0),
        ("other", target_date): np.full(96, 8.0),
    }

    candidate, remainder = component_curves(portfolio, users, {"candidate"})

    np.testing.assert_allclose(candidate[target_date], 2.0)
    np.testing.assert_allclose(remainder[target_date], 8.0)
    np.testing.assert_allclose(
        candidate[target_date] + remainder[target_date], portfolio[target_date]
    )


def test_calendar_frame_supports_an_earlier_operational_start() -> None:
    curves = {
        date(2026, 4, day): np.full(96, float(day))
        for day in range(1, 9)
    }

    result = calendar_frame(
        curves,
        model_name="full_cycle_test",
        target_start=date(2026, 4, 4),
    )

    assert result["target_date"].min() == date(2026, 4, 4)
    assert result["target_date"].max() == date(2026, 4, 8)
    assert result.groupby("target_date")["trade_interval"].nunique().eq(96).all()
    assert result.loc[
        result["target_date"].eq(date(2026, 4, 4)), "source_dates"
    ].eq("2026-04-01").all()


def test_daylight_guard_changes_only_intervals_25_to_72() -> None:
    combined = pd.DataFrame(
        {
            "trade_interval": [24, 25, 72, 73],
            "actual_load": [100.0] * 4,
            "core_prediction": [100.0] * 4,
            "segmented_prediction": [80.0] * 4,
        }
    )

    result = apply_guard(
        combined,
        model_name="test",
        alpha=0.5,
        guard="daylight",
    )

    np.testing.assert_allclose(result["predicted_load"], [100.0, 90.0, 90.0, 100.0])


def test_candidate_scope_includes_only_requested_evidence_tiers() -> None:
    candidates = pd.DataFrame(
        {
            "retail_user_name": ["direct", "high", "medium", "review"],
            "evidence_tier": [
                "DIRECT_EXPORT_SIGNATURE",
                "HIGH_STATISTICAL",
                "MEDIUM_STATISTICAL",
                "REVIEW_ONLY",
            ],
        }
    )

    assert members_for_scope(candidates, "direct_high") == {"direct", "high"}
    assert members_for_scope(candidates, "direct_high_medium") == {
        "direct",
        "high",
        "medium",
    }


def test_strict_features_never_read_target_day_da(tmp_path) -> None:
    target_date = date(2026, 5, 1)
    calendar_path = tmp_path / "calendar.csv"
    pd.DataFrame(
        {
            "model": ["calendar_similar_day_mean4"],
            "target_date": [target_date],
            "trade_interval": [1],
            "source_dates": ["2026-04-21"],
            "day_type": ["workday"],
            "actual_load": [100.0],
            "predicted_load": [95.0],
        }
    ).to_csv(calendar_path, index=False)
    market_path = tmp_path / "market.csv"
    rows = []
    for trade_date, da, rt in (
        ("2026-04-21", 900.0, 910.0),
        ("2026-04-24", 920.0, 930.0),
        ("2026-04-28", 940.0, 950.0),
        ("2026-04-29", 960.0, 970.0),
    ):
        for data_type, value in (("DA", da), ("RT", rt)):
            rows.append(
                {
                    "trade_date": trade_date,
                    "trade_interval": 1,
                    "data_name": "SYSTEM_LOAD_POWER",
                    "data_type": data_type,
                    "value": value,
                    "created_at": "2026-04-01 00:00:00",
                    "updated_at": "2026-04-01 00:00:00",
                }
            )
    # A conspicuous target-day DA row is present in the source but must not be read.
    rows.append(
        {
            "trade_date": target_date.isoformat(),
            "trade_interval": 1,
            "data_name": "SYSTEM_LOAD_POWER",
            "data_type": "DA",
            "value": 999999999.0,
            "created_at": "2026-05-01 00:00:00",
            "updated_at": "2026-05-01 00:00:00",
        }
    )
    pd.DataFrame(rows).to_csv(market_path, index=False)

    result = build_strict_features(calendar_path, market_path)

    assert not any(column.startswith("target_da_") for column in result.columns)
    assert result.loc[0, "market_source_max_date"] == date(2026, 4, 29)
    assert result.loc[0, "d2_rt_da_bias"] == 970.0 / 960.0


def test_strict_features_can_fall_back_only_to_an_earlier_market_date(tmp_path) -> None:
    target_date = date(2026, 5, 1)
    calendar_path = tmp_path / "calendar.csv"
    pd.DataFrame(
        {
            "model": ["calendar_similar_day_mean4"],
            "target_date": [target_date],
            "trade_interval": [1],
            "source_dates": ["2026-04-21"],
            "day_type": ["workday"],
            "actual_load": [100.0],
            "predicted_load": [95.0],
        }
    ).to_csv(calendar_path, index=False)
    market_path = tmp_path / "market.csv"
    rows = []
    for trade_date, da, rt in (
        ("2026-04-21", 900.0, 910.0),
        ("2026-04-24", 920.0, 930.0),
        ("2026-04-28", 940.0, 950.0),
        ("2026-04-29", 960.0, np.nan),
        ("2026-05-01", 999999999.0, 999999999.0),
    ):
        for data_type, value in (("DA", da), ("RT", rt)):
            rows.append(
                {
                    "trade_date": trade_date,
                    "trade_interval": 1,
                    "data_name": "SYSTEM_LOAD_POWER",
                    "data_type": data_type,
                    "value": value,
                    "created_at": "2026-04-01 00:00:00",
                    "updated_at": "2026-04-01 00:00:00",
                }
            )
    pd.DataFrame(rows).to_csv(market_path, index=False)

    result = build_strict_features(
        calendar_path,
        market_path,
        allow_causal_market_fallback=True,
    )

    assert result.loc[0, "market_d2_effective_date"] == date(2026, 4, 28)
    assert result.loc[0, "d2_market_age_days"] == 1.0
    assert result.loc[0, "market_source_max_date"] <= date(2026, 4, 29)
    assert result.loc[0, "d2_rt_da_bias"] == 950.0 / 940.0
