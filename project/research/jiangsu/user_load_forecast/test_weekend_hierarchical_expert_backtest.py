from datetime import date

import numpy as np
import pandas as pd

from project.research.jiangsu.user_load_forecast.strict_weekend_hierarchical_expert_backtest import (
    attach_same_weekday_recency,
    gate_mask,
)


def test_same_weekday_recency_respects_d_minus_three_cutoff() -> None:
    target = date(2026, 4, 18)
    frame = pd.DataFrame(
        {
            "target_date": [target] * 96,
            "trade_interval": np.arange(1, 97),
            "calendar_prediction": np.full(96, 20.0),
        }
    )
    curves = {
        date(2026, 4, 4): np.full(96, 10.0),
        date(2026, 4, 11): np.full(96, 15.0),
        date(2026, 4, 18): np.full(96, 999999.0),
    }

    result = attach_same_weekday_recency(frame, curves)

    np.testing.assert_allclose(result["recent_same_weekday_load"], 15.0)
    np.testing.assert_allclose(result["previous_same_weekday_load"], 10.0)
    np.testing.assert_allclose(result["recent_same_weekday_age_days"], 7.0)
    assert result["recent_same_weekday_load"].max() < 999999.0


def test_weekend_gate_never_changes_holiday_or_workday() -> None:
    frame = pd.DataFrame(
        {
            "day_type": ["weekend", "weekend", "holiday", "workday"],
            "trade_interval": [32, 60, 40, 40],
        }
    )

    assert gate_mask(frame, "weekend_morning_25_52").tolist() == [
        True,
        False,
        False,
        False,
    ]
